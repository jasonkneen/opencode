package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
	"github.com/opencode-ai/opencode/internal/config"
	"github.com/opencode-ai/opencode/internal/llm/models"
	"github.com/opencode-ai/opencode/internal/llm/tools"
	"bytes" // Added for http.NewRequest
	"net/http" // Added for manual HTTP client usage

	"github.com/opencode-ai/opencode/internal/logging"
	"github.com/opencode-ai/opencode/internal/message"
)

// --- Structs for /v1/responses API ---

type OpenAIResponsesInputItem struct {
	Type string `json:"type"` // "text", "image_url", etc.
	Text string `json:"text,omitempty"`
	// TODO: Add ImageURL OpenAIResponsesImageURL `json:"image_url,omitempty"` for future multi-modal
}

// type OpenAIResponsesImageURL struct {
// 	URL string `json:"url"`
// 	Detail string `json:"detail,omitempty"` // "auto", "low", "high"
// }

type OpenAIResponsesRequest struct {
	Model           string                     `json:"model"`
	Input           []OpenAIResponsesInputItem `json:"input"`
	Instructions    string                     `json:"instructions,omitempty"`
	MaxOutputTokens *int                       `json:"max_output_tokens,omitempty"`
	Stream          bool                       `json:"stream,omitempty"`
	// TODO: Add other params like Temperature, TopP, Tools, ToolChoice if supported by /v1/responses
	// For now, focusing on text generation for codex-mini-latest
}

type OpenAIResponsesResponseTextContent struct {
	Type string `json:"type"` // "output_text"
	Text string `json:"text"`
}

type OpenAIResponsesResponseOutputItem struct {
	Type    string                               `json:"type"` // "message" - this might vary, need to confirm from actual API response
	Role    string                               `json:"role"` // "assistant" - this might vary
	Content []OpenAIResponsesResponseTextContent `json:"content"`
}

// OpenAIUsageData mirrors openai.Usage but is defined here in case of subtle differences
// or if the library's Usage type isn't directly applicable for /v1/responses.
// For now, assuming it's similar to `openai.Usage`. If not, fields will be adjusted.
type OpenAIUsageData struct {
	InputTokens    int `json:"input_tokens"`
	OutputTokens   int `json:"output_tokens"`
	TotalTokens    int `json:"total_tokens"` // May not be provided by /v1/responses
	CachedTokens   int `json:"cached_tokens,omitempty"` // Specific to some OpenAI APIs
	BilledTokens   int `json:"billed_tokens,omitempty"` // Specific to some OpenAI APIs
}


type OpenAIResponsesResponse struct {
	ID     string                              `json:"id"`
	Object string                              `json:"object"` // e.g., "response", "response.chunk" for streaming
	Output []OpenAIResponsesResponseOutputItem `json:"output"`
	Usage  *OpenAIUsageData                    `json:"usage,omitempty"`
	Error  *struct { // Simplified error structure, can be expanded
		Code    string `json:"code"`
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
	// Status string `json:"status,omitempty"` // e.g. "completed", "streaming", "error"
}

// For streaming delta
type OpenAIResponsesStreamDelta struct {
	Type string `json:"type"` // "output_text.delta"
	Text string `json:"text"`
}

type OpenAIResponsesStreamOutputDelta struct {
	Type  string                       `json:"type"` // "output.delta"
	Delta OpenAIResponsesStreamDelta `json:"delta"`
}

type OpenAIResponsesStreamChunk struct {
	ID     string                           `json:"id"`
	Object string                           `json:"object"` // "response.chunk"
	Output []OpenAIResponsesStreamOutputDelta `json:"output"`
	Usage  *OpenAIUsageData                 `json:"usage,omitempty"` // Often in the last chunk
}

// --- End of Structs for /v1/responses API ---

type openaiOptions struct {
	baseURL         string
	disableCache    bool
	reasoningEffort string
	extraHeaders    map[string]string
}

type OpenAIOption func(*openaiOptions)

type openaiClient struct {
	providerOptions providerClientOptions
	options         openaiOptions
	client          openai.Client
}

type OpenAIClient ProviderClient

func newOpenAIClient(opts providerClientOptions) OpenAIClient {
	openaiOpts := openaiOptions{
		reasoningEffort: "medium",
	}
	for _, o := range opts.openaiOptions {
		o(&openaiOpts)
	}

	openaiClientOptions := []option.RequestOption{}
	if opts.apiKey != "" {
		openaiClientOptions = append(openaiClientOptions, option.WithAPIKey(opts.apiKey))
	}
	if openaiOpts.baseURL != "" {
		openaiClientOptions = append(openaiClientOptions, option.WithBaseURL(openaiOpts.baseURL))
	}

	if openaiOpts.extraHeaders != nil {
		for key, value := range openaiOpts.extraHeaders {
			openaiClientOptions = append(openaiClientOptions, option.WithHeader(key, value))
		}
	}

	client := openai.NewClient(openaiClientOptions...)
	return &openaiClient{
		providerOptions: opts,
		options:         openaiOpts,
		client:          client,
	}
}

func (o *openaiClient) convertMessages(messages []message.Message) (openaiMessages []openai.ChatCompletionMessageParamUnion) {
	// Add system message first
	openaiMessages = append(openaiMessages, openai.SystemMessage(o.providerOptions.systemMessage))

	for _, msg := range messages {
		switch msg.Role {
		case message.User:
			var content []openai.ChatCompletionContentPartUnionParam
			textBlock := openai.ChatCompletionContentPartTextParam{Text: msg.Content().String()}
			content = append(content, openai.ChatCompletionContentPartUnionParam{OfText: &textBlock})
			for _, binaryContent := range msg.BinaryContent() {
				imageURL := openai.ChatCompletionContentPartImageImageURLParam{URL: binaryContent.String(models.ProviderOpenAI)}
				imageBlock := openai.ChatCompletionContentPartImageParam{ImageURL: imageURL}

				content = append(content, openai.ChatCompletionContentPartUnionParam{OfImageURL: &imageBlock})
			}

			openaiMessages = append(openaiMessages, openai.UserMessage(content))

		case message.Assistant:
			assistantMsg := openai.ChatCompletionAssistantMessageParam{
				Role: "assistant",
			}

			if msg.Content().String() != "" {
				assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(msg.Content().String()),
				}
			}

			if len(msg.ToolCalls()) > 0 {
				assistantMsg.ToolCalls = make([]openai.ChatCompletionMessageToolCallParam, len(msg.ToolCalls()))
				for i, call := range msg.ToolCalls() {
					assistantMsg.ToolCalls[i] = openai.ChatCompletionMessageToolCallParam{
						ID:   call.ID,
						Type: "function",
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      call.Name,
							Arguments: call.Input,
						},
					}
				}
			}

			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{
				OfAssistant: &assistantMsg,
			})

		case message.Tool:
			for _, result := range msg.ToolResults() {
				openaiMessages = append(openaiMessages,
					openai.ToolMessage(result.Content, result.ToolCallID),
				)
			}
		}
	}

	return
}

// buildOpenAIResponsesInput constructs the input and instructions for the /v1/responses API.
// System messages are used as instructions. User messages are concatenated into text inputs.
// Assistant and Tool messages are currently ignored for this API.
func (o *openaiClient) buildOpenAIResponsesInput(messages []message.Message) ([]OpenAIResponsesInputItem, string) {
	var inputs []OpenAIResponsesInputItem
	var instructions strings.Builder
	var userContentBuilder strings.Builder

	if o.providerOptions.systemMessage != "" {
		instructions.WriteString(o.providerOptions.systemMessage)
	}

	for _, msg := range messages {
		switch msg.Role {
		case message.System: // System messages from history could also be appended to instructions
			if instructions.Len() > 0 {
				instructions.WriteString("\n\n") // Separate multiple instruction parts
			}
			instructions.WriteString(msg.Content().String())
		case message.User:
			// Concatenate user messages, as /v1/responses might prefer a single input block or handle multiple
			// For now, let's try concatenating them into one item.
			// Alternatively, each user message could be a separate OpenAIResponsesInputItem.
			// The API might be flexible here. Starting with concatenation.
			if userContentBuilder.Len() > 0 {
				userContentBuilder.WriteString("\n\n") // Separate user messages
			}
			userContentBuilder.WriteString(msg.Content().String())
			// TODO: Handle binary content if /v1/responses supports it for user messages
		case message.Assistant, message.Tool:
			// Currently ignoring assistant and tool messages for /v1/responses input.
			// This API might have a different way to handle conversation history or tool interactions.
		}
	}

	if userContentBuilder.Len() > 0 {
		inputs = append(inputs, OpenAIResponsesInputItem{
			Type: "text",
			Text: userContentBuilder.String(),
		})
	}

	return inputs, instructions.String()
}

// constructLegacyPrompt builds a single string prompt for legacy completion models.
// It concatenates the system message (if any) and all user messages.
// Other message types (assistant, tool) are ignored for legacy completion.
func (o *openaiClient) constructLegacyPrompt(messages []message.Message) string {
	var promptBuilder strings.Builder

	// Prepend system message
	if o.providerOptions.systemMessage != "" {
		promptBuilder.WriteString(o.providerOptions.systemMessage)
		promptBuilder.WriteString("\n\n") // Add separation
	}

	for _, msg := range messages {
		if msg.Role == message.User {
			promptBuilder.WriteString(msg.Content().String())
			promptBuilder.WriteString("\n") // Add separation between user messages
		}
	}
	return promptBuilder.String()
}

func (o *openaiClient) convertTools(tools []tools.BaseTool) []openai.ChatCompletionToolParam {
	openaiTools := make([]openai.ChatCompletionToolParam, len(tools))

	for i, tool := range tools {
		info := tool.Info()
		openaiTools[i] = openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        info.Name,
				Description: openai.String(info.Description),
				Parameters: openai.FunctionParameters{
					"type":       "object",
					"properties": info.Parameters,
					"required":   info.Required,
				},
			},
		}
	}

	return openaiTools
}

func (o *openaiClient) finishReason(reason string) message.FinishReason {
	switch reason {
	case "stop":
		return message.FinishReasonEndTurn
	case "length":
		return message.FinishReasonMaxTokens
	case "tool_calls":
		return message.FinishReasonToolUse
	default:
		return message.FinishReasonUnknown
	}
}

func (o *openaiClient) preparedParams(messages []openai.ChatCompletionMessageParamUnion, tools []openai.ChatCompletionToolParam) openai.ChatCompletionNewParams {
	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(o.providerOptions.model.APIModel),
		Messages: messages,
		Tools:    tools,
	}

	if o.providerOptions.model.CanReason == true {
		params.MaxCompletionTokens = openai.Int(o.providerOptions.maxTokens)
		switch o.options.reasoningEffort {
		case "low":
			params.ReasoningEffort = shared.ReasoningEffortLow
		case "medium":
			params.ReasoningEffort = shared.ReasoningEffortMedium
		case "high":
			params.ReasoningEffort = shared.ReasoningEffortHigh
		default:
			params.ReasoningEffort = shared.ReasoningEffortMedium
		}
	} else {
		params.MaxTokens = openai.Int(o.providerOptions.maxTokens)
	}

	return params
}

func (o *openaiClient) send(ctx context.Context, messages []message.Message, tools []tools.BaseTool) (response *ProviderResponse, err error) {
	cfg := config.Get()
	attempts := 0

	if o.providerOptions.model.UseOpenAIResponsesAPI {
		// Use /v1/responses API
		inputItems, instructions := o.buildOpenAIResponsesInput(messages)
		maxTokens := int(o.providerOptions.maxTokens)
		responsesAPIRequest := OpenAIResponsesRequest{
			Model:           o.providerOptions.model.APIModel,
			Input:           inputItems,
			Instructions:    instructions,
			MaxOutputTokens: &maxTokens,
			Stream:          false,
		}

		if cfg.Debug {
			jsonData, _ := json.Marshal(responsesAPIRequest)
			logging.Debug("Prepared /v1/responses request", "params", string(jsonData))
		}

		var httpClient *http.Client
		if o.client.HTTPClient != nil {
			httpClient = o.client.HTTPClient
		} else {
			httpClient = &http.Client{Timeout: 60 * time.Second} // Default timeout
		}

		apiURL := "https://api.openai.com/v1/responses"
		if o.options.baseURL != "" {
			// Ensure base URL does not end with a slash, and path does not start with one.
			trimmedBaseURL := strings.TrimRight(o.options.baseURL, "/")
			apiURL = trimmedBaseURL + "/v1/responses"
		}
		
		for {
			attempts++
			requestBody, err := json.Marshal(responsesAPIRequest)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal /v1/responses request: %w", err)
			}

			req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(requestBody))
			if err != nil {
				return nil, fmt.Errorf("failed to create /v1/responses HTTP request: %w", err)
			}

			req.Header.Set("Content-Type", "application/json")
			if o.providerOptions.apiKey != "" {
				req.Header.Set("Authorization", "Bearer "+o.providerOptions.apiKey)
			}
			// Add extra headers if any
			for key, value := range o.options.extraHeaders {
				req.Header.Set(key, value)
			}
			
			httpResponse, err := httpClient.Do(req)
			if err != nil {
				// Basic retry for network errors, could be expanded
				if attempts <= maxRetries {
					logging.WarnPersist(fmt.Sprintf("Retrying /v1/responses call due to network error: %v. Attempt %d of %d", err, attempts, maxRetries), logging.PersistTimeArg, time.Millisecond*2000)
					time.Sleep(time.Millisecond * 2000 * time.Duration(attempts)) // Exponential backoff might be better
					continue
				}
				return nil, fmt.Errorf("failed to execute /v1/responses request: %w", err)
			}
			defer httpResponse.Body.Close()

			if httpResponse.StatusCode == http.StatusTooManyRequests || httpResponse.StatusCode >= http.StatusInternalServerError {
				// Simplified retry logic, not using shouldRetry as it expects openai.Error
				if attempts <= maxRetries {
					retryAfter := 2000 // default retry after 2s
					if val := httpResponse.Header.Get("Retry-After"); val != "" {
						if seconds, err := fmt.Sscanf(val, "%d", &retryAfter); err == nil && seconds > 0 {
							retryAfter = retryAfter * 1000 // convert to ms
						}
					}
					logging.WarnPersist(fmt.Sprintf("Retrying /v1/responses call due to status %d. Attempt %d of %d. Retry after %dms", httpResponse.StatusCode, attempts, maxRetries, retryAfter), logging.PersistTimeArg, time.Millisecond*time.Duration(retryAfter+100))
					select {
					case <-ctx.Done():
						return nil, ctx.Err()
					case <-time.After(time.Duration(retryAfter) * time.Millisecond):
						continue
					}
				}
				bodyBytes, _ := io.ReadAll(httpResponse.Body)
				return nil, fmt.Errorf("/v1/responses API error: status %d, body: %s", httpResponse.StatusCode, string(bodyBytes))
			}

			if httpResponse.StatusCode != http.StatusOK {
				bodyBytes, _ := io.ReadAll(httpResponse.Body)
				return nil, fmt.Errorf("/v1/responses API error: status %d, body: %s", httpResponse.StatusCode, string(bodyBytes))
			}
			
			var responsesAPIResponse OpenAIResponsesResponse
			if err := json.NewDecoder(httpResponse.Body).Decode(&responsesAPIResponse); err != nil {
				return nil, fmt.Errorf("failed to decode /v1/responses response: %w", err)
			}

			if responsesAPIResponse.Error != nil {
				return nil, fmt.Errorf("/v1/responses API returned an error: %s - %s", responsesAPIResponse.Error.Code, responsesAPIResponse.Error.Message)
			}

			content := ""
			if len(responsesAPIResponse.Output) > 0 && len(responsesAPIResponse.Output[0].Content) > 0 {
				if responsesAPIResponse.Output[0].Type == "message" && responsesAPIResponse.Output[0].Role == "assistant" {
					for _, c := range responsesAPIResponse.Output[0].Content {
						if c.Type == "output_text" {
							content += c.Text
						}
					}
				}
			}
			
			var tokenUsage TokenUsage
			if responsesAPIResponse.Usage != nil {
				tokenUsage = TokenUsage{
					InputTokens:  int64(responsesAPIResponse.Usage.InputTokens),
					OutputTokens: int64(responsesAPIResponse.Usage.OutputTokens),
					// CacheCreationTokens, CacheReadTokens are not standard in this API's response
				}
			}

			// FinishReason is implicit for /v1/responses, assume EndTurn on success
			finishReason := message.FinishReasonEndTurn
			// TODO: Determine if there's a more specific finish reason from /v1/responses

			return &ProviderResponse{
				Content:      content,
				ToolCalls:    []message.ToolCall{}, // /v1/responses tool calls are TBD
				Usage:        tokenUsage,
				FinishReason: finishReason,
			}, nil
		}

	} else { // Default to Chat Completions API
		// The UsesLegacyCompletionsAPI path has been removed as the flag 
		// itself was removed from the model struct in a previous subtask.
		// All models not using UseOpenAIResponsesAPI will now fall through 
		// to the standard chat completions API.
		params := o.preparedParams(o.convertMessages(messages), o.convertTools(tools))
		if cfg.Debug {
			jsonData, _ := json.Marshal(params)
			logging.Debug("Prepared chat completion request", "params", string(jsonData))
		}
		// This inner loop is for retries for chat completions
		for {
			attempts++ // This attempts counter is for chat completion retries
			openaiResponse, err := o.client.Chat.Completions.New(
				ctx,
				params,
			)
			if err != nil {
				retry, after, retryErr := o.shouldRetry(attempts, err) // Pass the correct attempts counter
				if retryErr != nil {
					return nil, retryErr
				}
				if retry {
					logging.WarnPersist(fmt.Sprintf("Retrying chat call due to rate limit... attempt %d of %d", attempts, maxRetries), logging.PersistTimeArg, time.Millisecond*time.Duration(after+100))
					select {
					case <-ctx.Done():
						return nil, ctx.Err()
					case <-time.After(time.Duration(after) * time.Millisecond):
						continue
					}
				}
				return nil, retryErr // Should be err from shouldRetry or original if not retrying
			}

			content := ""
			if openaiResponse.Choices[0].Message.Content != "" {
				content = openaiResponse.Choices[0].Message.Content
			}

			toolCalls := o.toolCalls(*openaiResponse)
			finishReason := o.finishReason(string(openaiResponse.Choices[0].FinishReason))

			if len(toolCalls) > 0 {
				finishReason = message.FinishReasonToolUse
			}

			return &ProviderResponse{
				Content:      content,
				ToolCalls:    toolCalls,
				Usage:        o.usage(*openaiResponse),
				FinishReason: finishReason,
			}, nil
		}
	}
}

func (o *openaiClient) stream(ctx context.Context, messages []message.Message, tools []tools.BaseTool) <-chan ProviderEvent {
	params := o.preparedParams(o.convertMessages(messages), o.convertTools(tools))
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}

	cfg := config.Get()
	if cfg.Debug {
		jsonData, _ := json.Marshal(params)
		logging.Debug("Prepared messages", "messages", string(jsonData))
	}

	attempts := 0
	eventChan := make(chan ProviderEvent)
	cfg := config.Get()

	go func() {
		currentAttempt := 0
		for { // Outer loop for retries
			currentAttempt++
			if o.providerOptions.model.UseOpenAIResponsesAPI {
				inputItems, instructions := o.buildOpenAIResponsesInput(messages)
				maxTokens := int(o.providerOptions.maxTokens)
				responsesAPIRequest := OpenAIResponsesRequest{
					Model:           o.providerOptions.model.APIModel,
					Input:           inputItems,
					Instructions:    instructions,
					MaxOutputTokens: &maxTokens,
					Stream:          true,
				}

				if cfg.Debug {
					jsonData, _ := json.Marshal(responsesAPIRequest)
					logging.Debug("Prepared /v1/responses stream request", "params", string(jsonData))
				}

				var httpClient *http.Client
				if o.client.HTTPClient != nil {
					httpClient = o.client.HTTPClient
				} else {
					// For streaming, potentially longer timeout or allow configuration
					httpClient = &http.Client{Timeout: 300 * time.Second} 
				}
				
				apiURL := "https://api.openai.com/v1/responses"
				if o.options.baseURL != "" {
					trimmedBaseURL := strings.TrimRight(o.options.baseURL, "/")
					apiURL = trimmedBaseURL + "/v1/responses"
				}

				requestBody, err := json.Marshal(responsesAPIRequest)
				if err != nil {
					eventChan <- ProviderEvent{Type: EventError, Error: fmt.Errorf("failed to marshal /v1/responses stream request: %w", err)}
					close(eventChan)
					return
				}

				req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(requestBody))
				if err != nil {
					eventChan <- ProviderEvent{Type: EventError, Error: fmt.Errorf("failed to create /v1/responses stream HTTP request: %w", err)}
					close(eventChan)
					return
				}

				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("Accept", "text/event-stream") // Important for SSE
				req.Header.Set("Cache-Control", "no-cache")
				req.Header.Set("Connection", "keep-alive")


				if o.providerOptions.apiKey != "" {
					req.Header.Set("Authorization", "Bearer "+o.providerOptions.apiKey)
				}
				for key, value := range o.options.extraHeaders {
					req.Header.Set(key, value)
				}

				httpResponse, err := httpClient.Do(req)
				if err != nil {
					if currentAttempt <= maxRetries {
						logging.WarnPersist(fmt.Sprintf("Retrying /v1/responses stream call due to network error: %v. Attempt %d of %d", err, currentAttempt, maxRetries), logging.PersistTimeArg, time.Millisecond*2000)
						time.Sleep(time.Millisecond * 2000 * time.Duration(currentAttempt))
						continue // Retry the outer loop
					}
					eventChan <- ProviderEvent{Type: EventError, Error: fmt.Errorf("failed to execute /v1/responses stream request: %w", err)}
					close(eventChan)
					return
				}
				// defer httpResponse.Body.Close() // Moved into successful connection path

				if httpResponse.StatusCode != http.StatusOK {
					// Handle non-200 status codes for stream creation attempt
					bodyBytes, _ := io.ReadAll(httpResponse.Body)
					httpResponse.Body.Close() // Close body here for error cases
					if (httpResponse.StatusCode == http.StatusTooManyRequests || httpResponse.StatusCode >= http.StatusInternalServerError) && currentAttempt <= maxRetries {
						retryAfter := 2000 
						if val := httpResponse.Header.Get("Retry-After"); val != "" {
							if seconds, scanErr := fmt.Sscanf(val, "%d", &retryAfter); scanErr == nil && seconds > 0 {
								retryAfter = retryAfter * 1000 
							}
						}
						logging.WarnPersist(fmt.Sprintf("Retrying /v1/responses stream call due to status %d. Attempt %d of %d. Retry after %dms", httpResponse.StatusCode, currentAttempt, maxRetries, retryAfter), logging.PersistTimeArg, time.Millisecond*time.Duration(retryAfter+100))
						select {
						case <-ctx.Done():
							eventChan <- ProviderEvent{Type: EventError, Error: ctx.Err()}
							close(eventChan)
							return
						case <-time.After(time.Duration(retryAfter) * time.Millisecond):
							continue // Retry the outer loop
						}
					}
					eventChan <- ProviderEvent{Type: EventError, Error: fmt.Errorf("/v1/responses stream API error: status %d, body: %s", httpResponse.StatusCode, string(bodyBytes))}
					close(eventChan)
					return
				}
				
				// Successfully connected and got 200 OK for stream
				defer httpResponse.Body.Close()
				reader := bufio.NewReader(httpResponse.Body)
				currentContent := ""
				var finalUsage TokenUsage

				for { // Inner loop for reading stream data
					line, err := reader.ReadString('\n')
					if err != nil {
						if errors.Is(err, io.EOF) {
							// EOF is normal end of stream if [DONE] was processed or if stream just ends.
							// Ensure final complete event is sent if not already.
							eventChan <- ProviderEvent{
								Type: EventComplete,
								Response: &ProviderResponse{
									Content:      currentContent,
									ToolCalls:    []message.ToolCall{},
									Usage:        finalUsage,
									FinishReason: message.FinishReasonEndTurn, // Assume EndTurn
								},
							}
							close(eventChan)
							return
						}
						// Actual error reading stream
						eventChan <- ProviderEvent{Type: EventError, Error: fmt.Errorf("error reading /v1/responses stream: %w", err)}
						close(eventChan)
						return
					}

					line = strings.TrimSpace(line)
					if strings.HasPrefix(line, "data: ") {
						dataContent := strings.TrimPrefix(line, "data: ")
						if dataContent == "[DONE]" {
							// Stream finished signal
							eventChan <- ProviderEvent{
								Type: EventComplete,
								Response: &ProviderResponse{
									Content:      currentContent,
									ToolCalls:    []message.ToolCall{},
									Usage:        finalUsage, // Usage might be in the last chunk before [DONE]
									FinishReason: message.FinishReasonEndTurn,
								},
							}
							close(eventChan)
							return
						}

						var chunk OpenAIResponsesStreamChunk
						err := json.Unmarshal([]byte(dataContent), &chunk)
						if err != nil {
							logging.Warn("Failed to unmarshal /v1/responses stream chunk data", "error", err, "data", dataContent)
							continue
						}

						if chunk.Object == "response.chunk" && len(chunk.Output) > 0 {
							for _, outputDelta := range chunk.Output {
								if outputDelta.Type == "output.delta" && outputDelta.Delta.Type == "output_text.delta" {
									deltaText := outputDelta.Delta.Text
									eventChan <- ProviderEvent{
										Type:    EventContentDelta,
										Content: deltaText,
									}
									currentContent += deltaText
								}
							}
						}
						if chunk.Usage != nil { // Usage might come in the last data chunk
							finalUsage = TokenUsage{
								InputTokens:  int64(chunk.Usage.InputTokens),
								OutputTokens: int64(chunk.Usage.OutputTokens),
							}
						}
					} // end if strings.HasPrefix(line, "data: ")
				} // end inner stream reading loop
				// Should not be reached if stream ends correctly with [DONE] or EOF
				
			} else if o.providerOptions.model.UsesLegacyCompletionsAPI {
				prompt := o.constructLegacyPrompt(messages)
				legacyParams := openai.CompletionParams{
					Model:     o.providerOptions.model.APIModel,
					Prompt:    prompt,
					MaxTokens: int(o.providerOptions.maxTokens),
					Stream:    true,
					// TODO: Add other relevant params like Temperature, TopP if needed
				}
				if cfg.Debug {
					jsonData, _ := json.Marshal(legacyParams)
					logging.Debug("Prepared legacy completion stream request", "params", string(jsonData))
				}

				legacyStream, err := o.client.Completions.CreateStream(ctx, legacyParams)
				if err != nil {
					// This initial error is not a stream error, might not be retryable by shouldRetry in the same way
					// For now, let's assume shouldRetry can handle it or we fail fast.
					retry, after, retryErr := o.shouldRetry(currentAttempt, err)
					if retryErr != nil {
						eventChan <- ProviderEvent{Type: EventError, Error: retryErr}
						close(eventChan)
						return
					}
					if retry {
						logging.WarnPersist(fmt.Sprintf("Retrying legacy stream creation due to rate limit... attempt %d of %d", currentAttempt, maxRetries), logging.PersistTimeArg, time.Millisecond*time.Duration(after+100))
						select {
						case <-ctx.Done():
							if ctx.Err() != nil { // Check if context error is not nil
								eventChan <- ProviderEvent{Type: EventError, Error: ctx.Err()}
							}
							close(eventChan)
							return
						case <-time.After(time.Duration(after) * time.Millisecond):
							continue // Retry the loop
						}
					}
					eventChan <- ProviderEvent{Type: EventError, Error: err} // Original error if not retrying
					close(eventChan)
					return
				}
				defer legacyStream.Close()

				currentContent := ""
				var finalResponse openai.CompletionResponse
				for {
					chunk, streamErr := legacyStream.Recv()
					if streamErr != nil {
						if errors.Is(streamErr, io.EOF) {
							// Stream completed successfully
							var finishReason message.FinishReason
							if len(finalResponse.Choices) > 0 { // finalResponse might not be populated fully here, need to check chunk
								// The legacy stream gives finish_reason in the *last* chunk or not at all clearly.
								// We'll use the finish reason from the last non-empty choice if available.
								// This part is tricky with legacy; often it's just "stop" or "length".
								// For simplicity, we derive it from the last chunk that had a finish reason.
								// If chunk.Choices[0].FinishReason is available, use it.
								// This might need refinement based on actual API behavior.
								if len(chunk.Choices) > 0 { // Check current chunk too
									switch chunk.Choices[0].FinishReason {
									case "stop":
										finishReason = message.FinishReasonEndTurn
									case "length":
										finishReason = message.FinishReasonMaxTokens
									default:
										finishReason = message.FinishReasonUnknown
									}
								} else {
									finishReason = message.FinishReasonEndTurn // Assume EndTurn if no specific reason
								}
							} else {
								finishReason = message.FinishReasonEndTurn // Default if no choices recorded
							}


							eventChan <- ProviderEvent{
								Type: EventComplete,
								Response: &ProviderResponse{
									Content:   currentContent,
									ToolCalls: []message.ToolCall{}, // No tool calls for legacy
									Usage: TokenUsage{ // Usage from the last chunk if available
										InputTokens:         int64(chunk.Usage.PromptTokens), // This might be from the *last* chunk
										OutputTokens:        int64(chunk.Usage.CompletionTokens),
										CacheCreationTokens: 0,
										CacheReadTokens:     0,
									},
									FinishReason: finishReason,
								},
							}
							close(eventChan)
							return
						}
						// Non-EOF error, try to retry
						retry, after, retryErr := o.shouldRetry(currentAttempt, streamErr)
						if retryErr != nil {
							eventChan <- ProviderEvent{Type: EventError, Error: retryErr}
							close(eventChan)
							return
						}
						if retry {
							logging.WarnPersist(fmt.Sprintf("Retrying legacy stream due to error: %v. Attempt %d of %d", streamErr, currentAttempt, maxRetries), logging.PersistTimeArg, time.Millisecond*time.Duration(after+100))
							// Need to break this inner loop and restart the outer stream creation loop
							goto retryStreamCreation // Not ideal, but simple for now
						}
						eventChan <- ProviderEvent{Type: EventError, Error: streamErr}
						close(eventChan)
						return
					} // end streamErr handling

					if len(chunk.Choices) > 0 && chunk.Choices[0].Text != "" {
						eventChan <- ProviderEvent{
							Type:    EventContentDelta,
							Content: chunk.Choices[0].Text,
						}
						currentContent += chunk.Choices[0].Text
					}
					finalResponse = chunk // Store the last chunk to potentially get usage and finish_reason
				}
			retryStreamCreation: // Label for goto, to re-attempt stream creation
				// This will continue the outer for loop for retries
			} else {
				// Use chat completions streaming API (existing logic)
				params := o.preparedParams(o.convertMessages(messages), o.convertTools(tools))
				params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
					IncludeUsage: openai.Bool(true),
				}
				if cfg.Debug {
					jsonData, _ := json.Marshal(params)
					logging.Debug("Prepared chat completion stream request", "params", string(jsonData))
				}

				openaiStream := o.client.Chat.Completions.NewStreaming(
					ctx,
					params,
				)

				acc := openai.ChatCompletionAccumulator{}
				currentContent := ""
				toolCalls := make([]message.ToolCall, 0)

				for openaiStream.Next() {
					chunk := openaiStream.Current()
					acc.AddChunk(chunk)

					for _, choice := range chunk.Choices {
						if choice.Delta.Content != "" {
							eventChan <- ProviderEvent{
								Type:    EventContentDelta,
								Content: choice.Delta.Content,
							}
							currentContent += choice.Delta.Content
						}
					}
				}

				err := openaiStream.Err()
				if err == nil || errors.Is(err, io.EOF) {
					finishReason := o.finishReason(string(acc.ChatCompletion.Choices[0].FinishReason))
					if len(acc.ChatCompletion.Choices[0].Message.ToolCalls) > 0 {
						toolCalls = append(toolCalls, o.toolCalls(acc.ChatCompletion)...)
					}
					if len(toolCalls) > 0 {
						finishReason = message.FinishReasonToolUse
					}

					eventChan <- ProviderEvent{
						Type: EventComplete,
						Response: &ProviderResponse{
							Content:      currentContent,
							ToolCalls:    toolCalls,
							Usage:        o.usage(acc.ChatCompletion),
							FinishReason: finishReason,
						},
					}
					close(eventChan)
					return
				}

				retry, after, retryErr := o.shouldRetry(currentAttempt, err)
				if retryErr != nil {
					eventChan <- ProviderEvent{Type: EventError, Error: retryErr}
					close(eventChan)
					return
				}
				if retry {
					logging.WarnPersist(fmt.Sprintf("Retrying chat stream due to rate limit... attempt %d of %d", currentAttempt, maxRetries), logging.PersistTimeArg, time.Millisecond*time.Duration(after+100))
					select {
					case <-ctx.Done():
						if ctx.Err() != nil {
							eventChan <- ProviderEvent{Type: EventError, Error: ctx.Err()}
						}
						close(eventChan)
						return
					case <-time.After(time.Duration(after) * time.Millisecond):
						// continue the loop for retry
					}
				} else {
					eventChan <- ProviderEvent{Type: EventError, Error: retryErr} // or err if not retrying
					close(eventChan)
					return
				}
			}
		} // end for loop for retries
	}()

	return eventChan
}

func (o *openaiClient) shouldRetry(attempts int, err error) (bool, int64, error) {
	var apierr *openai.Error
	if !errors.As(err, &apierr) {
		return false, 0, err
	}

	if apierr.StatusCode != 429 && apierr.StatusCode != 500 {
		return false, 0, err
	}

	if attempts > maxRetries {
		return false, 0, fmt.Errorf("maximum retry attempts reached for rate limit: %d retries", maxRetries)
	}

	retryMs := 0
	retryAfterValues := apierr.Response.Header.Values("Retry-After")

	backoffMs := 2000 * (1 << (attempts - 1))
	jitterMs := int(float64(backoffMs) * 0.2)
	retryMs = backoffMs + jitterMs
	if len(retryAfterValues) > 0 {
		if _, err := fmt.Sscanf(retryAfterValues[0], "%d", &retryMs); err == nil {
			retryMs = retryMs * 1000
		}
	}
	return true, int64(retryMs), nil
}

func (o *openaiClient) toolCalls(completion openai.ChatCompletion) []message.ToolCall {
	var toolCalls []message.ToolCall

	if len(completion.Choices) > 0 && len(completion.Choices[0].Message.ToolCalls) > 0 {
		for _, call := range completion.Choices[0].Message.ToolCalls {
			toolCall := message.ToolCall{
				ID:       call.ID,
				Name:     call.Function.Name,
				Input:    call.Function.Arguments,
				Type:     "function",
				Finished: true,
			}
			toolCalls = append(toolCalls, toolCall)
		}
	}

	return toolCalls
}

// usage converts openai.ChatCompletionUsage to TokenUsage.
// For legacy completions, this might need a separate helper if usage stats are different.
// The current one is for ChatCompletions.
// For legacy, usage is simpler: openai.CompletionResponseUsage.
func (o *openaiClient) usage(completion openai.ChatCompletion) TokenUsage {
	var cachedTokens int
	if completion.Usage != nil && completion.Usage.PromptTokensDetails != nil { // Check for nil before accessing
		cachedTokens = completion.Usage.PromptTokensDetails.CachedTokens
	}

	var promptTokens int
	if completion.Usage != nil {
		promptTokens = completion.Usage.PromptTokens
	}
	inputTokens := promptTokens - cachedTokens

	var completionTokens int
	if completion.Usage != nil {
		completionTokens = completion.Usage.CompletionTokens
	}


	return TokenUsage{
		InputTokens:         int64(inputTokens),
		OutputTokens:        int64(completionTokens),
		CacheCreationTokens: 0, // OpenAI doesn't provide this directly
		CacheReadTokens:     int64(cachedTokens),
	}
}

func WithOpenAIBaseURL(baseURL string) OpenAIOption {
	return func(options *openaiOptions) {
		options.baseURL = baseURL
	}
}

func WithOpenAIExtraHeaders(headers map[string]string) OpenAIOption {
	return func(options *openaiOptions) {
		options.extraHeaders = headers
	}
}

func WithOpenAIDisableCache() OpenAIOption {
	return func(options *openaiOptions) {
		options.disableCache = true
	}
}

func WithReasoningEffort(effort string) OpenAIOption {
	return func(options *openaiOptions) {
		defaultReasoningEffort := "medium"
		switch effort {
		case "low", "medium", "high":
			defaultReasoningEffort = effort
		default:
			logging.Warn("Invalid reasoning effort, using default: medium")
		}
		options.reasoningEffort = defaultReasoningEffort
	}
}
