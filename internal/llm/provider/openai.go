package provider

import (
	"bufio" // Added for stream reading
	"bytes" // Added for http.NewRequest
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http" // Added for manual HTTP client usage
	"strings"  // Added for constructLegacyPrompt
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
	"github.com/opencode-ai/opencode/internal/config"
	"github.com/opencode-ai/opencode/internal/llm/models"
	"github.com/opencode-ai/opencode/internal/llm/tools"
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

type OpenAIUsageData struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"` // May not be provided by /v1/responses
	CachedTokens int `json:"cached_tokens,omitempty"` // Specific to some OpenAI APIs
	BilledTokens int `json:"billed_tokens,omitempty"` // Specific to some OpenAI APIs
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
			assistantMsg := openai.ChatCompletionAssistantMessageParam{Role: "assistant"}
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
			openaiMessages = append(openaiMessages, openai.ChatCompletionMessageParamUnion{OfAssistant: &assistantMsg})
		case message.Tool:
			for _, result := range msg.ToolResults() {
				openaiMessages = append(openaiMessages, openai.ToolMessage(result.Content, result.ToolCallID))
			}
		}
	}
	return
}

func (o *openaiClient) buildOpenAIResponsesInput(messages []message.Message) ([]OpenAIResponsesInputItem, string) {
	var inputs []OpenAIResponsesInputItem
	var instructions strings.Builder
	var userContentBuilder strings.Builder

	if o.providerOptions.systemMessage != "" {
		instructions.WriteString(o.providerOptions.systemMessage)
	}

	for _, msg := range messages {
		switch msg.Role {
		case message.System:
			if instructions.Len() > 0 {
				instructions.WriteString("\n\n")
			}
			instructions.WriteString(msg.Content().String())
		case message.User:
			if userContentBuilder.Len() > 0 {
				userContentBuilder.WriteString("\n\n")
			}
			userContentBuilder.WriteString(msg.Content().String())
		case message.Assistant, message.Tool:
			// Ignoring for /v1/responses
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
	if o.providerOptions.model.CanReason {
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

func (o *openaiClient) send(ctx context.Context, messages []message.Message, toolsList []tools.BaseTool) (response *ProviderResponse, err error) {
	cfg := config.Get()
	attempts := 0

	if o.providerOptions.model.UseOpenAIResponsesAPI {
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
			httpClient = &http.Client{Timeout: 60 * time.Second}
		}

		apiURL := "https://api.openai.com/v1/responses"
		if o.options.baseURL != "" {
			trimmedBaseURL := strings.TrimRight(o.options.baseURL, "/")
			apiURL = trimmedBaseURL + "/v1/responses"
		}
		
		for { // Retry loop
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
			for key, value := range o.options.extraHeaders {
				req.Header.Set(key, value)
			}
			
			httpResponse, err := httpClient.Do(req)
			if err != nil {
				if attempts <= maxRetries {
					logging.WarnPersist(fmt.Sprintf("Retrying /v1/responses call due to network error: %v. Attempt %d of %d", err, attempts, maxRetries), logging.PersistTimeArg, time.Millisecond*2000*time.Duration(attempts))
					time.Sleep(time.Millisecond * 2000 * time.Duration(attempts))
					continue
				}
				return nil, fmt.Errorf("failed to execute /v1/responses request: %w", err)
			}
			
			bodyBytes, readErr := io.ReadAll(httpResponse.Body)
			httpResponse.Body.Close() // Close body after reading or on error
			if readErr != nil {
				return nil, fmt.Errorf("failed to read /v1/responses response body: %w", readErr)
			}

			if httpResponse.StatusCode == http.StatusTooManyRequests || httpResponse.StatusCode >= http.StatusInternalServerError {
				if attempts <= maxRetries {
					retryAfter := 2000 
					if val := httpResponse.Header.Get("Retry-After"); val != "" {
						if seconds, e := fmt.Sscanf(val, "%d", &retryAfter); e == nil && seconds > 0 {
							retryAfter *= 1000 
						}
					}
					logging.WarnPersist(fmt.Sprintf("Retrying /v1/responses call due to status %d. Attempt %d of %d. Retry after %dms", httpResponse.StatusCode, attempts, maxRetries, retryAfter), logging.PersistTimeArg, time.Millisecond*time.Duration(retryAfter+100))
					select {
					case <-ctx.Done(): return nil, ctx.Err()
					case <-time.After(time.Duration(retryAfter) * time.Millisecond): continue
					}
				}
				return nil, fmt.Errorf("/v1/responses API error: status %d, body: %s", httpResponse.StatusCode, string(bodyBytes))
			}

			if httpResponse.StatusCode != http.StatusOK {
				return nil, fmt.Errorf("/v1/responses API error: status %d, body: %s", httpResponse.StatusCode, string(bodyBytes))
			}
			
			var responsesAPIResponse OpenAIResponsesResponse
			if err := json.Unmarshal(bodyBytes, &responsesAPIResponse); err != nil {
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
				}
			}

			return &ProviderResponse{
				Content:      content,
				ToolCalls:    []message.ToolCall{},
				Usage:        tokenUsage,
				FinishReason: message.FinishReasonEndTurn,
			}, nil
		} // End retry loop
	} else { // Default to Chat Completions API
		params := o.preparedParams(o.convertMessages(messages), o.convertTools(toolsList))
		if cfg.Debug {
			jsonData, _ := json.Marshal(params)
			logging.Debug("Prepared chat completion request", "params", string(jsonData))
		}
		for { // Retry loop
			attempts++
			openaiResponse, err := o.client.Chat.Completions.New(ctx, params)
			if err != nil {
				retry, after, retryErr := o.shouldRetry(attempts, err)
				if retryErr != nil { return nil, retryErr }
				if retry {
					logging.WarnPersist(fmt.Sprintf("Retrying chat call due to rate limit... attempt %d of %d", attempts, maxRetries), logging.PersistTimeArg, time.Millisecond*time.Duration(after+100))
					select {
					case <-ctx.Done(): return nil, ctx.Err()
					case <-time.After(time.Duration(after) * time.Millisecond): continue
					}
				}
				return nil, err // Return original error if not retrying
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
		} // End retry loop
	}
}

func (o *openaiClient) stream(ctx context.Context, messages []message.Message, toolsList []tools.BaseTool) <-chan ProviderEvent {
	eventChan := make(chan ProviderEvent)
	cfg := config.Get()

	go func() {
		defer close(eventChan)
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
					return
				}

				req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(requestBody))
				if err != nil {
					eventChan <- ProviderEvent{Type: EventError, Error: fmt.Errorf("failed to create /v1/responses stream HTTP request: %w", err)}
					return
				}

				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("Accept", "text/event-stream")
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
						logging.WarnPersist(fmt.Sprintf("Retrying /v1/responses stream call due to network error: %v. Attempt %d of %d", err, currentAttempt, maxRetries), logging.PersistTimeArg, time.Millisecond*2000*time.Duration(currentAttempt))
						time.Sleep(time.Millisecond * 2000 * time.Duration(currentAttempt))
						continue 
					}
					eventChan <- ProviderEvent{Type: EventError, Error: fmt.Errorf("failed to execute /v1/responses stream request: %w", err)}
					return
				}
				
				if httpResponse.StatusCode != http.StatusOK {
					bodyBytes, _ := io.ReadAll(httpResponse.Body)
					httpResponse.Body.Close()
					if (httpResponse.StatusCode == http.StatusTooManyRequests || httpResponse.StatusCode >= http.StatusInternalServerError) && currentAttempt <= maxRetries {
						retryAfter := 2000 
						if val := httpResponse.Header.Get("Retry-After"); val != "" {
							if seconds, e := fmt.Sscanf(val, "%d", &retryAfter); e == nil && seconds > 0 {
								retryAfter *= 1000
							}
						}
						logging.WarnPersist(fmt.Sprintf("Retrying /v1/responses stream call due to status %d. Attempt %d of %d. Retry after %dms", httpResponse.StatusCode, currentAttempt, maxRetries, retryAfter), logging.PersistTimeArg, time.Millisecond*time.Duration(retryAfter+100))
						select {
						case <-ctx.Done(): eventChan <- ProviderEvent{Type: EventError, Error: ctx.Err()}; return
						case <-time.After(time.Duration(retryAfter) * time.Millisecond): continue
						}
					}
					eventChan <- ProviderEvent{Type: EventError, Error: fmt.Errorf("/v1/responses stream API error: status %d, body: %s", httpResponse.StatusCode, string(bodyBytes))}
					return
				}
				
				defer httpResponse.Body.Close()
				reader := bufio.NewReader(httpResponse.Body)
				currentContent := ""
				var finalUsage TokenUsage

				for { // Inner loop for reading stream data
					line, readErr := reader.ReadString('\n')
					if readErr != nil {
						if errors.Is(readErr, io.EOF) {
							eventChan <- ProviderEvent{Type: EventComplete, Response: &ProviderResponse{Content: currentContent, ToolCalls: []message.ToolCall{}, Usage: finalUsage, FinishReason: message.FinishReasonEndTurn}}
							return
						}
						eventChan <- ProviderEvent{Type: EventError, Error: fmt.Errorf("error reading /v1/responses stream: %w", readErr)}
						return
					}

					line = strings.TrimSpace(line)
					if strings.HasPrefix(line, "data: ") {
						dataContent := strings.TrimPrefix(line, "data: ")
						if dataContent == "[DONE]" {
							eventChan <- ProviderEvent{Type: EventComplete, Response: &ProviderResponse{Content: currentContent, ToolCalls: []message.ToolCall{}, Usage: finalUsage, FinishReason: message.FinishReasonEndTurn}}
							return
						}

						var chunk OpenAIResponsesStreamChunk
						if err := json.Unmarshal([]byte(dataContent), &chunk); err != nil {
							logging.Warn("Failed to unmarshal /v1/responses stream chunk data", "error", err, "data", dataContent)
							continue
						}

						if chunk.Object == "response.chunk" && len(chunk.Output) > 0 {
							for _, outputDelta := range chunk.Output {
								if outputDelta.Type == "output.delta" && outputDelta.Delta.Type == "output_text.delta" {
									deltaText := outputDelta.Delta.Text
									eventChan <- ProviderEvent{Type: EventContentDelta, Content: deltaText}
									currentContent += deltaText
								}
							}
						}
						if chunk.Usage != nil {
							finalUsage = TokenUsage{InputTokens: int64(chunk.Usage.InputTokens), OutputTokens: int64(chunk.Usage.OutputTokens)}
						}
					} 
				} // End inner stream reading loop
			} else { // Default to Chat Completions streaming API
				params := o.preparedParams(o.convertMessages(messages), o.convertTools(toolsList))
				params.StreamOptions = openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(true)}
				if cfg.Debug {
					jsonData, _ := json.Marshal(params)
					logging.Debug("Prepared chat completion stream request", "params", string(jsonData))
				}

				for { // Retry loop for chat completions stream
					openaiStream := o.client.Chat.Completions.NewStreaming(ctx, params)
					acc := openai.ChatCompletionAccumulator{}
					currentContent := ""
					toolCalls := make([]message.ToolCall, 0)
					streamSuccessful := false

					for openaiStream.Next() {
						streamSuccessful = true // Mark as successful if we receive any data
						chunk := openaiStream.Current()
						acc.AddChunk(chunk)
						for _, choice := range chunk.Choices {
							if choice.Delta.Content != "" {
								eventChan <- ProviderEvent{Type: EventContentDelta, Content: choice.Delta.Content}
								currentContent += choice.Delta.Content
							}
						}
					}

					err := openaiStream.Err()
					if err == nil || errors.Is(err, io.EOF) {
						if !streamSuccessful && acc.ChatCompletion.ID == "" { // No data received and looks like an empty stream
							// This case might indicate an issue before actual streaming began (e.g. bad request not caught by initial status check)
							// Try to retry if attempts allow
							if currentAttempt <= maxRetries {
								logging.WarnPersist(fmt.Sprintf("Retrying chat stream due to empty response. Attempt %d of %d", currentAttempt, maxRetries), logging.PersistTimeArg, time.Millisecond*2000*time.Duration(currentAttempt))
								time.Sleep(time.Millisecond*2000*time.Duration(currentAttempt))
								currentAttempt++
								continue // Retry the outer loop
							}
							eventChan <- ProviderEvent{Type: EventError, Error: fmt.Errorf("chat stream failed with empty response after retries")}
							return
						}


						finishReason := o.finishReason(string(acc.ChatCompletion.Choices[0].FinishReason))
						if len(acc.ChatCompletion.Choices[0].Message.ToolCalls) > 0 {
							toolCalls = append(toolCalls, o.toolCalls(acc.ChatCompletion)...)
						}
						if len(toolCalls) > 0 {
							finishReason = message.FinishReasonToolUse
						}
						eventChan <- ProviderEvent{Type: EventComplete, Response: &ProviderResponse{Content: currentContent, ToolCalls: toolCalls, Usage: o.usage(acc.ChatCompletion), FinishReason: finishReason}}
						return
					}

					// Handle actual stream error
					retry, after, retryErr := o.shouldRetry(currentAttempt, err)
					if retryErr != nil { eventChan <- ProviderEvent{Type: EventError, Error: retryErr}; return }
					if retry {
						logging.WarnPersist(fmt.Sprintf("Retrying chat stream due to error: %v. Attempt %d of %d", err, currentAttempt, maxRetries), logging.PersistTimeArg, time.Millisecond*time.Duration(after+100))
						select {
						case <-ctx.Done(): eventChan <- ProviderEvent{Type: EventError, Error: ctx.Err()}; return
						case <-time.After(time.Duration(after) * time.Millisecond): 
							currentAttempt++
							continue // Retry the outer loop
						}
					}
					eventChan <- ProviderEvent{Type: EventError, Error: err} // Return original error if not retrying
					return
				} // End retry loop for chat completions stream
			}
		} // End outer loop for retries
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

	backoffMs := 2000 * (1 << (attempts - 1)) // Exponential backoff
	jitterMs := int(float64(backoffMs) * 0.2)  // Add jitter
	retryMs = backoffMs + jitterMs
	if len(retryAfterValues) > 0 {
		if ra, scanErr := fmt.Sscanf(retryAfterValues[0], "%d", &retryMs); scanErr == nil && ra > 0 {
			retryMs *= 1000 // Convert seconds to milliseconds
		}
	}
	return true, int64(retryMs), nil
}

func (o *openaiClient) toolCalls(completion openai.ChatCompletion) []message.ToolCall {
	var toolCalls []message.ToolCall
	if len(completion.Choices) > 0 && len(completion.Choices[0].Message.ToolCalls) > 0 {
		for _, call := range completion.Choices[0].Message.ToolCalls {
			toolCalls = append(toolCalls, message.ToolCall{
				ID:       call.ID,
				Name:     call.Function.Name,
				Input:    call.Function.Arguments,
				Type:     "function",
				Finished: true,
			})
		}
	}
	return toolCalls
}

func (o *openaiClient) usage(completion openai.ChatCompletion) TokenUsage {
	var cachedTokens int
	if completion.Usage != nil && completion.Usage.PromptTokensDetails != nil {
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
		CacheCreationTokens: 0, 
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
