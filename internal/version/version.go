package version

import (
	"runtime/debug"
	"strings"
)

// Build-time parameters set via -ldflags
var Version = "unknown"

// Get returns the version string. This function should be used instead
// of directly accessing the Version variable.
func Get() string {
	return Version
}

// A user may install OpenCode using `go install github.com/opencode-ai/opencode@latest`
// without -ldflags, in which case the version above is unset. As a workaround
// we use the embedded build version that *is* set when using `go install` (and
// is only set for `go install` and not for `go build`).
func init() {
	// If version is already set via -ldflags, keep it
	if Version != "unknown" {
		// Version is already set, possibly by dev.sh or release build
		return
	}

	// Try to read build info
	info, ok := debug.ReadBuildInfo()
	if !ok {
		// < go v1.18
		return
	}

	mainVersion := info.Main.Version
	if mainVersion == "" || mainVersion == "(devel)" {
		// If devel and we're in a git repo, try to use vcs info
		for _, setting := range info.Settings {
			// Look for vcs.revision
			if setting.Key == "vcs.revision" && setting.Value != "" {
				commitHash := setting.Value
				if len(commitHash) > 7 {
					commitHash = commitHash[:7]
				}
				
				dirty := false
				// Check if the build is dirty
				for _, s := range info.Settings {
					if s.Key == "vcs.modified" && s.Value == "true" {
						dirty = true
						break
					}
				}
				
				// Format: dev-{commit}[-dirty]
				if dirty {
					Version = "dev-" + commitHash + "-dirty"
				} else {
					Version = "dev-" + commitHash
				}
				return
			}
		}
		
		// If we couldn't determine version from VCS, use a generic dev tag
		Version = "dev"
		return
	}

	// For go install builds
	if strings.HasPrefix(mainVersion, "v") {
		Version = mainVersion
	}
}
