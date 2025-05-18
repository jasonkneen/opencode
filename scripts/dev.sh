#!/usr/bin/env bash
# Development script for OpenCode
# This script sets up the environment for local development and runs the local build

set -e

# Directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Root directory of the project
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default configuration file path (relative to project root)
CONFIG_FILE=".opencode/config.json"
# Default log level for development
LOG_LEVEL="debug"

# Colors for output
COLOR=true
if [[ "$COLOR" == "true" ]]; then
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  RED='\033[0;31m'
  NC='\033[0m' # No Color
else
  GREEN=''
  YELLOW=''
  RED=''
  NC=''
fi

# Function to print status messages
print_status() {
  echo -e "${GREEN}[DEV]${NC} $1"
}

# Function to print warning messages
print_warning() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

# Function to print error messages
print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

print_usage() {
  echo "Usage: $0 [options] [-- opencode_options]"
  echo "Options:"
  echo "  -h, --help          Show this help message"
  echo "  -c, --config FILE   Use specific config file (default: $CONFIG_FILE)"
  echo "  -d, --debug         Enable extra debug logging (sets OPENCODE_DEV_DEBUG=true)"
  echo "  -b, --build         Force rebuild before running"
  echo "  --no-color          Disable colored output"
  echo "  --install           Set up aliases to use the local dev version as default"
  echo "  --uninstall         Remove dev version aliases"
  echo "  --                  Pass remaining arguments to the opencode binary"
  echo ""
  echo "Setup Options:"
  echo "  --install           Adds aliases to your .zshrc file:"
  echo "                      - 'opencode' will point to your local dev build"
  echo "                      - 'opencode.brew' will point to the Homebrew installation"
  echo "                      You'll need to restart your terminal or run 'source ~/.zshrc'"
  echo ""
  echo "  --uninstall         Removes the aliases added by --install"
  echo ""
  echo "Examples:"
  echo "  $0 -d                           # Run with debug logging"
  echo "  $0 -- --version                 # Show OpenCode version"
  echo "  $0 -d -- --version              # Debug mode and show version"
  echo "  $0 --install                    # Set up aliases for development"
  echo "  $0 --uninstall                  # Remove development aliases"
}

# Process command line arguments
BUILD=false
DEBUG_LOGGING=false
OPENCODE_ARGS=()

# Function to install development aliases
install_dev_aliases() {
  # Check if homebrew version exists
  if [ ! -f "/opt/homebrew/bin/opencode" ]; then
    print_error "Homebrew version not found at /opt/homebrew/bin/opencode"
    print_error "Please install OpenCode with Homebrew first"
    exit 1
  fi

  local ZSHRC="$HOME/.zshrc"
  local START_MARKER="# === OpenCode Development Environment Start ==="
  local END_MARKER="# === OpenCode Development Environment End ==="

  # Check if aliases are already installed
  if grep -q "$START_MARKER" "$ZSHRC"; then
    print_warning "Development aliases are already installed in $ZSHRC"
    print_status "To reinstall, run: $0 --uninstall && $0 --install"
    exit 0
  fi

  # Add aliases to .zshrc
  cat >> "$ZSHRC" << EOF

$START_MARKER
# These aliases were added by OpenCode dev.sh script
# To remove them, run: $PROJECT_ROOT/scripts/dev.sh --uninstall

# Alias the Homebrew version to opencode.brew
alias opencode.brew="/opt/homebrew/bin/opencode"

# Alias the development version to opencode
alias opencode="$PROJECT_ROOT/opencode"

# Add environment variables for development
export OPENCODE_CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
$END_MARKER

EOF

  print_status "Development aliases installed in $ZSHRC"
  print_status "To use them, restart your terminal or run: source $ZSHRC"
  exit 0
}

# Function to uninstall development aliases
uninstall_dev_aliases() {
  local ZSHRC="$HOME/.zshrc"
  local START_MARKER="# === OpenCode Development Environment Start ==="
  local END_MARKER="# === OpenCode Development Environment End ==="

  # Check if aliases are installed
  if ! grep -q "$START_MARKER" "$ZSHRC"; then
    print_warning "Development aliases not found in $ZSHRC"
    exit 0
  fi

  # Remove everything between and including the markers
  sed -i.bak "/^$START_MARKER$/,/^$END_MARKER$/d" "$ZSHRC"

  print_status "Development aliases removed from $ZSHRC"
  print_status "To apply changes, restart your terminal or run: source $ZSHRC"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_usage
      exit 0
      ;;
    -c|--config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    -d|--debug)
      DEBUG_LOGGING=true
      shift
      ;;
    -b|--build)
      BUILD=true
      shift
      ;;
    --install)
      install_dev_aliases
      ;;
    --uninstall)
      uninstall_dev_aliases
      ;;
    --no-color)
      COLOR=false
      shift
      ;;
    --)
      # Stop processing script options and collect the rest for opencode
      shift
      OPENCODE_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

# Update colors if --no-color was specified
if [[ "$COLOR" == "false" ]]; then
  GREEN=''
  YELLOW=''
  RED=''
  NC=''
fi

# Check if the config file exists, create it if not
if [[ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]]; then
  # Create directory for config if it doesn't exist
  mkdir -p "$(dirname "$PROJECT_ROOT/$CONFIG_FILE")"
  
  # If there's a global config, copy it
  if [[ -f "$HOME/.opencode.json" ]]; then
    print_status "Copying global config to $CONFIG_FILE"
    cp "$HOME/.opencode.json" "$PROJECT_ROOT/$CONFIG_FILE"
  else
    # Create a minimal config
    print_status "Creating minimal config at $CONFIG_FILE"
    cat > "$PROJECT_ROOT/$CONFIG_FILE" << EOF
{
  "data": {
    "directory": ".opencode"
  },
  "debug": true,
  "autoCompact": true
}
EOF
  fi
fi

# Get detailed version info for development build
get_version() {
  # Use git describe to get position relative to last tag
  if git_describe=$(git describe --tags --long 2>/dev/null); then
    # Check for uncommitted changes
    if [[ -n "$(git status --porcelain)" ]]; then
      echo "${git_describe}-dev-dirty"
    else
      echo "${git_describe}-dev"
    fi
  else
    # Fallback if git describe fails
    git_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
    git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    
    # Check for uncommitted changes
    if [[ -n "$(git status --porcelain)" ]]; then
      echo "${git_tag}-${git_commit}-dev-dirty"
    else
      echo "${git_tag}-${git_commit}-dev"
    fi
  fi
}

# Build the project if requested or if binary doesn't exist
if [[ "$BUILD" == "true" ]] || [[ ! -f "$PROJECT_ROOT/opencode" ]]; then
  print_status "Building OpenCode..."
  cd "$PROJECT_ROOT"
  
  # Get version and mark as development build
  VERSION=$(get_version)
  print_status "Setting development version: $VERSION"
  
  # Build with version info - force the version by setting main.Version and disabling trimpath
  go build \
    -ldflags "\
      -X 'github.com/opencode-ai/opencode/internal/version.Version=$VERSION' \
      -X 'main.version=$VERSION'" \
    -o opencode
fi

# Set up environment variables for development
export OPENCODE_CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"

# Enable extra debug logging if requested
if [[ "$DEBUG_LOGGING" == "true" ]]; then
  export OPENCODE_DEV_DEBUG="true"
  print_status "Debug logging enabled (logs will be in .opencode/debug.log)"
fi

# Print development environment info
print_status "Starting OpenCode in development mode"
print_status "  • Config: $OPENCODE_CONFIG_FILE"
print_status "  • Working directory: $PROJECT_ROOT"
if [[ "$DEBUG_LOGGING" == "true" ]]; then
  print_status "  • Debug logging: Enabled"
else
  print_status "  • Debug logging: Disabled (use -d to enable)"
fi

# Display arguments being passed to opencode if any
if [[ ${#OPENCODE_ARGS[@]} -gt 0 ]]; then
  print_status "  • Passing arguments: ${OPENCODE_ARGS[*]}"
fi

# Check if the binary exists
if [[ ! -f "$PROJECT_ROOT/opencode" ]]; then
  print_error "OpenCode binary not found. Please build it with: $0 -b"
  exit 1
fi

# Run the local binary with the specified config
cd "$PROJECT_ROOT"
exec ./opencode "${OPENCODE_ARGS[@]}"

