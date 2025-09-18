#!/bin/bash

# nvbind Universal Installer
# Install nvbind GPU container runtime on Linux systems

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NVBIND_VERSION="0.1.0"
NVBIND_REPO="https://github.com/ghostkellz/nvbind"
INSTALL_DIR="/usr/local/bin"
CONFIG_DIR="/etc/nvbind"
USER_INSTALL=false

# Utility functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)
            USER_INSTALL=true
            INSTALL_DIR="$HOME/.local/bin"
            CONFIG_DIR="$HOME/.config/nvbind"
            shift
            ;;
        --version)
            NVBIND_VERSION="$2"
            shift 2
            ;;
        --help)
            echo "nvbind installer"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "OPTIONS:"
            echo "  --user      Install to user directory (~/.local/bin)"
            echo "  --version   Specify version to install (default: $NVBIND_VERSION)"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main installation flow
main() {
    echo "🚀 nvbind GPU Container Runtime Installer"
    echo "========================================"
    echo ""

    log_info "Installing nvbind $NVBIND_VERSION..."

    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    # Clone repository
    log_info "Cloning nvbind repository..."
    git clone "$NVBIND_REPO" nvbind
    cd nvbind

    # Build release binary
    log_info "Building nvbind (this may take a few minutes)..."
    cargo build --release

    # Create installation directories
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$CONFIG_DIR"

    # Install binary
    if [[ "$USER_INSTALL" == false ]]; then
        sudo cp target/release/nvbind "$INSTALL_DIR/"
        sudo chmod +x "$INSTALL_DIR/nvbind"
    else
        cp target/release/nvbind "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/nvbind"
    fi

    # Generate default configuration
    log_info "Generating default configuration..."
    "$INSTALL_DIR/nvbind" config --output "$CONFIG_DIR/nvbind.toml"

    # Cleanup
    cd /
    rm -rf "$TEMP_DIR"

    log_success "nvbind installed to $INSTALL_DIR/nvbind"
    echo ""
    echo "Next steps:"
    echo "1. Run 'nvbind info' to check GPU detection"
    echo "2. Run 'nvbind doctor' for system compatibility"
    echo "3. Run 'nvbind run --runtime bolt --gpu all ubuntu nvidia-smi'"
}

# Run main installation
main "$@"