#!/bin/bash
set -e

# nvbind Installation Script
# A lightweight, Rust-based alternative to NVIDIA Container Toolkit

INSTALL_DIR="/usr/local/bin"
CONFIG_DIR="/etc/nvbind"
REPO_URL="https://github.com/ghostkellz/nvbind"

echo "🚀 Installing nvbind - NVIDIA Container GPU Passthrough Tool"
echo ""

# Check if running as root for system install
if [[ $EUID -ne 0 && "$1" != "--user" ]]; then
    echo "❌ This script needs to run as root for system installation."
    echo "   Run with 'sudo $0' for system install, or '$0 --user' for user install"
    exit 1
fi

# Set install directory based on user/system install
if [[ "$1" == "--user" ]]; then
    INSTALL_DIR="$HOME/.local/bin"
    CONFIG_DIR="$HOME/.config/nvbind"
    echo "📁 Installing to user directory: $INSTALL_DIR"
else
    echo "📁 Installing to system directory: $INSTALL_DIR"
fi

# Check dependencies
echo "🔍 Checking dependencies..."

if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/Cargo not found. Please install Rust first:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Please install git first."
    exit 1
fi

# Check for container runtime
HAS_PODMAN=false
HAS_DOCKER=false

if command -v podman &> /dev/null; then
    echo "✅ Found Podman: $(podman --version)"
    HAS_PODMAN=true
fi

if command -v docker &> /dev/null; then
    echo "✅ Found Docker: $(docker --version)"
    HAS_DOCKER=true
fi

if [[ "$HAS_PODMAN" == false && "$HAS_DOCKER" == false ]]; then
    echo "⚠️  No container runtime found. Please install Podman or Docker:"
    echo "   - Podman: https://podman.io/getting-started/installation"
    echo "   - Docker: https://docs.docker.com/get-docker/"
fi

# Check NVIDIA drivers
echo "🔍 Checking NVIDIA drivers..."
if [[ -f "/proc/driver/nvidia/version" ]]; then
    NVIDIA_VERSION=$(head -n1 /proc/driver/nvidia/version | awk '{print $8}')
    echo "✅ NVIDIA Driver detected: $NVIDIA_VERSION"
elif command -v nvidia-smi &> /dev/null; then
    NVIDIA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    echo "✅ NVIDIA Driver detected: $NVIDIA_VERSION"
else
    echo "⚠️  NVIDIA drivers not detected. nvbind requires NVIDIA drivers to function."
    echo "   Install from: https://www.nvidia.com/drivers"
fi

# Build and install
echo ""
echo "🔨 Building nvbind..."

# Create temporary directory for build
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Clone repository
echo "📥 Downloading source code..."
git clone "$REPO_URL" .

# Build release binary
echo "⚙️  Compiling..."
cargo build --release

# Create directories
echo "📁 Creating directories..."
mkdir -p "$INSTALL_DIR"
if [[ "$1" != "--user" ]]; then
    mkdir -p "$CONFIG_DIR"
fi

# Install binary
echo "📦 Installing binary..."
cp target/release/nvbind "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/nvbind"

# Create default config
echo "⚙️  Creating default configuration..."
if [[ "$1" == "--user" ]]; then
    # User install - let nvbind create config in user's config dir
    "$INSTALL_DIR/nvbind" config --output "$CONFIG_DIR/config.toml" 2>/dev/null || true
else
    # System install - create system config
    "$INSTALL_DIR/nvbind" config --output "$CONFIG_DIR/config.toml" 2>/dev/null || true
fi

# Cleanup
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "✅ nvbind installed successfully!"
echo ""
echo "📋 Usage:"
echo "   nvbind info                    # Show GPU information"
echo "   nvbind config --show           # Show current configuration"
echo "   nvbind run ubuntu nvidia-smi   # Run container with GPU access"
echo ""

if [[ "$1" == "--user" ]]; then
    echo "📝 Note: Add $INSTALL_DIR to your PATH if not already present:"
    echo "   echo 'export PATH=\$PATH:$INSTALL_DIR' >> ~/.bashrc"
    echo "   source ~/.bashrc"
fi

echo "🔗 Documentation: $REPO_URL"
echo "🏁 Ready to use GPU containers with nvbind!"