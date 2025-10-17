#!/bin/bash
# Build script for Debian/Ubuntu DEB package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
VERSION=$(grep '^version = ' "$PROJECT_ROOT/Cargo.toml" | sed 's/version = "\(.*\)"/\1/')
ARCH="amd64"
PKG_NAME="nvbind_${VERSION}_${ARCH}"
BUILD_DIR="/tmp/${PKG_NAME}"

echo "Building nvbind DEB package v${VERSION}..."

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/DEBIAN"
mkdir -p "$BUILD_DIR/usr/bin"
mkdir -p "$BUILD_DIR/usr/share/doc/nvbind"
mkdir -p "$BUILD_DIR/usr/share/man/man1"

# Build the binary in release mode
cd "$PROJECT_ROOT"
echo "Building nvbind binary..."
cargo build --release

# Copy binary
cp "$PROJECT_ROOT/target/release/nvbind" "$BUILD_DIR/usr/bin/"
chmod 755 "$BUILD_DIR/usr/bin/nvbind"

# Copy control file
cp "$SCRIPT_DIR/control" "$BUILD_DIR/DEBIAN/"

# Update version in control file
sed -i "s/^Version:.*/Version: ${VERSION}/" "$BUILD_DIR/DEBIAN/control"

# Copy and set permissions for maintainer scripts
cp "$SCRIPT_DIR/postinst" "$BUILD_DIR/DEBIAN/"
cp "$SCRIPT_DIR/prerm" "$BUILD_DIR/DEBIAN/"
chmod 755 "$BUILD_DIR/DEBIAN/postinst"
chmod 755 "$BUILD_DIR/DEBIAN/prerm"

# Copy documentation
cp "$PROJECT_ROOT/README.md" "$BUILD_DIR/usr/share/doc/nvbind/"
cp "$PROJECT_ROOT/LICENSE" "$BUILD_DIR/usr/share/doc/nvbind/" 2>/dev/null || echo "MIT" > "$BUILD_DIR/usr/share/doc/nvbind/LICENSE"

# Generate man page (basic version)
cat > "$BUILD_DIR/usr/share/man/man1/nvbind.1" <<'EOF'
.TH NVBIND 1 "2025" "nvbind 0.1.0" "User Commands"
.SH NAME
nvbind \- High-performance NVIDIA container GPU runtime
.SH SYNOPSIS
.B nvbind
[\fIOPTION\fR]... [\fICOMMAND\fR]
.SH DESCRIPTION
nvbind is a cutting-edge, Rust-based GPU container runtime designed as a
high-performance alternative to NVIDIA's Container Toolkit.
.SH OPTIONS
.TP
\fB\-h\fR, \fB\-\-help\fR
Display help information
.TP
\fB\-V\fR, \fB\-\-version\fR
Display version information
.SH COMMANDS
.TP
\fBdoctor\fR
Run system diagnostics and compatibility checks
.TP
\fBruntime\fR
Execute as a container runtime (used by Docker/Podman)
.TP
\fBcdi\fR
Generate CDI specifications for GPU devices
.SH EXAMPLES
.TP
Check system compatibility:
.B nvbind doctor
.TP
Run container with GPU:
.B docker run --runtime=nvbind --gpus all nvidia/cuda:12.0-base nvidia-smi
.SH AUTHOR
Written by ghostkellz
.SH REPORTING BUGS
Report bugs at: https://github.com/ghostkellz/nvbind/issues
.SH COPYRIGHT
Copyright (c) 2025. Licensed under the MIT License.
EOF

gzip -9 "$BUILD_DIR/usr/share/man/man1/nvbind.1"

# Build the package
echo "Creating DEB package..."
dpkg-deb --build --root-owner-group "$BUILD_DIR"

# Move to project root
mv "${BUILD_DIR}.deb" "$PROJECT_ROOT/${PKG_NAME}.deb"

echo ""
echo "✅ DEB package created: ${PKG_NAME}.deb"
echo ""
echo "To install:"
echo "  sudo dpkg -i ${PKG_NAME}.deb"
echo "  sudo apt-get install -f  # Install dependencies if needed"
echo ""
echo "To verify:"
echo "  dpkg -c ${PKG_NAME}.deb  # List contents"
echo "  dpkg -I ${PKG_NAME}.deb  # Show package info"
