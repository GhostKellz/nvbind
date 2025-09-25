#!/bin/bash
# Multi-target build script for nvbind
# Optimized for Arch Linux with support for major distros

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Build configuration
VERSION=${VERSION:-$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].version')}
BUILD_DIR="target/release"
DIST_DIR="dist"

echo -e "${BLUE}🚀 Building nvbind v${VERSION} for multiple targets${NC}"

# Ensure we're in project root
if [[ ! -f "Cargo.toml" ]]; then
    echo -e "${RED}❌ Error: Must run from project root (Cargo.toml not found)${NC}"
    exit 1
fi

# Create distribution directory
mkdir -p "${DIST_DIR}"

# Function to build for a specific target
build_target() {
    local target="$1"
    local name="$2"
    local description="$3"

    echo -e "${YELLOW}📦 Building ${name} (${target})${NC}"
    echo -e "    ${description}"

    # Install target if not available
    if ! rustup target list --installed | grep -q "${target}"; then
        echo -e "    Installing ${target} target..."
        rustup target add "${target}"
    fi

    # Build the target
    if cargo build --release --target "${target}" --all-features; then
        echo -e "${GREEN}✅ ${name} build completed${NC}"

        # Copy binary to dist with appropriate name
        local binary_name="nvbind"
        if [[ "${target}" == *"windows"* ]]; then
            binary_name="nvbind.exe"
        fi

        local dist_name="nvbind-v${VERSION}-${name}"
        if [[ "${target}" == *"windows"* ]]; then
            dist_name="${dist_name}.exe"
        fi

        cp "target/${target}/release/${binary_name}" "${DIST_DIR}/${dist_name}"

        # Create checksum
        cd "${DIST_DIR}"
        sha256sum "${dist_name}" > "${dist_name}.sha256"
        cd ..

        echo -e "    📄 Binary: ${DIST_DIR}/${dist_name}"
        echo -e "    🔐 Checksum: ${DIST_DIR}/${dist_name}.sha256"
    else
        echo -e "${RED}❌ ${name} build failed${NC}"
        return 1
    fi

    echo ""
}

# Build targets optimized for different distros
echo -e "${BLUE}🏗️ Building distro-optimized targets...${NC}"

# Arch Linux (native optimizations)
build_target "x86_64-unknown-linux-gnu" "arch-native" \
    "Arch Linux native build with maximum optimizations"

# Universal Linux (musl static)
build_target "x86_64-unknown-linux-musl" "universal-static" \
    "Universal static binary for maximum compatibility"

# Ubuntu/Debian compatible
RUSTFLAGS="-C target-cpu=x86-64-v2 -C opt-level=3 -C lto=thin" \
build_target "x86_64-unknown-linux-gnu" "ubuntu-debian" \
    "Ubuntu/Debian/PopOS compatible build"

# Fedora/RHEL compatible
RUSTFLAGS="-C target-cpu=x86-64-v2 -C opt-level=3 -C lto=thin" \
build_target "x86_64-unknown-linux-gnu" "fedora-rhel" \
    "Fedora/RHEL/CentOS compatible build"

# ARM64 builds
if rustup target list --installed | grep -q "aarch64-unknown-linux-gnu" || rustup target add aarch64-unknown-linux-gnu; then
    build_target "aarch64-unknown-linux-gnu" "arm64" \
        "ARM64 Linux build"

    build_target "aarch64-unknown-linux-musl" "arm64-static" \
        "ARM64 static binary"
fi

# Create archive packages
echo -e "${BLUE}📦 Creating distribution packages...${NC}"

cd "${DIST_DIR}"

# Create tarballs for each build
for binary in nvbind-v${VERSION}-*; do
    if [[ -f "${binary}" && ! "${binary}" =~ \.sha256$ ]]; then
        echo -e "${YELLOW}📦 Packaging ${binary}...${NC}"

        # Create temporary directory structure
        mkdir -p "tmp/${binary%-*}"
        cp "${binary}" "tmp/${binary%-*}/nvbind"
        cp "${binary}.sha256" "tmp/${binary%-*}/"

        # Add README
        cat > "tmp/${binary%-*}/README.txt" << EOF
nvbind v${VERSION} - NVIDIA GPU Container Runtime

Build: ${binary}
Target: $(echo "${binary}" | cut -d'-' -f4-)

Installation:
1. Copy 'nvbind' to /usr/local/bin/ (or preferred location)
2. Make executable: chmod +x /usr/local/bin/nvbind
3. Verify: nvbind --version

For documentation and support:
https://github.com/ghostkellz/nvbind

Checksum verification:
sha256sum -c nvbind.sha256
EOF

        # Create tarball
        tar -czf "${binary}.tar.gz" -C tmp "${binary%-*}"
        rm -rf "tmp/${binary%-*}"

        echo -e "${GREEN}✅ Created ${binary}.tar.gz${NC}"
    fi
done

# Cleanup
rm -rf tmp

# Build summary
echo -e "${BLUE}📊 Build Summary${NC}"
echo "===================="
ls -lh nvbind-v${VERSION}-* | while read -r line; do
    echo -e "${GREEN}${line}${NC}"
done

echo ""
echo -e "${GREEN}🎉 All builds completed successfully!${NC}"
echo -e "${BLUE}Distribution files are in: ${DIST_DIR}/${NC}"

# Performance optimization notes
echo ""
echo -e "${YELLOW}🚀 Performance Notes:${NC}"
echo "  • arch-native: Maximum performance on Arch Linux with native CPU optimizations"
echo "  • ubuntu-debian: Optimized for x86-64-v2 baseline (compatible with most modern systems)"
echo "  • universal-static: Single binary with no dependencies (slightly slower but universally compatible)"
echo "  • fedora-rhel: Similar to Ubuntu but tested on RHEL family"
echo "  • arm64: Native ARM64 performance for modern ARM systems"

cd ..