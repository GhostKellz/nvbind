#!/bin/bash
# Build script for Fedora/RHEL RPM package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
VERSION=$(grep '^version = ' "$PROJECT_ROOT/Cargo.toml" | sed 's/version = "\(.*\)"/\1/')

echo "Building nvbind RPM package v${VERSION}..."

# Set up rpmbuild directory structure
RPMBUILD_DIR="$HOME/rpmbuild"
mkdir -p "$RPMBUILD_DIR"/{BUILD,RPMS,SOURCES,SPECS,SRPMS}

# Create source tarball
cd "$PROJECT_ROOT"
TARBALL_NAME="nvbind-${VERSION}.tar.gz"
echo "Creating source tarball..."
git archive --format=tar.gz --prefix="nvbind-${VERSION}/" HEAD > "$RPMBUILD_DIR/SOURCES/$TARBALL_NAME" 2>/dev/null || \
    tar czf "$RPMBUILD_DIR/SOURCES/$TARBALL_NAME" \
        --transform "s,^,nvbind-${VERSION}/," \
        --exclude='.git' \
        --exclude='target' \
        --exclude='*.deb' \
        --exclude='*.rpm' \
        *

# Copy spec file
cp "$SCRIPT_DIR/nvbind.spec" "$RPMBUILD_DIR/SPECS/"

# Update version in spec file
sed -i "s/^Version:.*/Version:        ${VERSION}/" "$RPMBUILD_DIR/SPECS/nvbind.spec"

# Build the RPM
echo "Building RPM..."
rpmbuild -ba "$RPMBUILD_DIR/SPECS/nvbind.spec"

# Copy resulting RPM to project root
cp "$RPMBUILD_DIR/RPMS/x86_64/nvbind-${VERSION}-"*.rpm "$PROJECT_ROOT/" 2>/dev/null || \
cp "$RPMBUILD_DIR/RPMS/noarch/nvbind-${VERSION}-"*.rpm "$PROJECT_ROOT/" 2>/dev/null || \
echo "Warning: Could not find built RPM in expected location"

echo ""
echo "✅ RPM package created!"
echo ""
echo "To install:"
echo "  sudo dnf install ./nvbind-${VERSION}-*.rpm"
echo "  # OR for RHEL/CentOS:"
echo "  sudo yum install ./nvbind-${VERSION}-*.rpm"
echo ""
echo "To verify:"
echo "  rpm -qilp ./nvbind-${VERSION}-*.rpm"
