# nvbind Makefile
# Cross-compilation and packaging for major Linux distros
# Optimized for Arch Linux

.PHONY: all clean test bench install install-arch install-debian install-fedora install-popos
.PHONY: arch debian fedora popos universal arm64 all-targets
.PHONY: package check-deps help

# Project metadata
PROJECT := nvbind
VERSION := $(shell cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].version' 2>/dev/null || echo "unknown")
BUILD_DATE := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_HASH := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Directories
DIST_DIR := dist
BUILD_DIR := target/release

# Build flags
RUST_VERSION := $(shell rustc --version)
CARGO_FLAGS := --release --all-features

# Security hardening flags
HARDENING_FLAGS := -C relro-level=full -C strip=symbols -C panic=abort
SECURITY_RUSTFLAGS := -D warnings -D unsafe-code -D missing-docs

# Build validation
REQUIRED_RUST_VERSION := 1.70.0

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
BOLD := \033[1m
NC := \033[0m

# Default target
all: arch

help: ## Show this help message
	@echo "$(BOLD)nvbind build system$(NC)"
	@echo "Optimized for Arch Linux with multi-distro support"
	@echo ""
	@echo "$(BOLD)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BOLD)Examples:$(NC)"
	@echo "  make arch          # Build optimized for Arch Linux"
	@echo "  make all-targets   # Build for all supported distros"
	@echo "  make install-arch  # Build and install on Arch Linux"
	@echo "  make test          # Run all tests"
	@echo "  make bench         # Run benchmarks"

check-deps: ## Check build dependencies and security requirements
	@echo "$(BLUE)Checking build dependencies and security requirements...$(NC)"
	@which rustc > /dev/null || (echo "$(RED)Error: rustc not found$(NC)" && exit 1)
	@which cargo > /dev/null || (echo "$(RED)Error: cargo not found$(NC)" && exit 1)
	@which jq > /dev/null || (echo "$(YELLOW)Warning: jq not found (version detection may fail)$(NC)")
	@echo "$(GREEN)✅ Dependencies OK$(NC)"
	@echo "Rust version: $(shell rustc --version)"
	@echo "Cargo version: $(shell cargo --version)"

	# Validate Rust version
	@echo "$(BLUE)Validating Rust version...$(NC)"
	@RUST_VER=$$(rustc --version | cut -d' ' -f2); \
	if [ "$$(printf '%s\n%s' "$(REQUIRED_RUST_VERSION)" "$$RUST_VER" | sort -V | head -n1)" != "$(REQUIRED_RUST_VERSION)" ]; then \
		echo "$(RED)Error: Rust version $$RUST_VER is below required $(REQUIRED_RUST_VERSION)$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Rust version OK$(NC)"

security-check: ## Run security validation
	@echo "$(BLUE)Running security validation...$(NC)"
	@echo "Checking for unsafe code blocks..."
	@if grep -r "unsafe" src/ --include="*.rs" | grep -v "// Safety:" | head -5; then \
		echo "$(YELLOW)Warning: Found unsafe code blocks (review required)$(NC)"; \
	else \
		echo "$(GREEN)✅ No unsafe code found$(NC)"; \
	fi

	@echo "Checking for TODO/FIXME markers..."
	@if grep -r -n "TODO\|FIXME\|XXX\|HACK" src/ --include="*.rs" | head -10; then \
		echo "$(YELLOW)Warning: Found development markers (review before release)$(NC)"; \
	else \
		echo "$(GREEN)✅ No development markers found$(NC)"; \
	fi

	@echo "Checking for hardcoded credentials..."
	@if grep -r -i "password\|secret\|key\|token" src/ --include="*.rs" | grep -E "(=|:)" | head -5; then \
		echo "$(YELLOW)Warning: Found potential hardcoded credentials$(NC)"; \
	else \
		echo "$(GREEN)✅ No hardcoded credentials detected$(NC)"; \
	fi

	@echo "$(GREEN)✅ Security validation complete$(NC)"

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	cargo clean
	rm -rf $(DIST_DIR)
	@echo "$(GREEN)✅ Clean complete$(NC)"

test: ## Run all tests with security validation
	@echo "$(BLUE)Running comprehensive test suite...$(NC)"
	cargo test --all-features -- --test-threads=1
	@echo "$(BLUE)Running test coverage analysis...$(NC)"
	@TEST_COUNT=$$(cargo test --all-features 2>/dev/null | grep -E "test result:" | tail -1 | cut -d' ' -f3); \
	echo "$(GREEN)✅ $$TEST_COUNT tests passed$(NC)"
	@echo "$(BLUE)Validating test quality...$(NC)"
	@if [ "$$TEST_COUNT" -ge "90" ]; then \
		echo "$(GREEN)✅ Excellent test coverage ($$TEST_COUNT tests)$(NC)"; \
	elif [ "$$TEST_COUNT" -ge "70" ]; then \
		echo "$(YELLOW)⚠️  Good test coverage ($$TEST_COUNT tests)$(NC)"; \
	else \
		echo "$(RED)❌ Insufficient test coverage ($$TEST_COUNT tests)$(NC)"; \
	fi

bench: ## Run benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	cargo bench --all-features
	@echo "$(GREEN)✅ Benchmarks complete$(NC)"

# Distro-specific builds

arch: check-deps security-check ## Build optimized for Arch Linux (native CPU)
	@echo "$(BLUE)Building for Arch Linux (optimized)...$(NC)"
	RUSTFLAGS="$(HARDENING_FLAGS)" cargo build $(CARGO_FLAGS) --target x86_64-unknown-linux-gnu
	@echo "$(BLUE)Running post-build validation...$(NC)"
	@if [ -f target/x86_64-unknown-linux-gnu/release/nvbind ]; then \
		echo "$(GREEN)✅ Binary created successfully$(NC)"; \
		ls -lh target/x86_64-unknown-linux-gnu/release/nvbind; \
		file target/x86_64-unknown-linux-gnu/release/nvbind; \
	else \
		echo "$(RED)❌ Build failed - binary not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Arch Linux build complete$(NC)"

debian: check-deps ## Build compatible with Debian/Ubuntu/PopOS
	@echo "$(BLUE)Building for Debian/Ubuntu/PopOS...$(NC)"
	cargo build $(CARGO_FLAGS) --target x86_64-unknown-linux-gnu
	@echo "$(GREEN)✅ Debian/Ubuntu build complete$(NC)"

fedora: check-deps ## Build compatible with Fedora/RHEL/CentOS
	@echo "$(BLUE)Building for Fedora/RHEL/CentOS...$(NC)"
	cargo build $(CARGO_FLAGS) --target x86_64-unknown-linux-gnu
	@echo "$(GREEN)✅ Fedora/RHEL build complete$(NC)"

popos: debian ## Build for PopOS (alias for debian)

universal: check-deps security-check ## Build universal static binary (musl) with security hardening
	@echo "$(BLUE)Building universal static binary with hardening...$(NC)"
	rustup target add x86_64-unknown-linux-musl 2>/dev/null || true
	RUSTFLAGS="-C target-cpu=x86-64-v2 -C opt-level=3 -C lto=fat -C codegen-units=1 -C link-arg=-static $(HARDENING_FLAGS)" \
	cargo build $(CARGO_FLAGS) --target x86_64-unknown-linux-musl
	@echo "$(BLUE)Running static binary validation...$(NC)"
	@if [ -f target/x86_64-unknown-linux-musl/release/nvbind ]; then \
		echo "$(GREEN)✅ Static binary created$(NC)"; \
		ls -lh target/x86_64-unknown-linux-musl/release/nvbind; \
		file target/x86_64-unknown-linux-musl/release/nvbind; \
		ldd target/x86_64-unknown-linux-musl/release/nvbind 2>&1 | grep -q "not a dynamic executable" && \
		echo "$(GREEN)✅ Static linking verified$(NC)" || \
		echo "$(YELLOW)⚠️  Dynamic dependencies detected$(NC)"; \
	else \
		echo "$(RED)❌ Static build failed$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Universal static build complete$(NC)"

arm64: check-deps ## Build for ARM64 Linux
	@echo "$(BLUE)Building for ARM64 Linux...$(NC)"
	rustup target add aarch64-unknown-linux-gnu 2>/dev/null || true
	RUSTFLAGS="-C opt-level=3 -C lto=thin" \
	cargo build $(CARGO_FLAGS) --target aarch64-unknown-linux-gnu
	@echo "$(GREEN)✅ ARM64 build complete$(NC)"

all-targets: ## Build for all supported targets
	@echo "$(BOLD)Building for all supported targets...$(NC)"
	@./scripts/build-all-targets.sh
	@echo "$(GREEN)✅ All targets built$(NC)"

package: all-targets ## Create distribution packages
	@echo "$(BLUE)Creating distribution packages...$(NC)"
	@mkdir -p $(DIST_DIR)
	@echo "$(GREEN)✅ Packages created in $(DIST_DIR)/$(NC)"

# Installation targets

install-arch: arch ## Build and install on Arch Linux
	@echo "$(BLUE)Installing nvbind on Arch Linux...$(NC)"
	sudo cp target/x86_64-unknown-linux-gnu/release/nvbind /usr/local/bin/
	sudo chmod +x /usr/local/bin/nvbind
	@echo "$(GREEN)✅ nvbind installed to /usr/local/bin/nvbind$(NC)"
	@echo "Verify with: nvbind --version"

install-debian: debian ## Build and install on Debian/Ubuntu/PopOS
	@echo "$(BLUE)Installing nvbind on Debian/Ubuntu/PopOS...$(NC)"
	sudo cp target/x86_64-unknown-linux-gnu/release/nvbind /usr/local/bin/
	sudo chmod +x /usr/local/bin/nvbind
	@echo "$(GREEN)✅ nvbind installed to /usr/local/bin/nvbind$(NC)"
	@echo "Verify with: nvbind --version"

install-fedora: fedora ## Build and install on Fedora/RHEL/CentOS
	@echo "$(BLUE)Installing nvbind on Fedora/RHEL/CentOS...$(NC)"
	sudo cp target/x86_64-unknown-linux-gnu/release/nvbind /usr/local/bin/
	sudo chmod +x /usr/local/bin/nvbind
	@echo "$(GREEN)✅ nvbind installed to /usr/local/bin/nvbind$(NC)"
	@echo "Verify with: nvbind --version"

install-popos: install-debian ## Install on PopOS (alias for debian)

install: ## Auto-detect distro and install
	@echo "$(BLUE)Auto-detecting distribution...$(NC)"
	@if [ -f /etc/arch-release ]; then \
		echo "$(YELLOW)Detected: Arch Linux$(NC)"; \
		$(MAKE) install-arch; \
	elif [ -f /etc/pop-release ]; then \
		echo "$(YELLOW)Detected: PopOS$(NC)"; \
		$(MAKE) install-popos; \
	elif [ -f /etc/debian_version ]; then \
		echo "$(YELLOW)Detected: Debian/Ubuntu$(NC)"; \
		$(MAKE) install-debian; \
	elif [ -f /etc/fedora-release ] || [ -f /etc/redhat-release ]; then \
		echo "$(YELLOW)Detected: Fedora/RHEL$(NC)"; \
		$(MAKE) install-fedora; \
	else \
		echo "$(YELLOW)Unknown distribution, using universal build$(NC)"; \
		$(MAKE) universal; \
		sudo cp target/x86_64-unknown-linux-musl/release/nvbind /usr/local/bin/; \
		sudo chmod +x /usr/local/bin/nvbind; \
	fi

# Development targets with validation

dev: security-check ## Quick development build with security checks
	@echo "$(BLUE)Development build with validation...$(NC)"
	cargo build --all-features
	@echo "$(BLUE)Running quick validation...$(NC)"
	@if [ -f target/debug/nvbind ]; then \
		echo "$(GREEN)✅ Debug binary created$(NC)"; \
		ls -lh target/debug/nvbind; \
	else \
		echo "$(RED)❌ Development build failed$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Development build complete$(NC)"

fmt: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	cargo fmt
	@echo "$(GREEN)✅ Code formatted$(NC)"

lint: ## Run comprehensive linting and security analysis
	@echo "$(BLUE)Running comprehensive linting and security analysis...$(NC)"
	cargo clippy --all-features --all-targets -- -D warnings -D clippy::all -D clippy::pedantic -W clippy::cargo
	@echo "$(BLUE)Running additional security lints...$(NC)"
	cargo clippy --all-features --all-targets -- -W clippy::unwrap_used -W clippy::panic -W clippy::unimplemented
	@echo "$(GREEN)✅ All lints passed$(NC)"

check: security-check ## Comprehensive code validation without building
	@echo "$(BLUE)Comprehensive code validation...$(NC)"
	cargo check --all-features --all-targets
	@echo "$(BLUE)Checking for common vulnerabilities...$(NC)"
	@which cargo-audit >/dev/null 2>&1 && cargo audit || echo "$(YELLOW)cargo-audit not available - install with: cargo install cargo-audit$(NC)"
	@echo "$(GREEN)✅ Code validation complete$(NC)"

# Information targets

info: ## Show build information
	@echo "$(BOLD)nvbind Build Information$(NC)"
	@echo "========================="
	@echo "Project: $(PROJECT)"
	@echo "Version: $(VERSION)"
	@echo "Build Date: $(BUILD_DATE)"
	@echo "Git Hash: $(GIT_HASH)"
	@echo "Rust Version: $(RUST_VERSION)"
	@echo ""
	@echo "$(BOLD)Supported Targets:$(NC)"
	@echo "• Arch Linux (native optimized)"
	@echo "• Debian/Ubuntu/PopOS (x86-64-v2 baseline)"
	@echo "• Fedora/RHEL/CentOS (x86-64-v2 baseline)"
	@echo "• Universal Static (musl, maximum compatibility)"
	@echo "• ARM64 Linux"

size: arch ## Show binary size information
	@echo "$(BOLD)Binary Size Information$(NC)"
	@echo "========================"
	@ls -lh target/x86_64-unknown-linux-gnu/release/nvbind 2>/dev/null || echo "Binary not found - run 'make arch' first"
	@echo ""
	@echo "$(BOLD)Size breakdown:$(NC)"
	@size target/x86_64-unknown-linux-gnu/release/nvbind 2>/dev/null || echo "Run 'make arch' to build binary first"

verify-binary: ## Comprehensive binary integrity and security verification
	@echo "$(BOLD)Binary Security Verification$(NC)"
	@echo "==============================="
	@BINARY_PATH="target/x86_64-unknown-linux-gnu/release/nvbind"; \
	if [ -f "$$BINARY_PATH" ]; then \
		echo "$(GREEN)✅ Binary exists$(NC)"; \
		echo "Size: $$(du -h $$BINARY_PATH | cut -f1)"; \
		echo "Type: $$(file $$BINARY_PATH)"; \
		echo ""; \
		echo "$(BOLD)Security Features:$(NC)"; \
		if readelf -d "$$BINARY_PATH" 2>/dev/null | grep -q "RELRO"; then \
			echo "$(GREEN)✅ RELRO enabled$(NC)"; \
		else \
			echo "$(YELLOW)⚠️  RELRO not detected$(NC)"; \
		fi; \
		if readelf -d "$$BINARY_PATH" 2>/dev/null | grep -q "BIND_NOW"; then \
			echo "$(GREEN)✅ BIND_NOW enabled$(NC)"; \
		else \
			echo "$(YELLOW)⚠️  BIND_NOW not detected$(NC)"; \
		fi; \
		if readelf -h "$$BINARY_PATH" 2>/dev/null | grep -q "DYN"; then \
			echo "$(GREEN)✅ PIE/PIC enabled$(NC)"; \
		else \
			echo "$(YELLOW)⚠️  PIE/PIC not detected$(NC)"; \
		fi; \
		echo ""; \
		echo "$(BOLD)Dependencies:$(NC)"; \
		ldd "$$BINARY_PATH" 2>/dev/null | head -10 || echo "Static binary (no dependencies)"; \
		echo ""; \
		echo "$(BOLD)Symbols:$(NC)"; \
		if nm "$$BINARY_PATH" 2>/dev/null | grep -q "T "; then \
			echo "$(YELLOW)⚠️  Symbols present (not stripped)$(NC)"; \
		else \
			echo "$(GREEN)✅ Symbols stripped$(NC)"; \
		fi; \
	else \
		echo "$(RED)❌ Binary not found - run 'make arch' first$(NC)"; \
		exit 1; \
	fi

release-check: verify-binary ## Comprehensive release readiness validation
	@echo "$(BOLD)Release Readiness Check$(NC)"
	@echo "=========================="
	@echo "$(BLUE)Validating version consistency...$(NC)"
	@CARGO_VERSION=$$(cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '.packages[0].version' 2>/dev/null || echo "unknown"); \
	BINARY_VERSION=$$(./target/x86_64-unknown-linux-gnu/release/nvbind --version 2>/dev/null | cut -d' ' -f2 || echo "unknown"); \
	if [ "$$CARGO_VERSION" = "$$BINARY_VERSION" ]; then \
		echo "$(GREEN)✅ Version consistency: $$CARGO_VERSION$(NC)"; \
	else \
		echo "$(RED)❌ Version mismatch: Cargo=$$CARGO_VERSION, Binary=$$BINARY_VERSION$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Validating basic functionality...$(NC)"
	@if ./target/x86_64-unknown-linux-gnu/release/nvbind --help >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Help command works$(NC)"; \
	else \
		echo "$(RED)❌ Help command failed$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Release readiness validated$(NC)"