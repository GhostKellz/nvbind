# Getting Started with nvbind

Complete guide to installing and using nvbind

**Difficulty:** Beginner

## Table of Contents

1. [Installation](#1-installation)
2. [First Steps](#2-first-steps)

## 1. Installation

nvbind can be installed in several ways:

### From Source
```bash
git clone https://github.com/ghostkellz/nvbind.git
cd nvbind
cargo build --release
sudo cp target/release/nvbind /usr/local/bin/
```

### Using Cargo
```bash
cargo install nvbind
```

### Package Managers
#### Arch Linux (AUR)
```bash
yay -S nvbind
```

#### Ubuntu/Debian
```bash
# Add repository
curl -s https://packagecloud.io/install/repositories/nvbind/stable/script.deb.sh | sudo bash
sudo apt-get install nvbind
```

#### Fedora
```bash
sudo dnf copr enable ghostkellz/nvbind
sudo dnf install nvbind
```

## 2. First Steps

After installation, verify nvbind is working:

### Check Installation
```bash
nvbind --version
```

### Run System Diagnostics
```bash
nvbind diagnose
```

This will check for:
- NVIDIA GPU availability
- Driver installation
- Container runtimes (Docker, Podman)
- System compatibility

### Interactive Setup
```bash
nvbind setup
```

This launches an interactive wizard to configure nvbind for your system.

