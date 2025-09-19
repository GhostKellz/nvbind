# Quick Installation Guide

Get nvbind up and running in under 5 minutes!

## 🚀 One-Line Install

### System-wide Installation (Recommended)
```bash
curl -sSL https://raw.githubusercontent.com/ghostkellz/nvbind/main/install.sh | sudo bash
```

### User Installation (No root required)
```bash
curl -sSL https://raw.githubusercontent.com/ghostkellz/nvbind/main/install.sh | bash -s -- --user
```

## 📦 Package Manager Installation

### Arch Linux (AUR)
```bash
# Using yay
yay -S nvbind

# Using paru
paru -S nvbind

# Manual AUR build
git clone https://aur.archlinux.org/nvbind.git
cd nvbind && makepkg -si
```

### Ubuntu/Debian (Coming Soon)
```bash
# Add repository
curl -fsSL https://packages.nvbind.dev/gpg | sudo gpg --dearmor -o /usr/share/keyrings/nvbind.gpg
echo "deb [signed-by=/usr/share/keyrings/nvbind.gpg] https://packages.nvbind.dev/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/nvbind.list

# Install
sudo apt update
sudo apt install nvbind
```

### Fedora/RHEL (Coming Soon)
```bash
# Add repository
sudo dnf config-manager --add-repo https://packages.nvbind.dev/fedora/nvbind.repo

# Install
sudo dnf install nvbind
```

## 🔨 Build from Source

### Prerequisites
- Rust 1.70+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Git
- NVIDIA drivers (if using NVIDIA GPUs)

### Build Steps
```bash
# Clone repository
git clone https://github.com/ghostkellz/nvbind.git
cd nvbind

# Build release binary
cargo build --release

# Install to system
sudo cp target/release/nvbind /usr/local/bin/
sudo chmod +x /usr/local/bin/nvbind

# Install with Bolt support
cargo build --release --features bolt
sudo cp target/release/nvbind /usr/local/bin/nvbind-bolt
```

## ✅ Verify Installation

```bash
# Check version
nvbind --version

# Verify GPU detection
nvbind info

# Run system diagnostics
nvbind doctor

# Generate sample configuration
nvbind config --show
```

Expected output:
```
nvbind 0.1.0

=== GPU Information ===
Found 1 NVIDIA GPU:
  GPU 0: NVIDIA GeForce RTX 4090 (24576 MB)
  Driver: 535.146.02 (NVIDIA Open Kernel Modules)

=== System Status ===
✅ NVIDIA drivers loaded
✅ Container runtimes available: podman, docker
✅ CDI support ready
✅ nvbind installation successful
```

## 🎯 Quick Test

Run your first GPU-accelerated container:

```bash
# Test with CUDA
nvbind run --runtime podman nvidia/cuda:latest nvidia-smi

# Test with gaming profile
nvbind run --runtime bolt --profile gaming --gpu all ubuntu:22.04 nvidia-smi

# Test with AI/ML profile
nvbind run --runtime docker --profile ai-ml pytorch/pytorch:latest python -c "import torch; print(torch.cuda.is_available())"
```

## 🔧 Configure Default Runtime

Create a configuration file to set your preferred defaults:

```bash
# Generate config file
nvbind config --output ~/.config/nvbind/config.toml

# Edit the generated file
nano ~/.config/nvbind/config.toml
```

Example configuration:
```toml
[runtime]
default_runtime = "bolt"
default_args = ["--rm", "-it"]

[gpu]
default_selection = "all"
enable_isolation = true

[bolt]
[bolt.capsule]
snapshot_gpu_state = true
isolation_level = "exclusive"

[bolt.gaming]
dlss_enabled = true
rt_cores_enabled = true
performance_profile = "ultra-low-latency"
```

## 🚨 Troubleshooting

### nvbind command not found
```bash
# Check if binary exists
ls -la /usr/local/bin/nvbind

# Add to PATH if needed
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Permission denied
```bash
# Fix binary permissions
sudo chmod +x /usr/local/bin/nvbind

# For user installation
chmod +x ~/.local/bin/nvbind
```

### No GPUs detected
```bash
# Check NVIDIA driver status
nvidia-smi

# Verify driver modules loaded
lsmod | grep nvidia

# Check device permissions
ls -la /dev/nvidia*
```

## 📖 Next Steps

- **[Configure for your container runtime](../runtimes/)**
- **[Set up gaming containers](../examples/gaming.md)**
- **[Run AI/ML workloads](../examples/ai-ml.md)**
- **[Optimize performance](../troubleshooting/performance.md)**

---

**Installation complete! Ready to experience blazing-fast GPU containers? 🚀**