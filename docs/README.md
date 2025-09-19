# nvbind Documentation

Welcome to the comprehensive documentation for **nvbind** - the lightning-fast, Rust-based alternative to NVIDIA Container Toolkit.

## 📚 Documentation Structure

### 🚀 [Installation](installation/)
- **[Quick Install](installation/quick-install.md)** - Get started in minutes
- **[System Requirements](installation/requirements.md)** - Hardware and software prerequisites
- **[Distribution Guides](installation/distros.md)** - Installation for specific Linux distributions
- **[WSL2 Setup](installation/wsl2.md)** - Windows Subsystem for Linux 2 configuration

### 🛠️ [Container Runtimes](runtimes/)
- **[Bolt Integration](runtimes/bolt.md)** - Native Bolt runtime support with GPU acceleration
- **[Docker Usage](runtimes/docker.md)** - Drop-in replacement for Docker GPU workflow
- **[Podman Usage](runtimes/podman.md)** - Enhanced Podman GPU container support

### 🎯 [Examples](examples/)
- **[Gaming Containers](examples/gaming.md)** - Steam, Lutris, Wine/Proton gaming
- **[AI/ML Workloads](examples/ai-ml.md)** - PyTorch, TensorFlow, Ollama training
- **[Development Setup](examples/development.md)** - CUDA development containers

### 🔧 [Troubleshooting](troubleshooting/)
- **[Common Issues](troubleshooting/common.md)** - Frequently encountered problems
- **[GPU Detection](troubleshooting/gpu-detection.md)** - Driver and hardware issues
- **[Performance](troubleshooting/performance.md)** - Optimization and benchmarking

### 📖 [API Reference](api/)
- **[CLI Commands](api/cli.md)** - Complete command reference
- **[Configuration](api/configuration.md)** - TOML configuration options
- **[CDI Specifications](api/cdi.md)** - Container Device Interface details

## ⚡ Quick Start

```bash
# Install nvbind
curl -sSL https://raw.githubusercontent.com/ghostkellz/nvbind/main/install.sh | sudo bash

# Check GPU detection
nvbind info

# Run your first GPU container
nvbind run --runtime bolt --gpu all --profile gaming steam:latest
```

## 🎮 Why nvbind?

| Feature | nvbind | NVIDIA Container Toolkit |
|---------|--------|-------------------------|
| **Performance** | ⚡ Sub-microsecond GPU passthrough | 🐌 ~10ms overhead |
| **Gaming** | 🎮 99%+ native performance | 📉 85-90% performance |
| **Runtimes** | 🚀 Bolt, Docker, Podman | 🐳 Docker-only |
| **Memory Safety** | 🦀 Rust - zero buffer overflows | ❌ C++ memory vulnerabilities |
| **Driver Support** | 🌐 NVIDIA Open, Proprietary, Nouveau | 🔒 NVIDIA proprietary only |

## 🔗 Quick Links

- **[GitHub Repository](https://github.com/ghostkellz/nvbind)**
- **[Issue Tracker](https://github.com/ghostkellz/nvbind/issues)**
- **[Performance Benchmarks](https://github.com/ghostkellz/nvbind/wiki/benchmarks)**
- **[Community Discord](https://discord.gg/nvbind)** *(coming soon)*

## 💡 Need Help?

1. **Check the [troubleshooting guide](troubleshooting/common.md)**
2. **Run diagnostics**: `nvbind doctor`
3. **Open an issue**: [GitHub Issues](https://github.com/ghostkellz/nvbind/issues)

---

**Ready to revolutionize your GPU container workflow? Let's get started! 🚀**