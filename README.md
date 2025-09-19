# nvbind

<div align="center">
  <img src="https://raw.githubusercontent.com/ghostkellz/nvbind/main/assets/nvbind-logo.png" alt="nvbind logo" width="128" height="128">

  **nvbind** – A lightning-fast, Rust-based alternative to NVIDIA Container Toolkit
  *Next-gen GPU passthrough for modern container workflows*

  <p>
    <img src="https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=green" alt="NVIDIA">
    <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=blue" alt="Docker">
    <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=black" alt="Rust">
    <img src="https://img.shields.io/badge/Podman-892CA0?style=for-the-badge&logo=podman&logoColor=white" alt="Podman">
  </p>
</div>

---

## ✨ Overview

`nvbind` is a **cutting-edge NVIDIA container runtime** engineered in Rust for maximum performance and reliability. It provides **blazing-fast, secure GPU passthrough** for modern container orchestration:

- ⚡ **Lightning Fast**: Sub-microsecond operations, zero-overhead design
- 🔐 **Memory Safe**: Built in Rust — immune to buffer overflows and memory corruption
- 🛠 **Universal**: Native support for **Docker**, **Podman**, and **Bolt**
- 🧠 **Intelligent**: Auto-detects NVIDIA Open, proprietary, and Nouveau drivers
- 🎯 **Production Ready**: Comprehensive testing, benchmarks, and CI/CD

**The future of GPU containerization** — simpler, faster, and more secure than legacy toolkits.

---

## 🚀 Features

### **🎯 Core Capabilities**
- ✅ **Smart GPU Discovery** - Auto-detects discrete GPUs, vGPUs, and multi-GPU setups
- ✅ **Universal Driver Support** - NVIDIA Open GPU Kernel Modules, proprietary, and Nouveau
- ✅ **Dynamic Library Mounting** - Automatic detection and bind-mounting of GPU libraries
- ✅ **Rootless Containers** - Full support for unprivileged container execution

### **🛠 Container Integration**
- ✅ **Docker Integration** - Native GPU passthrough with `--runtime nvbind`
- ✅ **Podman Support** - Seamless GPU access for Podman containers
- ✅ **Bolt Orchestration** - Works with `bolt surge up` for complex deployments
- ✅ **Drop-in Replacement** - Compatible with `nvidia-docker2` workflows

### **⚙️ Configuration & Management**
- ✅ **TOML Configuration** - Declarative GPU profiles and runtime settings
- ✅ **Runtime Validation** - Pre-flight checks for container runtime availability
- ✅ **Comprehensive Logging** - Detailed tracing for debugging and monitoring
- 🧪 **Advanced Features** - GPU isolation and sandboxing helpers (experimental)  

---

## 🔧 Installation

### Quick Install (Recommended)
```sh
# System-wide installation
curl -sSL https://raw.githubusercontent.com/ghostkellz/nvbind/main/install.sh | sudo bash

# User installation (no root required)
curl -sSL https://raw.githubusercontent.com/ghostkellz/nvbind/main/install.sh | bash -s -- --user
```

### Manual Build
```sh
# Clone and build
git clone https://github.com/ghostkellz/nvbind
cd nvbind
cargo build --release

# Install system-wide
sudo cp target/release/nvbind /usr/local/bin/
```

## 📦 Usage
```sh
# Show detected GPUs & driver info
nvbind info

# Generate default configuration
nvbind config

# Show current configuration
nvbind config --show

# Run a container with GPU passthrough
nvbind run --runtime docker --gpu all ubuntu nvidia-smi
nvbind run --runtime podman ubuntu:22.04 ls /dev/nvidia*

# Use default runtime from config
nvbind run ubuntu:22.04 nvidia-smi
```

## 🧩 Integration
Boltfile (TOML)
[services.ml]
image = "pytorch/pytorch:latest"
gpus = "all"
runtime = "nvbind"


Podman/Docker
Set --runtime=nvbind or configure default-runtime in your engine config.

## ⚡ Why nvbind?

<div align="center">

| Feature | nvbind | NVIDIA Container Toolkit |
|---------|--------|-------------------------|
| **Language** | Rust 🦀 | C++ |
| **Memory Safety** | ✅ Zero buffer overflows | ❌ Vulnerable to memory bugs |
| **Performance** | ⚡ Sub-microsecond operations | 🐌 Higher overhead |
| **Driver Support** | 🧠 NVIDIA Open, proprietary, Nouveau | 🔒 NVIDIA proprietary only |
| **Configuration** | 📝 Modern TOML | 🗂️ Legacy JSON/environment vars |
| **Container Support** | 🛠️ Docker, Podman, Bolt | 🐳 Docker-centric |
| **Testing** | ✅ 30+ unit tests, benchmarks | ❓ Limited public testing |

</div>

### **🚀 Built for the Future**
- **Cloud-Native**: Ready for Kubernetes, service meshes, and edge computing
- **Extensible**: Plugin architecture for custom GPU management
- **Standards-Compliant**: CDI (Container Device Interface) support planned
- **Security-First**: Designed with zero-trust and supply chain security in mind

---

## 🛣 Roadmap

- 🎯 **CDI Support** - Container Device Interface standard compliance
- 🔒 **Advanced GPU Isolation** - Namespace-based GPU sandboxing
- 🪟 **WSL2 Support** - Windows Subsystem for Linux integration
- 📦 **Secure Distribution** - Signed manifests for GPU driver packages
- 📊 **Performance Dashboard** - Real-time GPU utilization monitoring

