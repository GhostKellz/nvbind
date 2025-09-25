# nvbind Documentation

Welcome to the nvbind documentation! This guide will help you get started with using nvbind, a lightweight Rust-based alternative to NVIDIA Container Toolkit.

## Quick Links

- [Installation Guide](guides/getting_started.md#installation)
- [API Reference](api/)
- [Code Examples](examples/)
- [Advanced Configuration](guides/advanced_configuration.md)

## API Reference

- [GPU Management](api/gpu.md) - Core GPU discovery and management functionality
- [Role-Based Access Control](api/rbac.md) - User and group permission management for GPU resources
- [Container Runtime](api/runtime.md) - Container orchestration with GPU support
- [Advanced GPU Features](api/gpu_advanced.md) - MIG, scheduling, and enterprise features
- [Performance Optimization](api/performance.md) - System tuning and optimization
- [Kubernetes Integration](api/k8s.md) - Kubernetes and orchestration platform integration
- [Monitoring](api/monitoring.md) - Performance monitoring and metrics collection

## Examples

### GPU
- [Basic GPU Discovery](examples/basic_gpu_discovery.md) - Discover and list all available GPUs
- [GPU Monitoring](examples/gpu_monitoring.md) - Monitor GPU usage and performance

### RBAC
- [RBAC Setup](examples/rbac_setup.md) - Set up role-based access control
- [Permission Checking](examples/permission_checking.md) - Check user permissions

### CONTAINER
- [Container with GPU](examples/container_with_gpu.md) - Run a container with GPU access

### GPU_ADVANCED
- [MIG Configuration](examples/mig_configuration.md) - Configure Multi-Instance GPU (MIG)

## User Guides

- [Getting Started with nvbind](guides/getting_started.md) - Complete guide to installing and using nvbind (🟢 Beginner)
- [Advanced Configuration](guides/advanced_configuration.md) - Detailed guide for advanced nvbind configuration (🔴 Advanced)

## System Requirements

- Linux (Arch, Ubuntu, Debian, Fedora, PopOS)
- NVIDIA GPU with CUDA support
- NVIDIA Driver (Open or Proprietary)
- Container runtime (Docker, Podman, or Containerd)
- Rust 1.70+ (for building from source)

## Support

- **Issues:** [GitHub Issues](https://github.com/ghostkellz/nvbind/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ghostkellz/nvbind/discussions)
- **Documentation:** This site
