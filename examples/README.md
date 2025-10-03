# nvbind Examples

Practical examples demonstrating nvbind usage with various container runtimes and workloads.

## Quick Start

### Generate CDI Specification

```bash
nvbind cdi generate --output /etc/cdi
```

### Basic Docker Usage

```bash
# Run nvidia-smi with specific GPU
docker run --rm --device=nvidia.com/gpu=gpu0 nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi

# Use all GPUs
docker run --rm --device=nvidia.com/gpu=all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi

# Alternative: NVIDIA runtime
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

### Basic Podman Usage

```bash
podman run --rm --device=nvidia.com/gpu=gpu0 nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

## Examples

### 1. Basic Usage Script
[`basic-usage.sh`](basic-usage.sh) - Comprehensive script demonstrating all basic nvbind operations

```bash
./examples/basic-usage.sh
```

### 2. Docker Compose
[`docker-compose.yml`](docker-compose.yml) - Production-ready compose configurations

```bash
cd examples
docker-compose up cuda-test        # Basic GPU test
docker-compose up pytorch          # PyTorch training
docker-compose up distributed-training  # Multi-GPU training
docker-compose up jupyter          # Jupyter with GPU
```

### 3. PyTorch Training
[`pytorch-training.py`](pytorch-training.py) - Complete PyTorch training example

```bash
docker run --rm \
  --device=nvidia.com/gpu=gpu0 \
  -v $(pwd)/examples:/workspace \
  pytorch/pytorch:2.5.1-cuda12.6-cudnn9-runtime \
  python /workspace/pytorch-training.py
```

### 4. CUDA Development
[`cuda-dev/`](cuda-dev/) - CUDA C++ development examples

```bash
docker run --rm \
  --device=nvidia.com/gpu=gpu0 \
  -v $(pwd)/examples/cuda-dev:/workspace \
  nvidia/cuda:12.6.3-devel-ubuntu24.04 \
  bash -c "cd /workspace && nvcc vector_add.cu -o vector_add && ./vector_add"
```

## Container Images Tested

### CUDA Base Images
- `nvidia/cuda:12.6.3-base-ubuntu24.04` - Minimal CUDA runtime
- `nvidia/cuda:12.6.3-devel-ubuntu24.04` - CUDA development tools

### ML/AI Frameworks
- `pytorch/pytorch:2.5.1-cuda12.6-cudnn9-runtime`
- `tensorflow/tensorflow:latest-gpu`
- `nvcr.io/nvidia/pytorch:25.01-py3`

### Specialized
- `ollama/ollama:latest` - LLM inference
- `jupyter/pytorch-notebook:latest` - Interactive notebooks

## Multi-GPU Examples

### All GPUs
```bash
docker run --rm --device=nvidia.com/gpu=all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

### Specific GPUs
```bash
# Use GPU 0 and GPU 1
docker run --rm \
  --device=nvidia.com/gpu=gpu0 \
  --device=nvidia.com/gpu=gpu1 \
  nvidia/cuda:12.6.3-base-ubuntu24.04 \
  nvidia-smi
```

## Environment Variables

nvbind sets the following environment variables in containers:

- `NVIDIA_VISIBLE_DEVICES` - Which GPUs are visible (defaults to `void`)
- `NVIDIA_DRIVER_CAPABILITIES` - Driver capabilities (defaults to `all`)
- `NVIDIA_DRIVER_VERSION` - Driver version (e.g., `580.82.09`)
- `NVIDIA_DRIVER_TYPE` - Driver type (`NvidiaOpen` or `Proprietary`)

## Troubleshooting

### Check GPU Detection
```bash
nvbind info
```

### Validate CDI Spec
```bash
cat /etc/cdi/nvidia.com_gpu.json | jq '.'
```

### List Available Devices
```bash
docker info | grep -A 10 "CDI spec"
```

### Test Basic GPU Access
```bash
docker run --rm --device=nvidia.com/gpu=gpu0 \
  nvidia/cuda:12.6.3-base-ubuntu24.04 \
  nvidia-smi
```

## Advanced Usage

### Custom CDI Output Location
```bash
nvbind cdi generate --output /custom/path
```

### Debug Mode
```bash
RUST_LOG=debug nvbind cdi generate
```

### Regenerate After Driver Update
```bash
nvbind cdi generate --output /etc/cdi --force
```

## Performance Tips

1. **Use specific GPU devices** instead of `all` when possible
2. **Pin containers to GPUs** to avoid scheduling overhead
3. **Monitor GPU usage** with `nvidia-smi` or `nvbind monitor`
4. **Use persistent mode** for production workloads

## Contributing

Have a useful example? Please submit a PR with:
- Working code/configuration
- Clear documentation
- Dockerfile if needed
- Expected output

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.