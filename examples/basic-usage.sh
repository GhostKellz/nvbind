#!/bin/bash
# nvbind Basic Usage Examples

set -e

echo "=== nvbind Basic Usage Examples ==="
echo

# Example 1: Generate CDI specification
echo "1. Generating CDI specification..."
nvbind cdi generate --output /etc/cdi
echo "✓ CDI spec generated at /etc/cdi/nvidia.com_gpu.json"
echo

# Example 2: Check GPU information
echo "2. Checking GPU information..."
nvbind info
echo

# Example 3: List available CDI devices
echo "3. Available CDI devices:"
nvbind cdi list
echo

# Example 4: Run nvidia-smi in container with specific GPU
echo "4. Running nvidia-smi with GPU 0..."
docker run --rm \
  --device=nvidia.com/gpu=gpu0 \
  nvidia/cuda:12.6.3-base-ubuntu24.04 \
  nvidia-smi
echo

# Example 5: Run container with all GPUs
echo "5. Running with all GPUs..."
docker run --rm \
  --device=nvidia.com/gpu=all \
  nvidia/cuda:12.6.3-base-ubuntu24.04 \
  nvidia-smi -L
echo

# Example 6: Using NVIDIA runtime (alternative)
echo "6. Using NVIDIA runtime..."
docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  nvidia/cuda:12.6.3-base-ubuntu24.04 \
  nvidia-smi
echo

# Example 7: Podman usage
echo "7. Using with Podman..."
podman run --rm \
  --device=nvidia.com/gpu=gpu0 \
  nvidia/cuda:12.6.3-base-ubuntu24.04 \
  nvidia-smi
echo

echo "=== All examples completed successfully! ==="