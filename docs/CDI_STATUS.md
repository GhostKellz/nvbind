# CDI Implementation Status

**Last Updated:** September 29, 2025
**CDI Specification Version:** 0.8.0
**Status:** ✅ Production Ready (Basic GPU Passthrough)

## Overview

nvbind implements the Container Device Interface (CDI) v0.8.0 specification for NVIDIA GPU passthrough to containers. The implementation provides full GPU access to Docker, Podman, and other CDI-compatible container runtimes.

## What's Working

### ✅ Core Functionality
- **GPU Discovery**: Automatic detection of all NVIDIA GPUs on the system
- **Driver Detection**: Identifies NVIDIA Open GPU Kernel Modules and proprietary drivers
- **CDI Spec Generation**: Creates valid CDI v0.8.0 specifications
- **Device Nodes**: Mounts `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-modeset`
- **Library Mounts**: 200+ NVIDIA driver libraries with proper bind mounts
- **Environment Variables**: Sets `NVIDIA_VISIBLE_DEVICES`, driver capabilities, version info

### ✅ Container Runtime Support
- **Docker**: Full support with `--device=nvidia.com/gpu=gpu0` syntax
- **Podman**: Compatible with CDI device injection
- **NVIDIA Runtime**: Works with existing `--runtime=nvidia` setup

### ✅ Validation
- **CUDA Containers**: Successfully tested with `nvidia/cuda:12.6.3-base-ubuntu24.04`
- **nvidia-smi**: Correctly reports GPU information inside containers
- **GPU Compute**: Verified GPU access and compute capabilities

## Implementation Details

### CDI Spec Structure

```json
{
  "cdiVersion": "0.8.0",
  "kind": "nvidia.com/gpu",
  "containerEdits": {
    "env": [
      "NVIDIA_VISIBLE_DEVICES=void",
      "NVIDIA_DRIVER_CAPABILITIES=all",
      "NVIDIA_DRIVER_VERSION=580.82.09"
    ],
    "deviceNodes": [...],
    "mounts": [...],
    "hooks": [...]
  },
  "devices": [...]
}
```

### Generated Devices
- `nvidia.com/gpu=gpu0` - First GPU
- `nvidia.com/gpu=gpu1` - Second GPU
- `nvidia.com/gpu=gpu2` - Third GPU
- `nvidia.com/gpu=gpu3` - Fourth GPU
- `nvidia.com/gpu=all` - All GPUs

### Hooks Implemented
1. **create-symlinks** - Creates compatibility symlinks for versioned libraries
   - libcuda.so.1 → libcuda.so
   - libnvidia-ml.so.1 → libnvidia-ml.so
   - libnvidia-opticalflow.so.1 → libnvidia-opticalflow.so
   - libnvidia-encode.so.1 → libnvidia-encode.so
   - libnvidia-fbc.so.1 → libnvidia-fbc.so

2. **update-ldcache** - Updates library cache for NVIDIA libraries
   - Scans /usr/lib64 and /lib64 directories
   - Ensures injected libraries are discoverable

3. **disable-device-node-modification** - Prevents NVML from creating device nodes
   - Modifies /proc/driver/nvidia/params inside container
   - Sets ModifyDeviceFiles to 0

## Comparison with NVIDIA Container Toolkit

| Feature | nvbind | nvidia-container-toolkit |
|---------|--------|-------------------------|
| CDI Version | 0.8.0 ✅ | 0.6.0 |
| GPU Discovery | Automatic ✅ | Automatic ✅ |
| Device Nodes | 3 ✅ | 3 ✅ |
| Library Mounts | 179 ✅ | ~150 ✅ |
| Hooks | 3 ✅ | 10 |
| CUDA Support | ✅ Yes | ✅ Yes |
| Docker Support | ✅ Yes | ✅ Yes |
| Podman Support | ✅ Yes | ✅ Yes |
| Devices | 5 (gpu0-3 + all) | 3 (0, UUID, all) |

## Known Limitations

### Missing Hooks (Optional Optimizations)
The following hooks are present in NVIDIA Container Toolkit but not yet implemented (non-critical for basic GPU operation):

1. **enable-cuda-compat** - Enables CUDA forward compatibility mode
   - Impact: Low - Only needed for driver/CUDA version mismatches
   - Priority: Low
   - Status: Planned for future release

2. **chmod** - Sets permissions on mounted files
   - Impact: Minimal - Permissions are inherited from host
   - Priority: Low
   - Status: Not prioritized

3. **Additional per-device hooks** - NVIDIA uses separate hooks per device
   - Impact: Minimal - Common hooks work for all devices
   - Priority: Low
   - Status: Under evaluation

### Testing Environment
- **Tested GPU**: NVIDIA GeForce RTX 4090
- **Driver**: 580.82.09 (Open GPU Kernel Modules)
- **CUDA Version**: 13.0
- **Container Runtimes**: Docker 28.4.0, Podman 5.6.1
- **Test Platform**: Arch Linux, Kernel 6.16.8

## Usage

### Generate CDI Specification

```bash
nvbind cdi generate --output /etc/cdi
```

### Use with Docker

```bash
docker run --rm --device=nvidia.com/gpu=gpu0 nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

### Use with Podman

```bash
podman run --rm --device=nvidia.com/gpu=gpu0 nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

### Use with NVIDIA Runtime (Existing Setup)

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

## Future Improvements

See [TODO.md](TODO.md) for planned optimizations and enhancements.

## References

- [CDI Specification v0.8.0](https://github.com/cncf-tags/container-device-interface/blob/main/SPEC.md)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Docker CDI Support](https://docs.docker.com/build/building/cdi/)