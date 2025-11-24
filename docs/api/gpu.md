# GPU Management API Reference

Core GPU discovery and management functionality

## Table of Contents

- [Functions](#functions)
- [Structs](#structs)
- [Examples](#examples)

## Functions

### discover_gpus

Discover all available NVIDIA GPUs on the system

**Returns:** `Result<Vec<GpuDevice>>`

**Errors:**

- No NVIDIA GPUs found
- NVIDIA driver not available

**Example:**

```rust
use nvbind::gpu;

let gpus = gpu::discover_gpus().await?;
for gpu in gpus {
    println!("Found GPU: {} ({}MB)", gpu.name, gpu.memory_mb);
}
```

### get_driver_info

Get NVIDIA driver information and version

**Returns:** `Result<DriverInfo>`

**Errors:**
- Driver information unavailable

**Example:**

```rust
use nvbind::gpu;

let driver = gpu::get_driver_info().await?;
println!("Driver: {} ({})", driver.version, driver.driver_type);
```

## Structs

### GpuDevice

Represents a single GPU device

**Fields:**

- `id`: `String` - GPU identifier (required)
- `name`: `String` - GPU model name (required)
- `pci_address`: `String` - PCI bus address (required)
- `driver_version`: `Option<String>` - NVIDIA driver version (optional)
- `memory`: `Option<u64>` - GPU memory in megabytes (optional)
- `device_path`: `String` - Device file path (e.g., /dev/nvidia0) (required)
- `architecture`: `Option<GpuArchitecture>` - GPU architecture (optional, new in 1.0)
- `compute_capability`: `Option<(u32, u32)>` - CUDA compute capability (optional, new in 1.0)

**Example:**

```json
{
    "id": "0",
    "name": "NVIDIA GeForce RTX 5090",
    "pci_address": "0000:01:00.0",
    "driver_version": "580.105.08",
    "memory": 33554432,
    "device_path": "/dev/nvidia0",
    "architecture": "Blackwell",
    "compute_capability": [10, 0]
}
```

### GpuArchitecture

Represents GPU architecture generation

**Variants:**

- `Maxwell` - GTX 900 series, Compute 5.x (not supported)
- `Pascal` - GTX 10 series, Compute 6.x (not supported)
- `Volta` - TITAN V, Compute 7.0 (not supported)
- `Turing` - RTX 20 series, Compute 7.5 (not supported)
- `Ampere` - **RTX 30 series, Compute 8.6** (supported)
- `AdaLovelace` - **RTX 40 series, Compute 8.9** (supported)
- `Hopper` - H100, Compute 9.0 (supported)
- `Blackwell` - **RTX 50 series, Compute 10.0** (supported)
- `Unknown` - Architecture detection failed

**Methods:**

```rust
impl GpuArchitecture {
    /// Check if GPU supports Multi-Instance GPU (MIG)
    pub fn supports_mig(&self) -> bool;

    /// Check if GPU supports FP4 Tensor Core precision
    pub fn supports_fp4(&self) -> bool;

    /// Get Tensor Core generation (1-5, or None)
    pub fn tensor_core_generation(&self) -> Option<u8>;
}
```

**Example Usage:**

```rust
use nvbind::gpu::{GpuDevice, GpuArchitecture};

let gpu = discover_gpus().await?.first().unwrap();

if let Some(arch) = &gpu.architecture {
    println!("Architecture: {:?}", arch);

    if arch.supports_fp4() {
        println!("FP4 Tensor Cores available (Blackwell)");
    }

    if arch.supports_mig() {
        println!("MIG support available");
    }

    if let Some(tc_gen) = arch.tensor_core_generation() {
        println!("Tensor Core Generation: {}", tc_gen);
    }
}

if let Some((major, minor)) = gpu.compute_capability {
    println!("Compute Capability: {}.{}", major, minor);
}
```

## Examples

- [basic_gpu_discovery](../examples/basic_gpu_discovery.md)
- [gpu_monitoring](../examples/gpu_monitoring.md)

