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

- `index`: `u32` - GPU index (0-based) (required)
- `name`: `String` - GPU model name (required)
- `memory_mb`: `u64` - GPU memory in megabytes (required)
- `uuid`: `String` - Unique GPU identifier (required)

**Example:**

```json
{
    "index": 0,
    "name": "NVIDIA GeForce RTX 4090",
    "memory_mb": 24576,
    "uuid": "GPU-12345678-1234-1234-1234-123456789abc"
}
```

## Examples

- [basic_gpu_discovery](../examples/basic_gpu_discovery.md)
- [gpu_monitoring](../examples/gpu_monitoring.md)

