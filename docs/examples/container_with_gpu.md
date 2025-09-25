# CONTAINER WITH GPU

Run a container with GPU access

**Category:** runtime

## Code

```rust
use nvbind::{runtime, config::Config};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = Config::load("/etc/nvbind/config.toml")?;

    // Create container specification
    let container_spec = runtime::ContainerSpec {
        image: "nvidia/cuda:12.0-runtime-ubuntu22.04".to_string(),
        command: vec!["nvidia-smi".to_string()],
        gpu_request: Some(runtime::GpuRequest {
            count: 1,
            memory_mb: Some(4096),
            capabilities: vec!["compute".to_string(), "utility".to_string()],
        }),
        ..Default::default()
    };

    // Run container
    let runtime = runtime::create_runtime("podman", &config)?;
    let result = runtime.run_container(container_spec).await?;

    println!("Container output:\n{}", result.stdout);

    Ok(())
}
```

## Expected Output

```
Container output:
Wed Sep 25 10:30:00 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.42                 Driver Version: 580.42         CUDA Version: 12.0 |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M.     |
|                               |                      |               MIG M.     |
|=======================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   35C    P8    25W / 450W |      0MiB / 24576MiB |      0%      Default    |
|                               |                      |                  N/A     |
+---------------------------------------------------------------------------------------+
```

## Running the Example

1. Ensure you have NVIDIA GPU and drivers installed
2. Install nvbind: `cargo install nvbind`
3. Copy the code above to a file (e.g., `example.rs`)
4. Run: `cargo run --bin example`

