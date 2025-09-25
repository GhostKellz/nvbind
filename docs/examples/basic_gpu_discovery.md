# BASIC GPU DISCOVERY

Discover and list all available GPUs

**Category:** gpu

## Code

```rust
use nvbind::gpu;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    // Discover GPUs
    let gpus = gpu::discover_gpus().await?;

    println!("Found {} GPU(s):", gpus.len());
    for (i, gpu) in gpus.iter().enumerate() {
        println!("  GPU {}: {}", i, gpu.name);
        println!("    Memory: {} MB", gpu.memory_mb);
        println!("    UUID: {}", gpu.uuid);
        println!("    PCI Bus ID: {}", gpu.pci_bus_id);
    }

    Ok(())
}
```

## Expected Output

```
Found 1 GPU(s):
  GPU 0: NVIDIA GeForce RTX 4090
    Memory: 24576 MB
    UUID: GPU-12345678-1234-1234-1234-123456789abc
    PCI Bus ID: 0000:01:00.0
```

## Running the Example

1. Ensure you have NVIDIA GPU and drivers installed
2. Install nvbind: `cargo install nvbind`
3. Copy the code above to a file (e.g., `example.rs`)
4. Run: `cargo run --bin example`

