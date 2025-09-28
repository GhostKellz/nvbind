//! Basic GPU Discovery Example
//!
//! This example demonstrates how to discover and list all available NVIDIA GPUs
//! on the system using nvbind's GPU management API.

use anyhow::Result;
use nvbind::gpu;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🔍 Discovering NVIDIA GPUs...\n");

    // Discover GPUs
    let gpus = gpu::discover_gpus().await?;

    if gpus.is_empty() {
        println!("❌ No NVIDIA GPUs found on this system");
        println!("   Make sure you have:");
        println!("   - NVIDIA GPU installed");
        println!("   - NVIDIA drivers installed");
        println!("   - Proper permissions to access GPU");
        return Ok(());
    }

    println!("✅ Found {} GPU(s):\n", gpus.len());

    for (i, gpu) in gpus.iter().enumerate() {
        println!("🎮 GPU {}:", i);
        println!("   Name: {}", gpu.name);
        println!(
            "   Memory: {} MB ({:.1} GB)",
            gpu.memory.unwrap_or(0),
            gpu.memory.unwrap_or(0) as f64 / 1024.0 / 1024.0 / 1024.0
        );
        println!("   PCI Address: {}", gpu.pci_address);
        println!("   Driver Version: {}", gpu.driver_version.as_deref().unwrap_or("Unknown"));
        println!("   Device Path: {}", gpu.device_path);

        println!();
    }

    // Get driver information
    match gpu::get_driver_info().await {
        Ok(driver) => {
            println!("🖥️  Driver Information:");
            println!("   Version: {}", driver.version);
            println!("   Type: {:?}", driver.driver_type);
            println!(
                "   CUDA Version: {}",
                driver.cuda_version.unwrap_or_else(|| "Unknown".to_string())
            );
        }
        Err(e) => {
            println!("⚠️  Could not get driver information: {}", e);
        }
    }

    Ok(())
}
