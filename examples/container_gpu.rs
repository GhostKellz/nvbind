//! Container with GPU Example
//!
//! This example demonstrates how to run a container with GPU access
//! using nvbind's runtime API.

use nvbind::{runtime, config::Config};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    println!("🐳 Creating container with GPU access...\n");

    // Load configuration (or use defaults)
    let config = Config::load_or_default("/etc/nvbind/config.toml")?;

    // Create container specification
    let container_spec = runtime::ContainerSpec {
        image: "nvidia/cuda:12.0-runtime-ubuntu22.04".to_string(),
        command: vec!["nvidia-smi".to_string()],
        gpu_request: Some(runtime::GpuRequest {
            count: 1,
            memory_mb: Some(4096), // 4GB GPU memory
            capabilities: vec!["compute".to_string(), "utility".to_string()],
            device_ids: None, // Use any available GPU
        }),
        environment: vec![
            ("NVIDIA_VISIBLE_DEVICES".to_string(), "all".to_string()),
            ("NVIDIA_DRIVER_CAPABILITIES".to_string(), "compute,utility".to_string()),
        ],
        working_dir: Some("/workspace".to_string()),
        remove_on_exit: true,
        ..Default::default()
    };

    println!("📋 Container Specification:");
    println!("   Image: {}", container_spec.image);
    println!("   Command: {:?}", container_spec.command);
    println!("   GPU Count: {}", container_spec.gpu_request.as_ref().unwrap().count);
    println!("   GPU Memory: {} MB", container_spec.gpu_request.as_ref().unwrap().memory_mb.unwrap_or(0));
    println!();

    // Create runtime (prefer podman, fallback to docker)
    let runtime_name = if runtime::validate_runtime("podman").is_ok() {
        "podman"
    } else if runtime::validate_runtime("docker").is_ok() {
        "docker"
    } else {
        return Err(anyhow::anyhow!("No supported container runtime found (podman or docker)"));
    };

    println!("🚀 Using {} runtime", runtime_name);
    let runtime = runtime::create_runtime(runtime_name, &config)?;

    // Run container
    println!("⏳ Running container...\n");
    let result = runtime.run_container(container_spec).await?;

    println!("✅ Container execution completed!");
    println!("   Exit Code: {}", result.exit_code);
    println!();

    if !result.stdout.is_empty() {
        println!("📊 Container Output:");
        println!("{}", result.stdout);
    }

    if !result.stderr.is_empty() {
        println!("⚠️  Container Errors:");
        println!("{}", result.stderr);
    }

    println!("\n🎉 Example completed successfully!");

    Ok(())
}
