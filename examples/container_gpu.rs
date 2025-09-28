//! Container with GPU Example
//!
//! This example demonstrates how to run a container with GPU access
//! using nvbind's runtime API.

use anyhow::Result;
use nvbind::{config::Config, plugin::ContainerSpec};
use std::collections::HashMap;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🐳 Creating container with GPU access...\n");

    // Load configuration (or use defaults)
    let config = Config::load_from_file(Path::new("/etc/nvbind/config.toml")).unwrap_or_default();

    // Create container specification
    let mut environment = HashMap::new();
    environment.insert("NVIDIA_VISIBLE_DEVICES".to_string(), "all".to_string());
    environment.insert("NVIDIA_DRIVER_CAPABILITIES".to_string(), "compute,utility".to_string());

    let container_spec = ContainerSpec {
        image: "nvidia/cuda:12.0-runtime-ubuntu22.04".to_string(),
        name: Some("nvbind-test".to_string()),
        command: vec!["nvidia-smi".to_string()],
        environment,
    };

    println!("📋 Container Specification:");
    println!("   Image: {}", container_spec.image);
    println!("   Command: {:?}", container_spec.command);
    println!("   Environment vars: {} set", container_spec.environment.len());
    println!();

    // This is a demonstration of how to create container specifications
    // In practice, this would be used with a runtime adapter
    println!("📝 This container spec would be used with nvbind's runtime adapters:");
    println!("   - Docker adapter: nvbind run --runtime docker");
    println!("   - Podman adapter: nvbind run --runtime podman");
    println!("   - Bolt adapter: nvbind run --runtime bolt");

    println!("\n🎉 Container specification created successfully!");

    Ok(())
}
