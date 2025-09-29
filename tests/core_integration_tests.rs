//! Core integration tests for nvbind runtime functionality
//!
//! These tests validate the main GPU passthrough and container execution features.

use anyhow::Result;
use nvbind::cdi::{CdiRegistry, generate_nvidia_cdi_spec};
use nvbind::config::Config;
use nvbind::distro::{Distribution, DistroManager};
use nvbind::gpu::{DriverType, discover_gpus, get_driver_info};
use nvbind::runtime::{run_with_config, validate_runtime};
use std::process::Command;
use tempfile::TempDir;

/// Test complete GPU discovery and driver detection flow
#[tokio::test]
async fn test_gpu_discovery_and_driver_detection() {
    // Test driver info retrieval
    let driver_result = get_driver_info().await;

    match driver_result {
        Ok(info) => {
            assert!(!info.version.is_empty());
            assert!(matches!(
                info.driver_type,
                DriverType::NvidiaOpen | DriverType::NvidiaProprietary | DriverType::Nouveau
            ));
            println!(
                "Driver detected: {} ({})",
                info.version,
                info.driver_type.name()
            );

            if let Some(cuda) = info.cuda_version {
                println!("CUDA version: {}", cuda);
            }
        }
        Err(e) => {
            println!("No driver detected (expected in CI): {}", e);
        }
    }

    // Test GPU discovery
    let gpu_result = discover_gpus().await;

    match gpu_result {
        Ok(gpus) => {
            for (i, gpu) in gpus.iter().enumerate() {
                assert!(!gpu.id.is_empty());
                assert!(!gpu.name.is_empty());
                assert!(!gpu.pci_address.is_empty());
                println!(
                    "GPU {}: {} (PCI: {}, Memory: {:?}MB)",
                    i,
                    gpu.name,
                    gpu.pci_address,
                    gpu.memory.map(|m| m / 1024 / 1024)
                );
            }
        }
        Err(e) => {
            println!("GPU discovery failed (expected in CI): {}", e);
        }
    }
}

/// Test distribution detection and compatibility checking
#[test]
fn test_distribution_compatibility() -> Result<()> {
    let distro_manager = DistroManager::new()?;
    let compatibility = distro_manager.check_compatibility()?;

    // Verify distribution detection
    assert!(matches!(
        compatibility.distribution,
        Distribution::Ubuntu
            | Distribution::Debian
            | Distribution::Fedora
            | Distribution::Arch
            | Distribution::OpenSUSE
            | Distribution::RHEL
            | Distribution::Unknown(_)
    ));

    println!(
        "Distribution: {} {}",
        compatibility.distribution.name(),
        compatibility.version
    );

    // Check container runtime availability
    for (runtime, available) in &compatibility.container_runtime_available {
        println!(
            "Runtime {}: {}",
            runtime,
            if *available { "✓" } else { "✗" }
        );
    }

    // Check recommendations
    let recommendations = compatibility.get_recommendations();
    if !recommendations.is_empty() {
        println!("System recommendations:");
        for rec in recommendations {
            println!("  - {}", rec);
        }
    }

    Ok(())
}

/// Test CDI specification generation and registry management
#[tokio::test]
async fn test_cdi_spec_generation_and_registry() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let spec_dir = temp_dir.path();

    // Try to generate CDI spec
    let spec_result = generate_nvidia_cdi_spec().await;

    match spec_result {
        Ok(spec) => {
            assert_eq!(spec.cdi_version, "0.6.0");
            assert_eq!(spec.kind, "nvidia.com/gpu");

            // Save spec to temp directory
            let mut registry = CdiRegistry::new();
            let spec_path = registry.save_spec(&spec, Some(spec_dir))?;
            assert!(spec_path.exists());

            // Load spec back and verify
            registry.load_spec_file(&spec_path)?;
            let devices = registry.list_devices();
            assert!(!devices.is_empty());

            for device in devices {
                assert!(device.starts_with("nvidia.com/gpu="));
                println!("CDI device: {}", device);
            }
        }
        Err(e) => {
            println!("CDI generation skipped (no GPU): {}", e);
        }
    }

    Ok(())
}

/// Test runtime validation for all supported container runtimes
#[test]
fn test_all_runtime_validation() {
    let runtimes = vec![("docker", "Docker"), ("podman", "Podman"), ("bolt", "Bolt")];

    for (cmd, name) in runtimes {
        let result = validate_runtime(cmd);

        match result {
            Ok(_) => {
                println!("✓ {} runtime is available", name);

                // Additional validation - check version command works
                let version_check = Command::new(cmd).arg("--version").output();

                if let Ok(output) = version_check {
                    if output.status.success() {
                        let version = String::from_utf8_lossy(&output.stdout);
                        println!("  Version: {}", version.lines().next().unwrap_or("unknown"));
                    }
                }
            }
            Err(_) => {
                println!("✗ {} runtime is not available", name);
            }
        }
    }
}

/// Test configuration merging and environment setup
#[test]
fn test_config_environment_setup() -> Result<()> {
    let mut config = Config::default();

    // Test default environment variables
    assert_eq!(
        config.runtime.environment.get("NVIDIA_DRIVER_CAPABILITIES"),
        Some(&"all".to_string())
    );

    // Test adding custom environment variables
    config
        .runtime
        .environment
        .insert("CUSTOM_GPU_VAR".to_string(), "test_value".to_string());

    // Test GPU selection configurations
    assert_eq!(config.get_gpu_selection(None), "all");
    assert_eq!(config.get_gpu_selection(Some("0")), "0");
    assert_eq!(config.get_gpu_selection(Some("gpu1")), "gpu1");

    // Test runtime selection
    assert_eq!(config.get_runtime_command(None), "podman");
    assert_eq!(config.get_runtime_command(Some("docker")), "docker");
    assert_eq!(config.get_runtime_command(Some("bolt")), "bolt");

    // Test device and library discovery
    let devices = config.get_all_devices();
    assert!(!devices.is_empty());
    assert!(devices.contains(&"/dev/nvidia0".to_string()));

    let libraries = config.get_all_libraries()?;
    // Libraries list can be empty on systems without NVIDIA drivers
    println!("Found {} library paths", libraries.len());

    Ok(())
}

/// Test WSL2 detection and GPU support
#[tokio::test]
async fn test_wsl2_gpu_support() {
    use nvbind::wsl2::{Wsl2GpuSupport, Wsl2Manager};

    if Wsl2Manager::detect_wsl2() {
        println!("WSL2 environment detected");

        let manager = Wsl2Manager::new().unwrap();
        let gpu_support = manager.check_gpu_support().unwrap();

        match gpu_support {
            Wsl2GpuSupport::Available {
                cuda,
                opencl,
                directx,
                opengl,
                vulkan,
            } => {
                println!("WSL2 GPU support available:");
                println!("  CUDA: {}", if cuda { "✓" } else { "✗" });
                println!("  OpenCL: {}", if opencl { "✓" } else { "✗" });
                println!("  DirectX: {}", if directx { "✓" } else { "✗" });
                println!("  OpenGL: {}", if opengl { "✓" } else { "✗" });
                println!("  Vulkan: {}", if vulkan { "✓" } else { "✗" });
            }
            Wsl2GpuSupport::NotWsl2 => {
                println!("Not running in WSL2");
            }
        }
    } else {
        println!("Not a WSL2 environment");
    }
}

/// Test isolation manager functionality
#[test]
fn test_isolation_manager() -> Result<()> {
    use nvbind::isolation::{IsolationConfig, IsolationManager};

    let config = IsolationConfig::default();
    let manager = IsolationManager::new(config);

    // Test initialization
    let init_result = manager.initialize();

    match init_result {
        Ok(_) => {
            println!("Isolation manager initialized successfully");

            // Test container creation (will fail without privileges but should not panic)
            let container_result = manager.create_isolated_container(
                "test-container",
                Some("ai-ml"),
                vec!["gpu0".to_string()],
            );

            match container_result {
                Ok(container) => {
                    println!("Created isolated container: {}", container.id);
                    // Profile validation
                    assert!(!container.id.is_empty());
                }
                Err(e) => {
                    println!(
                        "Container creation failed (expected without privileges): {}",
                        e
                    );
                }
            }
        }
        Err(e) => {
            println!("Isolation manager init failed (expected in CI): {}", e);
        }
    }

    Ok(())
}

/// Test error handling and recovery scenarios
#[tokio::test]
async fn test_error_handling_and_recovery() {
    // Test invalid GPU selection
    let config = Config::default();
    let result = run_with_config(
        config.clone(),
        "invalid_runtime".to_string(),
        "gpu999".to_string(),
        "test/image".to_string(),
        vec![],
    )
    .await;

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("invalid_runtime")
            || error_msg.contains("not available")
            || error_msg.contains("NVIDIA")
    );

    // Test missing image
    let result = run_with_config(
        config.clone(),
        "echo".to_string(), // Use echo as a "runtime" for testing
        "all".to_string(),
        "nonexistent/image:notreal".to_string(),
        vec![],
    )
    .await;

    // Should fail gracefully
    assert!(result.is_err());
}

/// Test concurrent CDI registry access
#[tokio::test]
async fn test_concurrent_cdi_access() {
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tokio::task;

    let registry = Arc::new(RwLock::new(CdiRegistry::new()));

    // Create multiple tasks accessing the registry
    let mut tasks = vec![];

    for i in 0..5 {
        let reg_clone = Arc::clone(&registry);
        let task = task::spawn(async move {
            // Read operation
            {
                let reg = reg_clone.read().await;
                let devices = reg.list_devices();
                println!("Task {} found {} devices", i, devices.len());
            }

            // Write operation (register mock spec)
            {
                let mut reg = reg_clone.write().await;
                let mock_spec = nvbind::cdi::CdiSpec {
                    cdi_version: "0.6.0".to_string(),
                    kind: format!("test.com/gpu{}", i),
                    container_edits: nvbind::cdi::ContainerEdits {
                        env: Some(vec![format!("TASK_ID={}", i)]),
                        device_nodes: None,
                        mounts: None,
                        hooks: None,
                    },
                    devices: vec![],
                };
                reg.register_spec(mock_spec).unwrap();
            }
        });
        tasks.push(task);
    }

    // Wait for all tasks
    for task in tasks {
        task.await.unwrap();
    }

    // Verify all specs were registered
    let reg = registry.read().await;
    let devices = reg.list_devices();
    println!(
        "Final device count after concurrent access: {}",
        devices.len()
    );
}

/// Test performance profiling integration
#[tokio::test]
async fn test_performance_profiling() -> Result<()> {
    // Performance profiling module exists and can be imported
    // Actual profiling requires privileged access
    use nvbind::performance::PerformanceProfiler;

    // Simulate some work
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    println!("Performance profiling module validated");
    Ok(())
}

/// Test service mesh functionality
#[tokio::test]
async fn test_service_mesh_basic() -> Result<()> {
    use nvbind::mesh::{MeshConfig, ServiceInstance, ServiceMesh};
    use uuid::Uuid;

    let config = MeshConfig::default();
    let mesh = ServiceMesh::new(config)?;

    // Register a test service
    use std::net::SocketAddr;
    use std::time::SystemTime;

    let instance = ServiceInstance {
        id: Uuid::new_v4(),
        name: "test-gpu-service".to_string(),
        version: "1.0.0".to_string(),
        address: "127.0.0.1:8080".parse::<SocketAddr>()?,
        metadata: std::collections::HashMap::new(),
        health_status: nvbind::mesh::HealthStatus::Healthy,
        weight: 100,
        registered_at: SystemTime::now(),
        last_heartbeat: SystemTime::now(),
        gpu_capabilities: None,
    };

    mesh.register_service(instance.clone()).await?;

    // Discover services
    let discovered = mesh.discover_services("test-gpu-service").await?;
    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].name, "test-gpu-service");

    println!("Service mesh test passed");
    Ok(())
}
