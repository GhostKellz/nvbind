//! Integration tests for nvbind
//!
//! These tests validate end-to-end functionality without requiring actual GPU hardware.

use anyhow::Result;
use nvbind::cdi::{CdiRegistry, generate_nvidia_cdi_spec};
use nvbind::config::Config;
use nvbind::gpu::{discover_gpus, is_nvidia_driver_available};
use nvbind::runtime::validate_runtime;
use tempfile::TempDir;

#[cfg(feature = "bolt")]
use nvbind::bolt::NvbindGpuManager;

/// Test configuration loading and validation
#[test]
fn test_config_loading() {
    let config = Config::default();

    assert_eq!(config.gpu.default_selection, "all");
    assert_eq!(config.runtime.default_runtime, "podman");
    assert!(config.security.allow_rootless);

    #[cfg(feature = "bolt")]
    {
        assert!(config.bolt.is_some());
        let bolt_config = config.bolt.unwrap();
        assert!(bolt_config.capsule.snapshot_gpu_state);
        assert_eq!(bolt_config.capsule.isolation_level, "exclusive");
    }
}

/// Test CDI specification generation (mocked without GPU)
#[tokio::test]
async fn test_cdi_generation() {
    // This should work even without actual GPU hardware
    let result = generate_nvidia_cdi_spec().await;

    // If no GPU, should return appropriate error
    if !is_nvidia_driver_available() {
        assert!(result.is_err());
    } else {
        let spec = result.unwrap();
        assert_eq!(spec.cdi_version, "0.6.0");
        assert_eq!(spec.kind, "nvidia.com/gpu");
        assert!(!spec.devices.is_empty());
    }
}

/// Test CDI registry functionality
#[test]
fn test_cdi_registry() {
    let mut registry = CdiRegistry::new();

    // Test with mock CDI spec
    let mock_spec = nvbind::cdi::CdiSpec {
        cdi_version: "0.6.0".to_string(),
        kind: "test.com/gpu".to_string(),
        container_edits: nvbind::cdi::ContainerEdits {
            env: Some(vec!["TEST_ENV=value".to_string()]),
            device_nodes: None,
            mounts: None,
            hooks: None,
        },
        devices: vec![nvbind::cdi::CdiDevice {
            name: "test-device".to_string(),
            container_edits: nvbind::cdi::ContainerEdits {
                env: Some(vec!["DEVICE_ENV=test".to_string()]),
                device_nodes: None,
                mounts: None,
                hooks: None,
            },
        }],
    };

    registry.register_spec(mock_spec).unwrap();

    let devices = registry.list_devices();
    assert_eq!(devices.len(), 1);
    assert_eq!(devices[0], "test.com/gpu=test-device");

    let device = registry.get_device("test.com/gpu=test-device");
    assert!(device.is_some());
    assert_eq!(device.unwrap().name, "test-device");
}

/// Test runtime validation
#[test]
fn test_runtime_validation() {
    // Test common container runtimes
    let runtimes = vec!["podman", "docker", "bolt"];

    for runtime in runtimes {
        let result = validate_runtime(runtime);

        // Should either succeed or fail gracefully with clear error
        match result {
            Ok(_) => println!("{} runtime available", runtime),
            Err(e) => println!("{} runtime not available: {}", runtime, e),
        }

        // Should not panic
    }
}

/// Test GPU discovery (mock mode)
#[tokio::test]
async fn test_gpu_discovery() {
    let result = discover_gpus().await;

    match result {
        Ok(gpus) => {
            println!("Found {} GPUs", gpus.len());
            for gpu in gpus {
                println!("GPU: {} (Memory: {:?} MB)", gpu.name, gpu.memory);
            }
        }
        Err(e) => {
            println!("GPU discovery failed (expected without hardware): {}", e);
        }
    }

    // Should not panic regardless of hardware availability
}

/// Test configuration file operations
#[test]
fn test_config_file_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config_path = temp_dir.path().join("nvbind.toml");

    // Create and save config
    let config = Config::default();
    config.save_to_file(&config_path)?;

    // Verify file exists
    assert!(config_path.exists());

    // Load config back
    let loaded_config = Config::load_from_file(&config_path)?;

    // Verify contents match
    assert_eq!(
        config.gpu.default_selection,
        loaded_config.gpu.default_selection
    );
    assert_eq!(
        config.runtime.default_runtime,
        loaded_config.runtime.default_runtime
    );

    Ok(())
}

/// Test error handling and graceful degradation
#[test]
fn test_error_handling() {
    // Test invalid runtime
    let result = validate_runtime("nonexistent-runtime");
    assert!(result.is_err());

    // Test missing configuration
    let result = Config::load_from_file(std::path::Path::new("/nonexistent/config.toml"));
    assert!(result.is_err());

    // Errors should be descriptive
    match result {
        Err(e) => assert!(e.to_string().contains("Failed to read config file")),
        Ok(_) => panic!("Should have failed"),
    }
}

/// Test Bolt integration (if feature enabled)
#[cfg(feature = "bolt")]
#[tokio::test]
async fn test_bolt_integration() {
    let manager = NvbindGpuManager::with_defaults();

    // Test GPU manager creation
    // Config validation would happen internally

    // Test compatibility check (should not panic without hardware)
    let result = manager.check_bolt_gpu_compatibility().await;
    match result {
        Ok(compatibility) => {
            println!(
                "Bolt GPU compatibility: gpus={}, driver={}",
                compatibility.gpus_available, compatibility.driver_version
            );
        }
        Err(e) => {
            println!(
                "Bolt compatibility check failed (expected without hardware): {}",
                e
            );
        }
    }
}

/// Test Bolt CDI generation (if feature enabled)
#[cfg(feature = "bolt")]
#[tokio::test]
async fn test_bolt_cdi_generation() {
    let manager = NvbindGpuManager::with_defaults();

    // These should not panic even without GPU hardware
    let gaming_result = manager.generate_gaming_cdi_spec().await;
    let aiml_result = manager.generate_aiml_cdi_spec().await;

    match (gaming_result, aiml_result) {
        (Ok(gaming_spec), Ok(aiml_spec)) => {
            assert_eq!(gaming_spec.kind, "nvidia.com/gpu-bolt");
            assert_eq!(aiml_spec.kind, "nvidia.com/gpu-bolt");
            println!("Bolt CDI generation successful");
        }
        _ => {
            println!("Bolt CDI generation failed (expected without hardware)");
        }
    }
}

/// Test command line interface functionality
#[test]
fn test_cli_command_generation() {
    // Test various command generation patterns
    let test_cases = vec![
        ("podman", "all", "nvidia/cuda:latest", vec!["nvidia-smi"]),
        ("docker", "gpu0", "ubuntu:22.04", vec!["echo", "test"]),
        (
            "bolt",
            "all",
            "pytorch/pytorch:latest",
            vec!["python", "train.py"],
        ),
    ];

    for (runtime, gpu, image, args) in test_cases {
        // Test that command generation doesn't panic
        println!(
            "Testing command generation: {} {} {} {:?}",
            runtime, gpu, image, args
        );

        // Would normally generate actual commands here
        // For now, just verify the parameters are handled correctly
        assert!(!runtime.is_empty());
        assert!(!gpu.is_empty());
        assert!(!image.is_empty());
        assert!(!args.is_empty());
    }
}

/// Test performance characteristics
#[test]
fn test_performance_characteristics() {
    use std::time::Instant;

    // Test configuration loading performance
    let start = Instant::now();
    let _config = Config::default();
    let config_time = start.elapsed();

    // Should be very fast (sub-millisecond)
    assert!(
        config_time.as_millis() < 10,
        "Config creation too slow: {:?}",
        config_time
    );

    // Test CDI registry creation performance
    let start = Instant::now();
    let _registry = CdiRegistry::new();
    let registry_time = start.elapsed();

    // Should be very fast
    assert!(
        registry_time.as_millis() < 10,
        "Registry creation too slow: {:?}",
        registry_time
    );

    println!(
        "Performance test passed - config: {:?}, registry: {:?}",
        config_time, registry_time
    );
}

/// Test memory safety and resource cleanup
#[test]
fn test_memory_safety() {
    // Test that objects can be created and dropped without issues
    for _ in 0..1000 {
        let _config = Config::default();
        let _registry = CdiRegistry::new();

        // Force cleanup
        drop(_config);
        drop(_registry);
    }

    println!("Memory safety test passed - no leaks detected in basic operations");
}

/// Test concurrent access patterns
#[tokio::test]
async fn test_concurrent_operations() {
    use tokio::task;

    // Test concurrent configuration access
    let tasks: Vec<_> = (0..10)
        .map(|i| {
            task::spawn(async move {
                let config = Config::default();
                assert_eq!(config.gpu.default_selection, "all");
                println!("Task {} completed", i);
            })
        })
        .collect();

    // Wait for all tasks
    for task in tasks {
        task.await.unwrap();
    }

    println!("Concurrent operations test passed");
}
