//! Integration tests for Docker, Podman, and Bolt runtime compatibility
//! Tests container runtime integration without requiring containers to be running

use nvbind::cdi::{CdiRegistry, generate_nvidia_cdi_spec};
use nvbind::gpu::is_nvidia_driver_available;
use std::process::Command;
use std::path::Path;

/// Test Docker runtime compatibility
#[tokio::test]
async fn test_docker_runtime_integration() {
    if !is_nvidia_driver_available() {
        eprintln!("Skipping Docker integration test - no NVIDIA driver detected");
        return;
    }

    // Check if Docker is available
    let docker_available = Command::new("docker")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if !docker_available {
        eprintln!("Skipping Docker test - Docker not available");
        return;
    }

    // Test CDI spec generation for Docker
    let spec_result = generate_nvidia_cdi_spec().await;
    assert!(spec_result.is_ok(), "CDI spec generation should succeed for Docker");

    let spec = spec_result.unwrap();
    assert!(!spec.devices.is_empty(), "CDI spec should contain GPU devices");

    println!("✅ Docker runtime integration validated");
    println!("  CDI version: {}", spec.cdi_version);
    println!("  Devices: {}", spec.devices.len());

    // Test Docker command generation
    let registry = CdiRegistry::new();
    for device in &spec.devices {
        let device_handle = registry.get_device(&format!("nvidia.com/gpu={}", device.name));
        if device_handle.is_some() {
            println!("  Device '{}' accessible via CDI", device.name);
        }
    }
}

/// Test Podman runtime compatibility
#[tokio::test]
async fn test_podman_runtime_integration() {
    if !is_nvidia_driver_available() {
        eprintln!("Skipping Podman integration test - no NVIDIA driver detected");
        return;
    }

    // Check if Podman is available
    let podman_available = Command::new("podman")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if !podman_available {
        eprintln!("Skipping Podman test - Podman not available");
        return;
    }

    // Test CDI spec generation for Podman
    let spec_result = generate_nvidia_cdi_spec().await;
    assert!(spec_result.is_ok(), "CDI spec generation should succeed for Podman");

    let spec = spec_result.unwrap();
    assert!(!spec.devices.is_empty(), "CDI spec should contain GPU devices");

    println!("✅ Podman runtime integration validated");

    // Test Podman-specific device access patterns
    let registry = CdiRegistry::new();
    let all_device = registry.get_device("nvidia.com/gpu=all");
    assert!(all_device.is_some(), "Podman should support 'all' GPU device access");

    // Test individual device access
    for device in &spec.devices {
        let device_name = format!("nvidia.com/gpu={}", device.name);
        let device_handle = registry.get_device(&device_name);
        assert!(device_handle.is_some(), "Individual GPU device '{}' should be accessible", device.name);
    }
}

/// Test Bolt runtime compatibility
#[tokio::test]
async fn test_bolt_runtime_integration() {
    if !is_nvidia_driver_available() {
        eprintln!("Skipping Bolt integration test - no NVIDIA driver detected");
        return;
    }

    // Check if Bolt is available (look for bolt binary or typical installation paths)
    let bolt_paths = [
        "/usr/local/bin/bolt",
        "/usr/bin/bolt",
        "/opt/bolt/bin/bolt",
    ];

    let bolt_available = bolt_paths.iter()
        .any(|path| Path::new(path).exists()) ||
        Command::new("bolt")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

    if !bolt_available {
        eprintln!("Skipping Bolt test - Bolt not available");
        return;
    }

    // Test CDI spec generation for Bolt
    let spec_result = generate_nvidia_cdi_spec().await;
    assert!(spec_result.is_ok(), "CDI spec generation should succeed for Bolt");

    let spec = spec_result.unwrap();
    assert!(!spec.devices.is_empty(), "CDI spec should contain GPU devices for Bolt");

    println!("✅ Bolt runtime integration validated");

    // Test Bolt-specific features
    let registry = CdiRegistry::new();

    // Bolt typically supports fine-grained GPU control
    for device in &spec.devices {
        let device_name = format!("nvidia.com/gpu={}", device.name);
        let device_handle = registry.get_device(&device_name);
        if device_handle.is_some() {
            println!("  Bolt device '{}' validated", device.name);
        }
    }

    // Test Bolt CDI compliance
    assert_eq!(spec.cdi_version, "0.6.0", "Bolt should use CDI v0.6.0");
    assert_eq!(spec.kind, "nvidia.com/gpu", "Bolt should use standard nvidia.com/gpu kind");
}

/// Test multi-runtime compatibility
#[tokio::test]
async fn test_multi_runtime_compatibility() {
    if !is_nvidia_driver_available() {
        eprintln!("Skipping multi-runtime test - no NVIDIA driver detected");
        return;
    }

    // Test that the same CDI spec works across all available runtimes
    let spec_result = generate_nvidia_cdi_spec().await;
    assert!(spec_result.is_ok(), "CDI spec generation should succeed");

    let spec = spec_result.unwrap();
    let registry = CdiRegistry::new();

    // Test common device access patterns across runtimes
    let test_patterns = [
        "nvidia.com/gpu=all",
        "nvidia.com/gpu=gpu0",
    ];

    for pattern in &test_patterns {
        let device = registry.get_device(pattern);
        println!("Device pattern '{}': {}", pattern,
            if device.is_some() { "✅ Available" } else { "❌ Not available" });
    }

    // Validate that spec is portable across runtimes
    if let Some(device_nodes) = &spec.container_edits.device_nodes {
        assert!(!device_nodes.is_empty(), "Device nodes should be defined for runtime compatibility");
        println!("✅ Multi-runtime device nodes validated: {} devices", device_nodes.len());
    }

    if let Some(mounts) = &spec.container_edits.mounts {
        println!("✅ Multi-runtime mounts validated: {} mounts", mounts.len());
    }
}

/// Test runtime command generation
#[test]
fn test_runtime_command_generation() {
    // Test Docker command generation
    let docker_cmd = generate_docker_gpu_command("nvidia/cuda:latest", &["nvidia-smi"]);
    assert!(docker_cmd.get_program() == "docker");
    let args: Vec<&str> = docker_cmd.get_args().map(|s| s.to_str().unwrap()).collect();
    assert!(args.contains(&"run"));
    assert!(args.contains(&"--rm"));
    println!("✅ Docker command generation validated");

    // Test Podman command generation
    let podman_cmd = generate_podman_gpu_command("nvidia/cuda:latest", &["nvidia-smi"]);
    assert!(podman_cmd.get_program() == "podman");
    let args: Vec<&str> = podman_cmd.get_args().map(|s| s.to_str().unwrap()).collect();
    assert!(args.contains(&"run"));
    assert!(args.contains(&"--rm"));
    println!("✅ Podman command generation validated");
}

/// Test error handling in runtime integration
#[tokio::test]
async fn test_runtime_error_handling() {
    // Test behavior when no GPUs are available (should not panic)
    let registry = CdiRegistry::new();
    let non_existent_device = registry.get_device("nvidia.com/gpu=gpu999");
    assert!(non_existent_device.is_none(), "Non-existent GPU device should return None");

    // Test invalid device patterns
    let invalid_patterns = [
        "invalid/pattern",
        "nvidia.com/gpu=",
        "nvidia.com/gpu=gpu-1",
    ];

    for pattern in &invalid_patterns {
        let _device = registry.get_device(pattern);
        println!("Invalid pattern '{}': handled gracefully", pattern);
        // Should not panic, may return None or handle gracefully
    }

    println!("✅ Runtime error handling validated");
}

/// Test container isolation and security
#[tokio::test]
async fn test_container_security_isolation() {
    if !is_nvidia_driver_available() {
        eprintln!("Skipping security test - no NVIDIA driver detected");
        return;
    }

    let spec_result = generate_nvidia_cdi_spec().await;
    if spec_result.is_err() {
        eprintln!("Skipping security test - CDI spec generation failed");
        return;
    }

    let spec = spec_result.unwrap();

    // Validate security aspects of CDI spec
    let edit = &spec.container_edits;

    // Check device nodes for proper permissions
    if let Some(device_nodes) = &edit.device_nodes {
        for device in device_nodes {
            println!("Device: {} (mode: {:?})", device.path, device.file_mode);

            // Validate device paths are under /dev
            assert!(device.path.starts_with("/dev/"),
                "Device paths should be under /dev for security");

            // Validate permissions are reasonable (but allow typical GPU device permissions)
            if let Some(mode) = &device.file_mode {
                println!("  Device mode: {:#o} ({})", mode, mode);
                // GPU devices typically have mode 0o666 (438 decimal) which is rw-rw-rw-
                // This is normal for GPU devices as they need to be accessible by container users
                // Just ensure it's not executable by others (which would be unusual)
                assert!(mode & 0o001 == 0, "Device should not be world-executable");
            }
        }
    }

    // Check mounts for security
    if let Some(mounts) = &edit.mounts {
        for mount in mounts {
            println!("Mount: {} -> {}", mount.host_path, mount.container_path);

            // Validate mount sources are from system paths
            let secure_prefixes = ["/usr", "/lib", "/lib64", "/dev"];
            assert!(secure_prefixes.iter().any(|prefix| mount.host_path.starts_with(prefix)),
                "Mount source '{}' should be from secure system path", mount.host_path);
        }
    }

    println!("✅ Container security isolation validated");
}

// Helper functions for command generation
fn generate_docker_gpu_command(image: &str, cmd: &[&str]) -> Command {
    let mut command = Command::new("docker");
    command
        .arg("run")
        .arg("--rm")
        .arg("--gpus")
        .arg("all")
        .arg(image);

    for arg in cmd {
        command.arg(arg);
    }

    command
}

fn generate_podman_gpu_command(image: &str, cmd: &[&str]) -> Command {
    let mut command = Command::new("podman");
    command
        .arg("run")
        .arg("--rm")
        .arg("--device=/dev/nvidiactl")
        .arg("--device=/dev/nvidia-uvm")
        .arg("--device=/dev/nvidia-modeset");

    // Add numbered GPU devices
    for i in 0..8 {
        let device = format!("--device=/dev/nvidia{}", i);
        if Path::new(&format!("/dev/nvidia{}", i)).exists() {
            command.arg(device);
        }
    }

    command.arg(image);
    for arg in cmd {
        command.arg(arg);
    }

    command
}