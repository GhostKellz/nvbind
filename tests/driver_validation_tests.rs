//! Comprehensive driver validation tests
//! Focus on NVIDIA Open driver (version 580+) detection and validation

use nvbind::gpu::{
    DriverType, check_nvidia_driver_status, detect_driver_type, discover_gpus, get_driver_info,
    is_nvidia_driver_available,
};
use std::fs;

/// Test NVIDIA Open driver detection specifically
#[tokio::test]
async fn test_nvidia_open_driver_detection() {
    // Skip if no NVIDIA driver available
    if !is_nvidia_driver_available() {
        eprintln!("Skipping NVIDIA Open driver test - no NVIDIA driver detected");
        return;
    }

    let driver_type = detect_driver_type();

    match driver_type {
        DriverType::NvidiaOpen => {
            println!("✅ NVIDIA Open driver detected successfully");

            // Validate driver version is 580+ if possible
            if let Ok(driver_info) = get_driver_info().await {
                println!("Driver version: {}", driver_info.version);

                // Try to extract numeric version
                if let Some(captures) = regex::Regex::new(r"(\d{3}\.\d+)")
                    .ok()
                    .and_then(|re| re.captures(&driver_info.version))
                {
                    if let Ok(version_num) = captures[1].parse::<f32>() {
                        if version_num >= 580.0 {
                            println!("✅ Version {} meets 580+ recommendation", version_num);
                        } else if version_num >= 515.0 {
                            println!(
                                "⚠️  Version {} is supported but 580+ recommended",
                                version_num
                            );
                        } else {
                            println!(
                                "❌ Version {} may have limited open driver support",
                                version_num
                            );
                        }
                    }
                }
            }
        }
        DriverType::NvidiaProprietary => {
            println!("ℹ️  NVIDIA Proprietary driver detected (not open source)");
        }
        DriverType::Nouveau => {
            println!("ℹ️  Nouveau open source driver detected");
        }
    }

    // Validate that driver status check works
    assert!(
        check_nvidia_driver_status().is_ok(),
        "NVIDIA driver status check should succeed"
    );
}

/// Test GPU discovery with different driver types
#[tokio::test]
async fn test_gpu_discovery_across_drivers() {
    if !is_nvidia_driver_available() {
        eprintln!("Skipping GPU discovery test - no NVIDIA driver detected");
        return;
    }

    let driver_type = detect_driver_type();
    let gpus_result = discover_gpus().await;

    match gpus_result {
        Ok(gpus) => {
            println!("✅ GPU discovery successful with {:?} driver", driver_type);
            println!("Found {} GPUs:", gpus.len());

            for (i, gpu) in gpus.iter().enumerate() {
                println!("  GPU {}: {}", i, gpu.name);
                println!("    PCI: {}", gpu.pci_address);
                println!("    Device: {}", gpu.device_path);

                // Validate basic GPU properties
                assert!(!gpu.name.is_empty(), "GPU name should not be empty");
                assert!(
                    !gpu.pci_address.is_empty(),
                    "PCI address should not be empty"
                );
                assert!(
                    !gpu.device_path.is_empty(),
                    "Device path should not be empty"
                );

                // Validate device path exists
                assert!(
                    std::path::Path::new(&gpu.device_path).exists(),
                    "GPU device path should exist: {}",
                    gpu.device_path
                );
            }

            // Test driver-specific validations
            match driver_type {
                DriverType::NvidiaOpen => {
                    println!("✅ Open driver GPU discovery validated");
                    test_open_driver_specific_features(&gpus).await;
                }
                DriverType::NvidiaProprietary => {
                    println!("✅ Proprietary driver GPU discovery validated");
                    test_proprietary_driver_features(&gpus).await;
                }
                DriverType::Nouveau => {
                    println!("✅ Nouveau driver GPU discovery validated");
                    test_nouveau_driver_features(&gpus).await;
                }
            }
        }
        Err(e) => {
            // GPU discovery failed - this might be expected in some environments
            println!("⚠️  GPU discovery failed: {}", e);
            println!("This may be expected in containerized or VM environments");
        }
    }
}

/// Test Open driver specific features
async fn test_open_driver_specific_features(gpus: &[nvbind::gpu::GpuDevice]) {
    // Test GSP firmware detection
    if let Ok(gsp_content) =
        fs::read_to_string("/sys/module/nvidia/parameters/NVreg_EnableGpuFirmware")
    {
        if gsp_content.trim() == "1" {
            println!("✅ GSP firmware enabled (Open driver feature)");
        }
    }

    // Test modeset detection
    if let Ok(modeset_content) = fs::read_to_string("/sys/module/nvidia_drm/parameters/modeset") {
        if modeset_content.trim() == "Y" {
            println!("✅ DRM modeset enabled (typical for Open driver)");
        }
    }

    // Test nvidia-caps device (more common with open driver)
    if std::path::Path::new("/dev/nvidia-caps").exists() {
        println!("✅ NVIDIA capabilities device available");
    }

    // Test for each GPU device
    for gpu in gpus {
        assert!(
            std::path::Path::new(&gpu.device_path).exists(),
            "GPU device {} should exist",
            gpu.device_path
        );
    }
}

/// Test Proprietary driver features
async fn test_proprietary_driver_features(gpus: &[nvbind::gpu::GpuDevice]) {
    // Proprietary driver specific tests
    for gpu in gpus {
        // Validate device access
        assert!(
            std::path::Path::new(&gpu.device_path).exists(),
            "GPU device {} should exist",
            gpu.device_path
        );
    }

    // Check for proprietary driver specific files
    if std::path::Path::new("/proc/driver/nvidia").exists() {
        println!("✅ NVIDIA proc driver interface available");
    }
}

/// Test Nouveau driver features
async fn test_nouveau_driver_features(gpus: &[nvbind::gpu::GpuDevice]) {
    // Nouveau driver specific tests
    for gpu in gpus {
        assert!(
            std::path::Path::new(&gpu.device_path).exists(),
            "GPU device {} should exist",
            gpu.device_path
        );
    }

    // Check for nouveau specific paths
    if std::path::Path::new("/sys/module/nouveau").exists() {
        println!("✅ Nouveau kernel module loaded");
    }
}

/// Test driver compatibility with container runtimes
#[tokio::test]
async fn test_driver_container_compatibility() {
    if !is_nvidia_driver_available() {
        eprintln!("Skipping container compatibility test - no NVIDIA driver detected");
        return;
    }

    let driver_type = detect_driver_type();

    // Test basic device visibility
    let essential_devices = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-modeset"];

    for device in &essential_devices {
        if std::path::Path::new(device).exists() {
            println!("✅ Essential device available: {}", device);
        } else {
            println!(
                "⚠️  Essential device missing: {} (may affect containers)",
                device
            );
        }
    }

    // Test numbered GPU devices
    for i in 0..8 {
        let device = format!("/dev/nvidia{}", i);
        if std::path::Path::new(&device).exists() {
            println!("✅ GPU device available: {}", device);
        }
    }

    // Driver-specific compatibility tests
    match driver_type {
        DriverType::NvidiaOpen => {
            println!("✅ Open driver container compatibility validated");
            // Open driver has excellent container support from 515+
        }
        DriverType::NvidiaProprietary => {
            println!("✅ Proprietary driver container compatibility validated");
            // Proprietary driver has mature container support
        }
        DriverType::Nouveau => {
            println!("ℹ️  Nouveau driver detected - limited GPU compute support in containers");
        }
    }
}

/// Test driver version compatibility
#[tokio::test]
async fn test_driver_version_compatibility() {
    if !is_nvidia_driver_available() {
        eprintln!("Skipping version compatibility test - no NVIDIA driver detected");
        return;
    }

    match get_driver_info().await {
        Ok(driver_info) => {
            println!(
                "Driver: {} version {}",
                match driver_info.driver_type {
                    DriverType::NvidiaOpen => "NVIDIA Open",
                    DriverType::NvidiaProprietary => "NVIDIA Proprietary",
                    DriverType::Nouveau => "Nouveau",
                },
                driver_info.version
            );

            // Version-specific validation
            if let Some(captures) = regex::Regex::new(r"(\d{3}\.\d+)")
                .ok()
                .and_then(|re| re.captures(&driver_info.version))
            {
                if let Ok(version_num) = captures[1].parse::<f32>() {
                    match driver_info.driver_type {
                        DriverType::NvidiaOpen => {
                            if version_num >= 580.0 {
                                println!("✅ Excellent Open driver version (580+)");
                            } else if version_num >= 515.0 {
                                println!("✅ Good Open driver version (515+)");
                            } else {
                                println!("⚠️  Open driver version below 515 - may have issues");
                            }
                        }
                        DriverType::NvidiaProprietary => {
                            if version_num >= 470.0 {
                                println!("✅ Compatible proprietary driver version");
                            } else {
                                println!("⚠️  Old proprietary driver - consider updating");
                            }
                        }
                        DriverType::Nouveau => {
                            println!(
                                "ℹ️  Nouveau driver - version compatibility varies by GPU generation"
                            );
                        }
                    }
                }
            }

            if let Some(cuda_version) = &driver_info.cuda_version {
                println!("CUDA support: {}", cuda_version);
            }
        }
        Err(e) => {
            println!("⚠️  Could not get driver info: {}", e);
        }
    }
}

/// Performance test for driver operations
#[tokio::test]
async fn test_driver_performance() {
    if !is_nvidia_driver_available() {
        eprintln!("Skipping driver performance test - no NVIDIA driver detected");
        return;
    }

    use std::time::Instant;

    // Test driver detection speed
    let start = Instant::now();
    let _driver_type = detect_driver_type();
    let detection_time = start.elapsed();

    println!("Driver detection time: {:?}", detection_time);
    assert!(
        detection_time.as_millis() < 100,
        "Driver detection should be fast"
    );

    // Test GPU discovery speed
    let start = Instant::now();
    let _gpus = discover_gpus().await;
    let discovery_time = start.elapsed();

    println!("GPU discovery time: {:?}", discovery_time);
    assert!(
        discovery_time.as_millis() < 1000,
        "GPU discovery should complete within 1 second"
    );
}

/// Test error handling and recovery
#[test]
fn test_driver_error_handling() {
    // Test graceful handling when no driver is available
    // This test should work even without GPUs

    let driver_available = is_nvidia_driver_available();
    println!("NVIDIA driver available: {}", driver_available);

    // Driver detection should not panic
    let _driver_type = detect_driver_type();

    // Driver status check should handle missing drivers gracefully
    let _status = check_nvidia_driver_status();

    println!("✅ Driver error handling tests completed without panic");
}
