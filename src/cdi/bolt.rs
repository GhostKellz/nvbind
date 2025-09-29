//! Bolt-specific CDI generation and management
//!
//! This module contains the Bolt container runtime specific implementation
//! of Container Device Interface (CDI) specifications for GPU management.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::{CdiDevice, CdiSpec, ContainerEdits, DeviceNode, Mount};
use crate::gpu::GpuDevice;

/// Bolt-specific capsule configuration for GPU management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoltCapsuleConfig {
    /// Enable GPU state snapshot/restore for capsules
    pub snapshot_support: bool,
    /// GPU isolation level for Bolt capsules
    pub isolation_level: BoltGpuIsolation,
    /// Gaming-specific optimizations
    pub gaming_optimizations: Option<BoltGamingConfig>,
    /// WSL2 mode detection
    pub wsl2_mode: bool,
}

/// GPU isolation levels for Bolt capsules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoltGpuIsolation {
    /// Shared GPU access across multiple capsules
    Shared,
    /// Exclusive GPU access for single capsule
    Exclusive,
    /// Virtual GPU with resource limits
    Virtual {
        memory_limit: String,
        compute_limit: String,
    },
}

/// Gaming-specific configuration for Bolt capsules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoltGamingConfig {
    /// Enable DLSS support
    pub dlss_enabled: bool,
    /// Enable ray tracing cores
    pub rt_cores_enabled: bool,
    /// Gaming performance profile
    pub performance_profile: GamingProfile,
    /// Wine/Proton optimizations
    pub wine_optimizations: bool,
}

/// Gaming performance profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GamingProfile {
    /// Maximum performance, highest power consumption
    UltraLowLatency,
    /// High performance with power efficiency balance
    Performance,
    /// Balanced performance and power
    Balanced,
    /// Lower performance, power efficient
    Efficiency,
}

impl Default for BoltCapsuleConfig {
    fn default() -> Self {
        Self {
            snapshot_support: true,
            isolation_level: BoltGpuIsolation::Exclusive,
            gaming_optimizations: None,
            wsl2_mode: crate::wsl2::Wsl2Manager::detect_wsl2(),
        }
    }
}

/// Generate Bolt-optimized CDI specification for gaming workloads
pub async fn generate_bolt_gaming_cdi_spec() -> Result<CdiSpec> {
    debug!("Generating Bolt gaming CDI specification");

    let gaming_config = BoltGamingConfig {
        dlss_enabled: true,
        rt_cores_enabled: true,
        performance_profile: GamingProfile::UltraLowLatency,
        wine_optimizations: true,
    };

    let capsule_config = BoltCapsuleConfig {
        snapshot_support: true,
        isolation_level: BoltGpuIsolation::Exclusive,
        gaming_optimizations: Some(gaming_config),
        wsl2_mode: crate::wsl2::Wsl2Manager::detect_wsl2(),
    };

    generate_bolt_nvidia_cdi_spec(capsule_config).await
}

/// Generate Bolt-optimized CDI specification for AI/ML workloads
pub async fn generate_bolt_aiml_cdi_spec() -> Result<CdiSpec> {
    debug!("Generating Bolt AI/ML CDI specification");

    let capsule_config = BoltCapsuleConfig {
        snapshot_support: true,
        isolation_level: BoltGpuIsolation::Virtual {
            memory_limit: "16GB".to_string(),
            compute_limit: "75%".to_string(),
        },
        gaming_optimizations: None,
        wsl2_mode: crate::wsl2::Wsl2Manager::detect_wsl2(),
    };

    generate_bolt_nvidia_cdi_spec(capsule_config).await
}

/// Generate custom Bolt CDI specification with specific configuration
pub async fn generate_bolt_nvidia_cdi_spec(config: BoltCapsuleConfig) -> Result<CdiSpec> {
    debug!(
        "Generating Bolt NVIDIA CDI specification with config: {:?}",
        config
    );

    // Discover available GPUs
    let gpus = crate::gpu::discover_gpus().await?;
    if gpus.is_empty() {
        return Err(anyhow::anyhow!(
            "No NVIDIA GPUs found for Bolt CDI generation"
        ));
    }

    // Generate base CDI spec
    let mut spec = CdiSpec {
        cdi_version: "0.6.0".to_string(),
        kind: "nvidia.com/gpu-bolt".to_string(),
        container_edits: ContainerEdits {
            env: Some(generate_bolt_environment_variables(&config)?),
            device_nodes: None,
            mounts: Some(generate_bolt_library_mounts(&config)?),
            hooks: Some(generate_bolt_hooks(&config)?),
        },
        devices: Vec::new(),
    };

    // Generate device specifications for each GPU
    for (idx, gpu) in gpus.iter().enumerate() {
        let device = generate_bolt_gpu_device(idx, gpu, &config)?;
        spec.devices.push(device);
    }

    // Add "all" device for convenience
    let all_device = generate_bolt_all_device(&gpus, &config)?;
    spec.devices.push(all_device);

    debug!(
        "Generated Bolt CDI spec with {} devices",
        spec.devices.len()
    );
    Ok(spec)
}

/// Generate environment variables for Bolt containers
fn generate_bolt_environment_variables(config: &BoltCapsuleConfig) -> Result<Vec<String>> {
    let mut env = vec![
        "NVIDIA_VISIBLE_DEVICES=all".to_string(),
        "NVIDIA_DRIVER_CAPABILITIES=all".to_string(),
        "BOLT_GPU_ENABLED=1".to_string(),
    ];

    // Add isolation level
    match &config.isolation_level {
        BoltGpuIsolation::Shared => env.push("BOLT_GPU_ISOLATION=shared".to_string()),
        BoltGpuIsolation::Exclusive => env.push("BOLT_GPU_ISOLATION=exclusive".to_string()),
        BoltGpuIsolation::Virtual {
            memory_limit,
            compute_limit,
        } => {
            env.push("BOLT_GPU_ISOLATION=virtual".to_string());
            env.push(format!("BOLT_GPU_MEMORY_LIMIT={}", memory_limit));
            env.push(format!("BOLT_GPU_COMPUTE_LIMIT={}", compute_limit));
        }
    }

    // Add snapshot support
    if config.snapshot_support {
        env.push("BOLT_GPU_SNAPSHOT=1".to_string());
    }

    // Add gaming optimizations
    if let Some(gaming) = &config.gaming_optimizations {
        env.push("BOLT_WORKLOAD_TYPE=gaming".to_string());

        if gaming.dlss_enabled {
            env.push("NVIDIA_DLSS_ENABLED=1".to_string());
        }

        if gaming.rt_cores_enabled {
            env.push("NVIDIA_RT_CORES_ENABLED=1".to_string());
        }

        if gaming.wine_optimizations {
            env.push("BOLT_WINE_OPTIMIZATIONS=1".to_string());
            env.push("DXVK_NVAPI_DRIVER_VERSION=47103".to_string());
            env.push("VKD3D_CONFIG=force_static_cbv".to_string());
        }

        match gaming.performance_profile {
            GamingProfile::UltraLowLatency => {
                env.push("BOLT_GPU_PROFILE=ultra-low-latency".to_string());
                env.push("NVIDIA_TF32_OVERRIDE=0".to_string());
            }
            GamingProfile::Performance => {
                env.push("BOLT_GPU_PROFILE=performance".to_string());
                env.push("NVIDIA_TF32_OVERRIDE=0".to_string());
            }
            GamingProfile::Balanced => {
                env.push("BOLT_GPU_PROFILE=balanced".to_string());
            }
            GamingProfile::Efficiency => {
                env.push("BOLT_GPU_PROFILE=efficiency".to_string());
                env.push("CUDA_CACHE_MAXSIZE=268435456".to_string()); // 256MB
            }
        }
    } else {
        // AI/ML optimizations
        env.push("BOLT_WORKLOAD_TYPE=ai-ml".to_string());
        env.push("NVIDIA_TF32_OVERRIDE=1".to_string());
        env.push("CUDA_CACHE_MAXSIZE=4294967296".to_string()); // 4GB
        env.push("PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512".to_string());
    }

    // WSL2 optimizations
    if config.wsl2_mode {
        env.push("BOLT_WSL2_MODE=1".to_string());
        env.push("WSLENV=BOLT_GPU_ENABLED:BOLT_WORKLOAD_TYPE:NVIDIA_VISIBLE_DEVICES".to_string());
    }

    Ok(env)
}

/// Generate library mounts for Bolt containers
fn generate_bolt_library_mounts(_config: &BoltCapsuleConfig) -> Result<Vec<Mount>> {
    let mut mounts = Vec::new();

    // NVIDIA libraries
    let nvidia_lib_paths = vec![
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/local/nvidia/lib64",
        "/usr/local/nvidia/lib",
    ];

    for lib_path in nvidia_lib_paths {
        if std::path::Path::new(lib_path).exists() {
            mounts.push(Mount {
                host_path: lib_path.to_string(),
                container_path: lib_path.to_string(),
                options: Some(vec![
                    "ro".to_string(),
                    "nosuid".to_string(),
                    "nodev".to_string(),
                ]),
            });
        }
    }

    // CUDA toolkit paths
    let cuda_paths = vec!["/usr/local/cuda", "/opt/cuda"];

    for cuda_path in cuda_paths {
        if std::path::Path::new(cuda_path).exists() {
            mounts.push(Mount {
                host_path: cuda_path.to_string(),
                container_path: cuda_path.to_string(),
                options: Some(vec![
                    "ro".to_string(),
                    "nosuid".to_string(),
                    "nodev".to_string(),
                ]),
            });
        }
    }

    Ok(mounts)
}

/// Generate Bolt-specific hooks
fn generate_bolt_hooks(config: &BoltCapsuleConfig) -> Result<Vec<super::Hook>> {
    let mut hooks = Vec::new();

    // Pre-start hook for GPU setup
    hooks.push(super::Hook {
        hook_name: "prestart".to_string(),
        path: "/usr/local/bin/nvbind".to_string(),
        args: Some(vec![
            "bolt".to_string(),
            "setup".to_string(),
            "--isolation".to_string(),
            match &config.isolation_level {
                BoltGpuIsolation::Shared => "shared".to_string(),
                BoltGpuIsolation::Exclusive => "exclusive".to_string(),
                BoltGpuIsolation::Virtual { .. } => "virtual".to_string(),
            },
        ]),
        env: None,
        timeout: Some(30),
    });

    // Post-start hook for snapshot preparation
    if config.snapshot_support {
        hooks.push(super::Hook {
            hook_name: "poststart".to_string(),
            path: "/usr/local/bin/nvbind".to_string(),
            args: Some(vec!["bolt".to_string(), "snapshot-prepare".to_string()]),
            env: None,
            timeout: Some(10),
        });
    }

    // Pre-stop hook for GPU cleanup
    hooks.push(super::Hook {
        hook_name: "prestop".to_string(),
        path: "/usr/local/bin/nvbind".to_string(),
        args: Some(vec!["bolt".to_string(), "cleanup".to_string()]),
        env: None,
        timeout: Some(15),
    });

    Ok(hooks)
}

/// Generate CDI device specification for a specific GPU
fn generate_bolt_gpu_device(
    index: usize,
    _gpu: &GpuDevice,
    _config: &BoltCapsuleConfig,
) -> Result<CdiDevice> {
    let device_name = format!("gpu{}", index);

    let mut device_nodes = vec![DeviceNode {
        path: format!("/dev/nvidia{}", index),
        device_type: Some("c".to_string()),
        major: Some(195),
        minor: Some(index as u32),
        file_mode: Some(0o666),
        uid: None,
        gid: None,
    }];

    // Add NVIDIA control device if this is GPU 0
    if index == 0 {
        device_nodes.push(DeviceNode {
            path: "/dev/nvidiactl".to_string(),
            device_type: Some("c".to_string()),
            major: Some(195),
            minor: Some(255),
            file_mode: Some(0o666),
            uid: None,
            gid: None,
        });

        device_nodes.push(DeviceNode {
            path: "/dev/nvidia-uvm".to_string(),
            device_type: Some("c".to_string()),
            major: Some(510),
            minor: Some(0),
            file_mode: Some(0o666),
            uid: None,
            gid: None,
        });
    }

    Ok(CdiDevice {
        name: device_name,
        container_edits: ContainerEdits {
            env: Some(vec![
                format!("NVIDIA_VISIBLE_DEVICES={}", index),
                format!("BOLT_GPU_INDEX={}", index),
            ]),
            device_nodes: Some(device_nodes),
            mounts: None,
            hooks: None,
        },
    })
}

/// Generate CDI device specification for all GPUs
fn generate_bolt_all_device(gpus: &[GpuDevice], _config: &BoltCapsuleConfig) -> Result<CdiDevice> {
    let mut device_nodes = Vec::new();
    let mut env_vars = vec![
        "NVIDIA_VISIBLE_DEVICES=all".to_string(),
        "BOLT_GPU_INDEX=all".to_string(),
    ];

    // Add all GPU device nodes
    for (index, _gpu) in gpus.iter().enumerate() {
        device_nodes.push(DeviceNode {
            path: format!("/dev/nvidia{}", index),
            device_type: Some("c".to_string()),
            major: Some(195),
            minor: Some(index as u32),
            file_mode: Some(0o666),
            uid: None,
            gid: None,
        });
    }

    // Add control devices
    device_nodes.push(DeviceNode {
        path: "/dev/nvidiactl".to_string(),
        device_type: Some("c".to_string()),
        major: Some(195),
        minor: Some(255),
        file_mode: Some(0o666),
        uid: None,
        gid: None,
    });

    device_nodes.push(DeviceNode {
        path: "/dev/nvidia-uvm".to_string(),
        device_type: Some("c".to_string()),
        major: Some(510),
        minor: Some(0),
        file_mode: Some(0o666),
        uid: None,
        gid: None,
    });

    // Add GPU count to environment
    env_vars.push(format!("BOLT_GPU_COUNT={}", gpus.len()));

    Ok(CdiDevice {
        name: "all".to_string(),
        container_edits: ContainerEdits {
            env: Some(env_vars),
            device_nodes: Some(device_nodes),
            mounts: None,
            hooks: None,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bolt_gaming_cdi_generation() {
        // Should not panic even without GPU hardware
        let _result = generate_bolt_gaming_cdi_spec().await;
    }

    #[tokio::test]
    async fn test_bolt_aiml_cdi_generation() {
        // Should not panic even without GPU hardware
        let _result = generate_bolt_aiml_cdi_spec().await;
    }

    #[test]
    fn test_bolt_capsule_config_default() {
        let config = BoltCapsuleConfig::default();
        assert!(config.snapshot_support);
        matches!(config.isolation_level, BoltGpuIsolation::Exclusive);
    }

    #[test]
    fn test_gaming_environment_variables() {
        let gaming_config = BoltGamingConfig {
            dlss_enabled: true,
            rt_cores_enabled: true,
            performance_profile: GamingProfile::UltraLowLatency,
            wine_optimizations: true,
        };

        let config = BoltCapsuleConfig {
            snapshot_support: true,
            isolation_level: BoltGpuIsolation::Exclusive,
            gaming_optimizations: Some(gaming_config),
            wsl2_mode: false,
        };

        let env_vars = generate_bolt_environment_variables(&config).unwrap();

        assert!(env_vars.contains(&"BOLT_WORKLOAD_TYPE=gaming".to_string()));
        assert!(env_vars.contains(&"NVIDIA_DLSS_ENABLED=1".to_string()));
        assert!(env_vars.contains(&"NVIDIA_RT_CORES_ENABLED=1".to_string()));
        assert!(env_vars.contains(&"BOLT_WINE_OPTIMIZATIONS=1".to_string()));
    }
}
