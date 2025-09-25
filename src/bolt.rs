//! Bolt container runtime integration for nvbind
//!
//! This module provides the official integration layer between nvbind and the Bolt container runtime.
//! When the "bolt" feature is enabled, this module exposes traits and utilities that Bolt can use
//! to integrate nvbind's GPU management capabilities directly into its capsule architecture.

#[cfg(feature = "bolt")]
use anyhow::Result;
#[cfg(feature = "bolt")]
use async_trait::async_trait;
#[cfg(feature = "bolt")]
use crate::{config::BoltConfig, cdi::bolt::BoltCapsuleConfig};


/// Core trait that Bolt implements to integrate with nvbind
#[cfg(feature = "bolt")]
#[async_trait]
pub trait BoltRuntime {
    /// Container/Capsule identifier type used by Bolt
    type ContainerId: Send + Sync;

    /// Bolt's container specification type
    type ContainerSpec: Send + Sync;

    /// Initialize GPU support for a Bolt capsule
    async fn setup_gpu_for_capsule(
        &self,
        container_id: &Self::ContainerId,
        spec: &Self::ContainerSpec,
        gpu_config: &BoltConfig,
    ) -> Result<()>;

    /// Apply CDI devices to a Bolt capsule
    async fn apply_cdi_devices(
        &self,
        container_id: &Self::ContainerId,
        cdi_devices: &[String],
    ) -> Result<()>;

    /// Enable GPU state snapshot/restore for capsule
    async fn enable_gpu_snapshot(
        &self,
        container_id: &Self::ContainerId,
        snapshot_config: &BoltCapsuleConfig,
    ) -> Result<()>;

    /// Optimize GPU configuration for gaming workloads
    async fn setup_gaming_optimization(
        &self,
        container_id: &Self::ContainerId,
        gaming_config: &crate::config::BoltGamingGpuConfig,
    ) -> Result<()>;

    /// Configure GPU for AI/ML workloads
    async fn setup_aiml_optimization(
        &self,
        container_id: &Self::ContainerId,
        aiml_config: &crate::config::BoltAiMlGpuConfig,
    ) -> Result<()>;

    /// Handle GPU isolation levels for Bolt capsules
    async fn configure_gpu_isolation(
        &self,
        container_id: &Self::ContainerId,
        isolation_level: &str, // "shared", "exclusive", "virtual"
    ) -> Result<()>;
}

/// High-level GPU manager that Bolt can use directly
#[cfg(feature = "bolt")]
pub struct NvbindGpuManager {
    config: BoltConfig,
}

#[cfg(feature = "bolt")]
impl NvbindGpuManager {
    /// Create a new GPU manager with Bolt configuration
    pub fn new(config: BoltConfig) -> Self {
        Self { config }
    }

    /// Create GPU manager with default Bolt configuration
    pub fn with_defaults() -> Self {
        Self {
            config: BoltConfig::default(),
        }
    }

    /// Generate Bolt-optimized CDI specification for gaming
    pub async fn generate_gaming_cdi_spec(&self) -> Result<crate::cdi::CdiSpec> {
        crate::cdi::bolt::generate_bolt_gaming_cdi_spec().await
    }

    /// Generate Bolt-optimized CDI specification for AI/ML
    pub async fn generate_aiml_cdi_spec(&self) -> Result<crate::cdi::CdiSpec> {
        crate::cdi::bolt::generate_bolt_aiml_cdi_spec().await
    }

    /// Create custom CDI specification with specific capsule config
    pub async fn generate_custom_cdi_spec(&self, capsule_config: BoltCapsuleConfig) -> Result<crate::cdi::CdiSpec> {
        crate::cdi::bolt::generate_bolt_nvidia_cdi_spec(capsule_config).await
    }

    /// Get GPU information for Bolt's hardware detection
    pub async fn get_gpu_info(&self) -> Result<Vec<crate::gpu::GpuDevice>> {
        crate::gpu::discover_gpus().await
    }

    /// Check if system is ready for GPU acceleration with Bolt
    pub async fn check_bolt_gpu_compatibility(&self) -> Result<BoltGpuCompatibility> {
        let gpus = crate::gpu::discover_gpus().await?;
        let driver_info = crate::gpu::get_driver_info().await?;
        let wsl2_detected = crate::wsl2::Wsl2Manager::detect_wsl2();

        Ok(BoltGpuCompatibility {
            gpus_available: !gpus.is_empty(),
            gpu_count: gpus.len(),
            driver_version: driver_info.version,
            nvidia_open_driver: driver_info.driver_type == crate::gpu::DriverType::NvidiaOpen,
            wsl2_mode: wsl2_detected,
            bolt_optimizations_available: true,
        })
    }

    /// Execute nvbind with Bolt runtime
    pub async fn run_with_bolt_runtime(
        &self,
        image: String,
        args: Vec<String>,
        gpu_selection: Option<String>,
    ) -> Result<()> {
        let config = crate::config::Config {
            #[cfg(feature = "bolt")]
            bolt: Some(self.config.clone()),
            ..Default::default()
        };

        crate::runtime::run_with_config(
            config,
            "bolt".to_string(),
            gpu_selection.unwrap_or_else(|| "all".to_string()),
            image,
            args,
        ).await
    }
}

/// Compatibility information for Bolt GPU integration
#[cfg(feature = "bolt")]
#[derive(Debug, Clone)]
pub struct BoltGpuCompatibility {
    pub gpus_available: bool,
    pub gpu_count: usize,
    pub driver_version: String,
    pub nvidia_open_driver: bool,
    pub wsl2_mode: bool,
    pub bolt_optimizations_available: bool,
}

/// Convenience functions for Bolt integration
#[cfg(feature = "bolt")]
pub mod utils {
    use super::*;

    /// Quick setup for Bolt gaming containers
    pub async fn setup_bolt_gaming_container() -> Result<crate::cdi::CdiSpec> {
        crate::cdi::bolt::generate_bolt_gaming_cdi_spec().await
    }

    /// Quick setup for Bolt AI/ML containers
    pub async fn setup_bolt_aiml_container() -> Result<crate::cdi::CdiSpec> {
        crate::cdi::bolt::generate_bolt_aiml_cdi_spec().await
    }

    /// Validate Bolt GPU environment
    pub async fn validate_bolt_environment() -> Result<bool> {
        let gpus = crate::gpu::discover_gpus().await?;
        let _driver_info = crate::gpu::get_driver_info().await?;

        // Check if bolt command is available
        let bolt_available = crate::runtime::validate_runtime("bolt").is_ok();

        Ok(!gpus.is_empty() && bolt_available)
    }

    /// Get recommended Bolt configuration for current system
    pub async fn get_recommended_bolt_config() -> Result<BoltConfig> {
        let wsl2_detected = crate::wsl2::Wsl2Manager::detect_wsl2();
        let gpus = crate::gpu::discover_gpus().await?;

        let mut config = BoltConfig::default();

        // Adjust for WSL2 if detected
        if wsl2_detected {
            if let Some(gaming) = &mut config.gaming {
                gaming.wine_optimizations = true;
                gaming.performance_profile = "balanced".to_string();
            }
        }

        // Adjust for GPU count
        if gpus.len() > 1 {
            config.capsule.isolation_level = "virtual".to_string();
            if let Some(aiml) = &mut config.aiml {
                aiml.mig_enabled = true;
            }
        }

        Ok(config)
    }
}

/// Re-export important types for Bolt's convenience
#[cfg(feature = "bolt")]
pub use crate::{
    config::{BoltCapsuleGpuConfig, BoltGamingGpuConfig, BoltAiMlGpuConfig},
    cdi::bolt::{BoltGpuIsolation, BoltGamingConfig, GamingProfile},
    gpu::GpuDevice,
};

#[cfg(test)]
#[cfg(feature = "bolt")]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_nvbind_gpu_manager_creation() {
        let manager = NvbindGpuManager::with_defaults();
        assert!(manager.config.capsule.snapshot_gpu_state);
        assert_eq!(manager.config.capsule.isolation_level, "exclusive");
    }

    #[tokio::test]
    async fn test_bolt_compatibility_check() {
        let manager = NvbindGpuManager::with_defaults();

        // This will fail in CI without GPU, but that's expected
        let _result = manager.check_bolt_gpu_compatibility().await;
        // We just ensure it doesn't panic
    }

    #[tokio::test]
    async fn test_recommended_config_generation() {
        let _config = utils::get_recommended_bolt_config().await;
        // Should not panic
    }
}