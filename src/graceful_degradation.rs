//! Graceful degradation system for nvbind
//!
//! This module implements sophisticated fallback mechanisms when GPU resources
//! are unavailable, drivers are missing, or hardware requirements aren't met.

use crate::error::NvbindError;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

/// Degradation strategies available for different scenarios
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegradationStrategy {
    /// Complete operation without GPU acceleration
    CpuFallback,
    /// Use software rendering instead of hardware acceleration
    SoftwareRendering,
    /// Fallback to alternative container runtime
    AlternativeRuntime,
    /// Use reduced feature set
    FeatureReduction,
    /// Attempt operation with reduced performance expectations
    PerformanceReduction,
    /// Fail operation entirely
    FailFast,
}

/// Degradation configuration for different contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationConfig {
    /// Enable graceful degradation globally
    pub enabled: bool,
    /// Default strategy when specific strategy not defined
    pub default_strategy: DegradationStrategy,
    /// Context-specific strategies
    pub strategies: HashMap<String, DegradationStrategy>,
    /// Maximum degradation attempts before failing
    pub max_attempts: u32,
    /// Track performance impact of degradation
    pub track_performance: bool,
}

impl Default for DegradationConfig {
    fn default() -> Self {
        let mut strategies = HashMap::new();
        strategies.insert("gpu_missing".to_string(), DegradationStrategy::CpuFallback);
        strategies.insert(
            "driver_unavailable".to_string(),
            DegradationStrategy::SoftwareRendering,
        );
        strategies.insert(
            "runtime_failed".to_string(),
            DegradationStrategy::AlternativeRuntime,
        );
        strategies.insert(
            "insufficient_memory".to_string(),
            DegradationStrategy::PerformanceReduction,
        );
        strategies.insert(
            "security_violation".to_string(),
            DegradationStrategy::FailFast,
        );

        Self {
            enabled: true,
            default_strategy: DegradationStrategy::CpuFallback,
            strategies,
            max_attempts: 3,
            track_performance: true,
        }
    }
}

/// System state and available alternatives
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    /// Available container runtimes
    pub available_runtimes: Vec<String>,
    /// GPU availability status
    pub gpu_available: bool,
    /// Driver availability and type
    pub driver_status: DriverAvailability,
    /// System memory and CPU capabilities
    pub system_resources: SystemResources,
    /// Supported rendering modes
    pub rendering_capabilities: RenderingCapabilities,
}

#[derive(Debug, Clone)]
pub enum DriverAvailability {
    Available {
        version: String,
        driver_type: String,
    },
    Unavailable,
    Incompatible {
        version: String,
        required: String,
    },
}

#[derive(Debug, Clone)]
pub struct SystemResources {
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub cpu_cores: u32,
    pub cpu_features: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RenderingCapabilities {
    pub software_rendering: bool,
    pub opengl_version: Option<String>,
    pub vulkan_support: bool,
}

/// Graceful degradation manager
pub struct GracefulDegradationManager {
    config: DegradationConfig,
    capabilities: SystemCapabilities,
    attempt_history: HashMap<String, u32>,
}

impl GracefulDegradationManager {
    /// Create a new degradation manager with system capability detection
    pub async fn new(config: DegradationConfig) -> Result<Self> {
        let capabilities = Self::detect_system_capabilities().await?;

        info!(
            "Graceful degradation initialized - GPU: {}, Driver: {:?}, Runtimes: {:?}",
            capabilities.gpu_available, capabilities.driver_status, capabilities.available_runtimes
        );

        Ok(Self {
            config,
            capabilities,
            attempt_history: HashMap::new(),
        })
    }

    /// Handle operation failure with appropriate degradation strategy
    pub async fn handle_failure(
        &mut self,
        operation: &str,
        error: &NvbindError,
        context: &OperationContext,
    ) -> Result<DegradationResult> {
        if !self.config.enabled {
            return Ok(DegradationResult::NoActionTaken);
        }

        // Check if we've exceeded max attempts for this operation
        // Check current attempt count and increment
        let current_attempts = self.attempt_history.get(operation).copied().unwrap_or(0) + 1;

        if current_attempts > self.config.max_attempts {
            warn!(
                "Maximum degradation attempts exceeded for operation: {}",
                operation
            );
            return Ok(DegradationResult::MaxAttemptsExceeded);
        }

        // Determine appropriate degradation strategy
        let strategy = self.determine_strategy(error, context);

        // Update attempt history after borrowing
        self.attempt_history
            .insert(operation.to_string(), current_attempts);

        info!(
            "Attempting degradation for '{}' using strategy: {:?} (attempt {}/{})",
            operation, strategy, current_attempts, self.config.max_attempts
        );

        // Apply degradation strategy
        match strategy {
            DegradationStrategy::CpuFallback => self.apply_cpu_fallback(context).await,
            DegradationStrategy::SoftwareRendering => self.apply_software_rendering(context).await,
            DegradationStrategy::AlternativeRuntime => {
                self.apply_alternative_runtime(context).await
            }
            DegradationStrategy::FeatureReduction => self.apply_feature_reduction(context).await,
            DegradationStrategy::PerformanceReduction => {
                self.apply_performance_reduction(context).await
            }
            DegradationStrategy::FailFast => Ok(DegradationResult::FailureAccepted),
        }
    }

    /// Determine appropriate degradation strategy based on error type and context
    fn determine_strategy(
        &self,
        error: &NvbindError,
        context: &OperationContext,
    ) -> DegradationStrategy {
        // Check for context-specific strategy first
        if let Some(strategy) = context.preferred_strategy {
            return strategy;
        }

        // Check context-specific strategies first (by operation type)
        if let Some(strategy) = self.config.strategies.get(&context.operation_type) {
            return *strategy;
        }

        // Map error types to degradation strategies
        let strategy_key = match error {
            NvbindError::Gpu { .. } => "gpu_missing",
            NvbindError::Driver { .. } => "driver_unavailable",
            NvbindError::Runtime { .. } => "runtime_failed",
            NvbindError::Security { .. } => "security_violation",
            NvbindError::System { .. } => "system_error",
            _ => "default",
        };

        self.config
            .strategies
            .get(strategy_key)
            .copied()
            .unwrap_or(self.config.default_strategy)
    }

    /// Apply CPU-only fallback
    async fn apply_cpu_fallback(&self, context: &OperationContext) -> Result<DegradationResult> {
        warn!("Falling back to CPU-only mode");

        let mut modifications = Vec::new();

        // Remove GPU-related environment variables
        modifications.push("Remove NVIDIA_VISIBLE_DEVICES environment variable".to_string());
        modifications.push("Set CUDA_VISIBLE_DEVICES=-1 to disable CUDA".to_string());

        // Add CPU optimization flags
        modifications.push("Add CPU optimization flags".to_string());
        modifications.push("Enable multi-threading for CPU workloads".to_string());

        // Update container configuration
        let mut container_config = context.container_config.clone();
        container_config.remove("--gpus");
        container_config.remove("--device");
        container_config.insert("--cpu-optimization".to_string(), "true".to_string());

        Ok(DegradationResult::Applied {
            strategy: DegradationStrategy::CpuFallback,
            modifications,
            performance_impact: Some(PerformanceImpact {
                expected_slowdown: 5.0,    // 5x slower
                memory_usage_change: -0.5, // 50% less GPU memory usage
                cpu_usage_change: 2.0,     // 2x more CPU usage
            }),
            new_container_config: Some(container_config),
        })
    }

    /// Apply software rendering fallback
    async fn apply_software_rendering(
        &self,
        context: &OperationContext,
    ) -> Result<DegradationResult> {
        if !self.capabilities.rendering_capabilities.software_rendering {
            return Ok(DegradationResult::NotApplicable);
        }

        warn!("Falling back to software rendering");

        let mut modifications = Vec::new();
        modifications.push("Enable software rendering mode".to_string());
        modifications.push("Configure Mesa software renderer".to_string());

        let mut container_config = context.container_config.clone();
        container_config.insert("MESA_GL_VERSION_OVERRIDE".to_string(), "3.3".to_string());
        container_config.insert("LIBGL_ALWAYS_SOFTWARE".to_string(), "1".to_string());

        Ok(DegradationResult::Applied {
            strategy: DegradationStrategy::SoftwareRendering,
            modifications,
            performance_impact: Some(PerformanceImpact {
                expected_slowdown: 10.0,  // 10x slower for graphics
                memory_usage_change: 0.1, // Slightly more CPU memory
                cpu_usage_change: 3.0,    // 3x more CPU usage
            }),
            new_container_config: Some(container_config),
        })
    }

    /// Apply alternative container runtime
    async fn apply_alternative_runtime(
        &self,
        context: &OperationContext,
    ) -> Result<DegradationResult> {
        // Find alternative runtime
        let current_runtime = &context.runtime;
        let alternative = self
            .capabilities
            .available_runtimes
            .iter()
            .find(|&rt| rt != current_runtime)
            .cloned();

        if let Some(alt_runtime) = alternative {
            warn!(
                "Switching from {} to {} runtime",
                current_runtime, alt_runtime
            );

            let modifications = vec![
                format!(
                    "Switch container runtime from {} to {}",
                    current_runtime, alt_runtime
                ),
                "Update runtime-specific configuration".to_string(),
            ];

            let mut container_config = context.container_config.clone();
            container_config.insert("runtime".to_string(), alt_runtime);

            Ok(DegradationResult::Applied {
                strategy: DegradationStrategy::AlternativeRuntime,
                modifications,
                performance_impact: Some(PerformanceImpact {
                    expected_slowdown: 1.1, // Minimal performance impact
                    memory_usage_change: 0.0,
                    cpu_usage_change: 0.1,
                }),
                new_container_config: Some(container_config),
            })
        } else {
            Ok(DegradationResult::NotApplicable)
        }
    }

    /// Apply feature reduction
    async fn apply_feature_reduction(
        &self,
        context: &OperationContext,
    ) -> Result<DegradationResult> {
        warn!("Reducing feature set for compatibility");

        let modifications = vec![
            "Disable advanced GPU features".to_string(),
            "Reduce rendering quality settings".to_string(),
            "Limit concurrent operations".to_string(),
            "Disable experimental features".to_string(),
        ];

        let mut container_config = context.container_config.clone();
        container_config.insert("NVBIND_REDUCED_MODE".to_string(), "true".to_string());
        container_config.insert("MAX_CONCURRENT_OPS".to_string(), "1".to_string());

        Ok(DegradationResult::Applied {
            strategy: DegradationStrategy::FeatureReduction,
            modifications,
            performance_impact: Some(PerformanceImpact {
                expected_slowdown: 1.2,
                memory_usage_change: -0.3,
                cpu_usage_change: -0.1,
            }),
            new_container_config: Some(container_config),
        })
    }

    /// Apply performance reduction
    async fn apply_performance_reduction(
        &self,
        context: &OperationContext,
    ) -> Result<DegradationResult> {
        warn!("Reducing performance expectations");

        let modifications = vec![
            "Increase timeout values".to_string(),
            "Reduce batch sizes".to_string(),
            "Enable conservative memory allocation".to_string(),
            "Lower target framerate/throughput".to_string(),
        ];

        let mut container_config = context.container_config.clone();
        container_config.insert("TIMEOUT_MULTIPLIER".to_string(), "3.0".to_string());
        container_config.insert("BATCH_SIZE_REDUCTION".to_string(), "0.5".to_string());

        Ok(DegradationResult::Applied {
            strategy: DegradationStrategy::PerformanceReduction,
            modifications,
            performance_impact: Some(PerformanceImpact {
                expected_slowdown: 2.0,
                memory_usage_change: -0.2,
                cpu_usage_change: 0.0,
            }),
            new_container_config: Some(container_config),
        })
    }

    /// Detect system capabilities
    async fn detect_system_capabilities() -> Result<SystemCapabilities> {
        // Detect available runtimes
        let available_runtimes = Self::detect_available_runtimes().await;

        // Check GPU availability
        let gpu_available = crate::gpu::is_nvidia_driver_available();

        // Detect driver status
        let driver_status = if gpu_available {
            match crate::gpu::get_driver_info().await {
                Ok(info) => DriverAvailability::Available {
                    version: info.version,
                    driver_type: format!("{:?}", info.driver_type),
                },
                Err(_) => DriverAvailability::Incompatible {
                    version: "unknown".to_string(),
                    required: "515.0+".to_string(),
                },
            }
        } else {
            DriverAvailability::Unavailable
        };

        // Detect system resources
        let system_resources = Self::detect_system_resources();

        // Detect rendering capabilities
        let rendering_capabilities = Self::detect_rendering_capabilities();

        Ok(SystemCapabilities {
            available_runtimes,
            gpu_available,
            driver_status,
            system_resources,
            rendering_capabilities,
        })
    }

    /// Detect available container runtimes
    async fn detect_available_runtimes() -> Vec<String> {
        let mut runtimes = Vec::new();

        let candidates = ["podman", "docker", "containerd", "cri-o"];

        for runtime in &candidates {
            if crate::runtime::validate_runtime(runtime).is_ok() {
                runtimes.push(runtime.to_string());
            }
        }

        runtimes
    }

    /// Detect system resource information
    fn detect_system_resources() -> SystemResources {
        let total_memory_gb = Self::get_total_memory_gb();
        let available_memory_gb = Self::get_available_memory_gb();
        let cpu_cores = std::thread::available_parallelism()
            .map(|p| p.get() as u32)
            .unwrap_or(4); // Default to 4 cores if detection fails
        let cpu_features = Self::detect_cpu_features();

        SystemResources {
            total_memory_gb,
            available_memory_gb,
            cpu_cores,
            cpu_features,
        }
    }

    /// Detect rendering capabilities
    fn detect_rendering_capabilities() -> RenderingCapabilities {
        RenderingCapabilities {
            software_rendering: true, // Mesa is usually available
            opengl_version: Self::detect_opengl_version(),
            vulkan_support: Self::detect_vulkan_support(),
        }
    }

    // Helper methods for system detection
    fn get_total_memory_gb() -> f64 {
        // Simplified memory detection - in production, use proper system APIs
        16.0 // Default assumption
    }

    fn get_available_memory_gb() -> f64 {
        12.0 // Default assumption
    }

    fn detect_cpu_features() -> Vec<String> {
        vec!["sse4_2".to_string(), "avx2".to_string()] // Default assumptions
    }

    fn detect_opengl_version() -> Option<String> {
        Some("3.3".to_string()) // Common baseline
    }

    fn detect_vulkan_support() -> bool {
        std::path::Path::new("/usr/lib/libvulkan.so").exists()
            || std::path::Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so").exists()
    }

    /// Reset attempt history for successful operations
    pub fn reset_attempts(&mut self, operation: &str) {
        self.attempt_history.remove(operation);
    }

    /// Get degradation statistics
    pub fn get_statistics(&self) -> DegradationStatistics {
        DegradationStatistics {
            total_attempts: self.attempt_history.values().sum(),
            operations_with_attempts: self.attempt_history.len(),
            capabilities: self.capabilities.clone(),
        }
    }
}

/// Context for operation being degraded
#[derive(Debug, Clone)]
pub struct OperationContext {
    pub operation_type: String,
    pub runtime: String,
    pub container_config: HashMap<String, String>,
    pub preferred_strategy: Option<DegradationStrategy>,
    pub criticality: OperationCriticality,
}

/// Operation criticality levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OperationCriticality {
    Critical, // Must succeed, try all strategies
    High,     // Important, try multiple strategies
    Medium,   // Normal, try reasonable strategies
    Low,      // Optional, minimal degradation attempts
}

/// Result of degradation attempt
#[derive(Debug)]
pub enum DegradationResult {
    Applied {
        strategy: DegradationStrategy,
        modifications: Vec<String>,
        performance_impact: Option<PerformanceImpact>,
        new_container_config: Option<HashMap<String, String>>,
    },
    NotApplicable,
    MaxAttemptsExceeded,
    FailureAccepted,
    NoActionTaken,
}

/// Expected performance impact of degradation
#[derive(Debug)]
pub struct PerformanceImpact {
    /// Expected slowdown multiplier (1.0 = no change, 2.0 = 2x slower)
    pub expected_slowdown: f64,
    /// Memory usage change multiplier (-0.5 = 50% less, 0.2 = 20% more)
    pub memory_usage_change: f64,
    /// CPU usage change multiplier
    pub cpu_usage_change: f64,
}

/// Statistics about degradation usage
#[derive(Debug)]
pub struct DegradationStatistics {
    pub total_attempts: u32,
    pub operations_with_attempts: usize,
    pub capabilities: SystemCapabilities,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degradation_config_default() {
        let config = DegradationConfig::default();
        assert!(config.enabled);
        assert_eq!(config.default_strategy, DegradationStrategy::CpuFallback);
        assert!(config.strategies.contains_key("gpu_missing"));
    }

    #[tokio::test]
    async fn test_degradation_manager_creation() {
        let config = DegradationConfig::default();
        let manager = GracefulDegradationManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[test]
    fn test_operation_context() {
        let context = OperationContext {
            operation_type: "gpu_processing".to_string(),
            runtime: "podman".to_string(),
            container_config: HashMap::new(),
            preferred_strategy: Some(DegradationStrategy::CpuFallback),
            criticality: OperationCriticality::High,
        };

        assert_eq!(context.runtime, "podman");
        assert_eq!(
            context.preferred_strategy,
            Some(DegradationStrategy::CpuFallback)
        );
    }
}
