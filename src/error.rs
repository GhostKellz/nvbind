//! Production-grade error handling and recovery for nvbind
//!
//! This module provides comprehensive error handling, graceful degradation,
//! and detailed error reporting for production environments.

use std::fmt;
use thiserror::Error;

/// Main nvbind error type with detailed context and recovery suggestions
#[derive(Error, Debug)]
pub enum NvbindError {
    /// GPU-related errors
    #[error("GPU Error: {message}")]
    Gpu {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        recovery_suggestion: String,
    },

    /// Driver-related errors
    #[error("Driver Error: {message}")]
    Driver {
        message: String,
        driver_version: Option<String>,
        required_version: Option<String>,
        recovery_suggestion: String,
    },

    /// Runtime errors (Docker, Podman, Bolt)
    #[error("Runtime Error: {runtime} - {message}")]
    Runtime {
        runtime: String,
        message: String,
        recovery_suggestion: String,
    },

    /// Configuration errors
    #[error("Configuration Error: {message}")]
    Configuration {
        message: String,
        config_path: Option<String>,
        recovery_suggestion: String,
    },

    /// CDI (Container Device Interface) errors
    #[error("CDI Error: {message}")]
    Cdi {
        message: String,
        device_name: Option<String>,
        recovery_suggestion: String,
    },

    /// Isolation and security errors
    #[error("Security Error: {message}")]
    Security {
        message: String,
        security_context: String,
        recovery_suggestion: String,
    },

    /// WSL2-specific errors
    #[error("WSL2 Error: {message}")]
    Wsl2 {
        message: String,
        windows_build: Option<u32>,
        recovery_suggestion: String,
    },

    /// Bolt-specific errors
    #[cfg(feature = "bolt")]
    #[error("Bolt Error: {message}")]
    Bolt {
        message: String,
        operation: String,
        recovery_suggestion: String,
    },

    /// System-level errors
    #[error("System Error: {message}")]
    System {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        recovery_suggestion: String,
    },
}

impl NvbindError {
    /// Create a GPU error with recovery suggestion
    pub fn gpu(message: impl Into<String>, recovery: impl Into<String>) -> Self {
        Self::Gpu {
            message: message.into(),
            source: None,
            recovery_suggestion: recovery.into(),
        }
    }

    /// Create a GPU error with source and recovery suggestion
    pub fn gpu_with_source(
        message: impl Into<String>,
        source: Box<dyn std::error::Error + Send + Sync>,
        recovery: impl Into<String>,
    ) -> Self {
        Self::Gpu {
            message: message.into(),
            source: Some(source),
            recovery_suggestion: recovery.into(),
        }
    }

    /// Create a driver error
    pub fn driver(
        message: impl Into<String>,
        driver_version: Option<String>,
        required_version: Option<String>,
        recovery: impl Into<String>,
    ) -> Self {
        Self::Driver {
            message: message.into(),
            driver_version,
            required_version,
            recovery_suggestion: recovery.into(),
        }
    }

    /// Create a runtime error
    pub fn runtime(
        runtime: impl Into<String>,
        message: impl Into<String>,
        recovery: impl Into<String>,
    ) -> Self {
        Self::Runtime {
            runtime: runtime.into(),
            message: message.into(),
            recovery_suggestion: recovery.into(),
        }
    }

    /// Create a configuration error
    pub fn configuration(
        message: impl Into<String>,
        config_path: Option<String>,
        recovery: impl Into<String>,
    ) -> Self {
        Self::Configuration {
            message: message.into(),
            config_path,
            recovery_suggestion: recovery.into(),
        }
    }

    /// Get the recovery suggestion for this error
    pub fn recovery_suggestion(&self) -> &str {
        match self {
            NvbindError::Gpu {
                recovery_suggestion,
                ..
            } => recovery_suggestion,
            NvbindError::Driver {
                recovery_suggestion,
                ..
            } => recovery_suggestion,
            NvbindError::Runtime {
                recovery_suggestion,
                ..
            } => recovery_suggestion,
            NvbindError::Configuration {
                recovery_suggestion,
                ..
            } => recovery_suggestion,
            NvbindError::Cdi {
                recovery_suggestion,
                ..
            } => recovery_suggestion,
            NvbindError::Security {
                recovery_suggestion,
                ..
            } => recovery_suggestion,
            NvbindError::Wsl2 {
                recovery_suggestion,
                ..
            } => recovery_suggestion,
            #[cfg(feature = "bolt")]
            NvbindError::Bolt {
                recovery_suggestion,
                ..
            } => recovery_suggestion,
            NvbindError::System {
                recovery_suggestion,
                ..
            } => recovery_suggestion,
        }
    }

    /// Check if this error allows graceful degradation
    pub fn allows_graceful_degradation(&self) -> bool {
        matches!(
            self,
            NvbindError::Gpu { .. } | NvbindError::Driver { .. } | NvbindError::Wsl2 { .. }
        )
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            NvbindError::Security { .. } => ErrorSeverity::Critical,
            NvbindError::System { .. } => ErrorSeverity::Critical,
            NvbindError::Configuration { .. } => ErrorSeverity::High,
            NvbindError::Runtime { .. } => ErrorSeverity::High,
            NvbindError::Driver { .. } => ErrorSeverity::Medium,
            NvbindError::Gpu { .. } => ErrorSeverity::Medium,
            NvbindError::Cdi { .. } => ErrorSeverity::Medium,
            NvbindError::Wsl2 { .. } => ErrorSeverity::Low,
            #[cfg(feature = "bolt")]
            NvbindError::Bolt { .. } => ErrorSeverity::Medium,
        }
    }
}

/// Error severity levels for monitoring and alerting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Critical,
    High,
    Medium,
    Low,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::Low => write!(f, "LOW"),
        }
    }
}

/// Error recovery manager for graceful degradation
pub struct ErrorRecoveryManager {
    degradation_enabled: bool,
    fallback_runtime: Option<String>,
}

impl Default for ErrorRecoveryManager {
    fn default() -> Self {
        Self {
            degradation_enabled: true,
            fallback_runtime: Some("podman".to_string()),
        }
    }
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager
    pub fn new(degradation_enabled: bool, fallback_runtime: Option<String>) -> Self {
        Self {
            degradation_enabled,
            fallback_runtime,
        }
    }

    /// Attempt to recover from an error
    pub fn recover_from_error(&self, error: &NvbindError) -> RecoveryAction {
        if !self.degradation_enabled {
            return RecoveryAction::Fail;
        }

        match error {
            NvbindError::Gpu { .. } => {
                tracing::warn!("GPU unavailable, attempting CPU-only mode");
                RecoveryAction::DegradeToCpuOnly
            }

            NvbindError::Driver { .. } => {
                tracing::warn!("Driver issues detected, attempting software fallback");
                RecoveryAction::DegradeToSoftwareRendering
            }

            NvbindError::Runtime { runtime, .. } => {
                if let Some(fallback) = &self.fallback_runtime {
                    if runtime != fallback {
                        tracing::warn!("Runtime {} failed, falling back to {}", runtime, fallback);
                        return RecoveryAction::FallbackRuntime(fallback.clone());
                    }
                }
                RecoveryAction::Fail
            }

            NvbindError::Wsl2 { .. } => {
                tracing::warn!("WSL2 issues detected, attempting native Linux mode");
                RecoveryAction::DegradeToNativeLinux
            }

            _ => RecoveryAction::Fail,
        }
    }

    /// Generate detailed error report for debugging
    pub fn generate_error_report(&self, error: &NvbindError) -> ErrorReport {
        ErrorReport {
            error_type: format!("{:?}", error),
            message: error.to_string(),
            severity: error.severity(),
            recovery_suggestion: error.recovery_suggestion().to_string(),
            timestamp: chrono::Utc::now(),
            system_info: self.collect_system_info(),
            allows_degradation: error.allows_graceful_degradation(),
        }
    }

    /// Collect system information for error reports
    fn collect_system_info(&self) -> SystemInfo {
        SystemInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            hostname: hostname::get()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            nvidia_driver_available: crate::gpu::is_nvidia_driver_available(),
            wsl2_detected: crate::wsl2::Wsl2Manager::detect_wsl2(),
            available_runtimes: self.detect_available_runtimes(),
        }
    }

    /// Detect available container runtimes
    fn detect_available_runtimes(&self) -> Vec<String> {
        let mut available = Vec::new();

        for runtime in &["podman", "docker", "bolt"] {
            if crate::runtime::validate_runtime(runtime).is_ok() {
                available.push(runtime.to_string());
            }
        }

        available
    }
}

/// Recovery action recommendations
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Continue operation without GPU
    DegradeToCpuOnly,
    /// Fall back to software rendering
    DegradeToSoftwareRendering,
    /// Use alternative container runtime
    FallbackRuntime(String),
    /// Disable WSL2-specific optimizations
    DegradeToNativeLinux,
    /// Operation cannot be recovered
    Fail,
}

/// Detailed error report for debugging and monitoring
#[derive(Debug, Clone)]
pub struct ErrorReport {
    pub error_type: String,
    pub message: String,
    pub severity: ErrorSeverity,
    pub recovery_suggestion: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub system_info: SystemInfo,
    pub allows_degradation: bool,
}

/// System information for error context
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub hostname: String,
    pub nvidia_driver_available: bool,
    pub wsl2_detected: bool,
    pub available_runtimes: Vec<String>,
}

/// Result type alias for nvbind operations
pub type Result<T> = std::result::Result<T, NvbindError>;

/// Convenience macros for creating errors
#[macro_export]
macro_rules! gpu_error {
    ($msg:expr, $recovery:expr) => {
        $crate::error::NvbindError::gpu($msg, $recovery)
    };
}

#[macro_export]
macro_rules! runtime_error {
    ($runtime:expr, $msg:expr, $recovery:expr) => {
        $crate::error::NvbindError::runtime($runtime, $msg, $recovery)
    };
}

#[macro_export]
macro_rules! config_error {
    ($msg:expr, $path:expr, $recovery:expr) => {
        $crate::error::NvbindError::configuration($msg, $path, $recovery)
    };
}

/// Error handling utilities
pub mod utils {
    use super::*;
    use tracing::{error, info, warn};

    /// Log error with appropriate level based on severity
    pub fn log_error(error: &NvbindError) {
        match error.severity() {
            ErrorSeverity::Critical => {
                error!(
                    "CRITICAL ERROR: {}\nRecovery: {}",
                    error,
                    error.recovery_suggestion()
                );
            }
            ErrorSeverity::High => {
                error!(
                    "HIGH SEVERITY: {}\nRecovery: {}",
                    error,
                    error.recovery_suggestion()
                );
            }
            ErrorSeverity::Medium => {
                warn!(
                    "MEDIUM SEVERITY: {}\nRecovery: {}",
                    error,
                    error.recovery_suggestion()
                );
            }
            ErrorSeverity::Low => {
                info!(
                    "LOW SEVERITY: {}\nRecovery: {}",
                    error,
                    error.recovery_suggestion()
                );
            }
        }
    }

    /// Handle error with automatic recovery attempt
    pub fn handle_error_with_recovery(
        error: NvbindError,
        recovery_manager: &ErrorRecoveryManager,
    ) -> Result<RecoveryAction> {
        log_error(&error);

        let recovery_action = recovery_manager.recover_from_error(&error);

        match &recovery_action {
            RecoveryAction::Fail => {
                error!("Error recovery failed: {}", error);
                Err(error)
            }
            action => {
                info!("Attempting recovery: {:?}", action);
                Ok(recovery_action)
            }
        }
    }

    /// Validate error recovery was successful
    pub fn validate_recovery(action: &RecoveryAction) -> bool {
        match action {
            RecoveryAction::DegradeToCpuOnly => {
                info!("Successfully degraded to CPU-only mode");
                true
            }
            RecoveryAction::DegradeToSoftwareRendering => {
                info!("Successfully degraded to software rendering");
                true
            }
            RecoveryAction::FallbackRuntime(runtime) => {
                match crate::runtime::validate_runtime(runtime) {
                    Ok(_) => {
                        info!("Successfully fell back to runtime: {}", runtime);
                        true
                    }
                    Err(_) => {
                        error!("Fallback runtime {} not available", runtime);
                        false
                    }
                }
            }
            RecoveryAction::DegradeToNativeLinux => {
                info!("Successfully degraded to native Linux mode");
                true
            }
            RecoveryAction::Fail => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = NvbindError::gpu(
            "No GPU found",
            "Install NVIDIA drivers and ensure GPU is properly connected",
        );

        assert_eq!(error.severity(), ErrorSeverity::Medium);
        assert!(error.allows_graceful_degradation());
        assert!(error.recovery_suggestion().contains("NVIDIA drivers"));
    }

    #[test]
    fn test_error_recovery_manager() {
        let manager = ErrorRecoveryManager::default();

        let gpu_error = NvbindError::gpu("GPU not found", "Install drivers");
        let action = manager.recover_from_error(&gpu_error);

        assert!(matches!(action, RecoveryAction::DegradeToCpuOnly));
    }

    #[test]
    fn test_error_report_generation() {
        let manager = ErrorRecoveryManager::default();
        let error = NvbindError::runtime("docker", "Runtime failed", "Try podman");

        let report = manager.generate_error_report(&error);

        assert_eq!(report.severity, ErrorSeverity::High);
        assert!(!report.system_info.os.is_empty());
        assert!(!report.recovery_suggestion.is_empty());
    }
}
