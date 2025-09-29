//! User-friendly error handling
//!
//! Provides clear, actionable error messages for common issues.

use anyhow::Result;
use std::fmt;

/// User-friendly error wrapper
#[derive(Debug)]
pub struct UserError {
    pub message: String,
    pub suggestion: Option<String>,
    pub documentation_link: Option<String>,
    pub source: anyhow::Error,
}

impl UserError {
    pub fn new(source: anyhow::Error) -> Self {
        let (message, suggestion) = analyze_error(&source);

        Self {
            message,
            suggestion,
            documentation_link: get_relevant_docs(&source),
            source,
        }
    }

    pub fn display_friendly(&self) {
        eprintln!("❌ Error: {}", self.message);

        if let Some(ref suggestion) = self.suggestion {
            eprintln!("💡 Suggestion: {}", suggestion);
        }

        if let Some(ref link) = self.documentation_link {
            eprintln!("📚 Documentation: {}", link);
        }

        if std::env::var("NVBIND_DEBUG").is_ok() {
            eprintln!("\n🔍 Debug info:");
            eprintln!("{:?}", self.source);
        } else {
            eprintln!("\n💡 Tip: Set NVBIND_DEBUG=1 for detailed error information");
        }
    }
}

impl fmt::Display for UserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(ref suggestion) = self.suggestion {
            write!(f, "\n  Suggestion: {}", suggestion)?;
        }
        Ok(())
    }
}

impl std::error::Error for UserError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.source()
    }
}

/// Analyze error and provide user-friendly message and suggestions
fn analyze_error(error: &anyhow::Error) -> (String, Option<String>) {
    let error_str = error.to_string();

    // NVIDIA driver errors
    if error_str.contains("NVIDIA driver not found") || error_str.contains("no GPU driver modules")
    {
        return (
            "NVIDIA GPU driver is not installed or not loaded".to_string(),
            Some("Install NVIDIA drivers: sudo apt install nvidia-driver-535 (Ubuntu) or follow your distribution's guide".to_string())
        );
    }

    if error_str.contains("NVIDIA driver found but no GPUs detected") {
        return (
            "NVIDIA driver is installed but no GPUs are available".to_string(),
            Some("Check if GPUs are properly connected or if running in a VM/container without GPU passthrough".to_string())
        );
    }

    // Container runtime errors
    if error_str.contains("docker") && error_str.contains("not available") {
        return (
            "Docker runtime is not installed or not running".to_string(),
            Some("Install Docker: curl -fsSL https://get.docker.com | sh\nStart Docker: sudo systemctl start docker".to_string())
        );
    }

    if error_str.contains("podman") && error_str.contains("not available") {
        return (
            "Podman runtime is not installed".to_string(),
            Some(
                "Install Podman: sudo apt install podman (Ubuntu) or dnf install podman (Fedora)"
                    .to_string(),
            ),
        );
    }

    if error_str.contains("bolt") && error_str.contains("not available") {
        return (
            "Bolt runtime is not installed".to_string(),
            Some("Install Bolt from: https://github.com/your-bolt-repo".to_string()),
        );
    }

    // Permission errors
    if error_str.contains("Permission denied") || error_str.contains("EACCES") {
        if error_str.contains("/dev/nvidia") {
            return (
                "Permission denied accessing GPU devices".to_string(),
                Some("Add your user to the 'video' group: sudo usermod -aG video $USER\nThen logout and login again".to_string())
            );
        }
        if error_str.contains("docker.sock") {
            return (
                "Permission denied accessing Docker socket".to_string(),
                Some("Add your user to the 'docker' group: sudo usermod -aG docker $USER\nThen logout and login again".to_string())
            );
        }
        return (
            "Permission denied - insufficient privileges".to_string(),
            Some("Try running with sudo or check file permissions".to_string()),
        );
    }

    // CDI errors
    if error_str.contains("CDI") && error_str.contains("not found") {
        return (
            "CDI specifications not found".to_string(),
            Some("Generate CDI specs: nvbind cdi generate".to_string()),
        );
    }

    // WSL2 errors
    if error_str.contains("WSL2") && error_str.contains("GPU support") {
        return (
            "WSL2 GPU support is not available".to_string(),
            Some("Ensure Windows 11 or Windows 10 21H2+ with NVIDIA GPU drivers installed on Windows host".to_string())
        );
    }

    // Configuration errors
    if error_str.contains("Failed to read config file") {
        return (
            "Configuration file not found or invalid".to_string(),
            Some("Generate default config: nvbind config --output nvbind.toml".to_string()),
        );
    }

    // Network errors
    if error_str.contains("Connection refused") || error_str.contains("NetworkError") {
        return (
            "Network connection failed".to_string(),
            Some("Check network connectivity and firewall settings".to_string()),
        );
    }

    // Resource errors
    if error_str.contains("out of memory") || error_str.contains("OOM") {
        return (
            "Out of memory - insufficient GPU or system memory".to_string(),
            Some(
                "Reduce batch size, close other applications, or use a GPU with more memory"
                    .to_string(),
            ),
        );
    }

    // Default fallback
    (
        error_str.clone(),
        Some("Check the error message and ensure all requirements are met".to_string()),
    )
}

/// Get relevant documentation link based on error type
fn get_relevant_docs(error: &anyhow::Error) -> Option<String> {
    let error_str = error.to_string();

    if error_str.contains("NVIDIA") || error_str.contains("GPU") {
        Some("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/".to_string())
    } else if error_str.contains("docker") {
        Some("https://docs.docker.com/engine/install/".to_string())
    } else if error_str.contains("podman") {
        Some("https://podman.io/getting-started/installation".to_string())
    } else if error_str.contains("WSL2") {
        Some("https://docs.nvidia.com/cuda/wsl-user-guide/".to_string())
    } else if error_str.contains("CDI") {
        Some("https://github.com/cncf-tags/container-device-interface".to_string())
    } else {
        None
    }
}

/// Helper function to wrap any Result with user-friendly error handling
pub trait UserFriendlyError<T> {
    fn user_friendly(self) -> Result<T>;
}

impl<T> UserFriendlyError<T> for Result<T> {
    fn user_friendly(self) -> Result<T> {
        self.map_err(|e| {
            let user_error = UserError::new(e);
            user_error.display_friendly();
            anyhow::anyhow!("Operation failed")
        })
    }
}

/// Validate system requirements and provide clear feedback
pub fn validate_system_requirements() -> Result<()> {
    // Check for NVIDIA driver
    if let Err(e) = crate::gpu::check_nvidia_requirements() {
        return Err(anyhow::anyhow!(
            "System requirements not met: {}. Run 'nvbind doctor' for diagnostics",
            e
        ));
    }

    // Check for container runtime
    let runtimes = vec!["docker", "podman", "bolt"];
    let mut available_runtime = false;

    for runtime in &runtimes {
        if crate::runtime::validate_runtime(runtime).is_ok() {
            available_runtime = true;
            break;
        }
    }

    if !available_runtime {
        return Err(anyhow::anyhow!(
            "No container runtime found. Install Docker, Podman, or Bolt"
        ));
    }

    Ok(())
}

/// Check common issues and provide diagnostics
pub async fn run_diagnostics() -> Result<String> {
    let mut report = String::new();

    report.push_str("=== nvbind System Diagnostics ===\n\n");

    // Check GPU driver
    report.push_str("GPU Driver Status:\n");
    match crate::gpu::get_driver_info().await {
        Ok(info) => {
            report.push_str(&format!(
                "  ✅ Driver: {} ({})\n",
                info.version,
                info.driver_type.name()
            ));
            if let Some(cuda) = info.cuda_version {
                report.push_str(&format!("  ✅ CUDA: {}\n", cuda));
            }
        }
        Err(e) => {
            report.push_str(&format!("  ❌ Error: {}\n", e));
        }
    }

    // Check GPUs
    report.push_str("\nGPU Devices:\n");
    match crate::gpu::discover_gpus().await {
        Ok(gpus) => {
            if gpus.is_empty() {
                report.push_str("  ⚠️  No GPUs detected\n");
            } else {
                for gpu in gpus {
                    report.push_str(&format!(
                        "  ✅ {}: {} ({})\n",
                        gpu.id, gpu.name, gpu.pci_address
                    ));
                }
            }
        }
        Err(e) => {
            report.push_str(&format!("  ❌ Error: {}\n", e));
        }
    }

    // Check container runtimes
    report.push_str("\nContainer Runtimes:\n");
    for runtime in &["docker", "podman", "bolt"] {
        match crate::runtime::validate_runtime(runtime) {
            Ok(_) => report.push_str(&format!("  ✅ {}\n", runtime)),
            Err(_) => report.push_str(&format!("  ❌ {} (not installed)\n", runtime)),
        }
    }

    // Check permissions
    report.push_str("\nPermissions:\n");
    if std::path::Path::new("/dev/nvidia0").exists() {
        match std::fs::metadata("/dev/nvidia0") {
            Ok(_) => report.push_str("  ✅ GPU device access\n"),
            Err(_) => report.push_str("  ❌ Cannot access GPU devices (permission denied)\n"),
        }
    } else {
        report.push_str("  ⚠️  No GPU devices found in /dev/\n");
    }

    // Check CDI
    report.push_str("\nCDI Support:\n");
    let _cdi_registry = crate::cdi::CdiRegistry::new();
    if let Ok(devices) = crate::cdi::list_cdi_devices() {
        if devices.is_empty() {
            report.push_str("  ⚠️  No CDI devices configured\n");
        } else {
            report.push_str(&format!("  ✅ {} CDI devices available\n", devices.len()));
        }
    } else {
        report.push_str("  ❌ CDI not configured\n");
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_analysis() {
        let error = anyhow::anyhow!("NVIDIA driver not found");
        let (msg, suggestion) = analyze_error(&error);
        assert!(msg.contains("NVIDIA GPU driver"));
        assert!(suggestion.is_some());

        let error = anyhow::anyhow!("Permission denied: /dev/nvidia0");
        let (msg, suggestion) = analyze_error(&error);
        assert!(msg.contains("Permission"));
        assert!(suggestion.unwrap().contains("video"));
    }

    #[test]
    fn test_documentation_links() {
        let error = anyhow::anyhow!("NVIDIA GPU error");
        let link = get_relevant_docs(&error);
        assert!(link.is_some());
        assert!(link.unwrap().contains("nvidia"));
    }
}
