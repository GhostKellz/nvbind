use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use tracing::{info, warn};

/// Supported Linux distributions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Distribution {
    Arch,
    Debian,
    Ubuntu,
    Fedora,
    CentOS,
    RHEL,
    OpenSUSE,
    Manjaro,
    Unknown(String),
}

/// Distribution-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistroConfig {
    pub distribution: Distribution,
    pub version: String,
    pub package_manager: PackageManager,
    pub nvidia_packages: Vec<String>,
    pub opencl_packages: Vec<String>,
    pub vulkan_packages: Vec<String>,
    pub dev_packages: Vec<String>,
    pub library_paths: Vec<String>,
    pub service_manager: ServiceManager,
    pub kernel_modules: Vec<String>,
}

/// Package manager information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManager {
    pub name: String,
    pub install_cmd: Vec<String>,
    pub update_cmd: Vec<String>,
    pub search_cmd: Vec<String>,
    pub remove_cmd: Vec<String>,
}

/// Service manager type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceManager {
    Systemd,
    OpenRC,
    Runit,
    SysVInit,
}

impl Distribution {
    /// Detect the current Linux distribution
    pub fn detect() -> Result<Self> {
        // Check /etc/os-release first (most modern distros)
        if let Ok(os_release) = Self::parse_os_release() {
            return Ok(os_release);
        }

        // Fallback to /etc/lsb-release
        if let Ok(lsb_release) = Self::parse_lsb_release() {
            return Ok(lsb_release);
        }

        // Check distribution-specific files
        if Path::new("/etc/arch-release").exists() {
            return Ok(Distribution::Arch);
        }

        if Path::new("/etc/debian_version").exists() {
            return Ok(Distribution::Debian);
        }

        if Path::new("/etc/fedora-release").exists() {
            return Ok(Distribution::Fedora);
        }

        if Path::new("/etc/centos-release").exists() {
            return Ok(Distribution::CentOS);
        }

        if Path::new("/etc/redhat-release").exists() {
            return Ok(Distribution::RHEL);
        }

        if Path::new("/etc/SuSE-release").exists() {
            return Ok(Distribution::OpenSUSE);
        }

        // Try uname as last resort
        if let Ok(output) = Command::new("uname").arg("-a").output() {
            let uname_output = String::from_utf8_lossy(&output.stdout).to_lowercase();

            if uname_output.contains("arch") {
                return Ok(Distribution::Arch);
            }
            if uname_output.contains("ubuntu") {
                return Ok(Distribution::Ubuntu);
            }
            if uname_output.contains("debian") {
                return Ok(Distribution::Debian);
            }
            if uname_output.contains("fedora") {
                return Ok(Distribution::Fedora);
            }
        }

        Ok(Distribution::Unknown("linux".to_string()))
    }

    /// Parse /etc/os-release file
    fn parse_os_release() -> Result<Self> {
        let content =
            fs::read_to_string("/etc/os-release").context("Failed to read /etc/os-release")?;

        let mut id = None;
        let mut id_like = None;

        for line in content.lines() {
            if let Some(value) = line.strip_prefix("ID=") {
                id = Some(value.trim_matches('"').to_lowercase());
            } else if let Some(value) = line.strip_prefix("ID_LIKE=") {
                id_like = Some(value.trim_matches('"').to_lowercase());
            }
        }

        if let Some(id) = id {
            return Ok(match id.as_str() {
                "arch" => Distribution::Arch,
                "manjaro" => Distribution::Manjaro,
                "debian" => Distribution::Debian,
                "ubuntu" => Distribution::Ubuntu,
                "fedora" => Distribution::Fedora,
                "centos" => Distribution::CentOS,
                "rhel" => Distribution::RHEL,
                "opensuse" | "opensuse-leap" | "opensuse-tumbleweed" => Distribution::OpenSUSE,
                _ => {
                    // Check ID_LIKE for derived distributions
                    if let Some(id_like) = id_like {
                        if id_like.contains("arch") {
                            Distribution::Arch
                        } else if id_like.contains("debian") {
                            Distribution::Debian
                        } else if id_like.contains("ubuntu") {
                            Distribution::Ubuntu
                        } else if id_like.contains("fedora") || id_like.contains("rhel") {
                            Distribution::Fedora
                        } else {
                            Distribution::Unknown(id)
                        }
                    } else {
                        Distribution::Unknown(id)
                    }
                }
            });
        }

        Err(anyhow::anyhow!(
            "Could not determine distribution from os-release"
        ))
    }

    /// Parse /etc/lsb-release file
    fn parse_lsb_release() -> Result<Self> {
        let content =
            fs::read_to_string("/etc/lsb-release").context("Failed to read /etc/lsb-release")?;

        for line in content.lines() {
            if let Some(value) = line.strip_prefix("DISTRIB_ID=") {
                let distrib_id = value.trim_matches('"').to_lowercase();
                return Ok(match distrib_id.as_str() {
                    "arch" => Distribution::Arch,
                    "manjaro" => Distribution::Manjaro,
                    "debian" => Distribution::Debian,
                    "ubuntu" => Distribution::Ubuntu,
                    "fedora" => Distribution::Fedora,
                    "centos" => Distribution::CentOS,
                    "rhel" => Distribution::RHEL,
                    "opensuse" => Distribution::OpenSUSE,
                    _ => Distribution::Unknown(distrib_id),
                });
            }
        }

        Err(anyhow::anyhow!(
            "Could not determine distribution from lsb-release"
        ))
    }

    /// Get the distribution name as a string
    pub fn name(&self) -> &str {
        match self {
            Distribution::Arch => "Arch Linux",
            Distribution::Debian => "Debian",
            Distribution::Ubuntu => "Ubuntu",
            Distribution::Fedora => "Fedora",
            Distribution::CentOS => "CentOS",
            Distribution::RHEL => "Red Hat Enterprise Linux",
            Distribution::OpenSUSE => "openSUSE",
            Distribution::Manjaro => "Manjaro",
            Distribution::Unknown(name) => name,
        }
    }
}

impl DistroConfig {
    /// Create distribution configuration based on detected distro
    pub fn for_distribution(distro: Distribution, version: String) -> Self {
        match distro {
            Distribution::Arch | Distribution::Manjaro => Self::arch_config(distro, version),
            Distribution::Debian => Self::debian_config(version),
            Distribution::Ubuntu => Self::ubuntu_config(version),
            Distribution::Fedora => Self::fedora_config(version),
            Distribution::CentOS | Distribution::RHEL => Self::rhel_config(distro, version),
            Distribution::OpenSUSE => Self::opensuse_config(version),
            Distribution::Unknown(_) => Self::generic_config(distro, version),
        }
    }

    /// Arch Linux configuration
    fn arch_config(distro: Distribution, version: String) -> Self {
        Self {
            distribution: distro,
            version,
            package_manager: PackageManager {
                name: "pacman".to_string(),
                install_cmd: vec![
                    "pacman".to_string(),
                    "-S".to_string(),
                    "--noconfirm".to_string(),
                ],
                update_cmd: vec![
                    "pacman".to_string(),
                    "-Syu".to_string(),
                    "--noconfirm".to_string(),
                ],
                search_cmd: vec!["pacman".to_string(), "-Ss".to_string()],
                remove_cmd: vec![
                    "pacman".to_string(),
                    "-R".to_string(),
                    "--noconfirm".to_string(),
                ],
            },
            nvidia_packages: vec![
                "nvidia-open".to_string(), // Preferred open-source driver
                "nvidia".to_string(),      // Proprietary driver fallback
                "nvidia-utils".to_string(),
                "nvidia-settings".to_string(),
                "cuda".to_string(),
                "cuda-tools".to_string(),
            ],
            opencl_packages: vec![
                "opencl-nvidia".to_string(),
                "opencl-headers".to_string(),
                "opencl-clhpp".to_string(),
            ],
            vulkan_packages: vec![
                "vulkan-tools".to_string(),
                "vulkan-validation-layers".to_string(),
                "nvidia-vulkan-driver".to_string(),
            ],
            dev_packages: vec![
                "base-devel".to_string(),
                "linux-headers".to_string(),
                "dkms".to_string(),
            ],
            library_paths: vec![
                "/usr/lib".to_string(),
                "/usr/lib64".to_string(),
                "/opt/cuda/lib64".to_string(),
            ],
            service_manager: ServiceManager::Systemd,
            kernel_modules: vec![
                "nvidia".to_string(),
                "nvidia_modeset".to_string(),
                "nvidia_uvm".to_string(),
                "nvidia_drm".to_string(),
            ],
        }
    }

    /// Debian configuration
    fn debian_config(version: String) -> Self {
        Self {
            distribution: Distribution::Debian,
            version,
            package_manager: PackageManager {
                name: "apt".to_string(),
                install_cmd: vec!["apt".to_string(), "install".to_string(), "-y".to_string()],
                update_cmd: vec!["apt".to_string(), "update".to_string()],
                search_cmd: vec!["apt".to_string(), "search".to_string()],
                remove_cmd: vec!["apt".to_string(), "remove".to_string(), "-y".to_string()],
            },
            nvidia_packages: vec![
                "nvidia-driver".to_string(),
                "nvidia-cuda-toolkit".to_string(),
                "nvidia-settings".to_string(),
                "libnvidia-ml1".to_string(),
            ],
            opencl_packages: vec![
                "nvidia-opencl-icd".to_string(),
                "opencl-headers".to_string(),
                "ocl-icd-opencl-dev".to_string(),
            ],
            vulkan_packages: vec![
                "vulkan-tools".to_string(),
                "vulkan-validationlayers".to_string(),
                "nvidia-vulkan-icd".to_string(),
            ],
            dev_packages: vec![
                "build-essential".to_string(),
                "linux-headers-generic".to_string(),
                "dkms".to_string(),
            ],
            library_paths: vec![
                "/usr/lib/x86_64-linux-gnu".to_string(),
                "/usr/lib".to_string(),
                "/usr/local/cuda/lib64".to_string(),
            ],
            service_manager: ServiceManager::Systemd,
            kernel_modules: vec![
                "nvidia".to_string(),
                "nvidia_modeset".to_string(),
                "nvidia_uvm".to_string(),
                "nvidia_drm".to_string(),
            ],
        }
    }

    /// Ubuntu configuration
    fn ubuntu_config(version: String) -> Self {
        let mut config = Self::debian_config(version);
        config.distribution = Distribution::Ubuntu;

        // Ubuntu-specific NVIDIA packages
        config.nvidia_packages = vec![
            "nvidia-driver-535".to_string(), // Latest stable
            "nvidia-driver-525".to_string(), // LTS fallback
            "nvidia-cuda-toolkit".to_string(),
            "nvidia-settings".to_string(),
            "nvidia-prime".to_string(), // For hybrid graphics
        ];

        config
    }

    /// Fedora configuration
    fn fedora_config(version: String) -> Self {
        Self {
            distribution: Distribution::Fedora,
            version,
            package_manager: PackageManager {
                name: "dnf".to_string(),
                install_cmd: vec!["dnf".to_string(), "install".to_string(), "-y".to_string()],
                update_cmd: vec!["dnf".to_string(), "update".to_string(), "-y".to_string()],
                search_cmd: vec!["dnf".to_string(), "search".to_string()],
                remove_cmd: vec!["dnf".to_string(), "remove".to_string(), "-y".to_string()],
            },
            nvidia_packages: vec![
                "akmod-nvidia".to_string(), // RPM Fusion
                "xorg-x11-drv-nvidia".to_string(),
                "xorg-x11-drv-nvidia-cuda".to_string(),
                "nvidia-settings".to_string(),
            ],
            opencl_packages: vec![
                "nvidia-opencl".to_string(),
                "opencl-headers".to_string(),
                "ocl-icd-devel".to_string(),
            ],
            vulkan_packages: vec![
                "vulkan-tools".to_string(),
                "vulkan-validation-layers".to_string(),
                "nvidia-vulkan-driver".to_string(),
            ],
            dev_packages: vec![
                "gcc".to_string(),
                "gcc-c++".to_string(),
                "kernel-devel".to_string(),
                "kernel-headers".to_string(),
                "dkms".to_string(),
            ],
            library_paths: vec![
                "/usr/lib64".to_string(),
                "/usr/lib".to_string(),
                "/usr/local/cuda/lib64".to_string(),
            ],
            service_manager: ServiceManager::Systemd,
            kernel_modules: vec![
                "nvidia".to_string(),
                "nvidia_modeset".to_string(),
                "nvidia_uvm".to_string(),
                "nvidia_drm".to_string(),
            ],
        }
    }

    /// RHEL/CentOS configuration
    fn rhel_config(distro: Distribution, version: String) -> Self {
        let mut config = Self::fedora_config(version);
        config.distribution = distro;

        // RHEL might use yum instead of dnf on older versions
        if config.version.starts_with('7') {
            config.package_manager.name = "yum".to_string();
            config.package_manager.install_cmd[0] = "yum".to_string();
            config.package_manager.update_cmd[0] = "yum".to_string();
            config.package_manager.search_cmd[0] = "yum".to_string();
            config.package_manager.remove_cmd[0] = "yum".to_string();
        }

        config
    }

    /// openSUSE configuration
    fn opensuse_config(version: String) -> Self {
        Self {
            distribution: Distribution::OpenSUSE,
            version,
            package_manager: PackageManager {
                name: "zypper".to_string(),
                install_cmd: vec![
                    "zypper".to_string(),
                    "install".to_string(),
                    "-y".to_string(),
                ],
                update_cmd: vec!["zypper".to_string(), "update".to_string(), "-y".to_string()],
                search_cmd: vec!["zypper".to_string(), "search".to_string()],
                remove_cmd: vec!["zypper".to_string(), "remove".to_string(), "-y".to_string()],
            },
            nvidia_packages: vec![
                "nvidia-glG05".to_string(),
                "nvidia-computeG05".to_string(),
                "nvidia-utils".to_string(),
            ],
            opencl_packages: vec!["nvidia-opencl".to_string(), "opencl-headers".to_string()],
            vulkan_packages: vec![
                "vulkan-tools".to_string(),
                "vulkan-validationlayers".to_string(),
            ],
            dev_packages: vec![
                "gcc".to_string(),
                "gcc-c++".to_string(),
                "kernel-devel".to_string(),
                "kernel-default-devel".to_string(),
            ],
            library_paths: vec!["/usr/lib64".to_string(), "/usr/lib".to_string()],
            service_manager: ServiceManager::Systemd,
            kernel_modules: vec![
                "nvidia".to_string(),
                "nvidia_modeset".to_string(),
                "nvidia_uvm".to_string(),
                "nvidia_drm".to_string(),
            ],
        }
    }

    /// Generic configuration for unknown distributions
    fn generic_config(distro: Distribution, version: String) -> Self {
        Self {
            distribution: distro,
            version,
            package_manager: PackageManager {
                name: "unknown".to_string(),
                install_cmd: vec![],
                update_cmd: vec![],
                search_cmd: vec![],
                remove_cmd: vec![],
            },
            nvidia_packages: vec![],
            opencl_packages: vec![],
            vulkan_packages: vec![],
            dev_packages: vec![],
            library_paths: vec![
                "/usr/lib".to_string(),
                "/usr/lib64".to_string(),
                "/usr/local/lib".to_string(),
                "/usr/local/lib64".to_string(),
            ],
            service_manager: ServiceManager::Systemd,
            kernel_modules: vec![
                "nvidia".to_string(),
                "nvidia_modeset".to_string(),
                "nvidia_uvm".to_string(),
                "nvidia_drm".to_string(),
            ],
        }
    }

    /// Get installation instructions for the distribution
    pub fn get_install_instructions(&self) -> Vec<String> {
        let mut instructions = Vec::new();

        match self.distribution {
            Distribution::Arch | Distribution::Manjaro => {
                instructions.push("# Arch Linux / Manjaro Installation".to_string());
                instructions.push("sudo pacman -Syu".to_string());
                instructions
                    .push("sudo pacman -S nvidia-open nvidia-utils nvidia-settings".to_string());
                instructions.push("# For AUR packages (optional):".to_string());
                instructions.push("# yay -S cuda cuda-tools".to_string());
            }
            Distribution::Debian => {
                instructions.push("# Debian Installation".to_string());
                instructions.push("sudo apt update".to_string());
                instructions.push("sudo apt install nvidia-driver nvidia-cuda-toolkit".to_string());
                instructions.push("# Add non-free repository if needed:".to_string());
                instructions.push("# echo 'deb http://deb.debian.org/debian bookworm non-free-firmware' | sudo tee -a /etc/apt/sources.list".to_string());
            }
            Distribution::Ubuntu => {
                instructions.push("# Ubuntu Installation".to_string());
                instructions.push("sudo apt update".to_string());
                instructions.push("sudo ubuntu-drivers autoinstall".to_string());
                instructions.push("# Or manually:".to_string());
                instructions.push("sudo apt install nvidia-driver-535".to_string());
            }
            Distribution::Fedora => {
                instructions.push("# Fedora Installation".to_string());
                instructions.push("sudo dnf update".to_string());
                instructions.push("# Enable RPM Fusion:".to_string());
                instructions.push("sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm".to_string());
                instructions.push("sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm".to_string());
                instructions
                    .push("sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda".to_string());
            }
            Distribution::CentOS | Distribution::RHEL => {
                instructions.push("# RHEL/CentOS Installation".to_string());
                instructions.push("sudo dnf update".to_string());
                instructions.push("# Enable EPEL repository:".to_string());
                instructions.push("sudo dnf install epel-release".to_string());
                instructions.push("# Install NVIDIA repository:".to_string());
                instructions.push("sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo".to_string());
                instructions.push("sudo dnf install nvidia-driver cuda".to_string());
            }
            Distribution::OpenSUSE => {
                instructions.push("# openSUSE Installation".to_string());
                instructions.push("sudo zypper update".to_string());
                instructions.push("# Add NVIDIA repository:".to_string());
                instructions.push("sudo zypper addrepo --refresh https://download.nvidia.com/opensuse/leap/15.4 nvidia".to_string());
                instructions.push("sudo zypper install nvidia-glG05 nvidia-computeG05".to_string());
            }
            Distribution::Unknown(_) => {
                instructions.push("# Generic Installation Instructions".to_string());
                instructions.push(
                    "# Please refer to NVIDIA's documentation for your distribution:".to_string(),
                );
                instructions.push(
                    "# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/".to_string(),
                );
            }
        }

        instructions.push("".to_string());
        instructions.push("# After installation, reboot your system:".to_string());
        instructions.push("sudo reboot".to_string());

        instructions
    }

    /// Check if required packages are installed
    pub fn check_nvidia_packages(&self) -> Result<HashMap<String, bool>> {
        let mut status = HashMap::new();

        for package in &self.nvidia_packages {
            let installed = self.is_package_installed(package)?;
            status.insert(package.clone(), installed);
        }

        Ok(status)
    }

    /// Check if a package is installed
    fn is_package_installed(&self, package: &str) -> Result<bool> {
        match self.package_manager.name.as_str() {
            "pacman" => {
                let output = Command::new("pacman")
                    .args(["-Q", package])
                    .output()
                    .context("Failed to run pacman")?;
                Ok(output.status.success())
            }
            "apt" => {
                let output = Command::new("dpkg")
                    .args(["-l", package])
                    .output()
                    .context("Failed to run dpkg")?;
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    Ok(stdout.lines().any(|line| line.starts_with("ii")))
                } else {
                    Ok(false)
                }
            }
            "dnf" | "yum" => {
                let output = Command::new(&self.package_manager.name)
                    .args(["list", "installed", package])
                    .output()
                    .context(format!("Failed to run {}", self.package_manager.name))?;
                Ok(output.status.success())
            }
            "zypper" => {
                let output = Command::new("rpm")
                    .args(["-q", package])
                    .output()
                    .context("Failed to run rpm")?;
                Ok(output.status.success())
            }
            _ => {
                warn!("Unknown package manager: {}", self.package_manager.name);
                Ok(false)
            }
        }
    }

    /// Generate container modifications for this distribution
    pub fn get_container_modifications(&self) -> Vec<String> {
        let mut modifications = Vec::new();

        // Add distribution-specific library paths
        for lib_path in &self.library_paths {
            if Path::new(lib_path).exists() {
                modifications.push("--volume".to_string());
                modifications.push(format!("{}:{}:ro", lib_path, lib_path));
            }
        }

        // Distribution-specific environment variables
        match self.distribution {
            Distribution::Arch | Distribution::Manjaro => {
                modifications.push("--env".to_string());
                modifications.push("ARCH_DISTRO=1".to_string());
            }
            Distribution::Ubuntu => {
                modifications.push("--env".to_string());
                modifications.push("UBUNTU_DISTRO=1".to_string());
            }
            Distribution::Fedora => {
                modifications.push("--env".to_string());
                modifications.push("FEDORA_DISTRO=1".to_string());
            }
            _ => {}
        }

        modifications
    }
}

/// Cross-distribution compatibility manager
pub struct DistroManager {
    config: DistroConfig,
}

impl DistroManager {
    /// Create a new distribution manager
    pub fn new() -> Result<Self> {
        let distro = Distribution::detect()?;
        let version = Self::get_version(&distro)?;
        let config = DistroConfig::for_distribution(distro, version);

        info!(
            "Detected distribution: {} {}",
            config.distribution.name(),
            config.version
        );

        Ok(Self { config })
    }

    /// Get distribution version
    fn get_version(distro: &Distribution) -> Result<String> {
        match distro {
            Distribution::Arch | Distribution::Manjaro => {
                // Arch is rolling release
                Ok("rolling".to_string())
            }
            _ => {
                // Try to get version from /etc/os-release
                if let Ok(content) = fs::read_to_string("/etc/os-release") {
                    for line in content.lines() {
                        if let Some(version) = line.strip_prefix("VERSION_ID=") {
                            return Ok(version.trim_matches('"').to_string());
                        }
                    }
                }

                // Fallback to unknown
                Ok("unknown".to_string())
            }
        }
    }

    /// Get the distribution configuration
    pub fn config(&self) -> &DistroConfig {
        &self.config
    }

    /// Generate installation guide for current distribution
    pub fn generate_install_guide(&self) -> Vec<String> {
        self.config.get_install_instructions()
    }

    /// Check system compatibility
    pub fn check_compatibility(&self) -> Result<CompatibilityReport> {
        info!(
            "Checking system compatibility for {}",
            self.config.distribution.name()
        );

        let package_status = self.config.check_nvidia_packages()?;
        let kernel_modules = self.check_kernel_modules()?;
        let library_paths = self.check_library_paths();
        let container_runtime = self.check_container_runtime();

        Ok(CompatibilityReport {
            distribution: self.config.distribution.clone(),
            version: self.config.version.clone(),
            packages_installed: package_status,
            kernel_modules_loaded: kernel_modules,
            library_paths_available: library_paths,
            container_runtime_available: container_runtime,
        })
    }

    /// Check if kernel modules are loaded
    fn check_kernel_modules(&self) -> Result<HashMap<String, bool>> {
        let mut status = HashMap::new();

        if let Ok(modules) = fs::read_to_string("/proc/modules") {
            for module in &self.config.kernel_modules {
                let loaded = modules.lines().any(|line| line.starts_with(module));
                status.insert(module.clone(), loaded);
            }
        }

        Ok(status)
    }

    /// Check if library paths are available
    fn check_library_paths(&self) -> HashMap<String, bool> {
        let mut status = HashMap::new();

        for path in &self.config.library_paths {
            status.insert(path.clone(), Path::new(path).exists());
        }

        status
    }

    /// Check if container runtime is available
    fn check_container_runtime(&self) -> HashMap<String, bool> {
        let mut status = HashMap::new();
        let runtimes = ["docker", "podman", "containerd"];

        for runtime in &runtimes {
            let available = which::which(runtime).is_ok();
            status.insert(runtime.to_string(), available);
        }

        status
    }
}

impl Default for DistroManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            config: DistroConfig::generic_config(
                Distribution::Unknown("linux".to_string()),
                "unknown".to_string(),
            ),
        })
    }
}

/// System compatibility report
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    pub distribution: Distribution,
    pub version: String,
    pub packages_installed: HashMap<String, bool>,
    pub kernel_modules_loaded: HashMap<String, bool>,
    pub library_paths_available: HashMap<String, bool>,
    pub container_runtime_available: HashMap<String, bool>,
}

impl CompatibilityReport {
    /// Check if the system is ready for GPU containerization
    pub fn is_ready(&self) -> bool {
        // Check if at least one container runtime is available
        let has_runtime = self.container_runtime_available.values().any(|&v| v);

        // Check if essential NVIDIA modules are loaded
        let has_nvidia_module = self
            .kernel_modules_loaded
            .get("nvidia")
            .copied()
            .unwrap_or(false);

        // Check if at least one library path exists
        let has_library_path = self.library_paths_available.values().any(|&v| v);

        has_runtime && has_nvidia_module && has_library_path
    }

    /// Get recommendations for improving compatibility
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check container runtime
        if !self.container_runtime_available.values().any(|&v| v) {
            recommendations.push("Install a container runtime (Docker or Podman)".to_string());
        }

        // Check NVIDIA packages
        for (package, installed) in &self.packages_installed {
            if !installed {
                recommendations.push(format!("Install NVIDIA package: {}", package));
            }
        }

        // Check kernel modules
        for (module, loaded) in &self.kernel_modules_loaded {
            if !loaded {
                recommendations.push(format!("Load kernel module: {}", module));
            }
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_detection() {
        let distro = Distribution::detect().unwrap();
        println!("Detected distribution: {:?}", distro);
        // Just ensure it doesn't panic
    }

    #[test]
    fn test_distro_config_creation() {
        let config = DistroConfig::for_distribution(Distribution::Arch, "rolling".to_string());
        assert_eq!(config.distribution, Distribution::Arch);
        assert_eq!(config.package_manager.name, "pacman");
        assert!(!config.nvidia_packages.is_empty());
    }

    #[test]
    fn test_ubuntu_config() {
        let config = DistroConfig::for_distribution(Distribution::Ubuntu, "22.04".to_string());
        assert_eq!(config.distribution, Distribution::Ubuntu);
        assert_eq!(config.package_manager.name, "apt");
        assert!(
            config
                .nvidia_packages
                .iter()
                .any(|p| p.contains("nvidia-driver"))
        );
    }

    #[test]
    fn test_fedora_config() {
        let config = DistroConfig::for_distribution(Distribution::Fedora, "38".to_string());
        assert_eq!(config.distribution, Distribution::Fedora);
        assert_eq!(config.package_manager.name, "dnf");
        assert!(
            config
                .nvidia_packages
                .iter()
                .any(|p| p.contains("akmod-nvidia"))
        );
    }

    #[test]
    fn test_distribution_names() {
        assert_eq!(Distribution::Arch.name(), "Arch Linux");
        assert_eq!(Distribution::Ubuntu.name(), "Ubuntu");
        assert_eq!(Distribution::Fedora.name(), "Fedora");
    }

    #[test]
    fn test_install_instructions() {
        let config = DistroConfig::for_distribution(Distribution::Arch, "rolling".to_string());
        let instructions = config.get_install_instructions();
        assert!(!instructions.is_empty());
        assert!(instructions.iter().any(|i| i.contains("pacman")));
    }

    #[test]
    fn test_distro_manager_creation() {
        let manager = DistroManager::new();
        // Should not panic
        if let Ok(manager) = manager {
            println!(
                "Created manager for: {}",
                manager.config.distribution.name()
            );
        }
    }

    #[test]
    fn test_container_modifications() {
        let config = DistroConfig::for_distribution(Distribution::Arch, "rolling".to_string());
        let modifications = config.get_container_modifications();
        // Should return some modifications
        println!("Container modifications: {:?}", modifications);
    }
}
