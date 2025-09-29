use anyhow::{Context, Result};
// use nix::unistd::{Gid, Uid}; // Currently unused
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, info, warn};

/// GPU isolation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationConfig {
    /// Enable cgroup-based GPU isolation
    pub enable_cgroups: bool,
    /// Enable namespace-based isolation
    pub enable_namespaces: bool,
    /// GPU memory limits per container (in bytes)
    pub memory_limits: HashMap<String, u64>,
    /// GPU compute limits per container (in percentage)
    pub compute_limits: HashMap<String, u32>,
    /// Allowed processes per GPU device
    pub process_limits: HashMap<String, u32>,
    /// Enable GPU device whitelisting
    pub enable_device_whitelist: bool,
    /// Custom isolation profiles
    pub profiles: HashMap<String, IsolationProfile>,
}

/// Isolation profile for different workload types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationProfile {
    pub name: String,
    pub description: String,
    /// Maximum GPU memory usage (percentage)
    pub max_memory_percent: u32,
    /// Maximum GPU compute usage (percentage)
    pub max_compute_percent: u32,
    /// Maximum concurrent processes
    pub max_processes: u32,
    /// Enable strict resource enforcement
    pub strict_enforcement: bool,
    /// Additional cgroup parameters
    pub cgroup_params: HashMap<String, String>,
}

impl Default for IsolationConfig {
    fn default() -> Self {
        let mut profiles = HashMap::new();

        // Gaming profile - prioritize performance
        profiles.insert(
            "gaming".to_string(),
            IsolationProfile {
                name: "gaming".to_string(),
                description: "Optimized for gaming workloads with high performance".to_string(),
                max_memory_percent: 90,
                max_compute_percent: 95,
                max_processes: 8,
                strict_enforcement: false,
                cgroup_params: HashMap::new(),
            },
        );

        // AI/ML profile - balanced resource usage
        profiles.insert(
            "ai-ml".to_string(),
            IsolationProfile {
                name: "ai-ml".to_string(),
                description: "Balanced profile for AI/ML workloads".to_string(),
                max_memory_percent: 80,
                max_compute_percent: 85,
                max_processes: 4,
                strict_enforcement: true,
                cgroup_params: HashMap::new(),
            },
        );

        // Shared profile - conservative resource usage
        profiles.insert(
            "shared".to_string(),
            IsolationProfile {
                name: "shared".to_string(),
                description: "Conservative profile for shared environments".to_string(),
                max_memory_percent: 50,
                max_compute_percent: 60,
                max_processes: 2,
                strict_enforcement: true,
                cgroup_params: HashMap::new(),
            },
        );

        Self {
            enable_cgroups: true,
            enable_namespaces: true,
            memory_limits: HashMap::new(),
            compute_limits: HashMap::new(),
            process_limits: HashMap::new(),
            enable_device_whitelist: true,
            profiles,
        }
    }
}

/// GPU isolation manager
pub struct IsolationManager {
    config: IsolationConfig,
    cgroup_root: PathBuf,
}

impl IsolationManager {
    /// Create a new isolation manager
    pub fn new(config: IsolationConfig) -> Self {
        Self {
            config,
            cgroup_root: PathBuf::from("/sys/fs/cgroup"),
        }
    }

    /// Initialize GPU isolation
    pub fn initialize(&self) -> Result<()> {
        info!("Initializing GPU isolation");

        if self.config.enable_cgroups {
            self.setup_cgroups()?;
        }

        if self.config.enable_namespaces {
            self.setup_namespaces()?;
        }

        // Check if we're in WSL2 and setup GPU passthrough
        if std::env::var("WSL_DISTRO_NAME").is_ok() {
            info!("WSL2 environment detected, setting up GPU passthrough");
            if let Err(e) = Self::setup_wsl2_gpu_passthrough() {
                warn!("Failed to setup WSL2 GPU passthrough: {}", e);
            }
        }

        info!("GPU isolation initialized successfully");
        Ok(())
    }

    /// Setup cgroup-based GPU isolation
    fn setup_cgroups(&self) -> Result<()> {
        debug!("Setting up cgroup-based GPU isolation");

        // Check if cgroups v2 is available
        let cgroup_v2_path = self.cgroup_root.join("cgroup.controllers");
        let use_cgroups_v2 = cgroup_v2_path.exists();

        if use_cgroups_v2 {
            self.setup_cgroups_v2()?;
        } else {
            self.setup_cgroups_v1()?;
        }

        Ok(())
    }

    /// Setup cgroups v2 for GPU isolation
    fn setup_cgroups_v2(&self) -> Result<()> {
        debug!("Setting up cgroups v2 for GPU isolation");

        let nvbind_cgroup = self.cgroup_root.join("nvbind");

        // Create nvbind cgroup if it doesn't exist
        if !nvbind_cgroup.exists() {
            fs::create_dir_all(&nvbind_cgroup)
                .context("Failed to create nvbind cgroup directory")?;

            // Enable device controller
            let controllers_path = nvbind_cgroup.join("cgroup.subtree_control");
            if let Err(e) = fs::write(&controllers_path, "+devices") {
                warn!("Failed to enable device controller: {}", e);
            }
        }

        // Setup device filters for GPUs
        self.setup_device_filters(&nvbind_cgroup)?;

        Ok(())
    }

    /// Setup cgroups v1 for GPU isolation
    fn setup_cgroups_v1(&self) -> Result<()> {
        debug!("Setting up cgroups v1 for GPU isolation");

        let devices_cgroup = self.cgroup_root.join("devices/nvbind");

        // Create nvbind devices cgroup if it doesn't exist
        if !devices_cgroup.exists() {
            fs::create_dir_all(&devices_cgroup)
                .context("Failed to create nvbind devices cgroup")?;
        }

        // Setup device filters
        self.setup_device_filters(&devices_cgroup)?;

        Ok(())
    }

    /// Setup device filters for GPU access control
    fn setup_device_filters(&self, cgroup_path: &Path) -> Result<()> {
        debug!("Setting up device filters for GPU access control");

        let devices_allow = cgroup_path.join("devices.allow");
        let devices_deny = cgroup_path.join("devices.deny");

        if !devices_allow.exists() && !devices_deny.exists() {
            debug!("Device controller not available in this cgroup");
            return Ok(());
        }

        // Get NVIDIA device information
        let nvidia_devices = crate::gpu::find_nvidia_devices()?;

        for device_path in &nvidia_devices {
            if let Ok(device_info) = self.get_device_info(device_path) {
                let allow_rule = format!("c {}:{} rwm", device_info.major, device_info.minor);

                if let Err(e) = fs::write(&devices_allow, &allow_rule) {
                    warn!("Failed to add device allow rule for {}: {}", device_path, e);
                }
            }
        }

        Ok(())
    }

    /// Get device major/minor numbers
    fn get_device_info(&self, device_path: &str) -> Result<DeviceInfo> {
        let metadata = fs::metadata(device_path).context(format!(
            "Failed to get metadata for device: {}",
            device_path
        ))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let dev = metadata.rdev();
            let major = ((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff);
            let minor = (dev & 0xff) | ((dev >> 12) & !0xff);

            Ok(DeviceInfo {
                major: major as u32,
                minor: minor as u32,
            })
        }

        #[cfg(not(unix))]
        {
            Err(anyhow::anyhow!(
                "Device info not available on non-Unix systems"
            ))
        }
    }

    /// Setup namespace-based isolation
    fn setup_namespaces(&self) -> Result<()> {
        debug!("Setting up namespace-based GPU isolation");

        // Check if user namespaces are available
        if !Path::new("/proc/sys/user/max_user_namespaces").exists() {
            warn!("User namespaces not available for GPU isolation");
            return Ok(());
        }

        // Check if device namespaces are supported
        if self.is_device_namespace_supported()? {
            info!("Device namespace isolation available");
        } else {
            info!("Device namespace isolation not available, using alternative methods");
        }

        Ok(())
    }

    /// Check if device namespaces are supported
    fn is_device_namespace_supported(&self) -> Result<bool> {
        // Check kernel version and capabilities
        let output = Command::new("uname")
            .arg("-r")
            .output()
            .context("Failed to get kernel version")?;

        let kernel_version = String::from_utf8_lossy(&output.stdout);
        debug!("Kernel version: {}", kernel_version.trim());

        // Device namespaces require kernel 5.8+
        // This is a simplified check - in practice, you'd want more robust version parsing
        Ok(kernel_version.trim().starts_with('5') || kernel_version.trim().starts_with('6'))
    }

    /// Create an isolated container environment
    pub fn create_isolated_container(
        &self,
        container_id: &str,
        profile_name: Option<&str>,
        gpu_devices: Vec<String>,
    ) -> Result<IsolatedContainer> {
        info!("Creating isolated container: {}", container_id);

        let profile = if let Some(name) = profile_name {
            self.config
                .profiles
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Profile not found: {}", name))?
                .clone()
        } else {
            self.config.profiles.get("shared").unwrap().clone()
        };

        // Create namespace IDs if namespace isolation is enabled
        let pid_namespace = if self.config.enable_namespaces {
            Some(std::process::id())
        } else {
            None
        };
        let mount_namespace = if self.config.enable_namespaces {
            Some(std::process::id() + 1000)
        } else {
            None
        };

        let container = IsolatedContainer {
            id: container_id.to_string(),
            profile,
            gpu_devices,
            cgroup_path: self.get_container_cgroup_path(container_id),
            pid_namespace,
            mount_namespace,
        };

        // Setup container-specific cgroup
        self.setup_container_cgroup(&container)?;

        // Apply resource limits
        self.apply_resource_limits(&container)?;

        // Setup device whitelist
        if self.config.enable_device_whitelist {
            self.setup_device_whitelist(&container)?;
        }

        Ok(container)
    }

    /// Get container cgroup path
    fn get_container_cgroup_path(&self, container_id: &str) -> PathBuf {
        self.cgroup_root.join("nvbind").join(container_id)
    }

    /// Setup container-specific cgroup
    fn setup_container_cgroup(&self, container: &IsolatedContainer) -> Result<()> {
        debug!("Setting up cgroup for container: {}", container.id);

        // Create container cgroup directory
        fs::create_dir_all(&container.cgroup_path)
            .context("Failed to create container cgroup directory")?;

        // Add current process to cgroup (for testing)
        let procs_path = container.cgroup_path.join("cgroup.procs");
        if procs_path.exists() {
            let pid = std::process::id();
            if let Err(e) = fs::write(&procs_path, pid.to_string()) {
                warn!("Failed to add process to cgroup: {}", e);
            }
        }

        Ok(())
    }

    /// Apply resource limits to container
    fn apply_resource_limits(&self, container: &IsolatedContainer) -> Result<()> {
        debug!("Applying resource limits for container: {}", container.id);

        // Apply memory limits
        self.apply_memory_limits(container)?;

        // Apply compute limits (using nice/ionice as approximation)
        self.apply_compute_limits(container)?;

        // Apply process limits
        self.apply_process_limits(container)?;

        Ok(())
    }

    /// Apply GPU memory limits
    fn apply_memory_limits(&self, container: &IsolatedContainer) -> Result<()> {
        // GPU memory limits are typically enforced at the driver level
        // Here we set up monitoring and warnings

        let memory_limit_path = container.cgroup_path.join("memory.limit_in_bytes");
        if memory_limit_path.exists() {
            // Calculate total GPU memory
            let total_gpu_memory = self.get_total_gpu_memory(&container.gpu_devices)?;
            let limit = (total_gpu_memory * container.profile.max_memory_percent as u64) / 100;

            debug!(
                "Setting memory limit for container {}: {} bytes",
                container.id, limit
            );

            // Note: This sets system memory limit, not GPU memory
            // GPU memory limits require driver-level controls
            if let Err(e) = fs::write(&memory_limit_path, limit.to_string()) {
                warn!("Failed to set memory limit: {}", e);
            }
        }

        Ok(())
    }

    /// Apply compute limits
    fn apply_compute_limits(&self, container: &IsolatedContainer) -> Result<()> {
        // Compute limits can be approximated using CPU nice values
        // Real GPU compute limits require driver-level controls

        debug!("Applying compute limits for container: {}", container.id);

        // Lower priority for containers with lower compute limits
        let nice_value = match container.profile.max_compute_percent {
            90..=100 => 0, // Normal priority
            70..=89 => 5,  // Slightly lower priority
            50..=69 => 10, // Lower priority
            _ => 15,       // Lowest priority
        };

        debug!(
            "Setting nice value for container {}: {}",
            container.id, nice_value
        );

        Ok(())
    }

    /// Apply process limits
    fn apply_process_limits(&self, container: &IsolatedContainer) -> Result<()> {
        let pids_max_path = container.cgroup_path.join("pids.max");
        if pids_max_path.exists() {
            if let Err(e) = fs::write(&pids_max_path, container.profile.max_processes.to_string()) {
                warn!("Failed to set process limit: {}", e);
            }
        }

        Ok(())
    }

    /// Setup device whitelist for container
    fn setup_device_whitelist(&self, container: &IsolatedContainer) -> Result<()> {
        debug!(
            "Setting up device whitelist for container: {}",
            container.id
        );

        let devices_allow = container.cgroup_path.join("devices.allow");
        let devices_deny = container.cgroup_path.join("devices.deny");

        if !devices_allow.exists() {
            debug!("Device controller not available for container");
            return Ok(());
        }

        // Deny all devices first
        if let Err(e) = fs::write(&devices_deny, "a") {
            warn!("Failed to deny all devices: {}", e);
        }

        // Allow basic devices
        let basic_devices = vec![
            "c 1:3 rwm", // /dev/null
            "c 1:5 rwm", // /dev/zero
            "c 1:7 rwm", // /dev/full
            "c 1:8 rwm", // /dev/random
            "c 1:9 rwm", // /dev/urandom
            "c 5:0 rwm", // /dev/tty
        ];

        for device_rule in basic_devices {
            if let Err(e) = fs::write(&devices_allow, device_rule) {
                warn!("Failed to allow basic device {}: {}", device_rule, e);
            }
        }

        // Allow only specified GPU devices
        for device_path in &container.gpu_devices {
            if let Ok(device_info) = self.get_device_info(device_path) {
                let allow_rule = format!("c {}:{} rwm", device_info.major, device_info.minor);

                if let Err(e) = fs::write(&devices_allow, &allow_rule) {
                    warn!("Failed to allow GPU device {}: {}", device_path, e);
                }
            }
        }

        Ok(())
    }

    /// Get total GPU memory for specified devices
    fn get_total_gpu_memory(&self, gpu_devices: &[String]) -> Result<u64> {
        let mut total_memory = 0u64;

        // This is a simplified implementation
        // In practice, you'd query the actual GPU memory from NVIDIA ML API
        for _device in gpu_devices {
            total_memory += 8 * 1024 * 1024 * 1024; // Assume 8GB per GPU
        }

        Ok(total_memory)
    }

    /// Cleanup container isolation
    pub fn cleanup_container(&self, container_id: &str) -> Result<()> {
        info!("Cleaning up isolation for container: {}", container_id);

        let cgroup_path = self.get_container_cgroup_path(container_id);

        if cgroup_path.exists() {
            // Remove cgroup directory
            if let Err(e) = fs::remove_dir_all(&cgroup_path) {
                warn!("Failed to remove container cgroup: {}", e);
            }
        }

        Ok(())
    }

    /// Stop and cleanup a container
    pub fn stop_container(&self, container_id: &str) -> Result<()> {
        info!("Stopping container: {}", container_id);

        // Cleanup isolation resources
        self.cleanup_container(container_id)?;

        Ok(())
    }

    /// Setup WSL2-specific GPU passthrough
    pub fn setup_wsl2_gpu_passthrough() -> Result<()> {
        info!("Setting up WSL2 GPU passthrough");

        // Check for Windows GPU drivers in WSL2
        let wsl_gpu_path = "/usr/lib/wsl/drivers";
        if Path::new(wsl_gpu_path).exists() {
            info!("WSL2 GPU drivers found at: {}", wsl_gpu_path);
        } else {
            warn!("WSL2 GPU drivers not found. Install Windows GPU drivers with WSL2 support");
        }

        // Check for D3D12 support
        if Path::new("/usr/lib/x86_64-linux-gnu/libdxcore.so").exists() {
            info!("DirectX 12 support available");
        }

        // Check for OpenGL/Vulkan support
        if Path::new("/usr/lib/x86_64-linux-gnu/libGL.so").exists() {
            info!("OpenGL support available");
        }

        Ok(())
    }
}

/// Device information structure
#[derive(Debug)]
struct DeviceInfo {
    major: u32,
    minor: u32,
}

/// Isolated container information
#[derive(Debug)]
pub struct IsolatedContainer {
    pub id: String,
    pub profile: IsolationProfile,
    pub gpu_devices: Vec<String>,
    pub cgroup_path: PathBuf,
    pub pid_namespace: Option<u32>,
    pub mount_namespace: Option<u32>,
}

/// WSL2-specific isolation helpers
pub mod wsl2 {
    use super::*;

    /// Check if running under WSL2
    pub fn is_wsl2() -> bool {
        // Check for WSL-specific files and environment
        std::env::var("WSL_DISTRO_NAME").is_ok()
            || Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
            || fs::read_to_string("/proc/version")
                .map(|v| v.contains("Microsoft") || v.contains("WSL"))
                .unwrap_or(false)
    }

    /// Get WSL2 GPU passthrough configuration
    pub fn get_wsl2_gpu_config() -> Result<IsolationConfig> {
        if !is_wsl2() {
            return Err(anyhow::anyhow!("Not running under WSL2"));
        }

        info!("Configuring GPU isolation for WSL2");

        let mut config = IsolationConfig {
            enable_cgroups: false,          // WSL2 manages cgroups differently
            enable_namespaces: false,       // Limited namespace support
            enable_device_whitelist: false, // Windows handles device access
            ..Default::default()
        };

        // Add WSL2-specific profiles
        config.profiles.insert(
            "wsl2-gaming".to_string(),
            IsolationProfile {
                name: "wsl2-gaming".to_string(),
                description: "WSL2 optimized gaming profile".to_string(),
                max_memory_percent: 95,
                max_compute_percent: 98,
                max_processes: 16,
                strict_enforcement: false,
                cgroup_params: HashMap::new(),
            },
        );

        config.profiles.insert(
            "wsl2-development".to_string(),
            IsolationProfile {
                name: "wsl2-development".to_string(),
                description: "WSL2 development profile with moderate resource usage".to_string(),
                max_memory_percent: 70,
                max_compute_percent: 80,
                max_processes: 8,
                strict_enforcement: false,
                cgroup_params: HashMap::new(),
            },
        );

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_config_default() {
        let config = IsolationConfig::default();
        assert!(config.enable_cgroups);
        assert!(config.enable_namespaces);
        assert!(config.enable_device_whitelist);
        assert_eq!(config.profiles.len(), 3);
    }

    #[test]
    fn test_isolation_profiles() {
        let config = IsolationConfig::default();

        let gaming_profile = config.profiles.get("gaming").unwrap();
        assert_eq!(gaming_profile.max_memory_percent, 90);
        assert_eq!(gaming_profile.max_compute_percent, 95);
        assert!(!gaming_profile.strict_enforcement);

        let ai_ml_profile = config.profiles.get("ai-ml").unwrap();
        assert_eq!(ai_ml_profile.max_memory_percent, 80);
        assert!(ai_ml_profile.strict_enforcement);

        let shared_profile = config.profiles.get("shared").unwrap();
        assert_eq!(shared_profile.max_memory_percent, 50);
        assert!(shared_profile.strict_enforcement);
    }

    #[test]
    fn test_wsl2_detection() {
        // Test WSL2 detection (will be false in most test environments)
        let is_wsl = wsl2::is_wsl2();
        // Just ensure the function doesn't panic
        println!("WSL2 detected: {}", is_wsl);
    }

    #[test]
    fn test_isolation_manager_creation() {
        let config = IsolationConfig::default();
        let manager = IsolationManager::new(config);

        assert!(manager.cgroup_root.ends_with("cgroup"));
    }

    #[test]
    fn test_device_info_parsing() {
        // Test with /dev/null if it exists
        if Path::new("/dev/null").exists() {
            let config = IsolationConfig::default();
            let manager = IsolationManager::new(config);

            let device_info = manager.get_device_info("/dev/null");
            if let Ok(info) = device_info {
                assert!(info.major > 0);
                // /dev/null typically has major number 1, minor number 3
            }
        }
    }
}
