use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, RwLock};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

/// WSL2 Gaming Support Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wsl2GamingConfig {
    /// Enable WSL2 GPU passthrough validation
    pub enable_gpu_passthrough: bool,
    /// Windows gaming compatibility mode
    pub windows_gaming_mode: bool,
    /// Cross-platform development workflows
    pub cross_platform_workflows: bool,
    /// DirectX/Vulkan API passthrough
    pub directx_vulkan_passthrough: bool,
    /// Gaming-specific performance optimizations
    pub gaming_optimizations: bool,
    /// Windows driver compatibility testing
    pub driver_compatibility_testing: bool,
    /// WSL2 distribution preference
    pub wsl_distribution: String,
}

impl Default for Wsl2GamingConfig {
    fn default() -> Self {
        Self {
            enable_gpu_passthrough: true,
            windows_gaming_mode: true,
            cross_platform_workflows: true,
            directx_vulkan_passthrough: true,
            gaming_optimizations: true,
            driver_compatibility_testing: true,
            wsl_distribution: "Ubuntu-22.04".to_string(),
        }
    }
}

/// DirectX/Vulkan API Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiPassthroughConfig {
    /// DirectX 11 support
    pub directx11: bool,
    /// DirectX 12 support
    pub directx12: bool,
    /// Vulkan API support
    pub vulkan: bool,
    /// OpenGL support
    pub opengl: bool,
    /// GPU acceleration for rendering
    pub gpu_acceleration: bool,
    /// Hardware acceleration for video
    pub hardware_video_acceleration: bool,
}

impl Default for ApiPassthroughConfig {
    fn default() -> Self {
        Self {
            directx11: true,
            directx12: true,
            vulkan: true,
            opengl: true,
            gpu_acceleration: true,
            hardware_video_acceleration: true,
        }
    }
}

/// Gaming Performance Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingPerformanceProfile {
    /// Profile name
    pub name: String,
    /// Low latency mode
    pub low_latency_mode: bool,
    /// GPU scheduling priority
    pub gpu_scheduling_priority: GpuSchedulingPriority,
    /// Memory allocation strategy
    pub memory_allocation: MemoryAllocationStrategy,
    /// Thread affinity settings
    pub thread_affinity: ThreadAffinityConfig,
    /// Power management settings
    pub power_management: PowerManagementConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuSchedulingPriority {
    RealTime,
    High,
    Normal,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAllocationStrategy {
    Gaming,
    Performance,
    Balanced,
    PowerSaving,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadAffinityConfig {
    /// CPU cores for gaming threads
    pub gaming_cores: Vec<u32>,
    /// CPU cores for system threads
    pub system_cores: Vec<u32>,
    /// Enable hyperthreading
    pub hyperthreading: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerManagementConfig {
    /// Power profile
    pub profile: PowerProfile,
    /// GPU power limit
    pub gpu_power_limit: Option<u32>,
    /// CPU frequency scaling
    pub cpu_frequency_scaling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerProfile {
    Performance,
    Balanced,
    PowerSaver,
}

/// Windows Driver Compatibility Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverCompatibility {
    /// NVIDIA driver version
    pub nvidia_driver_version: Option<String>,
    /// WSL2 kernel version
    pub wsl_kernel_version: Option<String>,
    /// Windows version
    pub windows_version: Option<String>,
    /// DirectX runtime version
    pub directx_version: Option<String>,
    /// Vulkan runtime version
    pub vulkan_version: Option<String>,
    /// Compatibility status
    pub compatibility_status: CompatibilityStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityStatus {
    FullyCompatible,
    PartiallyCompatible,
    Incompatible,
    Unknown,
}

/// WSL2 Gaming Support Manager
pub struct Wsl2GamingManager {
    config: Wsl2GamingConfig,
    api_config: ApiPassthroughConfig,
    gaming_profiles: Arc<RwLock<HashMap<String, GamingPerformanceProfile>>>,
    driver_compatibility: Arc<RwLock<Option<DriverCompatibility>>>,
}

impl Wsl2GamingManager {
    /// Create a new WSL2 Gaming Manager
    pub fn new(config: Wsl2GamingConfig, api_config: ApiPassthroughConfig) -> Self {
        let mut gaming_profiles = HashMap::new();

        // Pre-configured gaming profiles
        gaming_profiles.insert("ultra_performance".to_string(), GamingPerformanceProfile {
            name: "Ultra Performance".to_string(),
            low_latency_mode: true,
            gpu_scheduling_priority: GpuSchedulingPriority::RealTime,
            memory_allocation: MemoryAllocationStrategy::Gaming,
            thread_affinity: ThreadAffinityConfig {
                gaming_cores: vec![0, 2, 4, 6],
                system_cores: vec![1, 3, 5, 7],
                hyperthreading: false,
            },
            power_management: PowerManagementConfig {
                profile: PowerProfile::Performance,
                gpu_power_limit: None,
                cpu_frequency_scaling: false,
            },
        });

        gaming_profiles.insert("balanced".to_string(), GamingPerformanceProfile {
            name: "Balanced".to_string(),
            low_latency_mode: false,
            gpu_scheduling_priority: GpuSchedulingPriority::High,
            memory_allocation: MemoryAllocationStrategy::Balanced,
            thread_affinity: ThreadAffinityConfig {
                gaming_cores: vec![0, 1, 2, 3],
                system_cores: vec![4, 5, 6, 7],
                hyperthreading: true,
            },
            power_management: PowerManagementConfig {
                profile: PowerProfile::Balanced,
                gpu_power_limit: Some(80),
                cpu_frequency_scaling: true,
            },
        });

        gaming_profiles.insert("power_efficient".to_string(), GamingPerformanceProfile {
            name: "Power Efficient".to_string(),
            low_latency_mode: false,
            gpu_scheduling_priority: GpuSchedulingPriority::Normal,
            memory_allocation: MemoryAllocationStrategy::PowerSaving,
            thread_affinity: ThreadAffinityConfig {
                gaming_cores: vec![0, 1],
                system_cores: vec![2, 3, 4, 5, 6, 7],
                hyperthreading: true,
            },
            power_management: PowerManagementConfig {
                profile: PowerProfile::PowerSaver,
                gpu_power_limit: Some(60),
                cpu_frequency_scaling: true,
            },
        });

        Self {
            config,
            api_config,
            gaming_profiles: Arc::new(RwLock::new(gaming_profiles)),
            driver_compatibility: Arc::new(RwLock::new(None)),
        }
    }

    /// Validate WSL2 GPU passthrough capability
    pub async fn validate_gpu_passthrough(&self) -> Result<bool> {
        info!("Validating WSL2 GPU passthrough capability");

        if !self.config.enable_gpu_passthrough {
            return Ok(false);
        }

        // Check if running in WSL2
        if !self.is_wsl2_environment()? {
            debug!("Not running in WSL2 environment");
            return Ok(false);
        }

        // Check for GPU device access
        let gpu_accessible = self.check_gpu_device_access().await?;
        if !gpu_accessible {
            warn!("GPU devices not accessible in WSL2");
            return Ok(false);
        }

        // Validate DirectX/Vulkan support
        let api_support = self.validate_api_support().await?;
        if !api_support {
            warn!("DirectX/Vulkan API support not available");
            return Ok(false);
        }

        info!("WSL2 GPU passthrough validation successful");
        Ok(true)
    }

    /// Check if running in WSL2 environment
    fn is_wsl2_environment(&self) -> Result<bool> {
        // Check for WSL environment variables
        if std::env::var("WSL_DISTRO_NAME").is_ok() {
            return Ok(true);
        }

        // Check /proc/version for WSL2 kernel
        let proc_version = std::fs::read_to_string("/proc/version")
            .unwrap_or_default();

        Ok(proc_version.contains("microsoft") || proc_version.contains("WSL"))
    }

    /// Check GPU device accessibility
    async fn check_gpu_device_access(&self) -> Result<bool> {
        // Check for GPU devices in /dev
        let gpu_devices = [
            "/dev/dxg",
            "/dev/nvidia0",
            "/dev/nvidiactl",
            "/dev/nvidia-modeset",
        ];

        for device in &gpu_devices {
            if std::path::Path::new(device).exists() {
                debug!("Found GPU device: {}", device);
                return Ok(true);
            }
        }

        // Check for GPU via lspci
        let output = Command::new("lspci")
            .arg("-nn")
            .output();

        if let Ok(output) = output {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if output_str.contains("VGA") || output_str.contains("3D") {
                debug!("GPU detected via lspci");
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Validate DirectX/Vulkan API support
    async fn validate_api_support(&self) -> Result<bool> {
        let mut api_available = false;

        // Check DirectX support
        if self.api_config.directx11 || self.api_config.directx12 {
            api_available |= self.check_directx_support().await?;
        }

        // Check Vulkan support
        if self.api_config.vulkan {
            api_available |= self.check_vulkan_support().await?;
        }

        // Check OpenGL support
        if self.api_config.opengl {
            api_available |= self.check_opengl_support().await?;
        }

        Ok(api_available)
    }

    /// Check DirectX support
    async fn check_directx_support(&self) -> Result<bool> {
        // Check for DirectX runtime
        let dxdiag_available = Command::new("which")
            .arg("dxdiag")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

        if dxdiag_available {
            debug!("DirectX support detected");
            return Ok(true);
        }

        // Check for Wine DirectX implementation
        let wine_dx_available = Command::new("which")
            .arg("winedxdiag")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

        Ok(wine_dx_available)
    }

    /// Check Vulkan support
    async fn check_vulkan_support(&self) -> Result<bool> {
        let vulkan_info = Command::new("vulkaninfo")
            .output();

        if let Ok(output) = vulkan_info {
            if output.status.success() {
                debug!("Vulkan support detected");
                return Ok(true);
            }
        }

        // Check for Vulkan loader
        let vulkan_loader = std::path::Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so.1").exists() ||
                           std::path::Path::new("/usr/lib/libvulkan.so.1").exists();

        Ok(vulkan_loader)
    }

    /// Check OpenGL support
    async fn check_opengl_support(&self) -> Result<bool> {
        let glx_info = Command::new("glxinfo")
            .arg("-B")
            .output();

        if let Ok(output) = glx_info {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if output_str.contains("direct rendering: Yes") {
                    debug!("OpenGL support with direct rendering detected");
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get driver compatibility information
    pub async fn get_driver_compatibility(&self) -> Result<DriverCompatibility> {
        info!("Checking driver compatibility");

        let nvidia_version = self.get_nvidia_driver_version().await?;
        let wsl_kernel = self.get_wsl_kernel_version().await?;
        let windows_version = self.get_windows_version().await?;
        let directx_version = self.get_directx_version().await?;
        let vulkan_version = self.get_vulkan_version().await?;

        let compatibility_status = self.assess_compatibility_status(
            &nvidia_version,
            &wsl_kernel,
            &windows_version,
        ).await?;

        let compatibility = DriverCompatibility {
            nvidia_driver_version: nvidia_version,
            wsl_kernel_version: wsl_kernel,
            windows_version,
            directx_version,
            vulkan_version,
            compatibility_status,
        };

        // Cache the compatibility information
        {
            let mut driver_compat = self.driver_compatibility.write().unwrap();
            *driver_compat = Some(compatibility.clone());
        }

        Ok(compatibility)
    }

    /// Get NVIDIA driver version
    async fn get_nvidia_driver_version(&self) -> Result<Option<String>> {
        let output = Command::new("nvidia-smi")
            .arg("--query-gpu=driver_version")
            .arg("--format=csv,noheader,nounits")
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .to_string();
                return Ok(Some(version));
            }
        }

        Ok(None)
    }

    /// Get WSL kernel version
    async fn get_wsl_kernel_version(&self) -> Result<Option<String>> {
        let kernel_version = std::fs::read_to_string("/proc/version")
            .map(|content| {
                content.split_whitespace()
                    .nth(2)
                    .unwrap_or("unknown")
                    .to_string()
            })
            .ok();

        Ok(kernel_version)
    }

    /// Get Windows version (from WSL)
    async fn get_windows_version(&self) -> Result<Option<String>> {
        let output = Command::new("powershell.exe")
            .arg("-c")
            .arg("Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion")
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let version_info = String::from_utf8_lossy(&output.stdout);
                // Parse the output to extract version information
                return Ok(Some(version_info.trim().to_string()));
            }
        }

        Ok(None)
    }

    /// Get DirectX version
    async fn get_directx_version(&self) -> Result<Option<String>> {
        let output = Command::new("dxdiag")
            .arg("/t")
            .arg("/tmp/dxdiag.txt")
            .output();

        if output.is_ok() {
            if let Ok(content) = std::fs::read_to_string("/tmp/dxdiag.txt") {
                // Parse DirectX version from dxdiag output
                for line in content.lines() {
                    if line.contains("DirectX Version:") {
                        return Ok(Some(line.split(':').nth(1).unwrap_or("").trim().to_string()));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Get Vulkan version
    async fn get_vulkan_version(&self) -> Result<Option<String>> {
        let output = Command::new("vulkaninfo")
            .arg("--summary")
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                for line in info.lines() {
                    if line.contains("Vulkan Instance Version:") {
                        return Ok(Some(line.split(':').nth(1).unwrap_or("").trim().to_string()));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Assess overall compatibility status
    async fn assess_compatibility_status(
        &self,
        nvidia_version: &Option<String>,
        wsl_kernel: &Option<String>,
        windows_version: &Option<String>,
    ) -> Result<CompatibilityStatus> {
        // Check for minimum required versions
        let mut compatibility_score = 0;
        let mut total_checks = 0;

        // NVIDIA driver version check
        if let Some(version) = nvidia_version {
            total_checks += 1;
            if self.is_nvidia_version_compatible(version) {
                compatibility_score += 1;
            }
        }

        // WSL kernel version check
        if let Some(kernel) = wsl_kernel {
            total_checks += 1;
            if self.is_wsl_kernel_compatible(kernel) {
                compatibility_score += 1;
            }
        }

        // Windows version check
        if let Some(windows) = windows_version {
            total_checks += 1;
            if self.is_windows_version_compatible(windows) {
                compatibility_score += 1;
            }
        }

        if total_checks == 0 {
            return Ok(CompatibilityStatus::Unknown);
        }

        let compatibility_ratio = compatibility_score as f64 / total_checks as f64;

        Ok(match compatibility_ratio {
            ratio if ratio >= 0.9 => CompatibilityStatus::FullyCompatible,
            ratio if ratio >= 0.5 => CompatibilityStatus::PartiallyCompatible,
            _ => CompatibilityStatus::Incompatible,
        })
    }

    /// Check if NVIDIA driver version is compatible
    fn is_nvidia_version_compatible(&self, version: &str) -> bool {
        // Minimum NVIDIA driver version for WSL2 GPU support: 470.xx
        if let Ok(version_num) = version.split('.').next().unwrap_or("0").parse::<u32>() {
            return version_num >= 470;
        }
        false
    }

    /// Check if WSL kernel version is compatible
    fn is_wsl_kernel_compatible(&self, kernel: &str) -> bool {
        // WSL2 kernels with GPU support typically start with 5.4+
        if kernel.contains("microsoft") || kernel.contains("WSL") {
            if let Some(version_part) = kernel.split('-').next() {
                if let Ok(major) = version_part.split('.').next().unwrap_or("0").parse::<u32>() {
                    if let Ok(minor) = version_part.split('.').nth(1).unwrap_or("0").parse::<u32>() {
                        return major > 5 || (major == 5 && minor >= 4);
                    }
                }
            }
        }
        false
    }

    /// Check if Windows version is compatible
    fn is_windows_version_compatible(&self, version: &str) -> bool {
        // Windows 10 version 21H2 (build 19044) or later, or Windows 11
        version.contains("Windows 11") ||
        version.contains("21H2") ||
        version.contains("22H2") ||
        version.contains("19044") ||
        version.contains("19045")
    }

    /// Apply gaming performance profile
    pub async fn apply_gaming_profile(&self, profile_name: &str) -> Result<()> {
        info!("Applying gaming performance profile: {}", profile_name);

        let profile = {
            let profiles = self.gaming_profiles.read().unwrap();
            profiles.get(profile_name)
                .ok_or_else(|| anyhow!("Gaming profile '{}' not found", profile_name))?
                .clone()
        };

        // Apply GPU scheduling priority
        self.apply_gpu_scheduling_priority(&profile.gpu_scheduling_priority).await?;

        // Apply memory allocation strategy
        self.apply_memory_allocation_strategy(&profile.memory_allocation).await?;

        // Apply thread affinity settings
        self.apply_thread_affinity(&profile.thread_affinity).await?;

        // Apply power management settings
        self.apply_power_management(&profile.power_management).await?;

        info!("Successfully applied gaming profile: {}", profile_name);
        Ok(())
    }

    /// Apply GPU scheduling priority
    async fn apply_gpu_scheduling_priority(&self, priority: &GpuSchedulingPriority) -> Result<()> {
        let priority_value = match priority {
            GpuSchedulingPriority::RealTime => "realtime",
            GpuSchedulingPriority::High => "high",
            GpuSchedulingPriority::Normal => "normal",
            GpuSchedulingPriority::Low => "low",
        };

        debug!("Setting GPU scheduling priority to: {}", priority_value);

        // Apply GPU scheduling settings (implementation would depend on specific GPU driver)
        // This is a placeholder for actual GPU scheduling implementation

        Ok(())
    }

    /// Apply memory allocation strategy
    async fn apply_memory_allocation_strategy(&self, strategy: &MemoryAllocationStrategy) -> Result<()> {
        debug!("Applying memory allocation strategy: {:?}", strategy);

        // Configure memory allocation based on strategy
        match strategy {
            MemoryAllocationStrategy::Gaming => {
                // Optimize for gaming workloads
                self.configure_gaming_memory().await?;
            },
            MemoryAllocationStrategy::Performance => {
                // Optimize for performance
                self.configure_performance_memory().await?;
            },
            MemoryAllocationStrategy::Balanced => {
                // Balanced configuration
                self.configure_balanced_memory().await?;
            },
            MemoryAllocationStrategy::PowerSaving => {
                // Power-saving configuration
                self.configure_power_saving_memory().await?;
            },
        }

        Ok(())
    }

    /// Configure gaming-optimized memory settings
    async fn configure_gaming_memory(&self) -> Result<()> {
        debug!("Configuring gaming-optimized memory settings");
        // Implementation for gaming-specific memory optimization
        Ok(())
    }

    /// Configure performance-optimized memory settings
    async fn configure_performance_memory(&self) -> Result<()> {
        debug!("Configuring performance-optimized memory settings");
        // Implementation for performance-specific memory optimization
        Ok(())
    }

    /// Configure balanced memory settings
    async fn configure_balanced_memory(&self) -> Result<()> {
        debug!("Configuring balanced memory settings");
        // Implementation for balanced memory configuration
        Ok(())
    }

    /// Configure power-saving memory settings
    async fn configure_power_saving_memory(&self) -> Result<()> {
        debug!("Configuring power-saving memory settings");
        // Implementation for power-saving memory configuration
        Ok(())
    }

    /// Apply thread affinity configuration
    async fn apply_thread_affinity(&self, config: &ThreadAffinityConfig) -> Result<()> {
        debug!("Applying thread affinity configuration");

        // Set CPU affinity for gaming threads
        for core in &config.gaming_cores {
            debug!("Setting gaming thread affinity to core: {}", core);
            // Implementation would use CPU affinity APIs
        }

        // Set CPU affinity for system threads
        for core in &config.system_cores {
            debug!("Setting system thread affinity to core: {}", core);
            // Implementation would use CPU affinity APIs
        }

        if config.hyperthreading {
            debug!("Enabling hyperthreading optimization");
        } else {
            debug!("Disabling hyperthreading for gaming optimization");
        }

        Ok(())
    }

    /// Apply power management configuration
    async fn apply_power_management(&self, config: &PowerManagementConfig) -> Result<()> {
        debug!("Applying power management configuration: {:?}", config.profile);

        // Set power profile
        self.set_power_profile(&config.profile).await?;

        // Set GPU power limit if specified
        if let Some(power_limit) = config.gpu_power_limit {
            self.set_gpu_power_limit(power_limit).await?;
        }

        // Configure CPU frequency scaling
        if config.cpu_frequency_scaling {
            self.enable_cpu_frequency_scaling().await?;
        } else {
            self.disable_cpu_frequency_scaling().await?;
        }

        Ok(())
    }

    /// Set system power profile
    async fn set_power_profile(&self, profile: &PowerProfile) -> Result<()> {
        let profile_name = match profile {
            PowerProfile::Performance => "performance",
            PowerProfile::Balanced => "balanced",
            PowerProfile::PowerSaver => "powersave",
        };

        debug!("Setting power profile to: {}", profile_name);

        // Set CPU governor
        let _ = Command::new("cpupower")
            .arg("frequency-set")
            .arg("-g")
            .arg(profile_name)
            .output();

        Ok(())
    }

    /// Set GPU power limit
    async fn set_gpu_power_limit(&self, limit: u32) -> Result<()> {
        debug!("Setting GPU power limit to: {}%", limit);

        // Use nvidia-ml-py or similar to set power limit
        let _ = Command::new("nvidia-smi")
            .arg("-pl")
            .arg(&format!("{}%", limit))
            .output();

        Ok(())
    }

    /// Enable CPU frequency scaling
    async fn enable_cpu_frequency_scaling(&self) -> Result<()> {
        debug!("Enabling CPU frequency scaling");

        let _ = Command::new("cpupower")
            .arg("frequency-set")
            .arg("-g")
            .arg("ondemand")
            .output();

        Ok(())
    }

    /// Disable CPU frequency scaling
    async fn disable_cpu_frequency_scaling(&self) -> Result<()> {
        debug!("Disabling CPU frequency scaling");

        let _ = Command::new("cpupower")
            .arg("frequency-set")
            .arg("-g")
            .arg("performance")
            .output();

        Ok(())
    }

    /// Get available gaming profiles
    pub fn get_gaming_profiles(&self) -> Vec<String> {
        let profiles = self.gaming_profiles.read().unwrap();
        profiles.keys().cloned().collect()
    }

    /// Get gaming profile details
    pub fn get_gaming_profile(&self, name: &str) -> Option<GamingPerformanceProfile> {
        let profiles = self.gaming_profiles.read().unwrap();
        profiles.get(name).cloned()
    }

    /// Add custom gaming profile
    pub fn add_gaming_profile(&self, profile: GamingPerformanceProfile) -> Result<()> {
        let mut profiles = self.gaming_profiles.write().unwrap();
        profiles.insert(profile.name.clone(), profile);
        Ok(())
    }

    /// Remove gaming profile
    pub fn remove_gaming_profile(&self, name: &str) -> Result<()> {
        let mut profiles = self.gaming_profiles.write().unwrap();
        profiles.remove(name)
            .ok_or_else(|| anyhow!("Gaming profile '{}' not found", name))?;
        Ok(())
    }

    /// Generate WSL2 gaming compatibility report
    pub async fn generate_compatibility_report(&self) -> Result<String> {
        info!("Generating WSL2 gaming compatibility report");

        let gpu_passthrough = self.validate_gpu_passthrough().await?;
        let driver_compat = self.get_driver_compatibility().await?;
        let api_support = self.validate_api_support().await?;

        let mut report = String::new();
        report.push_str("# WSL2 Gaming Compatibility Report\n\n");

        report.push_str(&format!("## GPU Passthrough Support: {}\n",
            if gpu_passthrough { "✅ Available" } else { "❌ Not Available" }));

        report.push_str(&format!("## Driver Compatibility: {:?}\n", driver_compat.compatibility_status));

        if let Some(nvidia_version) = &driver_compat.nvidia_driver_version {
            report.push_str(&format!("- NVIDIA Driver: {}\n", nvidia_version));
        }

        if let Some(wsl_kernel) = &driver_compat.wsl_kernel_version {
            report.push_str(&format!("- WSL Kernel: {}\n", wsl_kernel));
        }

        if let Some(windows_version) = &driver_compat.windows_version {
            report.push_str(&format!("- Windows Version: {}\n", windows_version));
        }

        report.push_str(&format!("\n## API Support: {}\n",
            if api_support { "✅ Available" } else { "❌ Limited" }));

        if let Some(directx_version) = &driver_compat.directx_version {
            report.push_str(&format!("- DirectX: {}\n", directx_version));
        }

        if let Some(vulkan_version) = &driver_compat.vulkan_version {
            report.push_str(&format!("- Vulkan: {}\n", vulkan_version));
        }

        report.push_str("\n## Gaming Performance Profiles Available:\n");
        for profile_name in self.get_gaming_profiles() {
            report.push_str(&format!("- {}\n", profile_name));
        }

        Ok(report)
    }
}