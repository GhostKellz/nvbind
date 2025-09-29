use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, RwLock};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

/// Multi-Vendor GPU Support Module (NVIDIA-focused)
/// Supports NVIDIA Open Kernel Module Driver (primary), Nouveau, and NVIDIA Proprietary Driver

/// GPU Driver Manager
pub struct GpuDriverManager {
    /// Detected GPU drivers
    detected_drivers: Arc<RwLock<HashMap<String, NvidiaDriverInfo>>>,
    /// Driver priority order
    driver_priority: Vec<NvidiaDriverType>,
    /// Driver capabilities cache
    capabilities_cache: Arc<RwLock<HashMap<String, DriverCapabilities>>>,
    /// Driver configuration
    driver_config: Arc<RwLock<DriverConfiguration>>,
}

/// NVIDIA Driver Type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum NvidiaDriverType {
    /// NVIDIA Open Kernel Module Driver (primary)
    OpenKernelModule,
    /// Nouveau open-source driver
    Nouveau,
    /// NVIDIA Proprietary Driver (fallback)
    Proprietary,
}

/// NVIDIA Driver Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NvidiaDriverInfo {
    /// Driver type
    pub driver_type: NvidiaDriverType,
    /// Driver version
    pub version: String,
    /// Kernel module name
    pub kernel_module: String,
    /// Driver path
    pub driver_path: PathBuf,
    /// Installation status
    pub installation_status: InstallationStatus,
    /// Supported GPU architectures
    pub supported_architectures: Vec<GpuArchitecture>,
    /// Driver capabilities
    pub capabilities: DriverCapabilities,
    /// Load priority
    pub priority: u32,
    /// Detection timestamp
    pub detected_at: chrono::DateTime<chrono::Utc>,
}

/// Installation Status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstallationStatus {
    /// Driver is installed and loaded
    Active,
    /// Driver is installed but not loaded
    Installed,
    /// Driver is available but not installed
    Available,
    /// Driver is not available
    NotAvailable,
    /// Driver installation failed
    Failed(String),
}

/// GPU Architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuArchitecture {
    /// Maxwell (GTX 900 series)
    Maxwell,
    /// Pascal (GTX 10 series)
    Pascal,
    /// Volta (Titan V, Tesla V100)
    Volta,
    /// Turing (RTX 20 series, GTX 16 series)
    Turing,
    /// Ampere (RTX 30 series)
    Ampere,
    /// Ada Lovelace (RTX 40 series)
    AdaLovelace,
    /// Hopper (H100)
    Hopper,
    /// Legacy architectures
    Legacy(String),
}

/// Driver Capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverCapabilities {
    /// CUDA support
    pub cuda_support: bool,
    /// CUDA version supported
    pub cuda_version: Option<String>,
    /// OpenCL support
    pub opencl_support: bool,
    /// Vulkan support
    pub vulkan_support: bool,
    /// OpenGL support
    pub opengl_support: bool,
    /// NVENC support (hardware encoding)
    pub nvenc_support: bool,
    /// NVDEC support (hardware decoding)
    pub nvdec_support: bool,
    /// NVLink support
    pub nvlink_support: bool,
    /// Multi-GPU support
    pub multi_gpu_support: bool,
    /// GPU virtualization support
    pub virtualization_support: bool,
    /// Container support
    pub container_support: bool,
    /// Dynamic power management
    pub power_management: bool,
    /// GPU monitoring capabilities
    pub monitoring_capabilities: MonitoringCapabilities,
}

/// Monitoring Capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringCapabilities {
    /// Temperature monitoring
    pub temperature_monitoring: bool,
    /// Power consumption monitoring
    pub power_monitoring: bool,
    /// GPU utilization monitoring
    pub utilization_monitoring: bool,
    /// Memory usage monitoring
    pub memory_monitoring: bool,
    /// Fan speed monitoring
    pub fan_monitoring: bool,
    /// Clock speed monitoring
    pub clock_monitoring: bool,
}

/// Driver Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverConfiguration {
    /// Preferred driver type
    pub preferred_driver: Option<NvidiaDriverType>,
    /// Automatic driver selection
    pub auto_select_driver: bool,
    /// Fallback behavior
    pub fallback_behavior: FallbackBehavior,
    /// Driver-specific configurations
    pub driver_configs: HashMap<NvidiaDriverType, DriverSpecificConfig>,
    /// Kernel module parameters
    pub kernel_module_params: HashMap<String, String>,
}

/// Fallback Behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackBehavior {
    /// Use next available driver in priority order
    NextAvailable,
    /// Use specific driver as fallback
    Specific(NvidiaDriverType),
    /// Fail if preferred driver not available
    Fail,
}

/// Driver-Specific Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverSpecificConfig {
    /// Enable/disable driver
    pub enabled: bool,
    /// Driver parameters
    pub parameters: HashMap<String, String>,
    /// Performance settings
    pub performance_settings: PerformanceSettings,
    /// Power management settings
    pub power_settings: PowerSettings,
}

/// Performance Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Performance mode
    pub performance_mode: PerformanceMode,
    /// GPU clock settings
    pub gpu_clock_settings: Option<ClockSettings>,
    /// Memory clock settings
    pub memory_clock_settings: Option<ClockSettings>,
    /// Power limit
    pub power_limit_percent: Option<u32>,
}

/// Performance Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMode {
    /// Maximum performance
    MaxPerformance,
    /// Balanced performance
    Balanced,
    /// Power efficient
    PowerEfficient,
    /// Custom settings
    Custom,
}

/// Clock Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClockSettings {
    /// Base clock offset in MHz
    pub base_clock_offset: i32,
    /// Boost clock offset in MHz
    pub boost_clock_offset: i32,
    /// Memory clock offset in MHz
    pub memory_clock_offset: i32,
}

/// Power Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSettings {
    /// Power management mode
    pub power_mode: PowerMode,
    /// Persistence mode
    pub persistence_mode: bool,
    /// Auto boost
    pub auto_boost: bool,
    /// GPU boost
    pub gpu_boost: bool,
}

/// Power Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerMode {
    /// Adaptive performance
    Adaptive,
    /// Prefer maximum performance
    PreferMaxPerformance,
    /// Optimal power
    OptimalPower,
}

/// GPU Device Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    /// GPU ID
    pub gpu_id: String,
    /// GPU name
    pub gpu_name: String,
    /// GPU architecture
    pub architecture: GpuArchitecture,
    /// PCI bus info
    pub pci_info: PciInfo,
    /// Memory information
    pub memory_info: MemoryInfo,
    /// Current driver
    pub current_driver: Option<NvidiaDriverType>,
    /// Supported drivers
    pub supported_drivers: Vec<NvidiaDriverType>,
    /// Device capabilities
    pub device_capabilities: DeviceCapabilities,
}

/// PCI Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PciInfo {
    /// PCI domain
    pub domain: u32,
    /// PCI bus
    pub bus: u32,
    /// PCI device
    pub device: u32,
    /// PCI function
    pub function: u32,
    /// Vendor ID
    pub vendor_id: u32,
    /// Device ID
    pub device_id: u32,
    /// Subsystem vendor ID
    pub subsystem_vendor_id: u32,
    /// Subsystem device ID
    pub subsystem_device_id: u32,
}

/// Memory Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total memory in bytes
    pub total_memory_bytes: u64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Memory type
    pub memory_type: MemoryType,
    /// Memory bus width
    pub memory_bus_width: u32,
}

/// Memory Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    GDDR5,
    GDDR5X,
    GDDR6,
    GDDR6X,
    HBM,
    HBM2,
    HBM2E,
    HBM3,
}

/// Device Capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// CUDA compute capability
    pub cuda_compute_capability: Option<ComputeCapability>,
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Maximum blocks per multiprocessor
    pub max_blocks_per_multiprocessor: u32,
    /// Number of multiprocessors
    pub multiprocessor_count: u32,
    /// Warp size
    pub warp_size: u32,
    /// Maximum grid size
    pub max_grid_size: [u32; 3],
    /// Maximum block size
    pub max_block_size: [u32; 3],
    /// Shared memory per block
    pub shared_memory_per_block: u32,
    /// Registers per block
    pub registers_per_block: u32,
}

/// CUDA Compute Capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapability {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
}

impl GpuDriverManager {
    /// Create a new GPU Driver Manager
    pub fn new() -> Self {
        let driver_priority = vec![
            NvidiaDriverType::OpenKernelModule, // Primary choice
            NvidiaDriverType::Proprietary,      // Fallback
            NvidiaDriverType::Nouveau,          // Last resort
        ];

        let driver_config = DriverConfiguration {
            preferred_driver: Some(NvidiaDriverType::OpenKernelModule),
            auto_select_driver: true,
            fallback_behavior: FallbackBehavior::NextAvailable,
            driver_configs: HashMap::new(),
            kernel_module_params: HashMap::new(),
        };

        Self {
            detected_drivers: Arc::new(RwLock::new(HashMap::new())),
            driver_priority,
            capabilities_cache: Arc::new(RwLock::new(HashMap::new())),
            driver_config: Arc::new(RwLock::new(driver_config)),
        }
    }

    /// Detect available NVIDIA drivers
    pub async fn detect_nvidia_drivers(&self) -> Result<Vec<NvidiaDriverInfo>> {
        info!("Detecting available NVIDIA drivers");

        let mut drivers = Vec::new();

        // Detect NVIDIA Open Kernel Module Driver
        if let Ok(open_driver) = self.detect_open_kernel_module_driver().await {
            drivers.push(open_driver);
        }

        // Detect NVIDIA Proprietary Driver
        if let Ok(proprietary_driver) = self.detect_proprietary_driver().await {
            drivers.push(proprietary_driver);
        }

        // Detect Nouveau Driver
        if let Ok(nouveau_driver) = self.detect_nouveau_driver().await {
            drivers.push(nouveau_driver);
        }

        // Cache detected drivers
        {
            let mut detected = self.detected_drivers.write().unwrap();
            for driver in &drivers {
                detected.insert(driver.kernel_module.clone(), driver.clone());
            }
        }

        info!("Detected {} NVIDIA drivers", drivers.len());
        Ok(drivers)
    }

    /// Detect NVIDIA Open Kernel Module Driver
    async fn detect_open_kernel_module_driver(&self) -> Result<NvidiaDriverInfo> {
        debug!("Detecting NVIDIA Open Kernel Module Driver");

        // Check if nvidia kernel module is loaded and is the open version
        let modinfo_output = Command::new("modinfo")
            .arg("nvidia")
            .output();

        if let Ok(output) = modinfo_output {
            if output.status.success() {
                let modinfo_str = String::from_utf8_lossy(&output.stdout);

                // Check if this is the open kernel module
                let is_open = modinfo_str.contains("NVIDIA Open GPU Kernel Module") ||
                             modinfo_str.contains("nvidia-open") ||
                             self.check_open_kernel_module_signature().await?;

                if is_open {
                    let version = self.extract_driver_version(&modinfo_str)?;
                    let capabilities = self.detect_open_kernel_capabilities().await?;

                    return Ok(NvidiaDriverInfo {
                        driver_type: NvidiaDriverType::OpenKernelModule,
                        version,
                        kernel_module: "nvidia".to_string(),
                        driver_path: PathBuf::from("/sys/module/nvidia"),
                        installation_status: InstallationStatus::Active,
                        supported_architectures: vec![
                            GpuArchitecture::Turing,
                            GpuArchitecture::Ampere,
                            GpuArchitecture::AdaLovelace,
                            GpuArchitecture::Hopper,
                        ],
                        capabilities,
                        priority: 1, // Highest priority
                        detected_at: chrono::Utc::now(),
                    });
                }
            }
        }

        // Check if open kernel module is available but not loaded
        let available = self.check_open_kernel_module_available().await?;
        if available {
            return Ok(NvidiaDriverInfo {
                driver_type: NvidiaDriverType::OpenKernelModule,
                version: "Unknown".to_string(),
                kernel_module: "nvidia-open".to_string(),
                driver_path: PathBuf::from("/lib/modules"),
                installation_status: InstallationStatus::Installed,
                supported_architectures: vec![
                    GpuArchitecture::Turing,
                    GpuArchitecture::Ampere,
                    GpuArchitecture::AdaLovelace,
                    GpuArchitecture::Hopper,
                ],
                capabilities: self.get_default_open_kernel_capabilities(),
                priority: 1,
                detected_at: chrono::Utc::now(),
            });
        }

        Err(anyhow!("NVIDIA Open Kernel Module Driver not found"))
    }

    /// Check if loaded nvidia module is the open kernel module
    async fn check_open_kernel_module_signature(&self) -> Result<bool> {
        // Check module signature or specific open kernel module indicators
        let module_path = PathBuf::from("/sys/module/nvidia");
        if !module_path.exists() {
            return Ok(false);
        }

        // Check for open kernel module specific files or attributes
        let open_indicators = vec![
            "/sys/module/nvidia/open_rm",
            "/sys/module/nvidia/version",
        ];

        for indicator in open_indicators {
            if PathBuf::from(indicator).exists() {
                return Ok(true);
            }
        }

        // Check dmesg for open kernel module messages
        let dmesg_output = Command::new("dmesg")
            .arg("-t")
            .output();

        if let Ok(output) = dmesg_output {
            let dmesg_str = String::from_utf8_lossy(&output.stdout);
            if dmesg_str.contains("nvidia-open") ||
               dmesg_str.contains("NVIDIA Open GPU Kernel Module") {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Check if open kernel module is available for installation
    async fn check_open_kernel_module_available(&self) -> Result<bool> {
        // Check for nvidia-open packages or modules
        let package_check = Command::new("dpkg")
            .arg("-l")
            .arg("nvidia-open")
            .output();

        if let Ok(output) = package_check {
            if output.status.success() && !output.stdout.is_empty() {
                return Ok(true);
            }
        }

        // Check for kernel modules
        let kernel_version = std::fs::read_to_string("/proc/version")?
            .split_whitespace()
            .nth(2)
            .unwrap_or("")
            .to_string();

        let module_path = format!("/lib/modules/{}/updates/dkms/nvidia-open.ko", kernel_version);
        if PathBuf::from(&module_path).exists() {
            return Ok(true);
        }

        Ok(false)
    }

    /// Detect open kernel module capabilities
    async fn detect_open_kernel_capabilities(&self) -> Result<DriverCapabilities> {
        let mut capabilities = self.get_default_open_kernel_capabilities();

        // Check for CUDA support
        if let Ok(cuda_version) = self.detect_cuda_version().await {
            capabilities.cuda_support = true;
            capabilities.cuda_version = Some(cuda_version);
        }

        // Check for container support
        capabilities.container_support = self.check_container_support().await.unwrap_or(true);

        Ok(capabilities)
    }

    /// Get default open kernel module capabilities
    fn get_default_open_kernel_capabilities(&self) -> DriverCapabilities {
        DriverCapabilities {
            cuda_support: true,
            cuda_version: None,
            opencl_support: true,
            vulkan_support: true,
            opengl_support: true,
            nvenc_support: true,
            nvdec_support: true,
            nvlink_support: true,
            multi_gpu_support: true,
            virtualization_support: true,
            container_support: true,
            power_management: true,
            monitoring_capabilities: MonitoringCapabilities {
                temperature_monitoring: true,
                power_monitoring: true,
                utilization_monitoring: true,
                memory_monitoring: true,
                fan_monitoring: true,
                clock_monitoring: true,
            },
        }
    }

    /// Detect NVIDIA Proprietary Driver
    async fn detect_proprietary_driver(&self) -> Result<NvidiaDriverInfo> {
        debug!("Detecting NVIDIA Proprietary Driver");

        // Check for nvidia-smi (proprietary driver indicator)
        let nvidia_smi_check = Command::new("nvidia-smi")
            .arg("--query-gpu=driver_version")
            .arg("--format=csv,noheader,nounits")
            .output();

        if let Ok(output) = nvidia_smi_check {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout).trim().to_string();

                // Verify this is proprietary driver (not open)
                let is_proprietary = !self.check_open_kernel_module_signature().await.unwrap_or(false);

                if is_proprietary {
                    let capabilities = self.detect_proprietary_capabilities().await?;

                    return Ok(NvidiaDriverInfo {
                        driver_type: NvidiaDriverType::Proprietary,
                        version,
                        kernel_module: "nvidia".to_string(),
                        driver_path: PathBuf::from("/usr/bin/nvidia-smi"),
                        installation_status: InstallationStatus::Active,
                        supported_architectures: vec![
                            GpuArchitecture::Maxwell,
                            GpuArchitecture::Pascal,
                            GpuArchitecture::Volta,
                            GpuArchitecture::Turing,
                            GpuArchitecture::Ampere,
                            GpuArchitecture::AdaLovelace,
                            GpuArchitecture::Hopper,
                        ],
                        capabilities,
                        priority: 2,
                        detected_at: chrono::Utc::now(),
                    });
                }
            }
        }

        // Check if proprietary driver is installed but not active
        let driver_installed = self.check_proprietary_driver_installed().await?;
        if driver_installed {
            return Ok(NvidiaDriverInfo {
                driver_type: NvidiaDriverType::Proprietary,
                version: "Unknown".to_string(),
                kernel_module: "nvidia".to_string(),
                driver_path: PathBuf::from("/usr/bin/nvidia-smi"),
                installation_status: InstallationStatus::Installed,
                supported_architectures: vec![
                    GpuArchitecture::Maxwell,
                    GpuArchitecture::Pascal,
                    GpuArchitecture::Volta,
                    GpuArchitecture::Turing,
                    GpuArchitecture::Ampere,
                    GpuArchitecture::AdaLovelace,
                    GpuArchitecture::Hopper,
                ],
                capabilities: self.get_default_proprietary_capabilities(),
                priority: 2,
                detected_at: chrono::Utc::now(),
            });
        }

        Err(anyhow!("NVIDIA Proprietary Driver not found"))
    }

    /// Check if proprietary driver is installed
    async fn check_proprietary_driver_installed(&self) -> Result<bool> {
        // Check for nvidia packages
        let package_check = Command::new("dpkg")
            .arg("-l")
            .arg("nvidia-driver-*")
            .output();

        if let Ok(output) = package_check {
            if output.status.success() && !output.stdout.is_empty() {
                return Ok(true);
            }
        }

        // Check for nvidia-smi binary
        Ok(PathBuf::from("/usr/bin/nvidia-smi").exists())
    }

    /// Detect proprietary driver capabilities
    async fn detect_proprietary_capabilities(&self) -> Result<DriverCapabilities> {
        let mut capabilities = self.get_default_proprietary_capabilities();

        // Check CUDA version
        if let Ok(cuda_version) = self.detect_cuda_version().await {
            capabilities.cuda_version = Some(cuda_version);
        }

        Ok(capabilities)
    }

    /// Get default proprietary driver capabilities
    fn get_default_proprietary_capabilities(&self) -> DriverCapabilities {
        DriverCapabilities {
            cuda_support: true,
            cuda_version: None,
            opencl_support: true,
            vulkan_support: true,
            opengl_support: true,
            nvenc_support: true,
            nvdec_support: true,
            nvlink_support: true,
            multi_gpu_support: true,
            virtualization_support: true,
            container_support: true,
            power_management: true,
            monitoring_capabilities: MonitoringCapabilities {
                temperature_monitoring: true,
                power_monitoring: true,
                utilization_monitoring: true,
                memory_monitoring: true,
                fan_monitoring: true,
                clock_monitoring: true,
            },
        }
    }

    /// Detect Nouveau Driver
    async fn detect_nouveau_driver(&self) -> Result<NvidiaDriverInfo> {
        debug!("Detecting Nouveau Driver");

        // Check if nouveau module is loaded
        let lsmod_output = Command::new("lsmod")
            .output();

        if let Ok(output) = lsmod_output {
            let lsmod_str = String::from_utf8_lossy(&output.stdout);
            if lsmod_str.contains("nouveau") {
                let capabilities = self.detect_nouveau_capabilities().await?;

                return Ok(NvidiaDriverInfo {
                    driver_type: NvidiaDriverType::Nouveau,
                    version: "Open Source".to_string(),
                    kernel_module: "nouveau".to_string(),
                    driver_path: PathBuf::from("/sys/module/nouveau"),
                    installation_status: InstallationStatus::Active,
                    supported_architectures: vec![
                        GpuArchitecture::Maxwell,
                        GpuArchitecture::Pascal,
                        GpuArchitecture::Volta,
                        GpuArchitecture::Turing,
                        // Limited support for newer architectures
                    ],
                    capabilities,
                    priority: 3, // Lowest priority
                    detected_at: chrono::Utc::now(),
                });
            }
        }

        // Check if nouveau is available but not loaded
        let available = self.check_nouveau_available().await?;
        if available {
            return Ok(NvidiaDriverInfo {
                driver_type: NvidiaDriverType::Nouveau,
                version: "Open Source".to_string(),
                kernel_module: "nouveau".to_string(),
                driver_path: PathBuf::from("/lib/modules"),
                installation_status: InstallationStatus::Installed,
                supported_architectures: vec![
                    GpuArchitecture::Maxwell,
                    GpuArchitecture::Pascal,
                    GpuArchitecture::Volta,
                    GpuArchitecture::Turing,
                ],
                capabilities: self.get_default_nouveau_capabilities(),
                priority: 3,
                detected_at: chrono::Utc::now(),
            });
        }

        Err(anyhow!("Nouveau Driver not found"))
    }

    /// Check if nouveau is available
    async fn check_nouveau_available(&self) -> Result<bool> {
        let kernel_version = std::fs::read_to_string("/proc/version")?
            .split_whitespace()
            .nth(2)
            .unwrap_or("")
            .to_string();

        let module_path = format!("/lib/modules/{}/kernel/drivers/gpu/drm/nouveau/nouveau.ko", kernel_version);
        Ok(PathBuf::from(&module_path).exists())
    }

    /// Detect nouveau capabilities
    async fn detect_nouveau_capabilities(&self) -> Result<DriverCapabilities> {
        Ok(self.get_default_nouveau_capabilities())
    }

    /// Get default nouveau capabilities
    fn get_default_nouveau_capabilities(&self) -> DriverCapabilities {
        DriverCapabilities {
            cuda_support: false, // Nouveau doesn't support CUDA
            cuda_version: None,
            opencl_support: true, // Limited OpenCL support
            vulkan_support: true, // Basic Vulkan support
            opengl_support: true,
            nvenc_support: false, // No NVENC support
            nvdec_support: false, // No NVDEC support
            nvlink_support: false, // No NVLink support
            multi_gpu_support: true,
            virtualization_support: false,
            container_support: false, // Limited container support
            power_management: true,
            monitoring_capabilities: MonitoringCapabilities {
                temperature_monitoring: true,
                power_monitoring: true,
                utilization_monitoring: true,
                memory_monitoring: true,
                fan_monitoring: false,
                clock_monitoring: true,
            },
        }
    }

    /// Extract driver version from modinfo output
    fn extract_driver_version(&self, modinfo_str: &str) -> Result<String> {
        for line in modinfo_str.lines() {
            if line.starts_with("version:") {
                return Ok(line.replace("version:", "").trim().to_string());
            }
        }
        Ok("Unknown".to_string())
    }

    /// Detect CUDA version
    async fn detect_cuda_version(&self) -> Result<String> {
        let nvcc_output = Command::new("nvcc")
            .arg("--version")
            .output();

        if let Ok(output) = nvcc_output {
            if output.status.success() {
                let version_str = String::from_utf8_lossy(&output.stdout);
                for line in version_str.lines() {
                    if line.contains("release") {
                        if let Some(version) = line.split("release").nth(1) {
                            return Ok(version.trim().split(',').next().unwrap_or("").to_string());
                        }
                    }
                }
            }
        }

        // Try alternative method using nvidia-smi
        let smi_output = Command::new("nvidia-smi")
            .arg("--query-gpu=driver_version")
            .arg("--format=csv,noheader,nounits")
            .output();

        if let Ok(output) = smi_output {
            if output.status.success() {
                let driver_version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                return Ok(self.map_driver_to_cuda_version(&driver_version));
            }
        }

        Err(anyhow!("Could not detect CUDA version"))
    }

    /// Map driver version to CUDA version
    fn map_driver_to_cuda_version(&self, driver_version: &str) -> String {
        // Simplified mapping - in real implementation would be more comprehensive
        if driver_version.starts_with("535") || driver_version.starts_with("545") {
            "12.2".to_string()
        } else if driver_version.starts_with("525") {
            "12.0".to_string()
        } else if driver_version.starts_with("515") {
            "11.7".to_string()
        } else if driver_version.starts_with("510") {
            "11.6".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    /// Check container support
    async fn check_container_support(&self) -> Result<bool> {
        // Check for nvidia-container-runtime
        let runtime_check = Command::new("which")
            .arg("nvidia-container-runtime")
            .output();

        if let Ok(output) = runtime_check {
            if output.status.success() {
                return Ok(true);
            }
        }

        // Check for libnvidia-container
        Ok(PathBuf::from("/usr/lib/x86_64-linux-gnu/libnvidia-container.so.1").exists())
    }

    /// Select best available driver
    pub async fn select_best_driver(&self) -> Result<NvidiaDriverType> {
        info!("Selecting best available NVIDIA driver");

        let drivers = self.detect_nvidia_drivers().await?;
        let config = self.driver_config.read().unwrap();

        // If user has a preference and it's available, use it
        if let Some(preferred) = &config.preferred_driver {
            for driver in &drivers {
                if &driver.driver_type == preferred &&
                   matches!(driver.installation_status, InstallationStatus::Active | InstallationStatus::Installed) {
                    info!("Selected preferred driver: {:?}", preferred);
                    return Ok(preferred.clone());
                }
            }
        }

        // Use priority order
        for driver_type in &self.driver_priority {
            for driver in &drivers {
                if &driver.driver_type == driver_type &&
                   matches!(driver.installation_status, InstallationStatus::Active | InstallationStatus::Installed) {
                    info!("Selected driver by priority: {:?}", driver_type);
                    return Ok(driver_type.clone());
                }
            }
        }

        Err(anyhow!("No suitable NVIDIA driver found"))
    }

    /// Load specified driver
    pub async fn load_driver(&self, driver_type: &NvidiaDriverType) -> Result<()> {
        info!("Loading NVIDIA driver: {:?}", driver_type);

        match driver_type {
            NvidiaDriverType::OpenKernelModule => {
                self.load_open_kernel_module().await?;
            },
            NvidiaDriverType::Proprietary => {
                self.load_proprietary_driver().await?;
            },
            NvidiaDriverType::Nouveau => {
                self.load_nouveau_driver().await?;
            },
        }

        info!("Successfully loaded driver: {:?}", driver_type);
        Ok(())
    }

    /// Load open kernel module
    async fn load_open_kernel_module(&self) -> Result<()> {
        debug!("Loading NVIDIA Open Kernel Module");

        // Ensure proprietary driver is not loaded
        self.unload_proprietary_driver().await?;

        // Load nvidia-open module
        let modprobe_result = Command::new("modprobe")
            .arg("nvidia-open")
            .output();

        if modprobe_result.is_err() {
            // Fallback to nvidia module if nvidia-open not available
            let fallback_result = Command::new("modprobe")
                .arg("nvidia")
                .output();

            if fallback_result.is_err() {
                return Err(anyhow!("Failed to load NVIDIA Open Kernel Module"));
            }
        }

        // Load additional modules
        let _ = Command::new("modprobe").arg("nvidia-uvm").output();
        let _ = Command::new("modprobe").arg("nvidia-modeset").output();

        Ok(())
    }

    /// Load proprietary driver
    async fn load_proprietary_driver(&self) -> Result<()> {
        debug!("Loading NVIDIA Proprietary Driver");

        // Ensure nouveau is not loaded
        self.unload_nouveau_driver().await?;

        // Load nvidia module
        let modprobe_result = Command::new("modprobe")
            .arg("nvidia")
            .output();

        if modprobe_result.is_err() {
            return Err(anyhow!("Failed to load NVIDIA Proprietary Driver"));
        }

        // Load additional modules
        let _ = Command::new("modprobe").arg("nvidia-uvm").output();
        let _ = Command::new("modprobe").arg("nvidia-modeset").output();

        Ok(())
    }

    /// Load nouveau driver
    async fn load_nouveau_driver(&self) -> Result<()> {
        debug!("Loading Nouveau Driver");

        // Ensure nvidia drivers are not loaded
        self.unload_proprietary_driver().await?;

        // Load nouveau module
        let modprobe_result = Command::new("modprobe")
            .arg("nouveau")
            .output();

        if modprobe_result.is_err() {
            return Err(anyhow!("Failed to load Nouveau Driver"));
        }

        Ok(())
    }

    /// Unload proprietary driver
    async fn unload_proprietary_driver(&self) -> Result<()> {
        debug!("Unloading NVIDIA proprietary driver modules");

        let modules_to_unload = vec!["nvidia-uvm", "nvidia-modeset", "nvidia"];
        for module in modules_to_unload {
            let _ = Command::new("rmmod").arg(module).output();
        }

        Ok(())
    }

    /// Unload nouveau driver
    async fn unload_nouveau_driver(&self) -> Result<()> {
        debug!("Unloading Nouveau driver");

        let _ = Command::new("rmmod").arg("nouveau").output();
        Ok(())
    }

    /// Get GPU device information
    pub async fn get_gpu_devices(&self) -> Result<Vec<GpuDeviceInfo>> {
        info!("Detecting GPU devices");

        let mut devices = Vec::new();

        // Use lspci to find NVIDIA GPUs
        let lspci_output = Command::new("lspci")
            .arg("-nn")
            .arg("-d")
            .arg("10de:") // NVIDIA vendor ID
            .output()?;

        let lspci_str = String::from_utf8_lossy(&lspci_output.stdout);

        for line in lspci_str.lines() {
            if line.contains("VGA") || line.contains("3D") {
                if let Ok(device) = self.parse_gpu_device_info(line).await {
                    devices.push(device);
                }
            }
        }

        // Try nvidia-ml-py for additional info if available
        if let Ok(nvidia_devices) = self.get_nvidia_ml_devices().await {
            for (i, nvidia_device) in nvidia_devices.iter().enumerate() {
                if i < devices.len() {
                    devices[i].memory_info = nvidia_device.memory_info.clone();
                    devices[i].device_capabilities = nvidia_device.device_capabilities.clone();
                }
            }
        }

        info!("Found {} GPU devices", devices.len());
        Ok(devices)
    }

    /// Parse GPU device info from lspci output
    async fn parse_gpu_device_info(&self, lspci_line: &str) -> Result<GpuDeviceInfo> {
        // Parse PCI info from lspci line
        let parts: Vec<&str> = lspci_line.split_whitespace().collect();
        let pci_address = parts[0];
        let pci_parts: Vec<&str> = pci_address.split(':').collect();

        let bus = u32::from_str_radix(pci_parts[0], 16)?;
        let device_func: Vec<&str> = pci_parts[1].split('.').collect();
        let device = u32::from_str_radix(device_func[0], 16)?;
        let function = u32::from_str_radix(device_func[1], 16)?;

        // Extract device name
        let gpu_name = lspci_line.split(':').skip(2).collect::<Vec<&str>>().join(":");
        let gpu_name = gpu_name.trim().to_string();

        // Determine architecture based on GPU name
        let architecture = self.determine_gpu_architecture(&gpu_name);

        let device_info = GpuDeviceInfo {
            gpu_id: format!("GPU-{:02x}:{:02x}.{}", bus, device, function),
            gpu_name,
            architecture,
            pci_info: PciInfo {
                domain: 0, // Default domain
                bus,
                device,
                function,
                vendor_id: 0x10de, // NVIDIA
                device_id: 0, // Would extract from lspci -n
                subsystem_vendor_id: 0,
                subsystem_device_id: 0,
            },
            memory_info: MemoryInfo {
                total_memory_bytes: 0, // Will be filled by nvidia-ml if available
                memory_bandwidth_gbps: 0.0,
                memory_type: MemoryType::GDDR6, // Default assumption
                memory_bus_width: 256,
            },
            current_driver: self.get_current_driver().await.ok(),
            supported_drivers: self.get_supported_drivers_for_architecture(&architecture),
            device_capabilities: DeviceCapabilities {
                cuda_compute_capability: None,
                max_threads_per_block: 1024,
                max_blocks_per_multiprocessor: 16,
                multiprocessor_count: 0,
                warp_size: 32,
                max_grid_size: [65535, 65535, 65535],
                max_block_size: [1024, 1024, 64],
                shared_memory_per_block: 49152,
                registers_per_block: 65536,
            },
        };

        Ok(device_info)
    }

    /// Determine GPU architecture from name
    fn determine_gpu_architecture(&self, gpu_name: &str) -> GpuArchitecture {
        let name_lower = gpu_name.to_lowercase();

        if name_lower.contains("rtx 40") || name_lower.contains("ada") {
            GpuArchitecture::AdaLovelace
        } else if name_lower.contains("rtx 30") || name_lower.contains("ampere") {
            GpuArchitecture::Ampere
        } else if name_lower.contains("rtx 20") || name_lower.contains("gtx 16") || name_lower.contains("turing") {
            GpuArchitecture::Turing
        } else if name_lower.contains("titan v") || name_lower.contains("tesla v") || name_lower.contains("volta") {
            GpuArchitecture::Volta
        } else if name_lower.contains("gtx 10") || name_lower.contains("pascal") {
            GpuArchitecture::Pascal
        } else if name_lower.contains("gtx 9") || name_lower.contains("maxwell") {
            GpuArchitecture::Maxwell
        } else if name_lower.contains("h100") || name_lower.contains("hopper") {
            GpuArchitecture::Hopper
        } else {
            GpuArchitecture::Legacy(gpu_name.to_string())
        }
    }

    /// Get current active driver
    async fn get_current_driver(&self) -> Result<NvidiaDriverType> {
        // Check which nvidia module is loaded
        let lsmod_output = Command::new("lsmod")
            .output()?;

        let lsmod_str = String::from_utf8_lossy(&lsmod_output.stdout);

        if lsmod_str.contains("nouveau") {
            return Ok(NvidiaDriverType::Nouveau);
        }

        if lsmod_str.contains("nvidia") {
            // Check if it's open kernel module
            if self.check_open_kernel_module_signature().await.unwrap_or(false) {
                return Ok(NvidiaDriverType::OpenKernelModule);
            } else {
                return Ok(NvidiaDriverType::Proprietary);
            }
        }

        Err(anyhow!("No NVIDIA driver currently loaded"))
    }

    /// Get supported drivers for GPU architecture
    fn get_supported_drivers_for_architecture(&self, architecture: &GpuArchitecture) -> Vec<NvidiaDriverType> {
        match architecture {
            GpuArchitecture::Turing |
            GpuArchitecture::Ampere |
            GpuArchitecture::AdaLovelace |
            GpuArchitecture::Hopper => {
                vec![
                    NvidiaDriverType::OpenKernelModule,
                    NvidiaDriverType::Proprietary,
                    NvidiaDriverType::Nouveau,
                ]
            },
            GpuArchitecture::Maxwell |
            GpuArchitecture::Pascal |
            GpuArchitecture::Volta => {
                vec![
                    NvidiaDriverType::Proprietary,
                    NvidiaDriverType::Nouveau,
                ]
            },
            GpuArchitecture::Legacy(_) => {
                vec![
                    NvidiaDriverType::Proprietary,
                    NvidiaDriverType::Nouveau,
                ]
            },
        }
    }

    /// Get device info using nvidia-ml (if available)
    async fn get_nvidia_ml_devices(&self) -> Result<Vec<GpuDeviceInfo>> {
        // Placeholder for nvidia-ml-py integration
        // In real implementation, would use nvidia-ml-py bindings
        Ok(Vec::new())
    }

    /// Get detected drivers
    pub fn get_detected_drivers(&self) -> Vec<NvidiaDriverInfo> {
        let drivers = self.detected_drivers.read().unwrap();
        drivers.values().cloned().collect()
    }

    /// Update driver configuration
    pub async fn update_driver_configuration(&self, config: DriverConfiguration) -> Result<()> {
        let mut driver_config = self.driver_config.write().unwrap();
        *driver_config = config;
        info!("Updated driver configuration");
        Ok(())
    }

    /// Get driver configuration
    pub fn get_driver_configuration(&self) -> DriverConfiguration {
        let config = self.driver_config.read().unwrap();
        config.clone()
    }

    /// Generate driver compatibility report
    pub async fn generate_compatibility_report(&self) -> Result<String> {
        info!("Generating NVIDIA driver compatibility report");

        let drivers = self.detect_nvidia_drivers().await?;
        let devices = self.get_gpu_devices().await?;

        let mut report = String::new();
        report.push_str("# NVIDIA Driver Compatibility Report\n\n");

        report.push_str("## Detected NVIDIA Drivers\n");
        for driver in &drivers {
            report.push_str(&format!("### {:?}\n", driver.driver_type));
            report.push_str(&format!("- Version: {}\n", driver.version));
            report.push_str(&format!("- Status: {:?}\n", driver.installation_status));
            report.push_str(&format!("- Priority: {}\n", driver.priority));
            report.push_str(&format!("- CUDA Support: {}\n", driver.capabilities.cuda_support));
            if let Some(cuda_version) = &driver.capabilities.cuda_version {
                report.push_str(&format!("- CUDA Version: {}\n", cuda_version));
            }
            report.push_str(&format!("- Container Support: {}\n", driver.capabilities.container_support));
            report.push_str("\n");
        }

        report.push_str("## Detected GPU Devices\n");
        for device in &devices {
            report.push_str(&format!("### {}\n", device.gpu_name));
            report.push_str(&format!("- GPU ID: {}\n", device.gpu_id));
            report.push_str(&format!("- Architecture: {:?}\n", device.architecture));
            report.push_str(&format!("- Current Driver: {:?}\n",
                device.current_driver.as_ref().map(|d| format!("{:?}", d)).unwrap_or("None".to_string())));
            report.push_str("- Supported Drivers: ");
            for (i, driver) in device.supported_drivers.iter().enumerate() {
                if i > 0 { report.push_str(", "); }
                report.push_str(&format!("{:?}", driver));
            }
            report.push_str("\n\n");
        }

        if let Ok(selected_driver) = self.select_best_driver().await {
            report.push_str(&format!("## Recommended Driver\n{:?}\n\n", selected_driver));
        }

        Ok(report)
    }
}

impl Default for GpuDriverManager {
    fn default() -> Self {
        Self::new()
    }
}