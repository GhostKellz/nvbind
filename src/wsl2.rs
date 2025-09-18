use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, info};

/// WSL2 configuration for nvbind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wsl2Config {
    /// Enable WSL2-specific optimizations
    pub enable_wsl2_optimizations: bool,
    /// WSL2 driver path
    pub driver_path: PathBuf,
    /// GPU compute API preferences
    pub preferred_apis: Vec<String>,
    /// DirectX 12 configuration
    pub dx12_config: Dx12Config,
    /// OpenGL configuration
    pub opengl_config: OpenGlConfig,
    /// Vulkan configuration
    pub vulkan_config: VulkanConfig,
    /// WSL2-specific environment variables
    pub environment: HashMap<String, String>,
}

/// DirectX 12 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dx12Config {
    pub enabled: bool,
    pub debug_layer: bool,
    pub gpu_validation: bool,
    pub dred_enabled: bool,
}

/// OpenGL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenGlConfig {
    pub enabled: bool,
    pub version: String,
    pub mesa_config: MesaConfig,
}

/// Mesa configuration for OpenGL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MesaConfig {
    pub enable_debug: bool,
    pub shader_cache: bool,
    pub optimization_level: u32,
}

/// Vulkan configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanConfig {
    pub enabled: bool,
    pub validation_layers: bool,
    pub api_dump: bool,
    pub instance_extensions: Vec<String>,
    pub device_extensions: Vec<String>,
}

impl Default for Wsl2Config {
    fn default() -> Self {
        let mut environment = HashMap::new();
        environment.insert("LIBGL_ALWAYS_INDIRECT".to_string(), "0".to_string());
        environment.insert("MESA_GL_VERSION_OVERRIDE".to_string(), "4.6".to_string());
        environment.insert("MESA_GLSL_VERSION_OVERRIDE".to_string(), "460".to_string());

        Self {
            enable_wsl2_optimizations: true,
            driver_path: PathBuf::from("/usr/lib/wsl/drivers"),
            preferred_apis: vec!["cuda".to_string(), "opencl".to_string(), "vulkan".to_string()],
            dx12_config: Dx12Config {
                enabled: true,
                debug_layer: false,
                gpu_validation: false,
                dred_enabled: false,
            },
            opengl_config: OpenGlConfig {
                enabled: true,
                version: "4.6".to_string(),
                mesa_config: MesaConfig {
                    enable_debug: false,
                    shader_cache: true,
                    optimization_level: 2,
                },
            },
            vulkan_config: VulkanConfig {
                enabled: true,
                validation_layers: false,
                api_dump: false,
                instance_extensions: vec![
                    "VK_KHR_surface".to_string(),
                    "VK_KHR_win32_surface".to_string(),
                ],
                device_extensions: vec![
                    "VK_KHR_swapchain".to_string(),
                ],
            },
            environment,
        }
    }
}

/// WSL2 detection and configuration manager
pub struct Wsl2Manager {
    config: Wsl2Config,
    is_wsl2: bool,
    wsl_version: Option<String>,
    windows_build: Option<u32>,
}

impl Wsl2Manager {
    /// Create a new WSL2 manager
    pub fn new() -> Result<Self> {
        let is_wsl2 = Self::detect_wsl2();
        let wsl_version = Self::get_wsl_version()?;
        let windows_build = Self::get_windows_build()?;

        info!("WSL2 detected: {}", is_wsl2);
        if let Some(ref version) = wsl_version {
            info!("WSL version: {}", version);
        }
        if let Some(build) = windows_build {
            info!("Windows build: {}", build);
        }

        Ok(Self {
            config: Wsl2Config::default(),
            is_wsl2,
            wsl_version,
            windows_build,
        })
    }

    /// Create WSL2 manager with custom config
    pub fn with_config(config: Wsl2Config) -> Result<Self> {
        let mut manager = Self::new()?;
        manager.config = config;
        Ok(manager)
    }

    /// Detect if running under WSL2
    pub fn detect_wsl2() -> bool {
        // Check for WSL environment variables
        if std::env::var("WSL_DISTRO_NAME").is_ok() {
            return true;
        }

        // Check for WSL interop file
        if Path::new("/proc/sys/fs/binfmt_misc/WSLInterop").exists() {
            return true;
        }

        // Check /proc/version for WSL indicators
        if let Ok(version) = fs::read_to_string("/proc/version") {
            if version.contains("Microsoft") || version.contains("WSL") {
                return true;
            }
        }

        // Check for WSL-specific mount points
        if let Ok(mounts) = fs::read_to_string("/proc/mounts") {
            if mounts.contains("9p") && mounts.contains("C:") {
                return true;
            }
        }

        false
    }

    /// Get WSL version information
    fn get_wsl_version() -> Result<Option<String>> {
        if let Ok(version) = std::env::var("WSL_DISTRO_NAME") {
            return Ok(Some(version));
        }

        // Try to get version from kernel
        if let Ok(version) = fs::read_to_string("/proc/version") {
            if let Some(start) = version.find("WSL") {
                let wsl_part = &version[start..];
                if let Some(end) = wsl_part.find(' ') {
                    return Ok(Some(wsl_part[..end].to_string()));
                }
            }
        }

        Ok(None)
    }

    /// Get Windows build number
    fn get_windows_build() -> Result<Option<u32>> {
        // Try to get Windows build from /proc/version
        if let Ok(version) = fs::read_to_string("/proc/version") {
            // Look for build number pattern
            let re = regex::Regex::new(r"(\d{5})")?;
            if let Some(captures) = re.captures(&version) {
                if let Ok(build) = captures[1].parse::<u32>() {
                    return Ok(Some(build));
                }
            }
        }

        Ok(None)
    }

    /// Check if WSL2 GPU support is available
    pub fn check_gpu_support(&self) -> Result<Wsl2GpuSupport> {
        if !self.is_wsl2 {
            return Ok(Wsl2GpuSupport::NotWsl2);
        }

        let mut support = Wsl2GpuSupport::Available {
            cuda: false,
            opencl: false,
            directx: false,
            opengl: false,
            vulkan: false,
        };

        // Check for CUDA support
        if self.check_cuda_support()? {
            if let Wsl2GpuSupport::Available { ref mut cuda, .. } = support {
                *cuda = true;
            }
        }

        // Check for OpenCL support
        if self.check_opencl_support()? {
            if let Wsl2GpuSupport::Available { ref mut opencl, .. } = support {
                *opencl = true;
            }
        }

        // Check for DirectX support
        if self.check_directx_support()? {
            if let Wsl2GpuSupport::Available { ref mut directx, .. } = support {
                *directx = true;
            }
        }

        // Check for OpenGL support
        if self.check_opengl_support()? {
            if let Wsl2GpuSupport::Available { ref mut opengl, .. } = support {
                *opengl = true;
            }
        }

        // Check for Vulkan support
        if self.check_vulkan_support()? {
            if let Wsl2GpuSupport::Available { ref mut vulkan, .. } = support {
                *vulkan = true;
            }
        }

        Ok(support)
    }

    /// Check CUDA support in WSL2
    fn check_cuda_support(&self) -> Result<bool> {
        // Check for NVIDIA CUDA libraries
        let cuda_libs = [
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcudart.so",
            "/usr/local/cuda/lib64/libcuda.so",
        ];

        for lib in &cuda_libs {
            if Path::new(lib).exists() {
                debug!("Found CUDA library: {}", lib);
                return Ok(true);
            }
        }

        // Check for nvidia-smi
        if which::which("nvidia-smi").is_ok() {
            if let Ok(output) = Command::new("nvidia-smi").output() {
                if output.status.success() {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Check OpenCL support in WSL2
    fn check_opencl_support(&self) -> Result<bool> {
        let opencl_libs = [
            "/usr/lib/x86_64-linux-gnu/libOpenCL.so",
            "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
        ];

        for lib in &opencl_libs {
            if Path::new(lib).exists() {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Check DirectX support in WSL2
    fn check_directx_support(&self) -> Result<bool> {
        let dx_libs = [
            "/usr/lib/x86_64-linux-gnu/libdxcore.so",
            "/usr/lib/x86_64-linux-gnu/libd3d12.so",
        ];

        for lib in &dx_libs {
            if Path::new(lib).exists() {
                return Ok(true);
            }
        }

        // Check WSL drivers directory
        let wsl_drivers = self.config.driver_path.join("dxgkrnl.sys");
        if wsl_drivers.exists() {
            return Ok(true);
        }

        Ok(false)
    }

    /// Check OpenGL support in WSL2
    fn check_opengl_support(&self) -> Result<bool> {
        let gl_libs = [
            "/usr/lib/x86_64-linux-gnu/libGL.so",
            "/usr/lib/x86_64-linux-gnu/libGL.so.1",
            "/usr/lib/x86_64-linux-gnu/mesa/libGL.so",
        ];

        for lib in &gl_libs {
            if Path::new(lib).exists() {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Check Vulkan support in WSL2
    fn check_vulkan_support(&self) -> Result<bool> {
        let vulkan_libs = [
            "/usr/lib/x86_64-linux-gnu/libvulkan.so",
            "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
        ];

        for lib in &vulkan_libs {
            if Path::new(lib).exists() {
                return Ok(true);
            }
        }

        // Check for vulkan-tools
        if which::which("vulkaninfo").is_ok() {
            return Ok(true);
        }

        Ok(false)
    }

    /// Setup WSL2 GPU environment
    pub fn setup_gpu_environment(&self) -> Result<HashMap<String, String>> {
        if !self.is_wsl2 {
            return Err(anyhow::anyhow!("Not running under WSL2"));
        }

        info!("Setting up WSL2 GPU environment");

        let mut env = self.config.environment.clone();

        // DirectX 12 environment
        if self.config.dx12_config.enabled {
            env.insert("D3D12_ENABLE_EXPERIMENTAL_FEATURES".to_string(), "1".to_string());

            if self.config.dx12_config.debug_layer {
                env.insert("D3D12_DEBUG".to_string(), "1".to_string());
            }

            if self.config.dx12_config.gpu_validation {
                env.insert("D3D12_GPU_VALIDATION".to_string(), "1".to_string());
            }

            if self.config.dx12_config.dred_enabled {
                env.insert("D3D12_DRED_ENABLE".to_string(), "1".to_string());
            }
        }

        // OpenGL/Mesa environment
        if self.config.opengl_config.enabled {
            env.insert(
                "MESA_GL_VERSION_OVERRIDE".to_string(),
                self.config.opengl_config.version.clone(),
            );

            if self.config.opengl_config.mesa_config.enable_debug {
                env.insert("MESA_DEBUG".to_string(), "1".to_string());
            }

            if self.config.opengl_config.mesa_config.shader_cache {
                env.insert("MESA_SHADER_CACHE_DISABLE".to_string(), "false".to_string());
            }

            env.insert(
                "MESA_OPTIMIZATION_LEVEL".to_string(),
                self.config.opengl_config.mesa_config.optimization_level.to_string(),
            );
        }

        // Vulkan environment
        if self.config.vulkan_config.enabled {
            if self.config.vulkan_config.validation_layers {
                env.insert("VK_LAYER_PATH".to_string(), "/usr/share/vulkan/explicit_layer.d".to_string());
                env.insert("VK_INSTANCE_LAYERS".to_string(), "VK_LAYER_KHRONOS_validation".to_string());
            }

            if self.config.vulkan_config.api_dump {
                env.insert("VK_LAYER_ENABLES".to_string(), "VK_VALIDATION_FEATURE_ENABLE_API_PARAMETERS_EXT".to_string());
            }
        }

        // WSL2-specific optimizations
        if self.config.enable_wsl2_optimizations {
            // Disable software rendering fallbacks
            env.insert("LIBGL_ALWAYS_SOFTWARE".to_string(), "0".to_string());
            env.insert("GALLIUM_DRIVER".to_string(), "d3d12".to_string());

            // Enable hardware acceleration
            env.insert("MESA_D3D12_DEFAULT_ADAPTER_NAME".to_string(), "NVIDIA".to_string());

            // WSL2 GPU scheduling optimizations
            env.insert("WSL_GPU_SCHEDULING".to_string(), "1".to_string());
        }

        Ok(env)
    }

    /// Apply WSL2-specific container modifications
    pub fn apply_container_modifications(&self, runtime: &str) -> Result<Vec<String>> {
        if !self.is_wsl2 {
            return Ok(Vec::new());
        }

        let mut args = Vec::new();

        // Mount WSL drivers directory
        if self.config.driver_path.exists() {
            args.push("--volume".to_string());
            args.push(format!("{}:/usr/lib/wsl/drivers:ro", self.config.driver_path.display()));
        }

        // Mount DRI devices for hardware acceleration
        let dri_devices = ["/dev/dri", "/dev/dxg"];
        for device in &dri_devices {
            if Path::new(device).exists() {
                args.push("--device".to_string());
                args.push(format!("{}:{}", device, device));
            }
        }

        // Add WSL2-specific environment variables
        let env_vars = self.setup_gpu_environment()?;
        for (key, value) in env_vars {
            args.push("--env".to_string());
            args.push(format!("{}={}", key, value));
        }

        // Runtime-specific modifications
        match runtime {
            "docker" => {
                // Docker Desktop on WSL2 specific settings
                if std::env::var("DOCKER_HOST").is_ok() {
                    args.push("--add-host".to_string());
                    args.push("host.docker.internal:host-gateway".to_string());
                }
            }
            "podman" => {
                // Podman on WSL2 specific settings
                args.push("--security-opt".to_string());
                args.push("label=disable".to_string());
            }
            _ => {}
        }

        Ok(args)
    }

    /// Check if Windows build supports GPU acceleration
    pub fn check_windows_gpu_support(&self) -> bool {
        if let Some(build) = self.windows_build {
            // GPU acceleration requires Windows 10 build 20150 or later
            build >= 20150
        } else {
            false
        }
    }

    /// Generate WSL2 GPU diagnostics report
    pub fn generate_diagnostics(&self) -> Result<Wsl2Diagnostics> {
        let gpu_support = self.check_gpu_support()?;
        let environment = self.setup_gpu_environment().unwrap_or_default();

        Ok(Wsl2Diagnostics {
            is_wsl2: self.is_wsl2,
            wsl_version: self.wsl_version.clone(),
            windows_build: self.windows_build,
            gpu_support,
            driver_path_exists: self.config.driver_path.exists(),
            environment,
            windows_gpu_support: self.check_windows_gpu_support(),
        })
    }
}

/// WSL2 GPU support status
#[derive(Debug, Clone)]
pub enum Wsl2GpuSupport {
    NotWsl2,
    Available {
        cuda: bool,
        opencl: bool,
        directx: bool,
        opengl: bool,
        vulkan: bool,
    },
}

/// WSL2 diagnostics information
#[derive(Debug, Clone)]
pub struct Wsl2Diagnostics {
    pub is_wsl2: bool,
    pub wsl_version: Option<String>,
    pub windows_build: Option<u32>,
    pub gpu_support: Wsl2GpuSupport,
    pub driver_path_exists: bool,
    pub environment: HashMap<String, String>,
    pub windows_gpu_support: bool,
}

impl Default for Wsl2Manager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            config: Wsl2Config::default(),
            is_wsl2: false,
            wsl_version: None,
            windows_build: None,
        })
    }
}

/// Gaming-specific WSL2 optimizations
pub mod gaming {
    use super::*;

    /// Setup WSL2 for gaming workloads
    pub fn setup_gaming_optimizations() -> Result<HashMap<String, String>> {
        info!("Setting up WSL2 gaming optimizations");

        let mut env = HashMap::new();

        // DirectX 12 optimizations for gaming
        env.insert("D3D12_ENABLE_EXPERIMENTAL_FEATURES".to_string(), "1".to_string());
        env.insert("D3D12_GPU_VALIDATION".to_string(), "0".to_string()); // Disable for performance
        env.insert("D3D12_DEBUG".to_string(), "0".to_string()); // Disable for performance

        // OpenGL optimizations
        env.insert("MESA_GL_VERSION_OVERRIDE".to_string(), "4.6".to_string());
        env.insert("MESA_GLSL_VERSION_OVERRIDE".to_string(), "460".to_string());
        env.insert("MESA_SHADER_CACHE_DISABLE".to_string(), "false".to_string());
        env.insert("MESA_OPTIMIZATION_LEVEL".to_string(), "3".to_string()); // Maximum optimization

        Ok(env)
    }

    /// Setup Bolt-specific gaming optimizations for WSL2
    #[cfg(feature = "bolt")]
    pub fn setup_bolt_gaming_optimizations(profile: &str) -> Result<HashMap<String, String>> {
        let mut env = setup_gaming_optimizations()?;

        // Add Bolt-specific gaming optimizations
        env.insert("BOLT_GAMING_MODE".to_string(), "enabled".to_string());
        env.insert("BOLT_GPU_PRIORITY".to_string(), "gaming".to_string());

        match profile {
            "performance" => {
                env.insert("NVIDIA_POWER_MANAGEMENT".to_string(), "performance".to_string());
                env.insert("__GL_SYNC_TO_VBLANK".to_string(), "0".to_string());
                env.insert("BOLT_CAPSULE_PRIORITY".to_string(), "realtime".to_string());
                env.insert("NVIDIA_DLSS_ENABLE".to_string(), "1".to_string());
            }
            "ultra-low-latency" => {
                env.insert("NVIDIA_POWER_MANAGEMENT".to_string(), "performance".to_string());
                env.insert("__GL_SYNC_TO_VBLANK".to_string(), "0".to_string());
                env.insert("NVIDIA_LOW_LATENCY_MODE".to_string(), "ultra".to_string());
                env.insert("BOLT_CAPSULE_PRIORITY".to_string(), "realtime".to_string());
                env.insert("BOLT_QUIC_ACCELERATION".to_string(), "ultra".to_string());
            }
            "balanced" => {
                env.insert("NVIDIA_POWER_MANAGEMENT".to_string(), "adaptive".to_string());
                env.insert("BOLT_CAPSULE_PRIORITY".to_string(), "high".to_string());
            }
            "efficiency" => {
                env.insert("NVIDIA_POWER_MANAGEMENT".to_string(), "adaptive".to_string());
                env.insert("BOLT_POWER_EFFICIENT_MODE".to_string(), "enabled".to_string());
            }
            _ => {}
        }

        // WSL2-specific Bolt gaming optimizations
        if Wsl2Manager::detect_wsl2() {
            env.insert("BOLT_WSL2_GAMING_MODE".to_string(), "enabled".to_string());
            env.insert("WSLG_USE_DXCORE".to_string(), "1".to_string());
            env.insert("LIBGL_ALWAYS_INDIRECT".to_string(), "0".to_string());
        }

        // Vulkan optimizations
        env.insert("VK_LAYER_PATH".to_string(), "".to_string()); // Disable validation layers for performance
        env.insert("RADV_PERFTEST".to_string(), "gpl".to_string()); // Enable graphics pipeline library

        // Gaming-specific optimizations
        env.insert("WINE_CPU_TOPOLOGY".to_string(), "4:2".to_string()); // Optimize for gaming VMs
        env.insert("DXVK_HUD".to_string(), "0".to_string()); // Disable HUD for performance
        env.insert("VKD3D_DEBUG".to_string(), "0".to_string()); // Disable debug for performance

        // Memory optimizations
        env.insert("MESA_HEAP_SIZE".to_string(), "2048".to_string()); // 2GB heap
        env.insert("GALLIUM_THREAD".to_string(), "8".to_string()); // Multi-threading

        Ok(env)
    }

    /// Get gaming-specific container arguments
    pub fn get_gaming_container_args() -> Vec<String> {
        vec![
            "--cap-add".to_string(),
            "SYS_NICE".to_string(), // Allow process priority adjustment
            "--security-opt".to_string(),
            "seccomp=unconfined".to_string(), // Gaming may need broader syscall access
            "--memory".to_string(),
            "8g".to_string(), // Allocate 8GB RAM for gaming
            "--cpus".to_string(),
            "4.0".to_string(), // Allocate 4 CPU cores
        ]
    }
}

/// AI/ML-specific WSL2 optimizations
pub mod ai_ml {
    use super::*;

    /// Setup WSL2 for AI/ML workloads
    pub fn setup_ai_ml_optimizations() -> Result<HashMap<String, String>> {
        info!("Setting up WSL2 AI/ML optimizations");

        let mut env = HashMap::new();

        // CUDA optimizations for AI/ML
        env.insert("CUDA_VISIBLE_DEVICES".to_string(), "all".to_string());
        env.insert("CUDA_DEVICE_ORDER".to_string(), "PCI_BUS_ID".to_string());
        env.insert("CUDA_CACHE_PATH".to_string(), "/tmp/.nv".to_string());

        // Memory management for large models
        env.insert("CUDA_LAUNCH_BLOCKING".to_string(), "0".to_string()); // Async execution
        env.insert("PYTORCH_CUDA_ALLOC_CONF".to_string(), "max_split_size_mb:512".to_string());

        // TensorFlow optimizations
        env.insert("TF_GPU_ALLOCATOR".to_string(), "cuda_malloc_async".to_string());
        env.insert("TF_FORCE_GPU_ALLOW_GROWTH".to_string(), "true".to_string());
        env.insert("TF_ENABLE_ONEDNN_OPTS".to_string(), "1".to_string());

        // OpenCL for alternative compute
        env.insert("OPENCL_VENDOR_PATH".to_string(), "/etc/OpenCL/vendors".to_string());

        // DirectML for Windows ML integration
        env.insert("DIRECTML_DEBUG".to_string(), "0".to_string());

        Ok(env)
    }

    /// Get AI/ML-specific container arguments
    pub fn get_ai_ml_container_args() -> Vec<String> {
        vec![
            "--memory".to_string(),
            "16g".to_string(), // Allocate 16GB RAM for AI/ML
            "--shm-size".to_string(),
            "4g".to_string(), // Shared memory for data loading
            "--ulimit".to_string(),
            "memlock=-1".to_string(), // Unlimited memory locking
            "--ulimit".to_string(),
            "stack=67108864".to_string(), // 64MB stack size
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wsl2_detection() {
        // Test WSL2 detection (will be false in most test environments)
        let is_wsl = Wsl2Manager::detect_wsl2();
        println!("WSL2 detected: {}", is_wsl);
        // Just ensure the function doesn't panic
    }

    #[test]
    fn test_wsl2_config_default() {
        let config = Wsl2Config::default();
        assert!(config.enable_wsl2_optimizations);
        assert!(config.dx12_config.enabled);
        assert!(config.opengl_config.enabled);
        assert!(config.vulkan_config.enabled);
        assert!(!config.preferred_apis.is_empty());
    }

    #[test]
    fn test_wsl2_manager_creation() {
        let result = Wsl2Manager::new();
        assert!(result.is_ok());

        let manager = result.unwrap();
        // Verify manager is created without panicking
        println!("WSL2 manager created successfully");
    }

    #[test]
    fn test_gaming_optimizations() {
        let env = gaming::setup_gaming_optimizations().unwrap();
        assert!(!env.is_empty());
        assert_eq!(env.get("MESA_OPTIMIZATION_LEVEL"), Some(&"3".to_string()));

        let args = gaming::get_gaming_container_args();
        assert!(args.contains(&"--cap-add".to_string()));
        assert!(args.contains(&"SYS_NICE".to_string()));
    }

    #[test]
    fn test_ai_ml_optimizations() {
        let env = ai_ml::setup_ai_ml_optimizations().unwrap();
        assert!(!env.is_empty());
        assert_eq!(env.get("CUDA_VISIBLE_DEVICES"), Some(&"all".to_string()));

        let args = ai_ml::get_ai_ml_container_args();
        assert!(args.contains(&"--memory".to_string()));
        assert!(args.contains(&"16g".to_string()));
    }

    #[test]
    fn test_dx12_config() {
        let config = Dx12Config {
            enabled: true,
            debug_layer: true,
            gpu_validation: true,
            dred_enabled: true,
        };

        assert!(config.enabled);
        assert!(config.debug_layer);
        assert!(config.gpu_validation);
        assert!(config.dred_enabled);
    }

    #[test]
    fn test_vulkan_config() {
        let config = VulkanConfig {
            enabled: true,
            validation_layers: true,
            api_dump: false,
            instance_extensions: vec!["VK_KHR_surface".to_string()],
            device_extensions: vec!["VK_KHR_swapchain".to_string()],
        };

        assert!(config.enabled);
        assert!(config.validation_layers);
        assert!(!config.api_dump);
        assert_eq!(config.instance_extensions.len(), 1);
        assert_eq!(config.device_extensions.len(), 1);
    }
}