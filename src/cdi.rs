use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// CDI (Container Device Interface) specification version
const CDI_VERSION: &str = "0.6.0";

/// Default CDI spec directory
const CDI_SPEC_DIR: &str = "/etc/cdi";

/// Alternative CDI spec directories
const CDI_SPEC_DIRS: &[&str] = &[
    "/etc/cdi",
    "/var/run/cdi",
    "/usr/local/etc/cdi",
];

/// CDI specification structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdiSpec {
    #[serde(rename = "cdiVersion")]
    pub cdi_version: String,
    pub kind: String,
    #[serde(rename = "containerEdits")]
    pub container_edits: ContainerEdits,
    pub devices: Vec<CdiDevice>,
}

/// Container edits for CDI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerEdits {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<Vec<String>>,
    #[serde(rename = "deviceNodes", skip_serializing_if = "Option::is_none")]
    pub device_nodes: Option<Vec<DeviceNode>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mounts: Option<Vec<Mount>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hooks: Option<Vec<Hook>>,
}

/// CDI device specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdiDevice {
    pub name: String,
    #[serde(rename = "containerEdits")]
    pub container_edits: ContainerEdits,
}

/// Device node specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceNode {
    pub path: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub device_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub major: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minor: Option<u32>,
    #[serde(rename = "fileMode", skip_serializing_if = "Option::is_none")]
    pub file_mode: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gid: Option<u32>,
}

/// Mount specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mount {
    #[serde(rename = "hostPath")]
    pub host_path: String,
    #[serde(rename = "containerPath")]
    pub container_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Vec<String>>,
}

/// Hook specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hook {
    #[serde(rename = "hookName")]
    pub hook_name: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u32>,
}

/// CDI registry for managing device specifications
pub struct CdiRegistry {
    spec_dirs: Vec<PathBuf>,
    specs: HashMap<String, CdiSpec>,
}

impl Default for CdiRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CdiRegistry {
    /// Create a new CDI registry
    pub fn new() -> Self {
        let spec_dirs = CDI_SPEC_DIRS
            .iter()
            .map(|dir| PathBuf::from(dir))
            .collect();

        Self {
            spec_dirs,
            specs: HashMap::new(),
        }
    }

    /// Load all CDI specifications from the spec directories
    pub fn load_specs(&mut self) -> Result<()> {
        self.specs.clear();

        let spec_dirs = self.spec_dirs.clone();
        for spec_dir in &spec_dirs {
            if spec_dir.exists() {
                self.load_specs_from_dir(spec_dir)?;
            }
        }

        info!("Loaded {} CDI specifications", self.specs.len());
        Ok(())
    }

    /// Load CDI specifications from a specific directory
    fn load_specs_from_dir(&mut self, dir: &Path) -> Result<()> {
        debug!("Loading CDI specs from: {}", dir.display());

        let entries = fs::read_dir(dir)
            .context(format!("Failed to read CDI spec directory: {}", dir.display()))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                match self.load_spec_file(&path) {
                    Ok(spec) => {
                        self.specs.insert(spec.kind.clone(), spec);
                    }
                    Err(e) => {
                        warn!("Failed to load CDI spec from {}: {}", path.display(), e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Load a single CDI specification file
    pub fn load_spec_file(&self, path: &Path) -> Result<CdiSpec> {
        debug!("Loading CDI spec file: {}", path.display());

        let content = fs::read_to_string(path)
            .context(format!("Failed to read CDI spec file: {}", path.display()))?;

        let spec: CdiSpec = serde_json::from_str(&content)
            .context(format!("Failed to parse CDI spec file: {}", path.display()))?;

        Ok(spec)
    }

    /// Get a device by its CDI name
    pub fn get_device(&self, device_name: &str) -> Option<&CdiDevice> {
        // Parse CDI device name format: vendor/class=device
        if let Some((kind, name)) = device_name.split_once('=') {
            if let Some(spec) = self.specs.get(kind) {
                return spec.devices.iter().find(|device| device.name == name);
            }
        }
        None
    }

    /// Get all available devices for a specific vendor/class
    pub fn get_devices_by_kind(&self, kind: &str) -> Vec<&CdiDevice> {
        if let Some(spec) = self.specs.get(kind) {
            spec.devices.iter().collect()
        } else {
            Vec::new()
        }
    }

    /// List all available CDI devices
    pub fn list_devices(&self) -> Vec<String> {
        let mut devices = Vec::new();
        for (kind, spec) in &self.specs {
            for device in &spec.devices {
                devices.push(format!("{}={}", kind, device.name));
            }
        }
        devices.sort();
        devices
    }

    /// Register a new CDI specification
    pub fn register_spec(&mut self, spec: CdiSpec) -> Result<()> {
        debug!("Registering CDI spec: {}", spec.kind);
        self.specs.insert(spec.kind.clone(), spec);
        Ok(())
    }

    /// Save a CDI specification to file
    pub fn save_spec(&self, spec: &CdiSpec, output_dir: Option<&Path>) -> Result<PathBuf> {
        let spec_dir = output_dir.unwrap_or_else(|| Path::new(CDI_SPEC_DIR));

        // Ensure the directory exists
        fs::create_dir_all(spec_dir)
            .context(format!("Failed to create CDI spec directory: {}", spec_dir.display()))?;

        // Generate filename from kind
        let filename = format!("{}.json", spec.kind.replace('/', "_"));
        let spec_path = spec_dir.join(filename);

        let content = serde_json::to_string_pretty(spec)
            .context("Failed to serialize CDI specification")?;

        fs::write(&spec_path, content)
            .context(format!("Failed to write CDI spec file: {}", spec_path.display()))?;

        info!("Saved CDI spec to: {}", spec_path.display());
        Ok(spec_path)
    }
}

/// Generate CDI specification for NVIDIA GPUs
pub async fn generate_nvidia_cdi_spec() -> Result<CdiSpec> {
    info!("Generating NVIDIA CDI specification");

    let gpus = crate::gpu::discover_gpus().await?;
    let driver_info = crate::gpu::get_driver_info().await?;

    let mut devices = Vec::new();
    let mut env_vars = vec![
        "NVIDIA_DRIVER_CAPABILITIES=all".to_string(),
        format!("NVIDIA_DRIVER_VERSION={}", driver_info.version),
    ];

    // Add CUDA version if available
    if let Some(cuda_version) = &driver_info.cuda_version {
        env_vars.push(format!("CUDA_VERSION={}", cuda_version));
        env_vars.push(format!("NVIDIA_REQUIRE_CUDA=cuda>={}", cuda_version));
    }

    // Add driver type information
    env_vars.push(format!("NVIDIA_DRIVER_TYPE={:?}", driver_info.driver_type));

    let mut container_edits = ContainerEdits {
        env: Some(env_vars),
        device_nodes: Some(Vec::new()),
        mounts: Some(Vec::new()),
        hooks: None,
    };

    // Add control devices to container edits with error handling
    let control_devices = vec!["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-modeset"];
    for device_path in control_devices {
        if Path::new(device_path).exists() {
            match create_device_node(device_path) {
                Ok(device_node) => {
                    if let Some(device_nodes) = &mut container_edits.device_nodes {
                        device_nodes.push(device_node);
                    }
                }
                Err(e) => {
                    warn!("Failed to create device node for {}: {}", device_path, e);
                    // Continue with other devices instead of failing completely
                }
            }
        } else {
            debug!("Control device not found: {}", device_path);
        }
    }

    // Add library mounts to container edits with validation
    for lib_path in &driver_info.libraries {
        // Validate library path exists and is readable
        if Path::new(lib_path).exists() {
            if let Some(mounts) = &mut container_edits.mounts {
                mounts.push(Mount {
                    host_path: lib_path.clone(),
                    container_path: lib_path.clone(),
                    options: Some(vec!["ro".to_string(), "nosuid".to_string(), "nodev".to_string()]),
                });
            }
        } else {
            warn!("Library path does not exist: {}", lib_path);
        }
    }

    // Create individual GPU devices with error handling
    for gpu in &gpus {
        match create_device_node(&gpu.device_path) {
            Ok(device_node) => {
                let mut device_edits = ContainerEdits {
                    env: Some(vec![
                        format!("NVIDIA_VISIBLE_DEVICES={}", gpu.id),
                        format!("GPU_DEVICE_ORDINAL={}", gpu.id),
                        format!("GPU_NAME={}", gpu.name),
                        format!("GPU_PCI_ADDRESS={}", gpu.pci_address),
                    ]),
                    device_nodes: Some(vec![device_node]),
                    mounts: None,
                    hooks: None,
                };

                // Add GPU-specific environment variables
                if let Some(memory) = gpu.memory {
                    if let Some(env) = &mut device_edits.env {
                        env.push(format!("GPU_MEMORY_SIZE={}", memory));
                        env.push(format!("GPU_MEMORY_SIZE_MB={}", memory / 1024 / 1024));
                    }
                }

                if let Some(driver_version) = &gpu.driver_version {
                    if let Some(env) = &mut device_edits.env {
                        env.push(format!("GPU_DRIVER_VERSION={}", driver_version));
                    }
                }

                devices.push(CdiDevice {
                    name: format!("gpu{}", gpu.id),
                    container_edits: device_edits,
                });
            }
            Err(e) => {
                warn!("Failed to create device node for GPU {}: {}", gpu.id, e);
                // Continue with other GPUs
            }
        }
    }

    // Create an "all" device that includes all GPUs with proper error handling
    if !gpus.is_empty() {
        let all_gpu_ids: Vec<String> = gpus.iter().map(|gpu| gpu.id.clone()).collect();
        let mut all_device_edits = ContainerEdits {
            env: Some(vec![
                "NVIDIA_VISIBLE_DEVICES=all".to_string(),
                format!("GPU_COUNT={}", gpus.len()),
                format!("GPU_IDS={}", all_gpu_ids.join(",")),
            ]),
            device_nodes: Some(Vec::new()),
            mounts: None,
            hooks: None,
        };

        // Add all GPU device nodes with error handling
        for gpu in &gpus {
            match create_device_node(&gpu.device_path) {
                Ok(device_node) => {
                    if let Some(device_nodes) = &mut all_device_edits.device_nodes {
                        device_nodes.push(device_node);
                    }
                }
                Err(e) => {
                    warn!("Failed to add GPU {} to 'all' device: {}", gpu.id, e);
                    // Continue with other GPUs for the 'all' device
                }
            }
        }

        // Only create "all" device if we have at least one valid GPU device
        if let Some(device_nodes) = &all_device_edits.device_nodes {
            if !device_nodes.is_empty() {
                devices.push(CdiDevice {
                    name: "all".to_string(),
                    container_edits: all_device_edits,
                });
            } else {
                warn!("No valid GPU devices found for 'all' device");
            }
        }
    }

    let spec = CdiSpec {
        cdi_version: CDI_VERSION.to_string(),
        kind: "nvidia.com/gpu".to_string(),
        container_edits,
        devices,
    };

    // Validate the generated spec before returning
    validate_cdi_spec(&spec)?;

    info!("Generated CDI spec with {} devices", spec.devices.len());
    Ok(spec)
}

/// Validate a CDI specification for correctness
fn validate_cdi_spec(spec: &CdiSpec) -> Result<()> {
    // Check CDI version
    if spec.cdi_version != CDI_VERSION {
        warn!("CDI version mismatch: spec={}, expected={}", spec.cdi_version, CDI_VERSION);
    }

    // Check kind format
    if !spec.kind.contains('/') {
        return Err(anyhow::anyhow!("Invalid CDI kind format: {}", spec.kind));
    }

    // Validate devices
    if spec.devices.is_empty() {
        return Err(anyhow::anyhow!("CDI spec must contain at least one device"));
    }

    // Check for duplicate device names
    let mut device_names = std::collections::HashSet::new();
    for device in &spec.devices {
        if device.name.is_empty() {
            return Err(anyhow::anyhow!("Device name cannot be empty"));
        }
        if !device_names.insert(&device.name) {
            return Err(anyhow::anyhow!("Duplicate device name: {}", device.name));
        }
    }

    // Validate device nodes
    for device in &spec.devices {
        if let Some(device_nodes) = &device.container_edits.device_nodes {
            for node in device_nodes {
                if !node.path.starts_with("/dev/") {
                    return Err(anyhow::anyhow!("Invalid device node path: {}", node.path));
                }
            }
        }
    }

    info!("CDI spec validation passed");
    Ok(())
}

/// Create a device node specification from a device path
fn create_device_node(device_path: &str) -> Result<DeviceNode> {
    let path = Path::new(device_path);

    // Validate device path for security
    if !device_path.starts_with("/dev/") {
        return Err(anyhow::anyhow!("Invalid device path: {}", device_path));
    }

    // Check if device exists before getting metadata
    if !path.exists() {
        warn!("Device does not exist: {}", device_path);
        return Err(anyhow::anyhow!("Device does not exist: {}", device_path));
    }

    // Get device information using stat
    let metadata = fs::metadata(path)
        .context(format!("Failed to get metadata for device: {}", device_path))?;

    // Extract device numbers on Unix systems with correct calculation
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        let dev = metadata.rdev();
        // Use correct Linux device number extraction
        let major = (dev >> 8) & 0xff;
        let minor = (dev & 0xff) | ((dev >> 12) & 0xfff00);

        Ok(DeviceNode {
            path: device_path.to_string(),
            device_type: Some("c".to_string()), // character device
            major: Some(major as u32),
            minor: Some(minor as u32),
            file_mode: Some(0o666), // rw-rw-rw-
            uid: None,
            gid: None,
        })
    }

    #[cfg(not(unix))]
    {
        Ok(DeviceNode {
            path: device_path.to_string(),
            device_type: Some("c".to_string()),
            major: None,
            minor: None,
            file_mode: Some(0o666),
            uid: None,
            gid: None,
        })
    }
}

/// Generate and save NVIDIA CDI specification
pub async fn generate_and_save_nvidia_cdi() -> Result<PathBuf> {
    let spec = generate_nvidia_cdi_spec().await?;
    let registry = CdiRegistry::new();
    registry.save_spec(&spec, None)
}

/// List all available CDI devices
pub fn list_cdi_devices() -> Result<Vec<String>> {
    let mut registry = CdiRegistry::new();
    registry.load_specs()?;
    Ok(registry.list_devices())
}

/// Apply CDI device to container configuration
pub fn apply_cdi_device(device_name: &str) -> Result<ContainerEdits> {
    let mut registry = CdiRegistry::new();
    registry.load_specs()?;

    if let Some(device) = registry.get_device(device_name) {
        Ok(device.container_edits.clone())
    } else {
        Err(anyhow::anyhow!("CDI device not found: {}", device_name))
    }
}

/// Bolt-specific CDI enhancements
pub mod bolt {
    use super::*;

    /// Bolt capsule-specific configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BoltCapsuleConfig {
        /// Enable snapshot/restore support for GPU state
        pub snapshot_support: bool,
        /// GPU isolation level for Bolt capsules
        pub isolation_level: BoltGpuIsolation,
        /// Gaming-specific optimizations
        pub gaming_optimizations: Option<BoltGamingConfig>,
        /// WSL2 compatibility mode
        pub wsl2_mode: bool,
    }

    /// GPU isolation levels for Bolt capsules
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum BoltGpuIsolation {
        /// Shared GPU access with other containers
        Shared,
        /// Exclusive GPU access for this capsule
        Exclusive,
        /// Virtual GPU with resource limits
        Virtual { memory_limit: String, compute_limit: String },
    }

    /// Gaming-specific configuration for Bolt
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BoltGamingConfig {
        /// Enable DLSS support
        pub dlss_enabled: bool,
        /// Ray tracing acceleration
        pub rt_cores_enabled: bool,
        /// Gaming performance profile
        pub performance_profile: GamingProfile,
        /// Wine/Proton compatibility optimizations
        pub wine_optimizations: bool,
    }

    /// Gaming performance profiles
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum GamingProfile {
        /// Maximum performance, highest power consumption
        Performance,
        /// Balanced performance and efficiency
        Balanced,
        /// Power-efficient gaming
        Efficiency,
        /// Ultra-low latency for competitive gaming
        UltraLowLatency,
    }

    /// Generate Bolt-optimized CDI specification for NVIDIA GPUs
    pub async fn generate_bolt_nvidia_cdi_spec(capsule_config: BoltCapsuleConfig) -> Result<CdiSpec> {
        info!("Generating Bolt-optimized NVIDIA CDI specification");

        let mut base_spec = generate_nvidia_cdi_spec().await?;

        // Enhance the base spec with Bolt-specific features
        base_spec.kind = "nvidia.com/gpu-bolt".to_string();

        // Add Bolt-specific environment variables
        if let Some(env) = &mut base_spec.container_edits.env {
            env.push("BOLT_GPU_RUNTIME=nvbind".to_string());
            env.push(format!("BOLT_GPU_ISOLATION={}",
                match capsule_config.isolation_level {
                    BoltGpuIsolation::Shared => "shared",
                    BoltGpuIsolation::Exclusive => "exclusive",
                    BoltGpuIsolation::Virtual { .. } => "virtual",
                }
            ));

            if capsule_config.snapshot_support {
                env.push("BOLT_CAPSULE_SNAPSHOT_GPU=enabled".to_string());
            }

            if capsule_config.wsl2_mode {
                env.push("BOLT_WSL2_MODE=enabled".to_string());
                env.push("BOLT_WSL2_GPU_ACCELERATION=enabled".to_string());
            }
        }

        // Add gaming-specific configuration
        if let Some(gaming_config) = &capsule_config.gaming_optimizations {
            if let Some(env) = &mut base_spec.container_edits.env {
                if gaming_config.dlss_enabled {
                    env.push("NVIDIA_DLSS_ENABLE=1".to_string());
                    env.push("BOLT_DLSS_OPTIMIZATION=enabled".to_string());
                }

                if gaming_config.rt_cores_enabled {
                    env.push("NVIDIA_RT_CORES_ENABLE=1".to_string());
                    env.push("BOLT_RT_ACCELERATION=enabled".to_string());
                }

                env.push(format!("BOLT_GAMING_PROFILE={}",
                    match gaming_config.performance_profile {
                        GamingProfile::Performance => "performance",
                        GamingProfile::Balanced => "balanced",
                        GamingProfile::Efficiency => "efficiency",
                        GamingProfile::UltraLowLatency => "ultra-low-latency",
                    }
                ));

                if gaming_config.wine_optimizations {
                    env.push("BOLT_WINE_GPU_OPTIMIZATION=enabled".to_string());
                    env.push("DXVK_HUD=1".to_string());
                    env.push("VKD3D_CONFIG=dxr".to_string());
                }
            }
        }

        // Add Bolt-specific hooks for capsule lifecycle
        let bolt_hooks = vec![
            Hook {
                hook_name: "prestart".to_string(),
                path: "/usr/local/bin/bolt-gpu-prestart".to_string(),
                args: Some(vec!["--capsule-isolation".to_string()]),
                env: None,
                timeout: Some(30),
            },
            Hook {
                hook_name: "poststop".to_string(),
                path: "/usr/local/bin/bolt-gpu-cleanup".to_string(),
                args: Some(vec!["--release-gpu".to_string()]),
                env: None,
                timeout: Some(10),
            },
        ];

        // Add snapshot hooks if enabled
        let mut hooks = bolt_hooks;
        if capsule_config.snapshot_support {
            hooks.extend(vec![
                Hook {
                    hook_name: "presnapshot".to_string(),
                    path: "/usr/local/bin/bolt-gpu-snapshot-prepare".to_string(),
                    args: Some(vec!["--save-gpu-state".to_string()]),
                    env: None,
                    timeout: Some(60),
                },
                Hook {
                    hook_name: "postrestore".to_string(),
                    path: "/usr/local/bin/bolt-gpu-snapshot-restore".to_string(),
                    args: Some(vec!["--restore-gpu-state".to_string()]),
                    env: None,
                    timeout: Some(60),
                },
            ]);
        }

        base_spec.container_edits.hooks = Some(hooks);

        // Enhance individual device configurations for Bolt
        for device in &mut base_spec.devices {
            if let Some(env) = &mut device.container_edits.env {
                env.push("BOLT_DEVICE_TYPE=gpu".to_string());

                // Add virtual GPU configuration if specified
                if let BoltGpuIsolation::Virtual { memory_limit, compute_limit } = &capsule_config.isolation_level {
                    env.push(format!("BOLT_GPU_MEMORY_LIMIT={}", memory_limit));
                    env.push(format!("BOLT_GPU_COMPUTE_LIMIT={}", compute_limit));
                }
            }
        }

        Ok(base_spec)
    }

    /// Generate Bolt CDI specification with default gaming configuration
    pub async fn generate_bolt_gaming_cdi_spec() -> Result<CdiSpec> {
        let gaming_config = BoltCapsuleConfig {
            snapshot_support: true,
            isolation_level: BoltGpuIsolation::Exclusive,
            gaming_optimizations: Some(BoltGamingConfig {
                dlss_enabled: true,
                rt_cores_enabled: true,
                performance_profile: GamingProfile::Performance,
                wine_optimizations: true,
            }),
            wsl2_mode: crate::wsl2::Wsl2Manager::detect_wsl2(),
        };

        generate_bolt_nvidia_cdi_spec(gaming_config).await
    }

    /// Generate Bolt CDI specification with AI/ML optimizations
    pub async fn generate_bolt_aiml_cdi_spec() -> Result<CdiSpec> {
        let aiml_config = BoltCapsuleConfig {
            snapshot_support: true,
            isolation_level: BoltGpuIsolation::Virtual {
                memory_limit: "16GB".to_string(),
                compute_limit: "80%".to_string(),
            },
            gaming_optimizations: None,
            wsl2_mode: crate::wsl2::Wsl2Manager::detect_wsl2(),
        };

        let mut spec = generate_bolt_nvidia_cdi_spec(aiml_config).await?;

        // Add AI/ML specific environment variables
        if let Some(env) = &mut spec.container_edits.env {
            env.push("BOLT_WORKLOAD_TYPE=ai-ml".to_string());
            env.push("CUDA_CACHE_DISABLE=0".to_string());
            env.push("CUDA_CACHE_MAXSIZE=2147483648".to_string()); // 2GB CUDA cache
            env.push("NVIDIA_TF32_OVERRIDE=1".to_string());
        }

        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cdi_spec_serialization() {
        let spec = CdiSpec {
            cdi_version: CDI_VERSION.to_string(),
            kind: "test.com/gpu".to_string(),
            container_edits: ContainerEdits {
                env: Some(vec!["TEST_ENV=value".to_string()]),
                device_nodes: None,
                mounts: None,
                hooks: None,
            },
            devices: vec![CdiDevice {
                name: "test-device".to_string(),
                container_edits: ContainerEdits {
                    env: Some(vec!["DEVICE_ENV=test".to_string()]),
                    device_nodes: None,
                    mounts: None,
                    hooks: None,
                },
            }],
        };

        let json = serde_json::to_string_pretty(&spec).unwrap();
        let parsed: CdiSpec = serde_json::from_str(&json).unwrap();

        assert_eq!(spec.cdi_version, parsed.cdi_version);
        assert_eq!(spec.kind, parsed.kind);
        assert_eq!(spec.devices.len(), parsed.devices.len());
    }

    #[test]
    fn test_cdi_registry() {
        let mut registry = CdiRegistry::new();

        let spec = CdiSpec {
            cdi_version: CDI_VERSION.to_string(),
            kind: "test.com/gpu".to_string(),
            container_edits: ContainerEdits {
                env: None,
                device_nodes: None,
                mounts: None,
                hooks: None,
            },
            devices: vec![CdiDevice {
                name: "test-device".to_string(),
                container_edits: ContainerEdits {
                    env: None,
                    device_nodes: None,
                    mounts: None,
                    hooks: None,
                },
            }],
        };

        registry.register_spec(spec).unwrap();

        let devices = registry.list_devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0], "test.com/gpu=test-device");

        let device = registry.get_device("test.com/gpu=test-device");
        assert!(device.is_some());
        assert_eq!(device.unwrap().name, "test-device");
    }

    #[test]
    fn test_device_node_creation() {
        // Test with /dev/null as it should exist on most systems
        if Path::new("/dev/null").exists() {
            let device_node = create_device_node("/dev/null").unwrap();
            assert_eq!(device_node.path, "/dev/null");
            assert_eq!(device_node.device_type, Some("c".to_string()));
        }
    }

    #[test]
    fn test_cdi_spec_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let mut registry = CdiRegistry::new();

        let spec = CdiSpec {
            cdi_version: CDI_VERSION.to_string(),
            kind: "test.com/device".to_string(),
            container_edits: ContainerEdits {
                env: Some(vec!["TEST=value".to_string()]),
                device_nodes: None,
                mounts: None,
                hooks: None,
            },
            devices: vec![],
        };

        // Save the spec
        let spec_path = registry.save_spec(&spec, Some(temp_dir.path())).unwrap();
        assert!(spec_path.exists());

        // Load it back
        let loaded_spec = registry.load_spec_file(&spec_path).unwrap();
        assert_eq!(spec.kind, loaded_spec.kind);
        assert_eq!(spec.cdi_version, loaded_spec.cdi_version);
    }
}