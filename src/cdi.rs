use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// CDI (Container Device Interface) specification version
const CDI_VERSION: &str = "0.8.0";

/// Default CDI spec directory
const CDI_SPEC_DIR: &str = "/etc/cdi";

/// Alternative CDI spec directories
const CDI_SPEC_DIRS: &[&str] = &["/etc/cdi", "/var/run/cdi", "/usr/local/etc/cdi"];

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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

impl Default for CdiSpec {
    fn default() -> Self {
        Self {
            cdi_version: CDI_VERSION.to_string(),
            kind: "nvidia.com/gpu".to_string(),
            container_edits: ContainerEdits::default(),
            devices: Vec::new(),
        }
    }
}

/// Runtime optimization settings
#[derive(Debug, Clone)]
pub struct RuntimeOptimizations {
    /// Enable Docker-specific optimizations
    pub docker: bool,
    /// Enable Podman-specific optimizations
    pub podman: bool,
    /// Use nvidia-ctk for hooks (default: true)
    pub use_nvidia_ctk: bool,
    /// Additional library paths to scan
    pub extra_lib_paths: Vec<String>,
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
        let spec_dirs = CDI_SPEC_DIRS.iter().map(PathBuf::from).collect();

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

        let entries = fs::read_dir(dir).context(format!(
            "Failed to read CDI spec directory: {}",
            dir.display()
        ))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().is_some_and(|ext| ext == "json") {
                match self.load_spec_file(&path) {
                    Ok(spec) => {
                        if let Err(e) = self.register_spec(spec) {
                            warn!("Failed to register CDI spec from {}: {}", path.display(), e);
                        }
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
        for kind in self.specs.keys() {
            let kind_devices = self.get_devices_by_kind(kind);
            for device in kind_devices {
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
        fs::create_dir_all(spec_dir).context(format!(
            "Failed to create CDI spec directory: {}",
            spec_dir.display()
        ))?;

        // Generate filename from kind
        let filename = format!("{}.json", spec.kind.replace('/', "_"));
        let spec_path = spec_dir.join(filename);

        let content =
            serde_json::to_string_pretty(spec).context("Failed to serialize CDI specification")?;

        fs::write(&spec_path, content).context(format!(
            "Failed to write CDI spec file: {}",
            spec_path.display()
        ))?;

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
        "NVIDIA_VISIBLE_DEVICES=void".to_string(),
        "NVIDIA_DRIVER_CAPABILITIES=all".to_string(),
        format!("NVIDIA_DRIVER_VERSION={}", driver_info.version),
    ];

    // Add CUDA version if available
    if let Some(cuda_version) = &driver_info.cuda_version {
        env_vars.push(format!("CUDA_VERSION={cuda_version}"));
        env_vars.push(format!("NVIDIA_REQUIRE_CUDA=cuda>={cuda_version}"));
    }

    // Add driver type information
    env_vars.push(format!("NVIDIA_DRIVER_TYPE={:?}", driver_info.driver_type));

    // Create hooks for CDI
    let mut hooks = Vec::new();

    // Add create-symlinks hook for library compatibility
    // Creates symlinks like libcuda.so.1 -> libcuda.so for compatibility
    let mut symlink_args = vec![
        "nvidia-ctk".to_string(),
        "hook".to_string(),
        "create-symlinks".to_string(),
    ];

    // Add common library symlinks
    for link in &[
        "libcuda.so.1::/usr/lib64/libcuda.so",
        "libnvidia-ml.so.1::/usr/lib64/libnvidia-ml.so",
        "libnvidia-opticalflow.so.1::/usr/lib64/libnvidia-opticalflow.so",
        "libnvidia-encode.so.1::/usr/lib64/libnvidia-encode.so",
        "libnvidia-fbc.so.1::/usr/lib64/libnvidia-fbc.so",
    ] {
        symlink_args.push("--link".to_string());
        symlink_args.push(link.to_string());
    }

    hooks.push(Hook {
        hook_name: "createContainer".to_string(),
        path: "/usr/bin/nvidia-ctk".to_string(),
        args: Some(symlink_args),
        env: None,
        timeout: Some(5),
    });

    // Add update-ldcache hook to ensure libraries are discoverable
    hooks.push(Hook {
        hook_name: "createContainer".to_string(),
        path: "/usr/bin/nvidia-ctk".to_string(),
        args: Some(vec![
            "nvidia-ctk".to_string(),
            "hook".to_string(),
            "update-ldcache".to_string(),
            "--folder".to_string(),
            "/usr/lib64:/lib64".to_string(),
        ]),
        env: None,
        timeout: Some(5),
    });

    // Add disable-device-node-modification hook
    // Prevents NVML from creating device nodes inside containers
    hooks.push(Hook {
        hook_name: "createContainer".to_string(),
        path: "/usr/bin/nvidia-ctk".to_string(),
        args: Some(vec![
            "nvidia-ctk".to_string(),
            "hook".to_string(),
            "disable-device-node-modification".to_string(),
        ]),
        env: None,
        timeout: Some(5),
    });

    let mut container_edits = ContainerEdits {
        env: Some(env_vars),
        device_nodes: Some(Vec::new()),
        mounts: Some(Vec::new()),
        hooks: Some(hooks),
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
                    options: Some(vec![
                        "ro".to_string(),
                        "nosuid".to_string(),
                        "nodev".to_string(),
                        "bind".to_string(),
                    ]),
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
                if let (Some(memory), Some(env)) = (gpu.memory, &mut device_edits.env) {
                    env.push(format!("GPU_MEMORY_SIZE={memory}"));
                    let memory_mb = memory / 1024 / 1024;
                    env.push(format!("GPU_MEMORY_SIZE_MB={memory_mb}"));
                }

                if let (Some(driver_version), Some(env)) =
                    (&gpu.driver_version, &mut device_edits.env)
                {
                    env.push(format!("GPU_DRIVER_VERSION={driver_version}"));
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
        warn!(
            "CDI version mismatch: spec={}, expected={}",
            spec.cdi_version, CDI_VERSION
        );
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
        return Err(anyhow::anyhow!("Invalid device path: {device_path}"));
    }

    // Check if device exists before getting metadata
    if !path.exists() {
        warn!("Device does not exist: {device_path}");
        return Err(anyhow::anyhow!("Device does not exist: {device_path}"));
    }

    // Get device information using stat
    let metadata = fs::metadata(path)
        .with_context(|| format!("Failed to get metadata for device: {device_path}"))?;

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
        Err(anyhow::anyhow!("CDI device not found: {device_name}"))
    }
}

/// Bolt-specific CDI enhancements
#[cfg(feature = "bolt")]
pub mod bolt;

#[cfg(not(feature = "bolt"))]
pub mod bolt {
    // Empty module when bolt feature is not enabled
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

        let json = serde_json::to_string_pretty(&spec).expect("Serialization should work in tests");
        let parsed: CdiSpec =
            serde_json::from_str(&json).expect("Deserialization should work in tests");

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

        registry
            .register_spec(spec)
            .expect("Spec registration should work in tests");

        let devices = registry.list_devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0], "test.com/gpu=test-device");

        let device = registry.get_device("test.com/gpu=test-device");
        assert!(device.is_some());
        assert_eq!(device.expect("Device should be found").name, "test-device");
    }

    #[test]
    fn test_device_node_creation() {
        // Test with /dev/null as it should exist on most systems
        if Path::new("/dev/null").exists() {
            let device_node =
                create_device_node("/dev/null").expect("Device node creation should work");
            assert_eq!(device_node.path, "/dev/null");
            assert_eq!(device_node.device_type, Some("c".to_string()));
        }
    }

    #[test]
    fn test_cdi_spec_save_load() {
        let temp_dir = TempDir::new().expect("Temp dir creation should work");
        let registry = CdiRegistry::new();

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
        let spec_path = registry
            .save_spec(&spec, Some(temp_dir.path()))
            .expect("Spec saving should work");
        assert!(spec_path.exists());

        // Load it back
        let loaded_spec = registry
            .load_spec_file(&spec_path)
            .expect("Spec loading should work");
        assert_eq!(spec.kind, loaded_spec.kind);
        assert_eq!(spec.cdi_version, loaded_spec.cdi_version);
    }

    #[tokio::test]
    async fn test_cdi_utility_functions() {
        // Test listing CDI devices
        let devices = list_cdi_devices().unwrap_or_default();
        println!("Found {} CDI devices", devices.len());

        // Test generating and saving NVIDIA CDI
        if let Ok(_path) = generate_and_save_nvidia_cdi().await {
            println!("Successfully generated NVIDIA CDI spec");
        }

        // Test applying CDI device (with fallback)
        if !devices.is_empty() {
            if let Ok(_edits) = apply_cdi_device(&devices[0]) {
                println!("Successfully applied CDI device");
            }
        }
    }
}
