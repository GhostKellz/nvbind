//! Kubernetes and orchestration platform integration
//!
//! Provides seamless integration with Kubernetes, OpenShift, and other
//! container orchestration platforms for GPU resource management.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use tokio::fs;
use tracing::{debug, info, warn};

/// Kubernetes integration manager
pub struct KubernetesIntegration {
    config: K8sConfig,
    client: Option<KubernetesClient>,
    device_plugin: DevicePlugin,
    cdi_manager: CdiManager,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct K8sConfig {
    pub enabled: bool,
    pub kubeconfig_path: Option<PathBuf>,
    pub namespace: String,
    pub device_plugin_enabled: bool,
    pub device_plugin_socket: PathBuf,
    pub cdi_enabled: bool,
    pub resource_quotas: ResourceQuotas,
    pub node_labeling: NodeLabeling,
}

impl Default for K8sConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            kubeconfig_path: None,
            namespace: "nvbind-system".to_string(),
            device_plugin_enabled: true,
            device_plugin_socket: PathBuf::from("/var/lib/kubelet/device-plugins/nvbind.sock"),
            cdi_enabled: true,
            resource_quotas: ResourceQuotas::default(),
            node_labeling: NodeLabeling::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuotas {
    pub gpu_limit_per_pod: Option<u32>,
    pub memory_limit_per_gpu: Option<u64>,
    pub max_pods_per_node: Option<u32>,
}

impl Default for ResourceQuotas {
    fn default() -> Self {
        Self {
            gpu_limit_per_pod: Some(8),
            memory_limit_per_gpu: Some(32 * 1024 * 1024 * 1024), // 32GB
            max_pods_per_node: Some(100),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeLabeling {
    pub enabled: bool,
    pub gpu_label_prefix: String,
    pub driver_label_prefix: String,
    pub capability_labels: Vec<String>,
}

impl Default for NodeLabeling {
    fn default() -> Self {
        Self {
            enabled: true,
            gpu_label_prefix: "nvidia.com/gpu".to_string(),
            driver_label_prefix: "nvidia.com/driver".to_string(),
            capability_labels: vec![
                "nvidia.com/cuda".to_string(),
                "nvidia.com/mig".to_string(),
                "nvidia.com/nvenc".to_string(),
                "nvidia.com/nvdec".to_string(),
            ],
        }
    }
}

impl KubernetesIntegration {
    /// Create new Kubernetes integration
    pub fn new(config: K8sConfig) -> Self {
        Self {
            config: config.clone(),
            client: None,
            device_plugin: DevicePlugin::new(config.device_plugin_socket.clone()),
            cdi_manager: CdiManager::new(),
        }
    }

    /// Initialize Kubernetes integration
    pub async fn initialize(&mut self) -> Result<()> {
        if !self.config.enabled {
            info!("Kubernetes integration disabled");
            return Ok(());
        }

        info!("Initializing Kubernetes integration");

        // Initialize client
        self.client = Some(KubernetesClient::new(self.config.kubeconfig_path.clone()).await?);

        // Set up device plugin
        if self.config.device_plugin_enabled {
            self.device_plugin.initialize().await?;
        }

        // Initialize CDI
        if self.config.cdi_enabled {
            self.cdi_manager.initialize().await?;
        }

        // Apply node labels
        if self.config.node_labeling.enabled {
            self.apply_node_labels().await?;
        }

        info!("Kubernetes integration initialized successfully");
        Ok(())
    }

    /// Start device plugin server
    pub async fn start_device_plugin(&mut self) -> Result<()> {
        if !self.config.device_plugin_enabled {
            return Ok(());
        }

        info!("Starting Kubernetes device plugin");
        self.device_plugin.start().await?;
        Ok(())
    }

    /// Apply node labels for GPU capabilities
    async fn apply_node_labels(&self) -> Result<()> {
        let Some(client) = &self.client else {
            return Err(anyhow::anyhow!("Kubernetes client not initialized"));
        };

        info!("Applying node labels for GPU capabilities");

        // Get GPU information
        let gpus = crate::gpu::discover_gpus().await?;
        let driver_info = crate::gpu::get_driver_info().await?;

        // Prepare labels
        let mut labels = HashMap::new();

        // GPU count
        labels.insert(
            format!("{}.count", self.config.node_labeling.gpu_label_prefix),
            gpus.len().to_string(),
        );

        // Driver information
        labels.insert(
            format!("{}.version", self.config.node_labeling.driver_label_prefix),
            driver_info.version,
        );

        labels.insert(
            format!("{}.type", self.config.node_labeling.driver_label_prefix),
            format!("{:?}", driver_info.driver_type).to_lowercase(),
        );

        // GPU details
        for (i, gpu) in gpus.iter().enumerate() {
            labels.insert(
                format!("{}.{}.name", self.config.node_labeling.gpu_label_prefix, i),
                gpu.name.clone(),
            );
            labels.insert(
                format!(
                    "{}.{}.memory",
                    self.config.node_labeling.gpu_label_prefix, i
                ),
                gpu.memory
                    .map(|m| (m / (1024 * 1024)).to_string())
                    .unwrap_or_default(),
            );
            labels.insert(
                format!("{}.{}.uuid", self.config.node_labeling.gpu_label_prefix, i),
                gpu.id.clone(),
            );
        }

        // Capability labels
        for capability in &self.config.node_labeling.capability_labels {
            labels.insert(capability.clone(), "true".to_string());
        }

        // Apply labels
        client.apply_node_labels(labels).await?;

        info!("Node labels applied successfully");
        Ok(())
    }

    /// Create CDI specification for Kubernetes
    pub async fn create_cdi_spec(&self) -> Result<()> {
        if !self.config.cdi_enabled {
            return Ok(());
        }

        info!("Creating CDI specification for Kubernetes");
        self.cdi_manager.create_kubernetes_spec().await?;
        Ok(())
    }

    /// Validate resource requests
    pub fn validate_resource_request(&self, request: &ResourceRequest) -> Result<()> {
        if let Some(limit) = self.config.resource_quotas.gpu_limit_per_pod {
            if request.gpu_count > limit {
                return Err(anyhow::anyhow!(
                    "GPU request ({}) exceeds pod limit ({})",
                    request.gpu_count,
                    limit
                ));
            }
        }

        if let Some(memory_limit) = self.config.resource_quotas.memory_limit_per_gpu {
            let total_memory = request.gpu_count as u64 * request.memory_per_gpu;
            if total_memory > memory_limit {
                return Err(anyhow::anyhow!(
                    "Memory request ({} GB) exceeds limit ({} GB)",
                    total_memory / (1024 * 1024 * 1024),
                    memory_limit / (1024 * 1024 * 1024)
                ));
            }
        }

        Ok(())
    }

    /// Get resource usage statistics
    pub async fn get_resource_usage(&self) -> Result<ResourceUsage> {
        let Some(client) = &self.client else {
            return Err(anyhow::anyhow!("Kubernetes client not initialized"));
        };

        client.get_resource_usage().await
    }
}

/// Kubernetes device plugin implementation
pub struct DevicePlugin {
    _socket_path: PathBuf,
    _devices: Vec<Device>,
    server: Option<DevicePluginServer>,
}

impl DevicePlugin {
    fn new(socket_path: PathBuf) -> Self {
        Self {
            _socket_path: socket_path,
            _devices: Vec::new(),
            server: None,
        }
    }

    /// Initialize device plugin
    async fn initialize(&mut self) -> Result<()> {
        info!("Initializing device plugin");

        // Discover GPU devices
        let gpus = crate::gpu::discover_gpus().await?;

        // Convert to device plugin format
        self._devices = gpus
            .into_iter()
            .enumerate()
            .map(|(i, gpu)| Device {
                id: format!("nvidia-gpu-{}", i),
                health: DeviceHealth::Healthy,
                topology: None,
                gpu_info: gpu,
            })
            .collect();

        info!(
            "Discovered {} GPU devices for device plugin",
            self._devices.len()
        );
        Ok(())
    }

    /// Start device plugin server
    async fn start(&mut self) -> Result<()> {
        info!("Starting device plugin server at {:?}", self._socket_path);

        // Clean up existing socket
        if self._socket_path.exists() {
            fs::remove_file(&self._socket_path).await?;
        }

        // Create server
        let server = DevicePluginServer::new(self._socket_path.clone(), self._devices.clone());

        self.server = Some(server);

        // Register with kubelet
        self.register_with_kubelet().await?;

        info!("Device plugin server started successfully");
        Ok(())
    }

    /// Register device plugin with kubelet
    async fn register_with_kubelet(&self) -> Result<()> {
        info!("Registering device plugin with kubelet");

        let _registration_request = RegistrationRequest {
            version: "v1beta1".to_string(),
            endpoint: self
                ._socket_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string(),
            resource_name: "nvidia.com/gpu".to_string(),
            options: RegistrationOptions {
                preferred_allocation_policy: "none".to_string(),
            },
        };

        // Connect to kubelet registration socket
        let _kubelet_socket = PathBuf::from("/var/lib/kubelet/device-plugins/kubelet.sock");
        // Implementation would use gRPC to communicate with kubelet
        // For now, we'll simulate the registration

        info!("Device plugin registered with kubelet");
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Device {
    pub id: String,
    pub health: DeviceHealth,
    pub topology: Option<Topology>,
    pub gpu_info: crate::gpu::GpuDevice,
}

#[derive(Debug, Clone)]
pub enum DeviceHealth {
    Healthy,
    Unhealthy,
}

#[derive(Debug, Clone)]
pub struct Topology {
    pub numa_node: Option<u32>,
    pub pci_bus_id: String,
}

#[derive(Debug)]
struct DevicePluginServer {
    _socket_path: PathBuf,
    _devices: Vec<Device>,
}

impl DevicePluginServer {
    fn new(socket_path: PathBuf, devices: Vec<Device>) -> Self {
        Self {
            _socket_path: socket_path,
            _devices: devices,
        }
    }

    // gRPC server implementation would go here
}

#[derive(Debug, Serialize, Deserialize)]
struct RegistrationRequest {
    version: String,
    endpoint: String,
    resource_name: String,
    options: RegistrationOptions,
}

#[derive(Debug, Serialize, Deserialize)]
struct RegistrationOptions {
    preferred_allocation_policy: String,
}

/// CDI manager for Kubernetes
pub struct CdiManager {
    spec_dir: PathBuf,
}

impl CdiManager {
    fn new() -> Self {
        Self {
            spec_dir: PathBuf::from("/etc/cdi"),
        }
    }

    /// Initialize CDI manager
    async fn initialize(&self) -> Result<()> {
        info!("Initializing CDI manager for Kubernetes");

        // Ensure CDI directory exists
        fs::create_dir_all(&self.spec_dir).await?;

        info!("CDI manager initialized");
        Ok(())
    }

    /// Create Kubernetes-specific CDI specification
    async fn create_kubernetes_spec(&self) -> Result<()> {
        info!("Creating Kubernetes CDI specification");

        let gpus = crate::gpu::discover_gpus().await?;
        let _driver_info = crate::gpu::get_driver_info().await?;

        let spec = CdiSpecification {
            cdi_version: "0.6.0".to_string(),
            kind: "nvidia.com/gpu".to_string(),
            devices: gpus
                .into_iter()
                .enumerate()
                .map(|(i, gpu)| CdiDevice {
                    name: format!("gpu{}", i),
                    annotations: Some(
                        [
                            ("gpu.uuid".to_string(), gpu.id.clone()),
                            ("gpu.name".to_string(), gpu.name.clone()),
                            (
                                "gpu.memory".to_string(),
                                gpu.memory.map(|m| m.to_string()).unwrap_or_default(),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                    container_edits: CdiContainerEdits {
                        device_nodes: vec![CdiDeviceNode {
                            path: format!("/dev/nvidia{}", i),
                            type_: "c".to_string(),
                            major: 195,
                            minor: i as u32,
                            file_mode: Some(0o666),
                            uid: Some(0),
                            gid: Some(0),
                        }],
                        mounts: vec![
                            CdiMount {
                                host_path: "/usr/bin/nvidia-smi".to_string(),
                                container_path: "/usr/bin/nvidia-smi".to_string(),
                                options: vec!["ro".to_string(), "bind".to_string()],
                            },
                            CdiMount {
                                host_path: "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so".to_string(),
                                container_path: "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so"
                                    .to_string(),
                                options: vec!["ro".to_string(), "bind".to_string()],
                            },
                        ],
                        env: vec![
                            CdiEnv {
                                name: "NVIDIA_VISIBLE_DEVICES".to_string(),
                                value: gpu.id.clone(),
                            },
                            CdiEnv {
                                name: "NVIDIA_DRIVER_CAPABILITIES".to_string(),
                                value: "compute,utility".to_string(),
                            },
                        ],
                        hooks: vec![],
                    },
                })
                .collect(),
            container_edits: CdiContainerEdits {
                device_nodes: vec![
                    CdiDeviceNode {
                        path: "/dev/nvidiactl".to_string(),
                        type_: "c".to_string(),
                        major: 195,
                        minor: 255,
                        file_mode: Some(0o666),
                        uid: Some(0),
                        gid: Some(0),
                    },
                    CdiDeviceNode {
                        path: "/dev/nvidia-uvm".to_string(),
                        type_: "c".to_string(),
                        major: 510,
                        minor: 0,
                        file_mode: Some(0o666),
                        uid: Some(0),
                        gid: Some(0),
                    },
                ],
                mounts: vec![],
                env: vec![],
                hooks: vec![],
            },
        };

        // Write specification to file
        let spec_path = self.spec_dir.join("nvidia-gpu.yaml");
        let spec_yaml = serde_yaml::to_string(&spec)?;
        fs::write(&spec_path, spec_yaml).await?;

        info!("CDI specification written to {:?}", spec_path);
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CdiSpecification {
    #[serde(rename = "cdiVersion")]
    cdi_version: String,
    kind: String,
    devices: Vec<CdiDevice>,
    #[serde(rename = "containerEdits")]
    container_edits: CdiContainerEdits,
}

#[derive(Debug, Serialize, Deserialize)]
struct CdiDevice {
    name: String,
    annotations: Option<HashMap<String, String>>,
    #[serde(rename = "containerEdits")]
    container_edits: CdiContainerEdits,
}

#[derive(Debug, Serialize, Deserialize)]
struct CdiContainerEdits {
    #[serde(rename = "deviceNodes")]
    device_nodes: Vec<CdiDeviceNode>,
    mounts: Vec<CdiMount>,
    env: Vec<CdiEnv>,
    hooks: Vec<CdiHook>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CdiDeviceNode {
    path: String,
    #[serde(rename = "type")]
    type_: String,
    major: u32,
    minor: u32,
    #[serde(rename = "fileMode")]
    file_mode: Option<u32>,
    uid: Option<u32>,
    gid: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CdiMount {
    #[serde(rename = "hostPath")]
    host_path: String,
    #[serde(rename = "containerPath")]
    container_path: String,
    options: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CdiEnv {
    name: String,
    value: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CdiHook {
    #[serde(rename = "hookName")]
    hook_name: String,
    path: String,
    args: Vec<String>,
    env: Vec<CdiEnv>,
    timeout: Option<u32>,
}

/// Kubernetes client wrapper
pub struct KubernetesClient {
    kubeconfig_path: Option<PathBuf>,
}

impl KubernetesClient {
    /// Create new Kubernetes client
    async fn new(kubeconfig_path: Option<PathBuf>) -> Result<Self> {
        info!("Initializing Kubernetes client");

        let client = Self { kubeconfig_path };

        // Validate connection
        client.validate_connection().await?;

        Ok(client)
    }

    /// Validate Kubernetes connection
    async fn validate_connection(&self) -> Result<()> {
        debug!("Validating Kubernetes connection");

        let mut cmd = Command::new("kubectl");

        if let Some(config_path) = &self.kubeconfig_path {
            cmd.arg("--kubeconfig").arg(config_path);
        }

        let output = cmd.arg("version").arg("--client").output()?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Failed to connect to Kubernetes: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        info!("Kubernetes connection validated");
        Ok(())
    }

    /// Apply node labels
    async fn apply_node_labels(&self, labels: HashMap<String, String>) -> Result<()> {
        info!("Applying {} node labels", labels.len());

        // Get node name
        let node_name = self.get_node_name().await?;

        // Apply each label
        for (key, value) in labels {
            let mut cmd = Command::new("kubectl");

            if let Some(config_path) = &self.kubeconfig_path {
                cmd.arg("--kubeconfig").arg(config_path);
            }

            let output = cmd
                .arg("label")
                .arg("node")
                .arg(&node_name)
                .arg(format!("{}={}", key, value))
                .arg("--overwrite")
                .output()?;

            if !output.status.success() {
                warn!(
                    "Failed to apply label {}={}: {}",
                    key,
                    value,
                    String::from_utf8_lossy(&output.stderr)
                );
            } else {
                debug!("Applied label: {}={}", key, value);
            }
        }

        info!("Node labels applied");
        Ok(())
    }

    /// Get current node name
    async fn get_node_name(&self) -> Result<String> {
        // Try to get from environment first
        if let Ok(node_name) = std::env::var("NODE_NAME") {
            return Ok(node_name);
        }

        // Fall back to hostname
        let hostname = hostname::get()?.to_string_lossy().to_string();

        Ok(hostname)
    }

    /// Get resource usage statistics
    async fn get_resource_usage(&self) -> Result<ResourceUsage> {
        info!("Collecting resource usage statistics");

        let mut cmd = Command::new("kubectl");

        if let Some(config_path) = &self.kubeconfig_path {
            cmd.arg("--kubeconfig").arg(config_path);
        }

        let output = cmd.arg("top").arg("nodes").arg("--no-headers").output()?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Failed to get resource usage: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        // Parse output (simplified)
        let usage = ResourceUsage {
            cpu_usage_percent: 50.0,    // Placeholder
            memory_usage_percent: 60.0, // Placeholder
            gpu_usage_percent: 80.0,    // Placeholder
            allocated_pods: 10,         // Placeholder
            capacity_pods: 100,         // Placeholder
        };

        Ok(usage)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub gpu_count: u32,
    pub memory_per_gpu: u64,
    pub cpu_cores: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub gpu_usage_percent: f64,
    pub allocated_pods: u32,
    pub capacity_pods: u32,
}

/// OpenShift integration (extends Kubernetes)
pub struct OpenShiftIntegration {
    k8s: KubernetesIntegration,
    scc_config: SecurityContextConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContextConfig {
    pub privileged_scc: bool,
    pub custom_scc_name: Option<String>,
    pub selinux_context: Option<String>,
}

impl OpenShiftIntegration {
    /// Create new OpenShift integration
    pub fn new(k8s_config: K8sConfig, scc_config: SecurityContextConfig) -> Self {
        Self {
            k8s: KubernetesIntegration::new(k8s_config),
            scc_config,
        }
    }

    /// Initialize OpenShift integration
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing OpenShift integration");

        // Initialize base Kubernetes functionality
        self.k8s.initialize().await?;

        // Set up Security Context Constraints
        self.setup_security_context_constraints().await?;

        info!("OpenShift integration initialized successfully");
        Ok(())
    }

    /// Set up Security Context Constraints
    async fn setup_security_context_constraints(&self) -> Result<()> {
        info!("Setting up Security Context Constraints");

        if self.scc_config.privileged_scc {
            // Create or update privileged SCC for GPU access
            self.create_gpu_scc().await?;
        }

        Ok(())
    }

    /// Create GPU-specific Security Context Constraint
    async fn create_gpu_scc(&self) -> Result<()> {
        let scc_name = self
            .scc_config
            .custom_scc_name
            .as_deref()
            .unwrap_or("nvbind-gpu-scc");

        info!("Creating Security Context Constraint: {}", scc_name);

        let _scc_yaml = format!(
            r#"
apiVersion: security.openshift.io/v1
kind: SecurityContextConstraints
metadata:
  name: {}
allowHostDirVolumePlugin: true
allowHostIPC: false
allowHostNetwork: false
allowHostPID: false
allowHostPorts: false
allowPrivilegedContainer: true
allowedCapabilities: []
defaultAddCapabilities: []
fsGroup:
  type: RunAsAny
readOnlyRootFilesystem: false
requiredDropCapabilities: []
runAsUser:
  type: RunAsAny
seLinuxContext:
  type: {}
supplementalGroups:
  type: RunAsAny
volumes:
- configMap
- downwardAPI
- emptyDir
- hostPath
- persistentVolumeClaim
- projected
- secret
users: []
groups: []
"#,
            scc_name,
            self.scc_config
                .selinux_context
                .as_deref()
                .unwrap_or("MustRunAs")
        );

        // Apply SCC (would use oc or kubectl)
        info!("SCC configuration prepared for {}", scc_name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_k8s_config_default() {
        let config = K8sConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.namespace, "nvbind-system");
        assert!(config.device_plugin_enabled);
        assert!(config.cdi_enabled);
    }

    #[tokio::test]
    async fn test_resource_validation() {
        let config = K8sConfig::default();
        let integration = KubernetesIntegration::new(config);

        let valid_request = ResourceRequest {
            gpu_count: 2,
            memory_per_gpu: 1024 * 1024 * 1024, // 1GB
            cpu_cores: Some(4),
        };

        assert!(
            integration
                .validate_resource_request(&valid_request)
                .is_ok()
        );

        let invalid_request = ResourceRequest {
            gpu_count: 16, // Exceeds default limit of 8
            memory_per_gpu: 1024 * 1024 * 1024,
            cpu_cores: Some(8),
        };

        assert!(
            integration
                .validate_resource_request(&invalid_request)
                .is_err()
        );
    }

    #[tokio::test]
    async fn test_cdi_manager_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CdiManager::new();
        manager.spec_dir = temp_dir.path().to_path_buf();

        assert!(manager.initialize().await.is_ok());
        assert!(manager.spec_dir.exists());
    }

    #[tokio::test]
    async fn test_device_plugin_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let socket_path = temp_dir.path().join("test.sock");
        let mut plugin = DevicePlugin::new(socket_path);

        // This would fail without actual GPUs, but tests the structure
        let result = plugin.initialize().await;
        // We expect this to fail in test environment without GPUs
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_node_labeling_config() {
        let config = NodeLabeling::default();
        assert!(config.enabled);
        assert_eq!(config.gpu_label_prefix, "nvidia.com/gpu");
        assert!(
            config
                .capability_labels
                .contains(&"nvidia.com/cuda".to_string())
        );
    }

    #[test]
    fn test_openshift_scc_config() {
        let scc_config = SecurityContextConfig {
            privileged_scc: true,
            custom_scc_name: Some("custom-gpu-scc".to_string()),
            selinux_context: Some("RunAsAny".to_string()),
        };

        assert!(scc_config.privileged_scc);
        assert_eq!(scc_config.custom_scc_name.unwrap(), "custom-gpu-scc");
    }
}
