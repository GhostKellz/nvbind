use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// CDI v0.7+ Specification Support
/// Implements the latest Container Device Interface specification
/// with enhanced features for dynamic allocation and device topology

/// CDI v0.7+ Device Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdiV07Spec {
    /// CDI specification version (0.7.0+)
    pub cdi_version: String,
    /// Device kind (e.g., "nvidia.com/gpu")
    pub kind: String,
    /// List of devices
    pub devices: Vec<CdiV07Device>,
    /// Container edits for all devices
    pub container_edits: Option<ContainerEdits>,
}

/// Enhanced CDI v0.7+ Device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdiV07Device {
    /// Device name
    pub name: String,
    /// Device annotations
    pub annotations: Option<HashMap<String, String>>,
    /// Container edits specific to this device
    pub container_edits: Option<ContainerEdits>,
    /// Device topology information (new in v0.7+)
    pub topology: Option<DeviceTopology>,
    /// Dynamic allocation capabilities (new in v0.7+)
    pub allocation: Option<DynamicAllocation>,
    /// Hot-plug capabilities (new in v0.7+)
    pub hot_plug: Option<HotPlugCapabilities>,
}

/// Device Topology Information (CDI v0.7+ feature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceTopology {
    /// NUMA node affinity
    pub numa_affinity: Option<Vec<u32>>,
    /// PCIe bus information
    pub pcie_info: Option<PcieInfo>,
    /// CPU affinity hints
    pub cpu_affinity: Option<Vec<u32>>,
    /// Memory affinity information
    pub memory_affinity: Option<MemoryAffinity>,
    /// Device interconnects
    pub interconnects: Option<Vec<DeviceInterconnect>>,
}

/// PCIe Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcieInfo {
    /// PCIe domain
    pub domain: u32,
    /// PCIe bus
    pub bus: u32,
    /// PCIe device
    pub device: u32,
    /// PCIe function
    pub function: u32,
    /// PCIe slot
    pub slot: Option<String>,
    /// PCIe generation
    pub generation: Option<u32>,
    /// PCIe lanes
    pub lanes: Option<u32>,
}

/// Memory Affinity Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAffinity {
    /// Preferred memory nodes
    pub preferred_nodes: Vec<u32>,
    /// Memory bandwidth characteristics
    pub bandwidth_gb_per_sec: Option<f64>,
    /// Memory latency characteristics
    pub latency_ns: Option<u64>,
}

/// Device Interconnect Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInterconnect {
    /// Interconnect type (NVLink, PCIe, etc.)
    pub interconnect_type: String,
    /// Connected device
    pub connected_device: String,
    /// Bandwidth in GB/s
    pub bandwidth_gb_per_sec: Option<f64>,
    /// Bidirectional capability
    pub bidirectional: bool,
}

/// Dynamic Device Allocation (CDI v0.7+ feature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicAllocation {
    /// Allocation strategy
    pub strategy: AllocationStrategy,
    /// Resource constraints
    pub constraints: Option<ResourceConstraints>,
    /// Allocation hints
    pub hints: Option<AllocationHints>,
    /// Sharing capabilities
    pub sharing: Option<SharingCapabilities>,
}

/// Allocation Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Exclusive allocation
    Exclusive,
    /// Shared allocation
    Shared,
    /// Time-sliced allocation
    TimeSliced,
    /// Multi-instance allocation
    MultiInstance,
    /// Fractional allocation
    Fractional,
}

/// Resource Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Memory constraints
    pub memory: Option<MemoryConstraints>,
    /// Compute constraints
    pub compute: Option<ComputeConstraints>,
    /// Bandwidth constraints
    pub bandwidth: Option<BandwidthConstraints>,
}

/// Memory Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConstraints {
    /// Minimum memory in bytes
    pub min_memory_bytes: Option<u64>,
    /// Maximum memory in bytes
    pub max_memory_bytes: Option<u64>,
    /// Memory type preference
    pub memory_type: Option<String>,
}

/// Compute Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeConstraints {
    /// Minimum compute units
    pub min_compute_units: Option<u32>,
    /// Maximum compute units
    pub max_compute_units: Option<u32>,
    /// Compute capability requirement
    pub compute_capability: Option<String>,
}

/// Bandwidth Constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthConstraints {
    /// Minimum bandwidth in GB/s
    pub min_bandwidth_gb_per_sec: Option<f64>,
    /// Maximum bandwidth in GB/s
    pub max_bandwidth_gb_per_sec: Option<f64>,
}

/// Allocation Hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationHints {
    /// Preferred devices
    pub preferred_devices: Option<Vec<String>>,
    /// Anti-affinity devices
    pub anti_affinity_devices: Option<Vec<String>>,
    /// Locality preference
    pub locality_preference: Option<LocalityPreference>,
}

/// Locality Preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LocalityPreference {
    /// Same NUMA node
    SameNuma,
    /// Same PCIe switch
    SamePcieSwitch,
    /// Same physical server
    SameServer,
    /// Any location
    Any,
}

/// Sharing Capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingCapabilities {
    /// Maximum concurrent users
    pub max_concurrent_users: Option<u32>,
    /// Sharing granularity
    pub granularity: Option<SharingGranularity>,
    /// Isolation level
    pub isolation_level: Option<IsolationLevel>,
}

/// Sharing Granularity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SharingGranularity {
    /// Device level sharing
    Device,
    /// Memory level sharing
    Memory,
    /// Compute unit sharing
    ComputeUnit,
    /// Time slice sharing
    TimeSlice,
}

/// Isolation Level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationLevel {
    /// Full hardware isolation
    Hardware,
    /// Software isolation
    Software,
    /// Process isolation
    Process,
    /// Namespace isolation
    Namespace,
}

/// Hot-Plug Capabilities (CDI v0.7+ feature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotPlugCapabilities {
    /// Support for hot-plug
    pub supported: bool,
    /// Hot-plug detection method
    pub detection_method: Option<HotPlugDetectionMethod>,
    /// Notification mechanism
    pub notification: Option<HotPlugNotification>,
    /// Hot-plug latency in milliseconds
    pub latency_ms: Option<u64>,
}

/// Hot-Plug Detection Method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HotPlugDetectionMethod {
    /// PCIe hot-plug
    Pcie,
    /// USB hot-plug
    Usb,
    /// Custom driver detection
    Driver,
    /// Polling-based detection
    Polling,
}

/// Hot-Plug Notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotPlugNotification {
    /// Notification type
    pub notification_type: NotificationType,
    /// Endpoint for notifications
    pub endpoint: Option<String>,
    /// Notification timeout
    pub timeout_ms: Option<u64>,
}

/// Notification Type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    /// Event-based notification
    Event,
    /// Webhook notification
    Webhook,
    /// File-based notification
    File,
    /// Signal-based notification
    Signal,
}

/// Enhanced Container Edits for CDI v0.7+
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerEdits {
    /// Environment variables
    pub env: Option<Vec<String>>,
    /// Device nodes
    pub device_nodes: Option<Vec<DeviceNode>>,
    /// Hooks
    pub hooks: Option<Vec<Hook>>,
    /// Mounts
    pub mounts: Option<Vec<Mount>>,
    /// Additional GIDs
    pub additional_gids: Option<Vec<u32>>,
    /// Annotations (new in v0.7+)
    pub annotations: Option<HashMap<String, String>>,
    /// Resource limits (new in v0.7+)
    pub resource_limits: Option<ResourceLimits>,
}

/// Device Node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceNode {
    /// Device path
    pub path: String,
    /// Device type
    pub device_type: Option<String>,
    /// Major number
    pub major: Option<u32>,
    /// Minor number
    pub minor: Option<u32>,
    /// File mode
    pub file_mode: Option<u32>,
    /// UID
    pub uid: Option<u32>,
    /// GID
    pub gid: Option<u32>,
}

/// Hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hook {
    /// Hook name
    pub hook_name: String,
    /// Executable path
    pub path: String,
    /// Arguments
    pub args: Option<Vec<String>>,
    /// Environment variables
    pub env: Option<Vec<String>>,
    /// Timeout
    pub timeout: Option<u32>,
}

/// Mount
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mount {
    /// Host path
    pub host_path: String,
    /// Container path
    pub container_path: String,
    /// Mount options
    pub options: Option<Vec<String>>,
    /// Mount type
    pub mount_type: Option<String>,
}

/// Resource Limits (CDI v0.7+ feature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Memory limit in bytes
    pub memory_bytes: Option<u64>,
    /// CPU limit in millicores
    pub cpu_millicores: Option<u32>,
    /// GPU memory limit in bytes
    pub gpu_memory_bytes: Option<u64>,
    /// Custom resource limits
    pub custom: Option<HashMap<String, String>>,
}

/// CDI v0.7+ Device Manager
pub struct CdiV07DeviceManager {
    /// Device specifications
    devices: Arc<RwLock<HashMap<String, CdiV07Spec>>>,
    /// Dynamic allocations
    allocations: Arc<RwLock<HashMap<String, DeviceAllocation>>>,
    /// Hot-plug detection
    hot_plug_enabled: bool,
    /// Topology awareness
    topology_aware: bool,
}

/// Device Allocation State
#[derive(Debug, Clone)]
pub struct DeviceAllocation {
    /// Allocation ID
    pub id: String,
    /// Device name
    pub device_name: String,
    /// Container ID
    pub container_id: String,
    /// Allocation strategy used
    pub strategy: AllocationStrategy,
    /// Resource allocation
    pub resources: AllocatedResources,
    /// Allocation timestamp
    pub allocated_at: chrono::DateTime<chrono::Utc>,
}

/// Allocated Resources
#[derive(Debug, Clone)]
pub struct AllocatedResources {
    /// Memory allocation
    pub memory_bytes: u64,
    /// Compute units
    pub compute_units: u32,
    /// Bandwidth allocation
    pub bandwidth_gb_per_sec: f64,
    /// Fractional allocation
    pub fraction: Option<f64>,
}

impl CdiV07DeviceManager {
    /// Create a new CDI v0.7+ Device Manager
    pub fn new() -> Self {
        Self {
            devices: Arc::new(RwLock::new(HashMap::new())),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            hot_plug_enabled: true,
            topology_aware: true,
        }
    }

    /// Register a CDI v0.7+ device specification
    pub fn register_device_spec(&self, spec: CdiV07Spec) -> Result<()> {
        let spec_name = format!("{}_{}", spec.kind, spec.cdi_version);

        info!("Registering CDI v0.7+ device spec: {}", spec_name);

        // Validate specification
        self.validate_spec(&spec)?;

        let mut devices = self.devices.write().unwrap();
        devices.insert(spec_name, spec);

        Ok(())
    }

    /// Validate CDI v0.7+ specification
    fn validate_spec(&self, spec: &CdiV07Spec) -> Result<()> {
        // Check CDI version compatibility
        if !self.is_version_compatible(&spec.cdi_version) {
            return Err(anyhow!("CDI version {} not supported", spec.cdi_version));
        }

        // Validate device definitions
        for device in &spec.devices {
            self.validate_device(device)?;
        }

        Ok(())
    }

    /// Check if CDI version is compatible
    fn is_version_compatible(&self, version: &str) -> bool {
        // Support CDI v0.7.0 and later
        if let Some(version_num) = self.parse_version(version) {
            return version_num >= (0, 7, 0);
        }
        false
    }

    /// Parse version string
    fn parse_version(&self, version: &str) -> Option<(u32, u32, u32)> {
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() >= 3 {
            if let (Ok(major), Ok(minor), Ok(patch)) = (
                parts[0].parse::<u32>(),
                parts[1].parse::<u32>(),
                parts[2].parse::<u32>(),
            ) {
                return Some((major, minor, patch));
            }
        }
        None
    }

    /// Validate individual device definition
    fn validate_device(&self, device: &CdiV07Device) -> Result<()> {
        // Validate device name
        if device.name.is_empty() {
            return Err(anyhow!("Device name cannot be empty"));
        }

        // Validate topology information if present
        if let Some(topology) = &device.topology {
            self.validate_topology(topology)?;
        }

        // Validate dynamic allocation if present
        if let Some(allocation) = &device.allocation {
            self.validate_allocation(allocation)?;
        }

        Ok(())
    }

    /// Validate device topology
    fn validate_topology(&self, topology: &DeviceTopology) -> Result<()> {
        // Validate NUMA affinity
        if let Some(numa_nodes) = &topology.numa_affinity {
            for &node in numa_nodes {
                if !self.is_numa_node_valid(node) {
                    warn!("NUMA node {} may not be valid", node);
                }
            }
        }

        // Validate PCIe information
        if let Some(pcie) = &topology.pcie_info {
            self.validate_pcie_info(pcie)?;
        }

        Ok(())
    }

    /// Validate PCIe information
    fn validate_pcie_info(&self, pcie: &PcieInfo) -> Result<()> {
        // Basic PCIe validation
        if pcie.bus > 255 {
            return Err(anyhow!("PCIe bus number {} exceeds maximum (255)", pcie.bus));
        }

        if pcie.device > 31 {
            return Err(anyhow!("PCIe device number {} exceeds maximum (31)", pcie.device));
        }

        if pcie.function > 7 {
            return Err(anyhow!("PCIe function number {} exceeds maximum (7)", pcie.function));
        }

        Ok(())
    }

    /// Check if NUMA node is valid
    fn is_numa_node_valid(&self, node: u32) -> bool {
        // Check if NUMA node exists on the system
        std::path::Path::new(&format!("/sys/devices/system/node/node{}", node)).exists()
    }

    /// Validate dynamic allocation configuration
    fn validate_allocation(&self, allocation: &DynamicAllocation) -> Result<()> {
        // Validate resource constraints
        if let Some(constraints) = &allocation.constraints {
            self.validate_constraints(constraints)?;
        }

        // Validate sharing capabilities
        if let Some(sharing) = &allocation.sharing {
            if let Some(max_users) = sharing.max_concurrent_users {
                if max_users == 0 {
                    return Err(anyhow!("Maximum concurrent users must be greater than 0"));
                }
            }
        }

        Ok(())
    }

    /// Validate resource constraints
    fn validate_constraints(&self, constraints: &ResourceConstraints) -> Result<()> {
        // Validate memory constraints
        if let Some(memory) = &constraints.memory {
            if let (Some(min), Some(max)) = (memory.min_memory_bytes, memory.max_memory_bytes) {
                if min > max {
                    return Err(anyhow!("Minimum memory cannot exceed maximum memory"));
                }
            }
        }

        // Validate compute constraints
        if let Some(compute) = &constraints.compute {
            if let (Some(min), Some(max)) = (compute.min_compute_units, compute.max_compute_units) {
                if min > max {
                    return Err(anyhow!("Minimum compute units cannot exceed maximum compute units"));
                }
            }
        }

        Ok(())
    }

    /// Allocate device dynamically
    pub async fn allocate_device(
        &self,
        device_name: &str,
        container_id: &str,
        requirements: &AllocationRequirements,
    ) -> Result<String> {
        info!("Allocating device {} for container {}", device_name, container_id);

        // Find device specification
        let device_spec = self.find_device_spec(device_name)?;

        // Check if device supports dynamic allocation
        let allocation_config = device_spec.allocation
            .as_ref()
            .ok_or_else(|| anyhow!("Device {} does not support dynamic allocation", device_name))?;

        // Select allocation strategy
        let strategy = self.select_allocation_strategy(allocation_config, requirements)?;

        // Perform resource allocation
        let resources = self.allocate_resources(&strategy, allocation_config, requirements).await?;

        // Create allocation record
        let allocation_id = Uuid::new_v4().to_string();
        let allocation = DeviceAllocation {
            id: allocation_id.clone(),
            device_name: device_name.to_string(),
            container_id: container_id.to_string(),
            strategy,
            resources,
            allocated_at: chrono::Utc::now(),
        };

        // Store allocation
        let mut allocations = self.allocations.write().unwrap();
        allocations.insert(allocation_id.clone(), allocation);

        info!("Successfully allocated device {} with ID {}", device_name, allocation_id);
        Ok(allocation_id)
    }

    /// Find device specification by name
    fn find_device_spec(&self, device_name: &str) -> Result<CdiV07Device> {
        let devices = self.devices.read().unwrap();

        for spec in devices.values() {
            for device in &spec.devices {
                if device.name == device_name {
                    return Ok(device.clone());
                }
            }
        }

        Err(anyhow!("Device {} not found", device_name))
    }

    /// Select appropriate allocation strategy
    fn select_allocation_strategy(
        &self,
        allocation_config: &DynamicAllocation,
        requirements: &AllocationRequirements,
    ) -> Result<AllocationStrategy> {
        // Use requested strategy if compatible
        if let Some(requested_strategy) = &requirements.preferred_strategy {
            if self.is_strategy_compatible(&allocation_config.strategy, requested_strategy) {
                return Ok(requested_strategy.clone());
            }
        }

        // Fall back to device default strategy
        Ok(allocation_config.strategy.clone())
    }

    /// Check if allocation strategies are compatible
    fn is_strategy_compatible(
        &self,
        device_strategy: &AllocationStrategy,
        requested_strategy: &AllocationStrategy,
    ) -> bool {
        use AllocationStrategy::*;

        match (device_strategy, requested_strategy) {
            (Shared, Shared) => true,
            (TimeSliced, TimeSliced) => true,
            (MultiInstance, MultiInstance) => true,
            (Fractional, Fractional) => true,
            (Exclusive, Exclusive) => true,
            // Allow shared strategies to be used for exclusive requests
            (Shared, Exclusive) => true,
            (TimeSliced, Exclusive) => true,
            (MultiInstance, Exclusive) => true,
            (Fractional, Exclusive) => true,
            _ => false,
        }
    }

    /// Allocate resources based on strategy
    async fn allocate_resources(
        &self,
        strategy: &AllocationStrategy,
        allocation_config: &DynamicAllocation,
        requirements: &AllocationRequirements,
    ) -> Result<AllocatedResources> {
        match strategy {
            AllocationStrategy::Exclusive => self.allocate_exclusive(requirements).await,
            AllocationStrategy::Shared => self.allocate_shared(allocation_config, requirements).await,
            AllocationStrategy::TimeSliced => self.allocate_time_sliced(allocation_config, requirements).await,
            AllocationStrategy::MultiInstance => self.allocate_multi_instance(allocation_config, requirements).await,
            AllocationStrategy::Fractional => self.allocate_fractional(allocation_config, requirements).await,
        }
    }

    /// Allocate exclusive resources
    async fn allocate_exclusive(&self, requirements: &AllocationRequirements) -> Result<AllocatedResources> {
        Ok(AllocatedResources {
            memory_bytes: requirements.memory_bytes.unwrap_or(0),
            compute_units: requirements.compute_units.unwrap_or(0),
            bandwidth_gb_per_sec: requirements.bandwidth_gb_per_sec.unwrap_or(0.0),
            fraction: Some(1.0),
        })
    }

    /// Allocate shared resources
    async fn allocate_shared(
        &self,
        allocation_config: &DynamicAllocation,
        requirements: &AllocationRequirements,
    ) -> Result<AllocatedResources> {
        let sharing = allocation_config.sharing
            .as_ref()
            .ok_or_else(|| anyhow!("Sharing configuration not available"))?;

        let max_users = sharing.max_concurrent_users.unwrap_or(1);
        let current_users = self.count_current_users().await?;

        if current_users >= max_users {
            return Err(anyhow!("Maximum concurrent users ({}) exceeded", max_users));
        }

        let fraction = 1.0 / max_users as f64;

        Ok(AllocatedResources {
            memory_bytes: (requirements.memory_bytes.unwrap_or(0) as f64 * fraction) as u64,
            compute_units: (requirements.compute_units.unwrap_or(0) as f64 * fraction) as u32,
            bandwidth_gb_per_sec: requirements.bandwidth_gb_per_sec.unwrap_or(0.0) * fraction,
            fraction: Some(fraction),
        })
    }

    /// Allocate time-sliced resources
    async fn allocate_time_sliced(
        &self,
        _allocation_config: &DynamicAllocation,
        requirements: &AllocationRequirements,
    ) -> Result<AllocatedResources> {
        // Time-sliced allocation implementation
        Ok(AllocatedResources {
            memory_bytes: requirements.memory_bytes.unwrap_or(0),
            compute_units: requirements.compute_units.unwrap_or(0),
            bandwidth_gb_per_sec: requirements.bandwidth_gb_per_sec.unwrap_or(0.0),
            fraction: Some(0.5), // Example: 50% time slice
        })
    }

    /// Allocate multi-instance resources
    async fn allocate_multi_instance(
        &self,
        _allocation_config: &DynamicAllocation,
        requirements: &AllocationRequirements,
    ) -> Result<AllocatedResources> {
        // Multi-instance allocation implementation
        Ok(AllocatedResources {
            memory_bytes: requirements.memory_bytes.unwrap_or(0),
            compute_units: requirements.compute_units.unwrap_or(0),
            bandwidth_gb_per_sec: requirements.bandwidth_gb_per_sec.unwrap_or(0.0),
            fraction: Some(0.25), // Example: 1/4 instance
        })
    }

    /// Allocate fractional resources
    async fn allocate_fractional(
        &self,
        _allocation_config: &DynamicAllocation,
        requirements: &AllocationRequirements,
    ) -> Result<AllocatedResources> {
        let fraction = requirements.fraction.unwrap_or(0.5);

        Ok(AllocatedResources {
            memory_bytes: (requirements.memory_bytes.unwrap_or(0) as f64 * fraction) as u64,
            compute_units: (requirements.compute_units.unwrap_or(0) as f64 * fraction) as u32,
            bandwidth_gb_per_sec: requirements.bandwidth_gb_per_sec.unwrap_or(0.0) * fraction,
            fraction: Some(fraction),
        })
    }

    /// Count current device users
    async fn count_current_users(&self) -> Result<u32> {
        let allocations = self.allocations.read().unwrap();
        Ok(allocations.len() as u32)
    }

    /// Deallocate device
    pub async fn deallocate_device(&self, allocation_id: &str) -> Result<()> {
        info!("Deallocating device with allocation ID: {}", allocation_id);

        let mut allocations = self.allocations.write().unwrap();
        allocations.remove(allocation_id)
            .ok_or_else(|| anyhow!("Allocation {} not found", allocation_id))?;

        info!("Successfully deallocated device with ID: {}", allocation_id);
        Ok(())
    }

    /// Enable hot-plug detection
    pub async fn enable_hot_plug_detection(&self) -> Result<()> {
        if !self.hot_plug_enabled {
            info!("Hot-plug detection already disabled");
            return Ok(());
        }

        info!("Enabling hot-plug detection for CDI v0.7+ devices");

        // Start hot-plug monitoring
        self.start_hot_plug_monitoring().await?;

        Ok(())
    }

    /// Start hot-plug monitoring
    async fn start_hot_plug_monitoring(&self) -> Result<()> {
        // Implementation would depend on the specific hot-plug mechanism
        // This is a placeholder for actual hot-plug detection
        debug!("Starting hot-plug monitoring");
        Ok(())
    }

    /// Get device topology information
    pub fn get_device_topology(&self, device_name: &str) -> Result<Option<DeviceTopology>> {
        let device_spec = self.find_device_spec(device_name)?;
        Ok(device_spec.topology)
    }

    /// Get all device allocations
    pub fn get_allocations(&self) -> Vec<DeviceAllocation> {
        let allocations = self.allocations.read().unwrap();
        allocations.values().cloned().collect()
    }

    /// Get allocation by ID
    pub fn get_allocation(&self, allocation_id: &str) -> Option<DeviceAllocation> {
        let allocations = self.allocations.read().unwrap();
        allocations.get(allocation_id).cloned()
    }

    /// Generate CDI v0.7+ specification
    pub fn generate_spec(
        &self,
        kind: &str,
        devices: Vec<CdiV07Device>,
        container_edits: Option<ContainerEdits>,
    ) -> CdiV07Spec {
        CdiV07Spec {
            cdi_version: "0.7.0".to_string(),
            kind: kind.to_string(),
            devices,
            container_edits,
        }
    }
}

/// Allocation Requirements
#[derive(Debug, Clone)]
pub struct AllocationRequirements {
    /// Preferred allocation strategy
    pub preferred_strategy: Option<AllocationStrategy>,
    /// Memory requirement in bytes
    pub memory_bytes: Option<u64>,
    /// Compute units requirement
    pub compute_units: Option<u32>,
    /// Bandwidth requirement in GB/s
    pub bandwidth_gb_per_sec: Option<f64>,
    /// Fractional allocation requirement
    pub fraction: Option<f64>,
    /// Locality preferences
    pub locality: Option<LocalityPreference>,
    /// Device preferences
    pub device_preferences: Option<Vec<String>>,
}

impl Default for AllocationRequirements {
    fn default() -> Self {
        Self {
            preferred_strategy: None,
            memory_bytes: None,
            compute_units: None,
            bandwidth_gb_per_sec: None,
            fraction: None,
            locality: None,
            device_preferences: None,
        }
    }
}