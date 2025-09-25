//! GPU state snapshot and restore for Bolt capsules
//!
//! This module provides GPU state management for container snapshots, allowing
//! Bolt capsules to save and restore GPU context during snapshot operations.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, info};


/// GPU state snapshot containing all recoverable GPU context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStateSnapshot {
    /// Timestamp when snapshot was created
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Container/capsule identifier
    pub container_id: String,
    /// GPU device information
    pub gpu_devices: Vec<GpuDeviceState>,
    /// NVIDIA driver state
    pub driver_state: DriverState,
    /// Process-specific GPU context
    pub process_contexts: Vec<ProcessGpuContext>,
    /// Memory allocations and mappings
    pub memory_state: GpuMemoryState,
    /// Performance and clock states
    pub performance_state: GpuPerformanceState,
}

/// Individual GPU device state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceState {
    /// GPU device ID (e.g., "gpu0")
    pub device_id: String,
    /// PCI bus ID
    pub pci_id: String,
    /// Device node path
    pub device_path: String,
    /// Current power state
    pub power_state: PowerState,
    /// Clock frequencies
    pub clock_state: ClockState,
    /// Thermal state
    pub thermal_state: ThermalState,
    /// Fan configuration
    pub fan_state: FanState,
    /// Display configuration
    pub display_state: Option<DisplayState>,
}

/// NVIDIA driver state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverState {
    /// Driver version
    pub version: String,
    /// Driver type (proprietary, open, nouveau)
    pub driver_type: String,
    /// Loaded kernel modules
    pub kernel_modules: Vec<String>,
    /// Driver capabilities
    pub capabilities: Vec<String>,
}

/// Per-process GPU context state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessGpuContext {
    /// Process ID
    pub pid: u32,
    /// Process name/command
    pub process_name: String,
    /// CUDA contexts
    pub cuda_contexts: Vec<CudaContext>,
    /// OpenGL contexts
    pub opengl_contexts: Vec<OpenGlContext>,
    /// Vulkan instances
    pub vulkan_instances: Vec<VulkanInstance>,
}

/// GPU memory state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryState {
    /// Total GPU memory
    pub total_memory: u64,
    /// Used GPU memory
    pub used_memory: u64,
    /// Memory allocations by process
    pub allocations: HashMap<u32, Vec<MemoryAllocation>>,
    /// Shared memory regions
    pub shared_memory: Vec<SharedMemoryRegion>,
}

/// GPU performance and clock state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPerformanceState {
    /// Current performance level
    pub performance_level: u8,
    /// GPU utilization
    pub gpu_utilization: f32,
    /// Memory utilization
    pub memory_utilization: f32,
    /// Power draw
    pub power_draw_watts: f32,
    /// Custom performance profiles
    pub custom_profiles: HashMap<String, PerformanceProfile>,
}

/// Power state enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerState {
    P0, // Maximum performance
    P1, // Balanced
    P2, // Power saving
    P3, // Minimum power
    P8, // Idle
}

/// Clock state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClockState {
    /// Graphics clock (MHz)
    pub graphics_clock: u32,
    /// Memory clock (MHz)
    pub memory_clock: u32,
    /// Shader clock (MHz)
    pub shader_clock: u32,
    /// Video clock (MHz)
    pub video_clock: u32,
}

/// Thermal state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalState {
    /// GPU temperature (Celsius)
    pub gpu_temp: i32,
    /// Memory temperature (Celsius)
    pub memory_temp: Option<i32>,
    /// Power limit temperature (Celsius)
    pub power_temp: Option<i32>,
    /// Thermal throttling state
    pub throttling: bool,
}

/// Fan state configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanState {
    /// Fan speed percentage
    pub speed_percent: u8,
    /// Auto fan control enabled
    pub auto_control: bool,
    /// Custom fan curve
    pub fan_curve: Option<Vec<(i32, u8)>>, // (temp, speed) pairs
}

/// Display state for GUI snapshots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayState {
    /// Connected displays
    pub displays: Vec<DisplayInfo>,
    /// Digital vibrance settings
    pub digital_vibrance: HashMap<String, i32>,
    /// Resolution and refresh rate
    pub display_modes: HashMap<String, DisplayMode>,
}

/// Display information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayInfo {
    pub name: String,
    pub connected: bool,
    pub primary: bool,
    pub resolution: (u32, u32),
    pub refresh_rate: f32,
}

/// Display mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayMode {
    pub width: u32,
    pub height: u32,
    pub refresh_rate: f32,
    pub color_depth: u8,
}

/// CUDA context state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaContext {
    pub context_id: u64,
    pub device_id: u32,
    pub memory_usage: u64,
    pub active: bool,
}

/// OpenGL context state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenGlContext {
    pub context_id: u64,
    pub version: String,
    pub renderer: String,
    pub memory_usage: u64,
}

/// Vulkan instance state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulkanInstance {
    pub instance_id: u64,
    pub api_version: String,
    pub device_count: u32,
    pub memory_usage: u64,
}

/// Memory allocation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    pub address: u64,
    pub size: u64,
    pub allocation_type: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Shared memory region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMemoryRegion {
    pub address: u64,
    pub size: u64,
    pub sharing_processes: Vec<u32>,
}

/// Performance profile configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub name: String,
    pub power_limit: u32,
    pub memory_clock_offset: i32,
    pub graphics_clock_offset: i32,
    pub fan_curve: Option<Vec<(i32, u8)>>,
}

/// GPU snapshot manager for Bolt capsules
pub struct GpuSnapshotManager {
    snapshot_dir: PathBuf,
}

impl GpuSnapshotManager {
    /// Create new snapshot manager
    pub fn new<P: AsRef<Path>>(snapshot_dir: P) -> Result<Self> {
        let snapshot_dir = snapshot_dir.as_ref().to_path_buf();
        fs::create_dir_all(&snapshot_dir)
            .context("Failed to create snapshot directory")?;

        Ok(Self { snapshot_dir })
    }

    /// Create GPU state snapshot for container
    pub async fn create_snapshot(&self, container_id: &str) -> Result<GpuStateSnapshot> {
        info!("Creating GPU state snapshot for container: {}", container_id);

        let snapshot = GpuStateSnapshot {
            timestamp: chrono::Utc::now(),
            container_id: container_id.to_string(),
            gpu_devices: self.capture_gpu_device_states().await?,
            driver_state: self.capture_driver_state().await?,
            process_contexts: self.capture_process_contexts(container_id).await?,
            memory_state: self.capture_memory_state().await?,
            performance_state: self.capture_performance_state().await?,
        };

        // Save snapshot to disk
        self.save_snapshot(&snapshot).await?;

        info!("GPU snapshot created successfully for container: {}", container_id);
        Ok(snapshot)
    }

    /// Restore GPU state from snapshot
    pub async fn restore_snapshot(&self, container_id: &str) -> Result<()> {
        info!("Restoring GPU state for container: {}", container_id);

        let snapshot = self.load_snapshot(container_id).await?;

        // Restore GPU device states
        self.restore_gpu_device_states(&snapshot.gpu_devices).await?;

        // Restore performance settings
        self.restore_performance_state(&snapshot.performance_state).await?;

        // Restore process contexts (if possible)
        self.restore_process_contexts(&snapshot.process_contexts).await?;

        info!("GPU state restored successfully for container: {}", container_id);
        Ok(())
    }

    /// Capture current GPU device states
    async fn capture_gpu_device_states(&self) -> Result<Vec<GpuDeviceState>> {
        debug!("Capturing GPU device states");

        let gpus = crate::gpu::discover_gpus().await?;
        let mut device_states = Vec::new();

        for gpu in gpus {
            let device_state = GpuDeviceState {
                device_id: gpu.id.clone(),
                pci_id: gpu.pci_address.clone(),
                device_path: gpu.device_path.clone(),
                power_state: self.get_power_state(&gpu.id).await?,
                clock_state: self.get_clock_state(&gpu.id).await?,
                thermal_state: self.get_thermal_state(&gpu.id).await?,
                fan_state: self.get_fan_state(&gpu.id).await?,
                display_state: self.get_display_state(&gpu.id).await.ok(),
            };

            device_states.push(device_state);
        }

        Ok(device_states)
    }

    /// Capture NVIDIA driver state
    async fn capture_driver_state(&self) -> Result<DriverState> {
        debug!("Capturing driver state");

        let driver_info = crate::gpu::get_driver_info().await?;

        // Get loaded kernel modules
        let lsmod_output = Command::new("lsmod")
            .output()
            .context("Failed to run lsmod")?;

        let lsmod_str = String::from_utf8_lossy(&lsmod_output.stdout);
        let nvidia_modules: Vec<String> = lsmod_str
            .lines()
            .filter(|line| line.contains("nvidia"))
            .map(|line| line.split_whitespace().next().unwrap_or("").to_string())
            .filter(|module| !module.is_empty())
            .collect();

        Ok(DriverState {
            version: driver_info.version,
            driver_type: format!("{:?}", driver_info.driver_type),
            kernel_modules: nvidia_modules,
            capabilities: vec![
                "CUDA".to_string(),
                "OpenGL".to_string(),
                "Vulkan".to_string(),
                "Video Decode".to_string(),
                "Video Encode".to_string(),
            ],
        })
    }

    /// Capture process GPU contexts for container
    async fn capture_process_contexts(&self, container_id: &str) -> Result<Vec<ProcessGpuContext>> {
        debug!("Capturing process contexts for container: {}", container_id);

        // Get processes running in container
        let processes = self.get_container_processes(container_id).await?;
        let mut contexts = Vec::new();

        for pid in processes {
            if let Ok(context) = self.get_process_gpu_context(pid).await {
                contexts.push(context);
            }
        }

        Ok(contexts)
    }

    /// Capture GPU memory state
    async fn capture_memory_state(&self) -> Result<GpuMemoryState> {
        debug!("Capturing GPU memory state");

        // Use nvidia-ml-py equivalent or nvidia-smi to get memory info
        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"])
            .output()
            .context("Failed to query GPU memory")?;

        let memory_info = String::from_utf8_lossy(&output.stdout);
        let (total, used) = if let Some(line) = memory_info.lines().next() {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 2 {
                let total = parts[0].trim().parse::<u64>().unwrap_or(0) * 1024 * 1024; // Convert MB to bytes
                let used = parts[1].trim().parse::<u64>().unwrap_or(0) * 1024 * 1024;
                (total, used)
            } else {
                (0, 0)
            }
        } else {
            (0, 0)
        };

        Ok(GpuMemoryState {
            total_memory: total,
            used_memory: used,
            allocations: HashMap::new(), // TODO: Implement detailed allocation tracking
            shared_memory: Vec::new(),
        })
    }

    /// Capture GPU performance state
    async fn capture_performance_state(&self) -> Result<GpuPerformanceState> {
        debug!("Capturing GPU performance state");

        // Query GPU utilization and power
        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=utilization.gpu,utilization.memory,power.draw", "--format=csv,noheader,nounits"])
            .output()
            .context("Failed to query GPU performance")?;

        let perf_info = String::from_utf8_lossy(&output.stdout);
        let (gpu_util, mem_util, power) = if let Some(line) = perf_info.lines().next() {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 3 {
                let gpu_util = parts[0].trim().parse::<f32>().unwrap_or(0.0);
                let mem_util = parts[1].trim().parse::<f32>().unwrap_or(0.0);
                let power = parts[2].trim().parse::<f32>().unwrap_or(0.0);
                (gpu_util, mem_util, power)
            } else {
                (0.0, 0.0, 0.0)
            }
        } else {
            (0.0, 0.0, 0.0)
        };

        Ok(GpuPerformanceState {
            performance_level: 0, // TODO: Implement P-state detection
            gpu_utilization: gpu_util,
            memory_utilization: mem_util,
            power_draw_watts: power,
            custom_profiles: HashMap::new(),
        })
    }

    /// Get power state for GPU
    async fn get_power_state(&self, _gpu_id: &str) -> Result<PowerState> {
        // TODO: Implement actual power state detection
        Ok(PowerState::P0)
    }

    /// Get clock state for GPU
    async fn get_clock_state(&self, _gpu_id: &str) -> Result<ClockState> {
        // TODO: Query actual clock frequencies
        Ok(ClockState {
            graphics_clock: 1500,
            memory_clock: 7000,
            shader_clock: 1500,
            video_clock: 1200,
        })
    }

    /// Get thermal state for GPU
    async fn get_thermal_state(&self, _gpu_id: &str) -> Result<ThermalState> {
        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
            .output()
            .context("Failed to query GPU temperature")?;

        let temp_str = String::from_utf8_lossy(&output.stdout);
        let gpu_temp = temp_str.trim().parse::<i32>().unwrap_or(0);

        Ok(ThermalState {
            gpu_temp,
            memory_temp: None,
            power_temp: None,
            throttling: false,
        })
    }

    /// Get fan state for GPU
    async fn get_fan_state(&self, _gpu_id: &str) -> Result<FanState> {
        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=fan.speed", "--format=csv,noheader,nounits"])
            .output()
            .context("Failed to query fan speed")?;

        let fan_str = String::from_utf8_lossy(&output.stdout);
        let speed = fan_str.trim().parse::<u8>().unwrap_or(0);

        Ok(FanState {
            speed_percent: speed,
            auto_control: true,
            fan_curve: None,
        })
    }

    /// Get display state for GPU
    async fn get_display_state(&self, _gpu_id: &str) -> Result<DisplayState> {
        // TODO: Implement display state capture using xrandr or wayland protocols
        Ok(DisplayState {
            displays: Vec::new(),
            digital_vibrance: HashMap::new(),
            display_modes: HashMap::new(),
        })
    }

    /// Get processes running in container
    async fn get_container_processes(&self, _container_id: &str) -> Result<Vec<u32>> {
        // TODO: Query container runtime for process list
        Ok(Vec::new())
    }

    /// Get GPU context for process
    async fn get_process_gpu_context(&self, pid: u32) -> Result<ProcessGpuContext> {
        // TODO: Use nvidia-ml-py or similar to get process GPU usage
        Ok(ProcessGpuContext {
            pid,
            process_name: format!("process-{}", pid),
            cuda_contexts: Vec::new(),
            opengl_contexts: Vec::new(),
            vulkan_instances: Vec::new(),
        })
    }

    /// Restore GPU device states
    async fn restore_gpu_device_states(&self, _device_states: &[GpuDeviceState]) -> Result<()> {
        debug!("Restoring GPU device states");
        // TODO: Implement state restoration
        Ok(())
    }

    /// Restore performance state
    async fn restore_performance_state(&self, _performance_state: &GpuPerformanceState) -> Result<()> {
        debug!("Restoring performance state");
        // TODO: Implement performance state restoration
        Ok(())
    }

    /// Restore process contexts
    async fn restore_process_contexts(&self, _contexts: &[ProcessGpuContext]) -> Result<()> {
        debug!("Restoring process contexts");
        // TODO: Implement context restoration where possible
        Ok(())
    }

    /// Save snapshot to disk
    async fn save_snapshot(&self, snapshot: &GpuStateSnapshot) -> Result<()> {
        let filename = format!("{}.snapshot.json", snapshot.container_id);
        let filepath = self.snapshot_dir.join(filename);

        let snapshot_json = serde_json::to_string_pretty(snapshot)
            .context("Failed to serialize snapshot")?;

        fs::write(&filepath, snapshot_json)
            .context("Failed to write snapshot file")?;

        debug!("Snapshot saved to: {:?}", filepath);
        Ok(())
    }

    /// Load snapshot from disk
    async fn load_snapshot(&self, container_id: &str) -> Result<GpuStateSnapshot> {
        let filename = format!("{}.snapshot.json", container_id);
        let filepath = self.snapshot_dir.join(filename);

        let snapshot_json = fs::read_to_string(&filepath)
            .context("Failed to read snapshot file")?;

        let snapshot: GpuStateSnapshot = serde_json::from_str(&snapshot_json)
            .context("Failed to deserialize snapshot")?;

        debug!("Snapshot loaded from: {:?}", filepath);
        Ok(snapshot)
    }

    /// List available snapshots
    pub async fn list_snapshots(&self) -> Result<Vec<String>> {
        let entries = fs::read_dir(&self.snapshot_dir)
            .context("Failed to read snapshot directory")?;

        let mut snapshots = Vec::new();
        for entry in entries {
            let entry = entry?;
            let filename = entry.file_name();
            let filename_str = filename.to_string_lossy();

            if filename_str.ends_with(".snapshot.json") {
                let container_id = filename_str
                    .strip_suffix(".snapshot.json")
                    .unwrap_or("")
                    .to_string();
                snapshots.push(container_id);
            }
        }

        Ok(snapshots)
    }

    /// Delete snapshot
    pub async fn delete_snapshot(&self, container_id: &str) -> Result<()> {
        let filename = format!("{}.snapshot.json", container_id);
        let filepath = self.snapshot_dir.join(filename);

        if filepath.exists() {
            fs::remove_file(&filepath)
                .context("Failed to delete snapshot file")?;
            info!("Deleted snapshot for container: {}", container_id);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_snapshot_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = GpuSnapshotManager::new(temp_dir.path()).unwrap();

        // Test directory creation
        assert!(temp_dir.path().exists());
    }

    #[tokio::test]
    async fn test_snapshot_serialization() {
        let snapshot = GpuStateSnapshot {
            timestamp: chrono::Utc::now(),
            container_id: "test-container".to_string(),
            gpu_devices: Vec::new(),
            driver_state: DriverState {
                version: "580.00".to_string(),
                driver_type: "NvidiaOpen".to_string(),
                kernel_modules: vec!["nvidia".to_string()],
                capabilities: vec!["CUDA".to_string()],
            },
            process_contexts: Vec::new(),
            memory_state: GpuMemoryState {
                total_memory: 8 * 1024 * 1024 * 1024, // 8GB
                used_memory: 2 * 1024 * 1024 * 1024,  // 2GB
                allocations: HashMap::new(),
                shared_memory: Vec::new(),
            },
            performance_state: GpuPerformanceState {
                performance_level: 0,
                gpu_utilization: 50.0,
                memory_utilization: 25.0,
                power_draw_watts: 200.0,
                custom_profiles: HashMap::new(),
            },
        };

        // Test JSON serialization
        let json = serde_json::to_string(&snapshot).unwrap();
        let deserialized: GpuStateSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(snapshot.container_id, deserialized.container_id);
        assert_eq!(snapshot.memory_state.total_memory, deserialized.memory_state.total_memory);
    }
}