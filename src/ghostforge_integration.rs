use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// GhostForge Integration Module
/// Provides gaming container profile API, real-time performance metrics,
/// Wine/Proton optimization, Steam library detection, and container status reporting

/// Gaming Container Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingContainerProfile {
    /// Profile ID
    pub id: String,
    /// Profile name
    pub name: String,
    /// Game title
    pub game_title: Option<String>,
    /// Container configuration
    pub container_config: ContainerConfig,
    /// GPU optimization settings
    pub gpu_optimization: GpuOptimization,
    /// Wine/Proton configuration
    pub wine_proton_config: Option<WineProtonConfig>,
    /// Performance settings
    pub performance_settings: PerformanceSettings,
    /// Anti-cheat compatibility
    pub anti_cheat_config: Option<AntiCheatConfig>,
    /// VR support configuration
    pub vr_config: Option<VrConfig>,
    /// Custom environment variables
    pub environment_variables: HashMap<String, String>,
    /// Profile creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Profile last modified timestamp
    pub modified_at: chrono::DateTime<chrono::Utc>,
}

/// Container Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    /// Base image
    pub base_image: String,
    /// Container runtime
    pub runtime: ContainerRuntime,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Volume mounts
    pub volume_mounts: Vec<VolumeMount>,
    /// Network configuration
    pub network_config: NetworkConfig,
    /// Security context
    pub security_context: SecurityContext,
}

/// Container Runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerRuntime {
    Docker,
    Podman,
    Bolt,
}

/// Resource Limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// CPU cores
    pub cpu_cores: Option<f64>,
    /// Memory in bytes
    pub memory_bytes: Option<u64>,
    /// GPU memory in bytes
    pub gpu_memory_bytes: Option<u64>,
    /// Disk space in bytes
    pub disk_space_bytes: Option<u64>,
}

/// Volume Mount
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeMount {
    /// Host path
    pub host_path: String,
    /// Container path
    pub container_path: String,
    /// Mount options
    pub options: Vec<String>,
    /// Read-only flag
    pub read_only: bool,
}

/// Network Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Network mode
    pub mode: NetworkMode,
    /// Port mappings
    pub port_mappings: Vec<PortMapping>,
    /// DNS servers
    pub dns_servers: Vec<String>,
}

/// Network Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkMode {
    Bridge,
    Host,
    None,
    Custom(String),
}

/// Port Mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortMapping {
    /// Host port
    pub host_port: u16,
    /// Container port
    pub container_port: u16,
    /// Protocol
    pub protocol: Protocol,
}

/// Protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Protocol {
    Tcp,
    Udp,
}

/// Security Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// Run as user
    pub run_as_user: Option<u32>,
    /// Run as group
    pub run_as_group: Option<u32>,
    /// Privileged mode
    pub privileged: bool,
    /// Capabilities to add
    pub capabilities_add: Vec<String>,
    /// Capabilities to drop
    pub capabilities_drop: Vec<String>,
}

/// GPU Optimization Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOptimization {
    /// GPU selection strategy
    pub gpu_selection: GpuSelectionStrategy,
    /// Memory allocation strategy
    pub memory_strategy: GpuMemoryStrategy,
    /// Performance profile
    pub performance_profile: GpuPerformanceProfile,
    /// Power management
    pub power_management: GpuPowerManagement,
    /// Multi-GPU configuration
    pub multi_gpu_config: Option<MultiGpuConfig>,
}

/// GPU Selection Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuSelectionStrategy {
    /// Use fastest available GPU
    Fastest,
    /// Use least utilized GPU
    LeastUtilized,
    /// Use specific GPU by ID
    Specific(String),
    /// Use all available GPUs
    All,
    /// Custom selection criteria
    Custom(GpuSelectionCriteria),
}

/// GPU Selection Criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSelectionCriteria {
    /// Minimum VRAM in bytes
    pub min_vram_bytes: Option<u64>,
    /// Minimum compute capability
    pub min_compute_capability: Option<String>,
    /// Preferred vendor
    pub preferred_vendor: Option<String>,
    /// Maximum power consumption
    pub max_power_watts: Option<u32>,
}

/// GPU Memory Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuMemoryStrategy {
    /// Allocate all available memory
    All,
    /// Allocate fixed amount
    Fixed(u64),
    /// Allocate percentage of total
    Percentage(f64),
    /// Dynamic allocation based on usage
    Dynamic,
}

/// GPU Performance Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuPerformanceProfile {
    /// Maximum performance
    MaxPerformance,
    /// Balanced performance and efficiency
    Balanced,
    /// Power efficient
    PowerEfficient,
    /// Custom profile
    Custom(CustomGpuProfile),
}

/// Custom GPU Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomGpuProfile {
    /// GPU clock speed in MHz
    pub gpu_clock_mhz: Option<u32>,
    /// Memory clock speed in MHz
    pub memory_clock_mhz: Option<u32>,
    /// Power limit in watts
    pub power_limit_watts: Option<u32>,
    /// Fan curve settings
    pub fan_curve: Option<Vec<FanCurvePoint>>,
}

/// Fan Curve Point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanCurvePoint {
    /// Temperature in Celsius
    pub temperature_celsius: u32,
    /// Fan speed percentage
    pub fan_speed_percent: u32,
}

/// GPU Power Management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPowerManagement {
    /// Power profile
    pub profile: PowerProfile,
    /// Automatic power management
    pub auto_power_management: bool,
    /// Idle timeout in seconds
    pub idle_timeout_seconds: Option<u32>,
}

/// Power Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerProfile {
    MaxPerformance,
    Balanced,
    PowerSaver,
}

/// Multi-GPU Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiGpuConfig {
    /// Multi-GPU mode
    pub mode: MultiGpuMode,
    /// GPU synchronization
    pub synchronization: GpuSynchronization,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Multi-GPU Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiGpuMode {
    /// SLI/CrossFire
    Sli,
    /// Independent GPUs
    Independent,
    /// Compute mode
    Compute,
}

/// GPU Synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuSynchronization {
    /// Hardware synchronization
    Hardware,
    /// Software synchronization
    Software,
    /// No synchronization
    None,
}

/// Load Balancing Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round robin
    RoundRobin,
    /// Least loaded
    LeastLoaded,
    /// Workload specific
    WorkloadSpecific,
}

/// Wine/Proton Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WineProtonConfig {
    /// Wine/Proton version
    pub version: WineProtonVersion,
    /// Wine prefix path
    pub wine_prefix: PathBuf,
    /// DXVK configuration
    pub dxvk_config: Option<DxvkConfig>,
    /// VKD3D configuration
    pub vkd3d_config: Option<Vkd3dConfig>,
    /// Wine registry settings
    pub registry_settings: HashMap<String, String>,
    /// DLL overrides
    pub dll_overrides: HashMap<String, String>,
    /// Environment variables
    pub environment_variables: HashMap<String, String>,
    /// Proton-specific settings
    pub proton_settings: Option<ProtonSettings>,
}

/// Wine/Proton Version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WineProtonVersion {
    /// System Wine
    SystemWine,
    /// Wine Staging
    WineStaging(String),
    /// Proton
    Proton(String),
    /// Proton GE
    ProtonGe(String),
    /// Custom build
    Custom(String),
}

/// DXVK Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DxvkConfig {
    /// DXVK version
    pub version: String,
    /// DXVK HUD settings
    pub hud: Option<String>,
    /// Memory allocation options
    pub memory_allocation: DxvkMemoryAllocation,
    /// Debug options
    pub debug_options: Vec<String>,
    /// Performance options
    pub performance_options: HashMap<String, String>,
}

/// DXVK Memory Allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DxvkMemoryAllocation {
    /// Device memory allocation
    pub device_memory_mb: Option<u32>,
    /// Shared memory allocation
    pub shared_memory_mb: Option<u32>,
}

/// VKD3D Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vkd3dConfig {
    /// VKD3D version
    pub version: String,
    /// Debug layer
    pub debug_layer: bool,
    /// Feature level
    pub feature_level: Option<String>,
    /// Configuration options
    pub options: HashMap<String, String>,
}

/// Proton Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtonSettings {
    /// Use large address aware
    pub large_address_aware: bool,
    /// Disable esync
    pub disable_esync: bool,
    /// Disable fsync
    pub disable_fsync: bool,
    /// Force Windows version
    pub force_windows_version: Option<String>,
    /// Steam app ID
    pub steam_app_id: Option<u32>,
}

/// Performance Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Target FPS
    pub target_fps: Option<u32>,
    /// VSync mode
    pub vsync_mode: VsyncMode,
    /// Render scaling
    pub render_scaling: f64,
    /// Texture quality
    pub texture_quality: TextureQuality,
    /// Shadow quality
    pub shadow_quality: ShadowQuality,
    /// Anti-aliasing
    pub anti_aliasing: AntiAliasing,
    /// Anisotropic filtering
    pub anisotropic_filtering: AnisotropicFiltering,
}

/// VSync Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VsyncMode {
    Off,
    On,
    Adaptive,
    FastSync,
}

/// Texture Quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TextureQuality {
    Low,
    Medium,
    High,
    Ultra,
}

/// Shadow Quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShadowQuality {
    Off,
    Low,
    Medium,
    High,
    Ultra,
}

/// Anti-Aliasing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AntiAliasing {
    Off,
    Fxaa,
    Msaa2x,
    Msaa4x,
    Msaa8x,
    Taa,
    Dlss,
    Fsr,
}

/// Anisotropic Filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnisotropicFiltering {
    Off,
    X2,
    X4,
    X8,
    X16,
}

/// Anti-Cheat Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiCheatConfig {
    /// Anti-cheat system
    pub system: AntiCheatSystem,
    /// Compatibility mode
    pub compatibility_mode: AntiCheatCompatibilityMode,
    /// Bypass options
    pub bypass_options: Vec<String>,
    /// Kernel-level support
    pub kernel_level_support: bool,
}

/// Anti-Cheat System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AntiCheatSystem {
    EasyAntiCheat,
    BattlEye,
    Vanguard,
    Fairfight,
    PunkBuster,
    Custom(String),
}

/// Anti-Cheat Compatibility Mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AntiCheatCompatibilityMode {
    /// Native Linux support
    Native,
    /// Wine compatibility layer
    Wine,
    /// Virtualized environment
    Virtualized,
    /// Bypass mode
    Bypass,
}

/// VR Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrConfig {
    /// VR runtime
    pub runtime: VrRuntime,
    /// Headset passthrough
    pub headset_passthrough: bool,
    /// Render target resolution
    pub render_resolution: VrResolution,
    /// Refresh rate
    pub refresh_rate: u32,
    /// IPD adjustment
    pub ipd_mm: Option<f64>,
    /// Room scale setup
    pub room_scale: bool,
}

/// VR Runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VrRuntime {
    SteamVr,
    OpenXr,
    Oculus,
    WMR,
}

/// VR Resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrResolution {
    /// Width per eye
    pub width: u32,
    /// Height per eye
    pub height: u32,
}

/// Real-time Performance Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Container ID
    pub container_id: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// GPU metrics
    pub gpu_metrics: GpuMetrics,
    /// CPU metrics
    pub cpu_metrics: CpuMetrics,
    /// Memory metrics
    pub memory_metrics: MemoryMetrics,
    /// Network metrics
    pub network_metrics: NetworkMetrics,
    /// Game-specific metrics
    pub game_metrics: Option<GameMetrics>,
}

/// GPU Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU utilization percentage
    pub utilization_percent: f64,
    /// GPU memory usage in bytes
    pub memory_used_bytes: u64,
    /// GPU memory total in bytes
    pub memory_total_bytes: u64,
    /// GPU temperature in Celsius
    pub temperature_celsius: f64,
    /// GPU power consumption in watts
    pub power_consumption_watts: f64,
    /// GPU clock speed in MHz
    pub clock_speed_mhz: u32,
    /// Memory clock speed in MHz
    pub memory_clock_speed_mhz: u32,
    /// Fan speed percentage
    pub fan_speed_percent: f64,
}

/// CPU Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    /// CPU utilization percentage
    pub utilization_percent: f64,
    /// CPU temperature in Celsius
    pub temperature_celsius: f64,
    /// CPU frequency in MHz
    pub frequency_mhz: u32,
    /// Load average
    pub load_average: [f64; 3],
}

/// Memory Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Memory usage in bytes
    pub used_bytes: u64,
    /// Total memory in bytes
    pub total_bytes: u64,
    /// Available memory in bytes
    pub available_bytes: u64,
    /// Swap usage in bytes
    pub swap_used_bytes: u64,
    /// Total swap in bytes
    pub swap_total_bytes: u64,
}

/// Network Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Bytes received
    pub bytes_received: u64,
    /// Bytes transmitted
    pub bytes_transmitted: u64,
    /// Packets received
    pub packets_received: u64,
    /// Packets transmitted
    pub packets_transmitted: u64,
    /// Network latency in milliseconds
    pub latency_ms: Option<f64>,
}

/// Game-specific Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameMetrics {
    /// Current FPS
    pub current_fps: f64,
    /// Average FPS
    pub average_fps: f64,
    /// Minimum FPS
    pub minimum_fps: f64,
    /// Maximum FPS
    pub maximum_fps: f64,
    /// Frame time in milliseconds
    pub frame_time_ms: f64,
    /// Input latency in milliseconds
    pub input_latency_ms: Option<f64>,
    /// Render latency in milliseconds
    pub render_latency_ms: Option<f64>,
}

/// Container Status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerStatus {
    /// Container ID
    pub container_id: String,
    /// Container name
    pub container_name: String,
    /// Current state
    pub state: ContainerState,
    /// Health status
    pub health: ContainerHealth,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Exit code if stopped
    pub exit_code: Option<i32>,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Container State
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerState {
    Starting,
    Running,
    Paused,
    Stopping,
    Stopped,
    Error(String),
}

/// Container Health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerHealth {
    Healthy,
    Unhealthy,
    Starting,
    Unknown,
}

/// Resource Usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage percentage
    pub cpu_percent: f64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// GPU usage percentage
    pub gpu_percent: f64,
    /// Network I/O in bytes per second
    pub network_io_bps: u64,
    /// Disk I/O in bytes per second
    pub disk_io_bps: u64,
}

/// Steam Library Detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteamLibraryInfo {
    /// Steam installation path
    pub steam_path: PathBuf,
    /// Library folders
    pub library_folders: Vec<SteamLibraryFolder>,
    /// Installed games
    pub installed_games: Vec<SteamGame>,
    /// User profiles
    pub user_profiles: Vec<SteamUserProfile>,
}

/// Steam Library Folder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteamLibraryFolder {
    /// Folder path
    pub path: PathBuf,
    /// Available space in bytes
    pub available_space_bytes: u64,
    /// Total space in bytes
    pub total_space_bytes: u64,
}

/// Steam Game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteamGame {
    /// App ID
    pub app_id: u32,
    /// Game name
    pub name: String,
    /// Installation path
    pub install_path: PathBuf,
    /// Executable path
    pub executable_path: Option<PathBuf>,
    /// Size in bytes
    pub size_bytes: u64,
    /// Last played timestamp
    pub last_played: Option<chrono::DateTime<chrono::Utc>>,
    /// Play time in minutes
    pub play_time_minutes: u64,
    /// ProtonDB compatibility rating
    pub protondb_rating: Option<ProtonDbRating>,
}

/// ProtonDB Compatibility Rating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtonDbRating {
    Native,
    Platinum,
    Gold,
    Silver,
    Bronze,
    Borked,
    Unknown,
}

/// Steam User Profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteamUserProfile {
    /// User ID
    pub user_id: String,
    /// Display name
    pub display_name: String,
    /// Profile path
    pub profile_path: PathBuf,
    /// Last login timestamp
    pub last_login: Option<chrono::DateTime<chrono::Utc>>,
}

/// GhostForge Integration Manager
pub struct GhostForgeIntegration {
    /// Gaming container profiles
    profiles: Arc<RwLock<HashMap<String, GamingContainerProfile>>>,
    /// Performance metrics cache
    metrics_cache: Arc<RwLock<HashMap<String, PerformanceMetrics>>>,
    /// Container status cache
    status_cache: Arc<RwLock<HashMap<String, ContainerStatus>>>,
    /// Steam library information
    steam_library: Arc<RwLock<Option<SteamLibraryInfo>>>,
    /// Wine/Proton installations
    wine_proton_installations: Arc<RwLock<HashMap<String, PathBuf>>>,
}

impl GhostForgeIntegration {
    /// Create a new GhostForge Integration manager
    pub fn new() -> Self {
        Self {
            profiles: Arc::new(RwLock::new(HashMap::new())),
            metrics_cache: Arc::new(RwLock::new(HashMap::new())),
            status_cache: Arc::new(RwLock::new(HashMap::new())),
            steam_library: Arc::new(RwLock::new(None)),
            wine_proton_installations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new gaming container profile
    pub async fn create_gaming_profile(
        &self,
        name: String,
        game_title: Option<String>,
        container_config: ContainerConfig,
        gpu_optimization: GpuOptimization,
    ) -> Result<String> {
        let profile_id = Uuid::new_v4().to_string();
        let now = chrono::Utc::now();

        let profile = GamingContainerProfile {
            id: profile_id.clone(),
            name,
            game_title,
            container_config,
            gpu_optimization,
            wine_proton_config: None,
            performance_settings: PerformanceSettings {
                target_fps: Some(60),
                vsync_mode: VsyncMode::Adaptive,
                render_scaling: 1.0,
                texture_quality: TextureQuality::High,
                shadow_quality: ShadowQuality::Medium,
                anti_aliasing: AntiAliasing::Fxaa,
                anisotropic_filtering: AnisotropicFiltering::X8,
            },
            anti_cheat_config: None,
            vr_config: None,
            environment_variables: HashMap::new(),
            created_at: now,
            modified_at: now,
        };

        let mut profiles = self.profiles.write().unwrap();
        profiles.insert(profile_id.clone(), profile);

        info!("Created gaming container profile: {}", profile_id);
        Ok(profile_id)
    }

    /// Get gaming container profile
    pub fn get_gaming_profile(&self, profile_id: &str) -> Option<GamingContainerProfile> {
        let profiles = self.profiles.read().unwrap();
        profiles.get(profile_id).cloned()
    }

    /// Update gaming container profile
    pub async fn update_gaming_profile(
        &self,
        profile_id: &str,
        updates: GamingProfileUpdates,
    ) -> Result<()> {
        let mut profiles = self.profiles.write().unwrap();
        let profile = profiles.get_mut(profile_id)
            .ok_or_else(|| anyhow!("Gaming profile {} not found", profile_id))?;

        if let Some(name) = updates.name {
            profile.name = name;
        }

        if let Some(game_title) = updates.game_title {
            profile.game_title = Some(game_title);
        }

        if let Some(gpu_optimization) = updates.gpu_optimization {
            profile.gpu_optimization = gpu_optimization;
        }

        if let Some(wine_proton_config) = updates.wine_proton_config {
            profile.wine_proton_config = Some(wine_proton_config);
        }

        if let Some(performance_settings) = updates.performance_settings {
            profile.performance_settings = performance_settings;
        }

        profile.modified_at = chrono::Utc::now();

        info!("Updated gaming profile: {}", profile_id);
        Ok(())
    }

    /// Delete gaming container profile
    pub async fn delete_gaming_profile(&self, profile_id: &str) -> Result<()> {
        let mut profiles = self.profiles.write().unwrap();
        profiles.remove(profile_id)
            .ok_or_else(|| anyhow!("Gaming profile {} not found", profile_id))?;

        info!("Deleted gaming profile: {}", profile_id);
        Ok(())
    }

    /// List all gaming profiles
    pub fn list_gaming_profiles(&self) -> Vec<GamingContainerProfile> {
        let profiles = self.profiles.read().unwrap();
        profiles.values().cloned().collect()
    }

    /// Collect real-time performance metrics
    pub async fn collect_performance_metrics(&self, container_id: &str) -> Result<PerformanceMetrics> {
        debug!("Collecting performance metrics for container: {}", container_id);

        // Collect GPU metrics
        let gpu_metrics = self.collect_gpu_metrics(container_id).await?;

        // Collect CPU metrics
        let cpu_metrics = self.collect_cpu_metrics(container_id).await?;

        // Collect memory metrics
        let memory_metrics = self.collect_memory_metrics(container_id).await?;

        // Collect network metrics
        let network_metrics = self.collect_network_metrics(container_id).await?;

        // Collect game-specific metrics if available
        let game_metrics = self.collect_game_metrics(container_id).await.ok();

        let metrics = PerformanceMetrics {
            container_id: container_id.to_string(),
            timestamp: chrono::Utc::now(),
            gpu_metrics,
            cpu_metrics,
            memory_metrics,
            network_metrics,
            game_metrics,
        };

        // Cache metrics
        let mut cache = self.metrics_cache.write().unwrap();
        cache.insert(container_id.to_string(), metrics.clone());

        Ok(metrics)
    }

    /// Collect GPU metrics
    async fn collect_gpu_metrics(&self, _container_id: &str) -> Result<GpuMetrics> {
        // Implementation would use nvidia-ml-py or similar
        Ok(GpuMetrics {
            utilization_percent: 75.0,
            memory_used_bytes: 4 * 1024 * 1024 * 1024, // 4GB
            memory_total_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            temperature_celsius: 68.0,
            power_consumption_watts: 180.0,
            clock_speed_mhz: 1800,
            memory_clock_speed_mhz: 7000,
            fan_speed_percent: 60.0,
        })
    }

    /// Collect CPU metrics
    async fn collect_cpu_metrics(&self, _container_id: &str) -> Result<CpuMetrics> {
        // Implementation would use system APIs
        Ok(CpuMetrics {
            utilization_percent: 45.0,
            temperature_celsius: 55.0,
            frequency_mhz: 3600,
            load_average: [1.2, 1.5, 1.8],
        })
    }

    /// Collect memory metrics
    async fn collect_memory_metrics(&self, _container_id: &str) -> Result<MemoryMetrics> {
        // Implementation would use system APIs
        Ok(MemoryMetrics {
            used_bytes: 8 * 1024 * 1024 * 1024,  // 8GB
            total_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            available_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            swap_used_bytes: 0,
            swap_total_bytes: 4 * 1024 * 1024 * 1024, // 4GB
        })
    }

    /// Collect network metrics
    async fn collect_network_metrics(&self, _container_id: &str) -> Result<NetworkMetrics> {
        // Implementation would use network monitoring
        Ok(NetworkMetrics {
            bytes_received: 1024 * 1024 * 100, // 100MB
            bytes_transmitted: 1024 * 1024 * 50, // 50MB
            packets_received: 10000,
            packets_transmitted: 8000,
            latency_ms: Some(25.0),
        })
    }

    /// Collect game-specific metrics
    async fn collect_game_metrics(&self, _container_id: &str) -> Result<GameMetrics> {
        // Implementation would hook into game engines or use overlays
        Ok(GameMetrics {
            current_fps: 75.0,
            average_fps: 72.5,
            minimum_fps: 60.0,
            maximum_fps: 85.0,
            frame_time_ms: 13.3,
            input_latency_ms: Some(35.0),
            render_latency_ms: Some(25.0),
        })
    }

    /// Get container status
    pub async fn get_container_status(&self, container_id: &str) -> Result<ContainerStatus> {
        debug!("Getting container status for: {}", container_id);

        // Check cache first
        {
            let cache = self.status_cache.read().unwrap();
            if let Some(status) = cache.get(container_id) {
                return Ok(status.clone());
            }
        }

        // Collect fresh status
        let status = self.collect_container_status(container_id).await?;

        // Cache status
        let mut cache = self.status_cache.write().unwrap();
        cache.insert(container_id.to_string(), status.clone());

        Ok(status)
    }

    /// Collect container status
    async fn collect_container_status(&self, container_id: &str) -> Result<ContainerStatus> {
        // Implementation would use container runtime APIs
        Ok(ContainerStatus {
            container_id: container_id.to_string(),
            container_name: format!("gaming-container-{}", &container_id[..8]),
            state: ContainerState::Running,
            health: ContainerHealth::Healthy,
            uptime_seconds: 3600, // 1 hour
            exit_code: None,
            resource_usage: ResourceUsage {
                cpu_percent: 45.0,
                memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
                gpu_percent: 75.0,
                network_io_bps: 1024 * 1024, // 1MB/s
                disk_io_bps: 512 * 1024,     // 512KB/s
            },
            last_updated: chrono::Utc::now(),
        })
    }

    /// Detect Steam library
    pub async fn detect_steam_library(&self) -> Result<SteamLibraryInfo> {
        info!("Detecting Steam library installation");

        let steam_path = self.find_steam_installation().await?;
        let library_folders = self.discover_steam_library_folders(&steam_path).await?;
        let installed_games = self.scan_installed_games(&library_folders).await?;
        let user_profiles = self.discover_user_profiles(&steam_path).await?;

        let library_info = SteamLibraryInfo {
            steam_path,
            library_folders,
            installed_games,
            user_profiles,
        };

        // Cache library information
        let mut cache = self.steam_library.write().unwrap();
        *cache = Some(library_info.clone());

        info!("Steam library detection completed");
        Ok(library_info)
    }

    /// Find Steam installation
    async fn find_steam_installation(&self) -> Result<PathBuf> {
        let possible_paths = vec![
            PathBuf::from("~/.steam"),
            PathBuf::from("~/.local/share/Steam"),
            PathBuf::from("/usr/share/steam"),
            PathBuf::from("/opt/steam"),
        ];

        for path in possible_paths {
            if path.exists() {
                return Ok(path);
            }
        }

        Err(anyhow!("Steam installation not found"))
    }

    /// Discover Steam library folders
    async fn discover_steam_library_folders(&self, steam_path: &PathBuf) -> Result<Vec<SteamLibraryFolder>> {
        let mut folders = Vec::new();

        // Add default Steam library folder
        let default_folder = steam_path.join("steamapps");
        if default_folder.exists() {
            folders.push(SteamLibraryFolder {
                path: default_folder.clone(),
                available_space_bytes: 0, // Would be calculated in real implementation
                total_space_bytes: 0,     // Would be calculated in real implementation
            });
        }

        // Scan for additional library folders in libraryfolders.vdf
        let library_config = steam_path.join("config").join("libraryfolders.vdf");
        if library_config.exists() {
            // Parse VDF file to find additional library folders
            // Implementation would parse the VDF format
        }

        Ok(folders)
    }

    /// Scan installed games
    async fn scan_installed_games(&self, library_folders: &[SteamLibraryFolder]) -> Result<Vec<SteamGame>> {
        let mut games = Vec::new();

        for folder in library_folders {
            let steamapps_path = folder.path.join("steamapps");
            if !steamapps_path.exists() {
                continue;
            }

            // Scan for .acf files (Steam app cache files)
            if let Ok(entries) = std::fs::read_dir(&steamapps_path) {
                for entry in entries.flatten() {
                    if let Some(extension) = entry.path().extension() {
                        if extension == "acf" {
                            if let Ok(game) = self.parse_acf_file(&entry.path()).await {
                                games.push(game);
                            }
                        }
                    }
                }
            }
        }

        Ok(games)
    }

    /// Parse Steam ACF file
    async fn parse_acf_file(&self, acf_path: &PathBuf) -> Result<SteamGame> {
        // Implementation would parse ACF file format
        // This is a placeholder implementation
        Ok(SteamGame {
            app_id: 12345,
            name: "Example Game".to_string(),
            install_path: acf_path.parent().unwrap().join("common").join("ExampleGame"),
            executable_path: None,
            size_bytes: 1024 * 1024 * 1024, // 1GB
            last_played: None,
            play_time_minutes: 0,
            protondb_rating: Some(ProtonDbRating::Gold),
        })
    }

    /// Discover Steam user profiles
    async fn discover_user_profiles(&self, steam_path: &PathBuf) -> Result<Vec<SteamUserProfile>> {
        let mut profiles = Vec::new();

        let userdata_path = steam_path.join("userdata");
        if !userdata_path.exists() {
            return Ok(profiles);
        }

        if let Ok(entries) = std::fs::read_dir(&userdata_path) {
            for entry in entries.flatten() {
                if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                    if let Some(user_id) = entry.file_name().to_str() {
                        let profile = SteamUserProfile {
                            user_id: user_id.to_string(),
                            display_name: format!("User {}", user_id),
                            profile_path: entry.path(),
                            last_login: None,
                        };
                        profiles.push(profile);
                    }
                }
            }
        }

        Ok(profiles)
    }

    /// Get Steam library information
    pub fn get_steam_library(&self) -> Option<SteamLibraryInfo> {
        let cache = self.steam_library.read().unwrap();
        cache.clone()
    }

    /// Configure Wine/Proton for a game
    pub async fn configure_wine_proton(
        &self,
        profile_id: &str,
        config: WineProtonConfig,
    ) -> Result<()> {
        let mut profiles = self.profiles.write().unwrap();
        let profile = profiles.get_mut(profile_id)
            .ok_or_else(|| anyhow!("Gaming profile {} not found", profile_id))?;

        profile.wine_proton_config = Some(config);
        profile.modified_at = chrono::Utc::now();

        info!("Configured Wine/Proton for profile: {}", profile_id);
        Ok(())
    }

    /// Get cached performance metrics
    pub fn get_cached_metrics(&self, container_id: &str) -> Option<PerformanceMetrics> {
        let cache = self.metrics_cache.read().unwrap();
        cache.get(container_id).cloned()
    }

    /// Get cached container status
    pub fn get_cached_status(&self, container_id: &str) -> Option<ContainerStatus> {
        let cache = self.status_cache.read().unwrap();
        cache.get(container_id).cloned()
    }

    /// Clear cache for container
    pub fn clear_container_cache(&self, container_id: &str) {
        {
            let mut metrics_cache = self.metrics_cache.write().unwrap();
            metrics_cache.remove(container_id);
        }

        {
            let mut status_cache = self.status_cache.write().unwrap();
            status_cache.remove(container_id);
        }
    }
}

/// Gaming Profile Updates
#[derive(Debug, Clone)]
pub struct GamingProfileUpdates {
    pub name: Option<String>,
    pub game_title: Option<String>,
    pub gpu_optimization: Option<GpuOptimization>,
    pub wine_proton_config: Option<WineProtonConfig>,
    pub performance_settings: Option<PerformanceSettings>,
}

impl Default for GhostForgeIntegration {
    fn default() -> Self {
        Self::new()
    }
}