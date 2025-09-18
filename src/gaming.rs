//! Gaming-specific optimizations for nvbind
//! Provides Wayland, Wine, Proton, and gaming performance optimizations

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Gaming runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingConfig {
    /// Wayland display optimizations
    pub wayland: WaylandConfig,
    /// Wine/Proton configuration
    pub wine: WineConfig,
    /// GPU gaming optimizations
    pub gpu: GamingGpuConfig,
    /// Audio configuration
    pub audio: AudioConfig,
    /// Input device configuration
    pub input: InputConfig,
}

/// Wayland display server optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaylandConfig {
    /// Enable Wayland native support
    pub enabled: bool,
    /// Force Wayland over X11
    pub force_wayland: bool,
    /// VRR (Variable Refresh Rate) support
    pub vrr_enabled: bool,
    /// HDR support
    pub hdr_enabled: bool,
    /// Fractional scaling
    pub fractional_scaling: bool,
    /// Explicit sync protocol
    pub explicit_sync: bool,
}

/// Wine/Proton configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WineConfig {
    /// Proton version (GE, experimental, stable)
    pub proton_version: ProtonVersion,
    /// DXVK configuration
    pub dxvk: DxvkConfig,
    /// VKD3D-Proton configuration
    pub vkd3d: Vkd3dConfig,
    /// Wine-specific optimizations
    pub wine_optimizations: WineOptimizations,
    /// Game-specific fixes
    pub game_fixes: GameFixes,
}

/// Proton version selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProtonVersion {
    /// Proton GE (community builds with latest patches)
    ProtonGE(String), // version like "8-32"
    /// Steam Proton Experimental
    Experimental,
    /// Steam Proton Stable
    Stable(String), // version like "8.0"
    /// System Wine
    SystemWine,
    /// Custom Wine build
    Custom(String), // path to wine binary
}

/// DXVK (DirectX to Vulkan) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DxvkConfig {
    /// Enable DXVK
    pub enabled: bool,
    /// DXVK version
    pub version: String,
    /// Async shader compilation
    pub async_shaders: bool,
    /// HUD display
    pub hud: DxvkHud,
    /// Memory allocation strategy
    pub memory_allocation: DxvkMemoryAllocation,
    /// Frame rate limit
    pub frame_limit: Option<u32>,
}

/// DXVK HUD configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DxvkHud {
    /// Enable HUD
    pub enabled: bool,
    /// HUD elements to display
    pub elements: Vec<String>, // fps, memory, gpuload, etc.
    /// HUD position
    pub position: String, // top-left, top-right, etc.
}

/// DXVK memory allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DxvkMemoryAllocation {
    /// Automatic allocation
    Auto,
    /// Conservative allocation (less VRAM usage)
    Conservative,
    /// Aggressive allocation (maximum performance)
    Aggressive,
}

/// VKD3D-Proton (DirectX 12 to Vulkan) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vkd3dConfig {
    /// Enable VKD3D-Proton
    pub enabled: bool,
    /// DirectX Raytracing support
    pub dxr_enabled: bool,
    /// Shader model version
    pub shader_model: String, // "6_6", "6_5", etc.
    /// Debug layer
    pub debug_enabled: bool,
}

/// Wine-specific optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WineOptimizations {
    /// Large address aware support
    pub large_address_aware: bool,
    /// CPU topology optimization
    pub cpu_topology: Option<String>, // "8:2" for 8 cores, 2 threads each
    /// Memory management
    pub memory_management: MemoryManagement,
    /// Registry optimizations
    pub registry_optimizations: bool,
}

/// Memory management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagement {
    /// Enable ESync
    pub esync: bool,
    /// Enable FSync
    pub fsync: bool,
    /// Wine heap size
    pub heap_size: String, // "1024m", "2048m", etc.
}

/// Game-specific fixes and compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameFixes {
    /// Anti-cheat compatibility
    pub anti_cheat: AntiCheatConfig,
    /// Launcher fixes
    pub launcher_fixes: LauncherFixes,
    /// Performance fixes
    pub performance_fixes: PerformanceFixes,
}

/// Anti-cheat system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiCheatConfig {
    /// BattlEye compatibility
    pub battleye: bool,
    /// Easy Anti-Cheat compatibility
    pub easy_anticheat: bool,
    /// Steam VAC compatibility
    pub vac_compatible: bool,
}

/// Launcher-specific fixes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LauncherFixes {
    /// Steam compatibility
    pub steam_fixes: bool,
    /// Epic Games Launcher fixes
    pub epic_fixes: bool,
    /// Origin/EA App fixes
    pub origin_fixes: bool,
    /// Battle.net fixes
    pub battlenet_fixes: bool,
}

/// Performance-specific fixes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceFixes {
    /// CPU scheduling optimization
    pub cpu_scheduling: bool,
    /// Memory defragmentation
    pub memory_defrag: bool,
    /// I/O priority optimization
    pub io_priority: bool,
}

/// Gaming GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamingGpuConfig {
    /// DLSS configuration
    pub dlss: DlssConfig,
    /// Ray tracing configuration
    pub raytracing: RaytracingConfig,
    /// Variable Rate Shading
    pub vrs: VrsConfig,
    /// Power management
    pub power_management: PowerManagement,
    /// GPU scheduling
    pub gpu_scheduling: GpuScheduling,
}

/// DLSS (Deep Learning Super Sampling) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DlssConfig {
    /// Enable DLSS
    pub enabled: bool,
    /// DLSS quality mode
    pub quality: DlssQuality,
    /// Frame generation (DLSS 3)
    pub frame_generation: bool,
    /// Ray reconstruction
    pub ray_reconstruction: bool,
}

/// DLSS quality settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DlssQuality {
    /// Performance mode (fastest)
    Performance,
    /// Balanced mode
    Balanced,
    /// Quality mode (best quality)
    Quality,
    /// Ultra Performance mode (DLSS 3)
    UltraPerformance,
}

/// Ray tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaytracingConfig {
    /// Enable ray tracing
    pub enabled: bool,
    /// RT cores utilization
    pub rt_cores_enabled: bool,
    /// Ray tracing quality
    pub quality: RaytracingQuality,
}

/// Ray tracing quality levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaytracingQuality {
    /// Low quality, high performance
    Low,
    /// Medium quality
    Medium,
    /// High quality
    High,
    /// Ultra quality (maximum visual fidelity)
    Ultra,
}

/// Variable Rate Shading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VrsConfig {
    /// Enable VRS
    pub enabled: bool,
    /// VRS tier support
    pub tier: VrsTier,
}

/// VRS tier levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VrsTier {
    /// Tier 1 (basic VRS)
    Tier1,
    /// Tier 2 (advanced VRS)
    Tier2,
}

/// GPU power management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerManagement {
    /// Power profile
    pub profile: PowerProfile,
    /// Temperature target
    pub temp_target: Option<u32>,
    /// Power limit adjustment
    pub power_limit: Option<u32>, // percentage
}

/// Power profiles for gaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowerProfile {
    /// Maximum performance
    Maximum,
    /// Balanced performance/power
    Balanced,
    /// Power saving mode
    PowerSaver,
    /// Quiet mode (reduced fan noise)
    Quiet,
}

/// GPU scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuScheduling {
    /// Hardware-accelerated GPU scheduling
    pub hardware_scheduling: bool,
    /// GPU priority level
    pub priority: GpuPriority,
}

/// GPU priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuPriority {
    /// Real-time priority (competitive gaming)
    Realtime,
    /// High priority
    High,
    /// Normal priority
    Normal,
    /// Low priority (background tasks)
    Low,
}

/// Audio configuration for gaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Audio driver
    pub driver: AudioDriver,
    /// Low-latency audio
    pub low_latency: bool,
    /// Spatial audio
    pub spatial_audio: bool,
    /// Sample rate
    pub sample_rate: u32, // 44100, 48000, 96000
    /// Buffer size
    pub buffer_size: u32, // 64, 128, 256, 512
}

/// Audio driver options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioDriver {
    /// PulseAudio
    PulseAudio,
    /// PipeWire
    PipeWire,
    /// ALSA
    Alsa,
    /// JACK
    Jack,
}

/// Input device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    /// Mouse configuration
    pub mouse: MouseConfig,
    /// Keyboard configuration
    pub keyboard: KeyboardConfig,
    /// Controller configuration
    pub controller: ControllerConfig,
}

/// Mouse configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MouseConfig {
    /// Raw input
    pub raw_input: bool,
    /// DPI scaling
    pub dpi_scaling: f32,
    /// Acceleration
    pub acceleration: bool,
    /// Polling rate
    pub polling_rate: u32, // Hz
}

/// Keyboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyboardConfig {
    /// Repeat rate
    pub repeat_rate: u32,
    /// Repeat delay
    pub repeat_delay: u32,
}

/// Controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerConfig {
    /// Steam Input support
    pub steam_input: bool,
    /// XInput emulation
    pub xinput_emulation: bool,
    /// Force feedback
    pub force_feedback: bool,
}

impl Default for GamingConfig {
    fn default() -> Self {
        Self {
            wayland: WaylandConfig {
                enabled: true,
                force_wayland: false,
                vrr_enabled: true,
                hdr_enabled: false,
                fractional_scaling: true,
                explicit_sync: true,
            },
            wine: WineConfig {
                proton_version: ProtonVersion::ProtonGE("8-32".to_string()),
                dxvk: DxvkConfig {
                    enabled: true,
                    version: "2.3".to_string(),
                    async_shaders: true,
                    hud: DxvkHud {
                        enabled: false,
                        elements: vec!["fps".to_string(), "memory".to_string()],
                        position: "top-left".to_string(),
                    },
                    memory_allocation: DxvkMemoryAllocation::Auto,
                    frame_limit: None,
                },
                vkd3d: Vkd3dConfig {
                    enabled: true,
                    dxr_enabled: true,
                    shader_model: "6_6".to_string(),
                    debug_enabled: false,
                },
                wine_optimizations: WineOptimizations {
                    large_address_aware: true,
                    cpu_topology: Some("8:2".to_string()),
                    memory_management: MemoryManagement {
                        esync: true,
                        fsync: true,
                        heap_size: "1024m".to_string(),
                    },
                    registry_optimizations: true,
                },
                game_fixes: GameFixes {
                    anti_cheat: AntiCheatConfig {
                        battleye: true,
                        easy_anticheat: true,
                        vac_compatible: true,
                    },
                    launcher_fixes: LauncherFixes {
                        steam_fixes: true,
                        epic_fixes: true,
                        origin_fixes: true,
                        battlenet_fixes: true,
                    },
                    performance_fixes: PerformanceFixes {
                        cpu_scheduling: true,
                        memory_defrag: true,
                        io_priority: true,
                    },
                },
            },
            gpu: GamingGpuConfig {
                dlss: DlssConfig {
                    enabled: true,
                    quality: DlssQuality::Balanced,
                    frame_generation: false,
                    ray_reconstruction: true,
                },
                raytracing: RaytracingConfig {
                    enabled: true,
                    rt_cores_enabled: true,
                    quality: RaytracingQuality::Medium,
                },
                vrs: VrsConfig {
                    enabled: true,
                    tier: VrsTier::Tier2,
                },
                power_management: PowerManagement {
                    profile: PowerProfile::Maximum,
                    temp_target: Some(83),
                    power_limit: Some(120),
                },
                gpu_scheduling: GpuScheduling {
                    hardware_scheduling: true,
                    priority: GpuPriority::High,
                },
            },
            audio: AudioConfig {
                driver: AudioDriver::PipeWire,
                low_latency: true,
                spatial_audio: true,
                sample_rate: 48000,
                buffer_size: 128,
            },
            input: InputConfig {
                mouse: MouseConfig {
                    raw_input: true,
                    dpi_scaling: 1.0,
                    acceleration: false,
                    polling_rate: 1000,
                },
                keyboard: KeyboardConfig {
                    repeat_rate: 25,
                    repeat_delay: 300,
                },
                controller: ControllerConfig {
                    steam_input: true,
                    xinput_emulation: true,
                    force_feedback: true,
                },
            },
        }
    }
}

impl GamingConfig {
    /// Generate environment variables for gaming optimization
    pub fn to_environment_vars(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();

        // Wayland optimizations
        if self.wayland.enabled {
            env.insert("WAYLAND_DISPLAY".to_string(), "wayland-1".to_string());
            env.insert("XDG_SESSION_TYPE".to_string(), "wayland".to_string());
            env.insert("QT_QPA_PLATFORM".to_string(), "wayland".to_string());
            env.insert("GDK_BACKEND".to_string(), "wayland".to_string());
            env.insert("SDL_VIDEODRIVER".to_string(), "wayland".to_string());

            if self.wayland.force_wayland {
                env.insert("WAYLAND_FORCE".to_string(), "1".to_string());
            }

            if self.wayland.vrr_enabled {
                env.insert("WAYLAND_VRR".to_string(), "1".to_string());
            }

            if self.wayland.explicit_sync {
                env.insert("WAYLAND_EXPLICIT_SYNC".to_string(), "1".to_string());
            }
        }

        // Wine/Proton configuration
        match &self.wine.proton_version {
            ProtonVersion::ProtonGE(version) => {
                env.insert("PROTON_VERSION".to_string(), format!("GE-{}", version));
                env.insert("PROTON_USE_WINED3D".to_string(), "0".to_string());
            }
            ProtonVersion::Experimental => {
                env.insert("PROTON_VERSION".to_string(), "experimental".to_string());
            }
            ProtonVersion::Stable(version) => {
                env.insert("PROTON_VERSION".to_string(), version.clone());
            }
            ProtonVersion::SystemWine => {
                env.insert("WINE_USE_SYSTEM".to_string(), "1".to_string());
            }
            ProtonVersion::Custom(path) => {
                env.insert("WINE_CUSTOM_PATH".to_string(), path.clone());
            }
        }

        // DXVK configuration
        if self.wine.dxvk.enabled {
            env.insert("DXVK_ENABLE".to_string(), "1".to_string());
            env.insert("DXVK_VERSION".to_string(), self.wine.dxvk.version.clone());

            if self.wine.dxvk.async_shaders {
                env.insert("DXVK_ASYNC".to_string(), "1".to_string());
            }

            if self.wine.dxvk.hud.enabled {
                env.insert("DXVK_HUD".to_string(), self.wine.dxvk.hud.elements.join(","));
            }

            match self.wine.dxvk.memory_allocation {
                DxvkMemoryAllocation::Conservative => {
                    env.insert("DXVK_MEMORY_ALLOCATION".to_string(), "conservative".to_string());
                }
                DxvkMemoryAllocation::Aggressive => {
                    env.insert("DXVK_MEMORY_ALLOCATION".to_string(), "aggressive".to_string());
                }
                _ => {}
            }

            if let Some(limit) = self.wine.dxvk.frame_limit {
                env.insert("DXVK_FRAME_RATE".to_string(), limit.to_string());
            }
        }

        // VKD3D configuration
        if self.wine.vkd3d.enabled {
            env.insert("VKD3D_CONFIG".to_string(), "dxr".to_string());
            env.insert("VKD3D_SHADER_MODEL".to_string(), self.wine.vkd3d.shader_model.clone());

            if self.wine.vkd3d.dxr_enabled {
                env.insert("VKD3D_ENABLE_DXR".to_string(), "1".to_string());
            }
        }

        // Wine optimizations
        if self.wine.wine_optimizations.large_address_aware {
            env.insert("WINE_LARGE_ADDRESS_AWARE".to_string(), "1".to_string());
        }

        if let Some(ref topology) = self.wine.wine_optimizations.cpu_topology {
            env.insert("WINE_CPU_TOPOLOGY".to_string(), topology.clone());
        }

        if self.wine.wine_optimizations.memory_management.esync {
            env.insert("WINEESYNC".to_string(), "1".to_string());
        }

        if self.wine.wine_optimizations.memory_management.fsync {
            env.insert("WINEFSYNC".to_string(), "1".to_string());
        }

        // GPU gaming optimizations
        if self.gpu.dlss.enabled {
            env.insert("NVIDIA_DLSS_ENABLE".to_string(), "1".to_string());
            env.insert("DLSS_QUALITY".to_string(), match self.gpu.dlss.quality {
                DlssQuality::Performance => "performance".to_string(),
                DlssQuality::Balanced => "balanced".to_string(),
                DlssQuality::Quality => "quality".to_string(),
                DlssQuality::UltraPerformance => "ultra_performance".to_string(),
            });

            if self.gpu.dlss.frame_generation {
                env.insert("DLSS_FRAME_GENERATION".to_string(), "1".to_string());
            }
        }

        if self.gpu.raytracing.enabled {
            env.insert("NVIDIA_RT_CORES_ENABLE".to_string(), "1".to_string());
            env.insert("RT_QUALITY".to_string(), match self.gpu.raytracing.quality {
                RaytracingQuality::Low => "low".to_string(),
                RaytracingQuality::Medium => "medium".to_string(),
                RaytracingQuality::High => "high".to_string(),
                RaytracingQuality::Ultra => "ultra".to_string(),
            });
        }

        if self.gpu.vrs.enabled {
            env.insert("NVIDIA_VRS_ENABLE".to_string(), "1".to_string());
        }

        // Power management
        env.insert("NVIDIA_POWER_MANAGEMENT".to_string(), match self.gpu.power_management.profile {
            PowerProfile::Maximum => "performance".to_string(),
            PowerProfile::Balanced => "balanced".to_string(),
            PowerProfile::PowerSaver => "power_saver".to_string(),
            PowerProfile::Quiet => "quiet".to_string(),
        });

        // GPU scheduling
        if self.gpu.gpu_scheduling.hardware_scheduling {
            env.insert("GPU_HARDWARE_SCHEDULING".to_string(), "1".to_string());
        }

        // Audio optimizations
        env.insert("AUDIO_DRIVER".to_string(), match self.audio.driver {
            AudioDriver::PulseAudio => "pulse".to_string(),
            AudioDriver::PipeWire => "pipewire".to_string(),
            AudioDriver::Alsa => "alsa".to_string(),
            AudioDriver::Jack => "jack".to_string(),
        });

        if self.audio.low_latency {
            env.insert("AUDIO_LOW_LATENCY".to_string(), "1".to_string());
            env.insert("PULSE_LATENCY_MSEC".to_string(), "20".to_string());
        }

        env.insert("AUDIO_SAMPLE_RATE".to_string(), self.audio.sample_rate.to_string());
        env.insert("AUDIO_BUFFER_SIZE".to_string(), self.audio.buffer_size.to_string());

        // Input optimizations
        if self.input.mouse.raw_input {
            env.insert("MOUSE_RAW_INPUT".to_string(), "1".to_string());
        }

        env.insert("MOUSE_POLLING_RATE".to_string(), self.input.mouse.polling_rate.to_string());

        env
    }

    /// Create gaming profile for competitive gaming
    pub fn competitive_gaming() -> Self {
        let mut config = Self::default();

        // Ultra-low latency settings
        config.gpu.power_management.profile = PowerProfile::Maximum;
        config.gpu.gpu_scheduling.priority = GpuPriority::Realtime;
        config.gpu.dlss.quality = DlssQuality::Performance;
        config.gpu.raytracing.enabled = false; // Disable for maximum FPS

        // Audio latency optimization
        config.audio.low_latency = true;
        config.audio.buffer_size = 64;

        // Input optimization
        config.input.mouse.raw_input = true;
        config.input.mouse.acceleration = false;
        config.input.mouse.polling_rate = 1000;

        config
    }

    /// Create gaming profile for AAA single-player games
    pub fn aaa_gaming() -> Self {
        let mut config = Self::default();

        // Visual quality settings
        config.gpu.dlss.quality = DlssQuality::Quality;
        config.gpu.raytracing.enabled = true;
        config.gpu.raytracing.quality = RaytracingQuality::High;
        config.gpu.dlss.frame_generation = true;

        config
    }

    /// Create gaming profile for handheld/battery gaming
    pub fn handheld_gaming() -> Self {
        let mut config = Self::default();

        // Power efficiency settings
        config.gpu.power_management.profile = PowerProfile::PowerSaver;
        config.gpu.dlss.quality = DlssQuality::Performance;
        config.gpu.raytracing.enabled = false;
        config.gpu.power_management.power_limit = Some(80); // 80% power limit

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaming_config_default() {
        let config = GamingConfig::default();
        assert!(config.wayland.enabled);
        assert!(config.wine.dxvk.enabled);
        assert!(config.gpu.dlss.enabled);
    }

    #[test]
    fn test_competitive_gaming_profile() {
        let config = GamingConfig::competitive_gaming();
        assert_eq!(config.gpu.gpu_scheduling.priority, GpuPriority::Realtime);
        assert!(!config.gpu.raytracing.enabled);
        assert_eq!(config.audio.buffer_size, 64);
    }

    #[test]
    fn test_environment_variables() {
        let config = GamingConfig::default();
        let env = config.to_environment_vars();

        assert!(env.contains_key("WAYLAND_DISPLAY"));
        assert!(env.contains_key("NVIDIA_DLSS_ENABLE"));
        assert!(env.contains_key("DXVK_ENABLE"));
    }

    #[test]
    fn test_aaa_gaming_profile() {
        let config = GamingConfig::aaa_gaming();
        assert!(config.gpu.raytracing.enabled);
        assert_eq!(config.gpu.dlss.quality, DlssQuality::Quality);
        assert!(config.gpu.dlss.frame_generation);
    }
}