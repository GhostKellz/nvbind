//! One-Click Gaming Profiles for GhostForge
//!
//! Pre-configured GPU settings for popular games

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameProfile {
    pub game_name: String,
    pub game_id: Option<String>, // Steam App ID, etc.
    pub gpu_config: GpuProfileConfig,
    pub nvidia_settings: Option<NvidiaGameSettings>,
    pub wine_settings: Option<WineSettings>,
    pub performance_hints: PerformanceHints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfileConfig {
    pub power_limit_watts: Option<u32>,
    pub gpu_clock_offset_mhz: Option<i32>,
    pub memory_clock_offset_mhz: Option<i32>,
    pub fan_speed_percent: Option<u32>,
    pub performance_mode: String, // "maximum", "balanced", "power_save"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NvidiaGameSettings {
    pub dlss_mode: Option<DlssMode>,
    pub ray_tracing: RayTracingLevel,
    pub reflex_mode: ReflexMode,
    pub max_frame_rate: Option<u32>,
    pub vsync: VsyncMode,
    pub shader_cache_enabled: bool,
    pub threaded_optimization: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DlssMode {
    Off,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RayTracingLevel {
    Off,
    Low,
    Medium,
    High,
    Ultra,
    Psycho, // For games like Cyberpunk 2077
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReflexMode {
    Off,
    On,
    OnBoost,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VsyncMode {
    Off,
    On,
    Adaptive,
    FastSync,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WineSettings {
    pub wine_version: Option<String>,
    pub dxvk_enabled: bool,
    pub dxvk_async: bool,
    pub esync: bool,
    pub fsync: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHints {
    pub recommended_vram_gb: u32,
    pub recommended_system_ram_gb: u32,
    pub cpu_thread_count: Option<u32>,
    pub requires_high_bandwidth_pcie: bool,
    pub notes: Vec<String>,
}

/// Gaming profile manager
pub struct GamingProfileManager {
    profiles: HashMap<String, GameProfile>,
}

impl GamingProfileManager {
    pub fn new() -> Self {
        let mut manager = Self {
            profiles: HashMap::new(),
        };

        // Load built-in profiles
        manager.load_builtin_profiles();
        manager
    }

    fn load_builtin_profiles(&mut self) {
        self.profiles.insert("cyberpunk2077".to_string(), Self::cyberpunk_2077());
        self.profiles.insert("eldenring".to_string(), Self::elden_ring());
        self.profiles.insert("cs2".to_string(), Self::counter_strike_2());
        self.profiles.insert("baldursgate3".to_string(), Self::baldurs_gate_3());
        self.profiles.insert("starfield".to_string(), Self::starfield());
        self.profiles.insert("hogwarts".to_string(), Self::hogwarts_legacy());
        self.profiles.insert("rdr2".to_string(), Self::red_dead_redemption_2());
        self.profiles.insert("witcher3".to_string(), Self::witcher_3());
    }

    pub fn get_profile(&self, game: &str) -> Option<&GameProfile> {
        self.profiles.get(game)
    }

    pub fn list_profiles(&self) -> Vec<String> {
        self.profiles.keys().cloned().collect()
    }

    // ==================== GAME PROFILES ====================

    /// Cyberpunk 2077 - RTX showcase
    pub fn cyberpunk_2077() -> GameProfile {
        GameProfile {
            game_name: "Cyberpunk 2077".to_string(),
            game_id: Some("1091500".to_string()),
            gpu_config: GpuProfileConfig {
                power_limit_watts: None, // Use maximum
                gpu_clock_offset_mhz: Some(100),
                memory_clock_offset_mhz: Some(500),
                fan_speed_percent: Some(75),
                performance_mode: "maximum".to_string(),
            },
            nvidia_settings: Some(NvidiaGameSettings {
                dlss_mode: Some(DlssMode::Quality),
                ray_tracing: RayTracingLevel::Psycho,
                reflex_mode: ReflexMode::On,
                max_frame_rate: None,
                vsync: VsyncMode::Off,
                shader_cache_enabled: true,
                threaded_optimization: true,
            }),
            wine_settings: None, // Native on Linux
            performance_hints: PerformanceHints {
                recommended_vram_gb: 12,
                recommended_system_ram_gb: 16,
                cpu_thread_count: Some(8),
                requires_high_bandwidth_pcie: true,
                notes: vec![
                    "Path tracing requires RTX 3080+ or RX 7900 XTX".to_string(),
                    "DLSS 3 Frame Generation on RTX 40 series".to_string(),
                ],
            },
        }
    }

    /// Counter-Strike 2 - Competitive FPS
    pub fn counter_strike_2() -> GameProfile {
        GameProfile {
            game_name: "Counter-Strike 2".to_string(),
            game_id: Some("730".to_string()),
            gpu_config: GpuProfileConfig {
                power_limit_watts: None,
                gpu_clock_offset_mhz: Some(200), // Max clocks for FPS
                memory_clock_offset_mhz: Some(800),
                fan_speed_percent: Some(80),
                performance_mode: "maximum".to_string(),
            },
            nvidia_settings: Some(NvidiaGameSettings {
                dlss_mode: None, // Not supported
                ray_tracing: RayTracingLevel::Off, // Competitive = performance
                reflex_mode: ReflexMode::OnBoost, // Critical for competitive
                max_frame_rate: Some(400), // High refresh rate
                vsync: VsyncMode::Off, // Never use in competitive
                shader_cache_enabled: true,
                threaded_optimization: true,
            }),
            wine_settings: None,
            performance_hints: PerformanceHints {
                recommended_vram_gb: 4,
                recommended_system_ram_gb: 16,
                cpu_thread_count: Some(6),
                requires_high_bandwidth_pcie: false,
                notes: vec![
                    "Disable V-Sync for lowest latency".to_string(),
                    "Target 300+ FPS for competitive play".to_string(),
                    "Use Reflex + Boost for best input latency".to_string(),
                ],
            },
        }
    }

    /// Elden Ring - Optimized for stable 60 FPS
    pub fn elden_ring() -> GameProfile {
        GameProfile {
            game_name: "Elden Ring".to_string(),
            game_id: Some("1245620".to_string()),
            gpu_config: GpuProfileConfig {
                power_limit_watts: Some(250),
                gpu_clock_offset_mhz: Some(50),
                memory_clock_offset_mhz: Some(300),
                fan_speed_percent: Some(65),
                performance_mode: "balanced".to_string(),
            },
            nvidia_settings: Some(NvidiaGameSettings {
                dlss_mode: Some(DlssMode::Quality),
                ray_tracing: RayTracingLevel::Off, // Not supported
                reflex_mode: ReflexMode::Off,
                max_frame_rate: Some(60), // Capped at 60
                vsync: VsyncMode::On, // Needed for 60 FPS lock
                shader_cache_enabled: true,
                threaded_optimization: true,
            }),
            wine_settings: None,
            performance_hints: PerformanceHints {
                recommended_vram_gb: 6,
                recommended_system_ram_gb: 12,
                cpu_thread_count: Some(4),
                requires_high_bandwidth_pcie: false,
                notes: vec![
                    "Game is capped at 60 FPS".to_string(),
                    "Use quality settings over performance".to_string(),
                ],
            },
        }
    }

    /// Baldur's Gate 3 - RPG optimized
    pub fn baldurs_gate_3() -> GameProfile {
        GameProfile {
            game_name: "Baldur's Gate 3".to_string(),
            game_id: Some("1086940".to_string()),
            gpu_config: GpuProfileConfig {
                power_limit_watts: Some(275),
                gpu_clock_offset_mhz: Some(100),
                memory_clock_offset_mhz: Some(400),
                fan_speed_percent: Some(70),
                performance_mode: "balanced".to_string(),
            },
            nvidia_settings: Some(NvidiaGameSettings {
                dlss_mode: Some(DlssMode::Quality),
                ray_tracing: RayTracingLevel::High,
                reflex_mode: ReflexMode::Off,
                max_frame_rate: Some(120),
                vsync: VsyncMode::Adaptive,
                shader_cache_enabled: true,
                threaded_optimization: true,
            }),
            wine_settings: None,
            performance_hints: PerformanceHints {
                recommended_vram_gb: 8,
                recommended_system_ram_gb: 16,
                cpu_thread_count: Some(6),
                requires_high_bandwidth_pcie: false,
                notes: vec![
                    "Act 3 is CPU intensive in city".to_string(),
                    "DLSS recommended for 4K gaming".to_string(),
                ],
            },
        }
    }

    /// Starfield - Bethesda RPG
    pub fn starfield() -> GameProfile {
        GameProfile {
            game_name: "Starfield".to_string(),
            game_id: Some("1716740".to_string()),
            gpu_config: GpuProfileConfig {
                power_limit_watts: None,
                gpu_clock_offset_mhz: Some(150),
                memory_clock_offset_mhz: Some(600),
                fan_speed_percent: Some(75),
                performance_mode: "maximum".to_string(),
            },
            nvidia_settings: Some(NvidiaGameSettings {
                dlss_mode: Some(DlssMode::Balanced),
                ray_tracing: RayTracingLevel::Medium, // Performance impact
                reflex_mode: ReflexMode::On,
                max_frame_rate: Some(120),
                vsync: VsyncMode::Off,
                shader_cache_enabled: true,
                threaded_optimization: true,
            }),
            wine_settings: None,
            performance_hints: PerformanceHints {
                recommended_vram_gb: 12,
                recommended_system_ram_gb: 16,
                cpu_thread_count: Some(8),
                requires_high_bandwidth_pcie: true,
                notes: vec![
                    "Heavy on VRAM usage".to_string(),
                    "Cities are CPU intensive".to_string(),
                ],
            },
        }
    }

    /// Hogwarts Legacy
    pub fn hogwarts_legacy() -> GameProfile {
        GameProfile {
            game_name: "Hogwarts Legacy".to_string(),
            game_id: Some("990080".to_string()),
            gpu_config: GpuProfileConfig {
                power_limit_watts: Some(300),
                gpu_clock_offset_mhz: Some(100),
                memory_clock_offset_mhz: Some(500),
                fan_speed_percent: Some(70),
                performance_mode: "balanced".to_string(),
            },
            nvidia_settings: Some(NvidiaGameSettings {
                dlss_mode: Some(DlssMode::Quality),
                ray_tracing: RayTracingLevel::High,
                reflex_mode: ReflexMode::On,
                max_frame_rate: Some(120),
                vsync: VsyncMode::Off,
                shader_cache_enabled: true,
                threaded_optimization: true,
            }),
            wine_settings: None,
            performance_hints: PerformanceHints {
                recommended_vram_gb: 8,
                recommended_system_ram_gb: 16,
                cpu_thread_count: Some(6),
                requires_high_bandwidth_pcie: false,
                notes: vec![
                    "Supports DLSS 3 Frame Generation on RTX 40 series".to_string(),
                ],
            },
        }
    }

    /// Red Dead Redemption 2
    pub fn red_dead_redemption_2() -> GameProfile {
        GameProfile {
            game_name: "Red Dead Redemption 2".to_string(),
            game_id: Some("1174180".to_string()),
            gpu_config: GpuProfileConfig {
                power_limit_watts: None,
                gpu_clock_offset_mhz: Some(120),
                memory_clock_offset_mhz: Some(500),
                fan_speed_percent: Some(75),
                performance_mode: "maximum".to_string(),
            },
            nvidia_settings: Some(NvidiaGameSettings {
                dlss_mode: Some(DlssMode::Quality),
                ray_tracing: RayTracingLevel::Off, // Not supported
                reflex_mode: ReflexMode::Off,
                max_frame_rate: None,
                vsync: VsyncMode::Off,
                shader_cache_enabled: true,
                threaded_optimization: true,
            }),
            wine_settings: Some(WineSettings {
                wine_version: Some("proton-ge".to_string()),
                dxvk_enabled: true,
                dxvk_async: true,
                esync: true,
                fsync: true,
            }),
            performance_hints: PerformanceHints {
                recommended_vram_gb: 8,
                recommended_system_ram_gb: 12,
                cpu_thread_count: Some(6),
                requires_high_bandwidth_pcie: false,
                notes: vec![
                    "Very demanding at max settings".to_string(),
                    "Use Hardware Unboxed optimized settings".to_string(),
                ],
            },
        }
    }

    /// The Witcher 3: Wild Hunt (Next-Gen)
    pub fn witcher_3() -> GameProfile {
        GameProfile {
            game_name: "The Witcher 3: Wild Hunt".to_string(),
            game_id: Some("292030".to_string()),
            gpu_config: GpuProfileConfig {
                power_limit_watts: Some(275),
                gpu_clock_offset_mhz: Some(100),
                memory_clock_offset_mhz: Some(400),
                fan_speed_percent: Some(70),
                performance_mode: "balanced".to_string(),
            },
            nvidia_settings: Some(NvidiaGameSettings {
                dlss_mode: Some(DlssMode::Quality),
                ray_tracing: RayTracingLevel::Ultra, // Next-gen RT
                reflex_mode: ReflexMode::Off,
                max_frame_rate: Some(144),
                vsync: VsyncMode::Adaptive,
                shader_cache_enabled: true,
                threaded_optimization: true,
            }),
            wine_settings: None,
            performance_hints: PerformanceHints {
                recommended_vram_gb: 8,
                recommended_system_ram_gb: 8,
                cpu_thread_count: Some(4),
                requires_high_bandwidth_pcie: false,
                notes: vec![
                    "Next-gen update adds ray tracing".to_string(),
                    "DLSS recommended for RT performance".to_string(),
                ],
            },
        }
    }
}

impl Default for GamingProfileManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_loading() {
        let manager = GamingProfileManager::new();
        assert!(manager.get_profile("cyberpunk2077").is_some());
        assert!(manager.get_profile("cs2").is_some());
    }

    #[test]
    fn test_cyberpunk_profile() {
        let profile = GamingProfileManager::cyberpunk_2077();
        assert_eq!(profile.game_name, "Cyberpunk 2077");
        assert_eq!(profile.performance_hints.recommended_vram_gb, 12);
    }
}
