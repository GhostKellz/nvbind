# nvbind → Bolt/GhostForge Integration Roadmap
## Making nvbind THE Preferred GPU Passthrough Solution

**Target**: Become the default GPU runtime for Bolt containers and GhostForge gaming platform

---

## 🎯 Phase 1: Real-Time Monitoring (GhostForge Critical)

### GPU Metrics WebSocket API
```rust
// File: src/ghostforge_api.rs
pub struct GhostForgeMetricsServer {
    /// WebSocket server for real-time GPU metrics
    ws_server: WsServer,
    /// Per-container GPU monitors
    monitors: HashMap<String, GpuMonitor>,
}

pub struct RealtimeGpuMetrics {
    pub timestamp: SystemTime,
    pub container_id: String,

    // Performance metrics
    pub fps: f32,
    pub frame_time_ms: f32,
    pub frame_time_p99: f32,

    // GPU metrics
    pub gpu_utilization: f32,
    pub gpu_temp_c: f32,
    pub gpu_clock_mhz: u32,
    pub memory_clock_mhz: u32,

    // Memory metrics
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub vram_pressure: f32,  // 0.0-1.0

    // Power metrics
    pub power_draw_w: f32,
    pub power_limit_w: f32,
    pub thermal_throttling: bool,

    // Gaming-specific
    pub rtx_utilization: f32,
    pub tensor_core_utilization: f32,
    pub dlss_active: bool,
    pub reflex_enabled: bool,
}

impl GhostForgeMetricsServer {
    /// Stream metrics to GhostForge GUI
    pub async fn stream_container_metrics(
        &self,
        container_id: &str,
        interval_ms: u64,
    ) -> impl Stream<Item = RealtimeGpuMetrics> {
        // Sample GPU metrics at interval
        // Send via WebSocket to GhostForge
    }
}
```

**API Endpoints for GhostForge**:
```
ws://localhost:9090/metrics/{container_id}  # WebSocket stream
GET /api/gpu/status                          # Current GPU status
GET /api/containers/{id}/gpu                 # Container GPU info
POST /api/containers/{id}/gpu/profile        # Change GPU profile
```

---

## 🎮 Phase 2: One-Click Gaming Profiles

### Pre-configured Game Profiles
```rust
// File: src/gaming_profiles.rs
pub struct GameProfile {
    pub game_name: String,
    pub gpu_config: GpuConfiguration,
    pub nvidia_settings: NvidiaSettings,
    pub amd_settings: AmdSettings,
    pub performance_hints: PerformanceHints,
}

pub struct NvidiaSettings {
    pub dlss_mode: DlssMode,        // Quality, Balanced, Performance, Ultra Performance
    pub ray_tracing: RayTracingLevel, // Off, Medium, High, Ultra, Psycho
    pub reflex_mode: ReflexMode,    // Off, On, OnBoost
    pub max_frame_rate: Option<u32>,
    pub vsync: VsyncMode,
    pub shader_cache: bool,
}

pub struct AmdSettings {
    pub fsr_mode: FsrMode,           // Quality, Balanced, Performance, Ultra Performance
    pub radeon_boost: bool,
    pub radeon_chill: Option<(u32, u32)>, // (min_fps, max_fps)
    pub anti_lag: bool,
}

// Pre-built profiles
impl GameProfile {
    pub fn cyberpunk_2077() -> Self { /* ... */ }
    pub fn elden_ring() -> Self { /* ... */ }
    pub fn counter_strike_2() -> Self { /* ... */ }
    pub fn baldurs_gate_3() -> Self { /* ... */ }
}
```

**GhostForge Integration**:
```bash
# GhostForge calls this:
nvbind gaming apply-profile --game "Cyberpunk 2077" --container gaming-001

# Or via API:
POST /api/gaming/profile
{
  "container_id": "gaming-001",
  "game": "Cyberpunk 2077",
  "quality_preset": "ultra"
}
```

---

## 🔴 Phase 3: AMD GPU Support (Critical!)

### Multi-Vendor GPU Detection
```rust
// File: src/gpu_vendor.rs
pub enum GpuVendor {
    Nvidia { driver_version: String },
    Amd { rocm_version: String },
    Intel { level_zero_version: String },
}

pub struct MultiVendorGpuManager {
    nvidia_manager: Option<NvidiaGpuManager>,
    amd_manager: Option<AmdGpuManager>,
    intel_manager: Option<IntelGpuManager>,
}

impl MultiVendorGpuManager {
    /// Detect all available GPUs
    pub async fn discover_all_gpus() -> Result<Vec<GpuDevice>> {
        let mut gpus = Vec::new();

        // NVIDIA detection
        if let Ok(nvidia_gpus) = discover_nvidia_gpus().await {
            gpus.extend(nvidia_gpus);
        }

        // AMD detection (ROCm)
        if let Ok(amd_gpus) = discover_amd_gpus().await {
            gpus.extend(amd_gpus);
        }

        // Intel detection
        if let Ok(intel_gpus) = discover_intel_gpus().await {
            gpus.extend(intel_gpus);
        }

        Ok(gpus)
    }
}

// AMD-specific implementation
pub struct AmdGpuManager {
    pub fn generate_amd_cdi_spec() -> Result<CdiSpec> {
        // ROCm library detection
        // AMDGPU kernel module
        // HIP runtime support
    }
}
```

**ROCm Integration**:
- Detect `/dev/kfd`, `/dev/dri/renderD*`
- Mount ROCm libraries: `libamdhip64.so`, `librocm_smi64.so`
- Set `HSA_OVERRIDE_GFX_VERSION` for older GPUs
- Support AMD-specific features (FSR, FreeSync)

---

## 📸 Phase 4: Snapshot-Aware GPU State (Bolt BTRFS/ZFS)

### GPU State Persistence
```rust
// File: src/snapshot.rs (enhance existing)
pub struct GpuSnapshotManager {
    pub fn snapshot_gpu_state(&self, container_id: &str) -> Result<GpuSnapshot> {
        GpuSnapshot {
            // Memory allocations
            allocated_memory: self.get_memory_regions()?,

            // GPU configuration
            compute_mode: self.get_compute_mode()?,
            persistence_mode: self.get_persistence_mode()?,
            power_limit: self.get_power_limit()?,

            // Application state
            shader_cache_path: self.locate_shader_cache()?,
            vulkan_state: self.export_vulkan_state()?,

            // Performance settings
            clock_offsets: self.get_clock_offsets()?,
            fan_curve: self.get_fan_curve()?,
        }
    }

    pub fn restore_gpu_state(&self, snapshot: &GpuSnapshot) -> Result<()> {
        // Restore all GPU settings from snapshot
        // Called when Bolt restores a capsule from BTRFS/ZFS
    }
}
```

**Bolt Integration**:
```bash
# Bolt snapshot with GPU state
bolt snapshot create gaming-capsule --include-gpu-state

# Restore includes GPU config
bolt snapshot restore gaming-capsule-20241003 --restore-gpu-state
```

---

## ⚡ Phase 5: QUIC-Aware GPU Scheduling (Bolt Networking)

### Network-Aware GPU Allocation
```rust
// File: src/quic_gpu_scheduling.rs
pub struct QuicGpuScheduler {
    pub fn schedule_with_network_topology(
        &self,
        workload: &Workload,
        network_info: &QuicNetworkInfo,
    ) -> Result<GpuAllocation> {
        // Consider QUIC connection locality
        // Prefer GPUs on same NUMA node as network interface
        // Optimize for RDMA/GPU Direct if available
    }
}

pub struct QuicNetworkInfo {
    pub interface: String,
    pub numa_node: u32,
    pub rdma_capable: bool,
    pub gpu_direct_capable: bool,
}
```

---

## 🔥 Phase 6: Hot-Reload GPU Configuration

### Dynamic GPU Reconfiguration
```rust
// File: src/hot_reload.rs
pub struct HotReloadGpuManager {
    pub async fn update_gpu_config(
        &self,
        container_id: &str,
        new_config: GpuConfiguration,
    ) -> Result<()> {
        // Apply without container restart:
        // - VRAM limits
        // - Clock speeds
        // - Power limits
        // - Performance profiles

        // Use NVIDIA NVML or AMD ROCm SMI
        self.apply_runtime_changes(&new_config).await?;

        // Notify container of changes
        self.send_configuration_event(container_id, &new_config).await?;

        Ok(())
    }
}
```

**GhostForge Use Case**:
```
User clicks "Performance Mode" in GUI
→ GhostForge calls: POST /api/containers/{id}/gpu/profile
→ nvbind applies changes WITHOUT restarting game
→ Instant performance boost
```

---

## 📊 Phase 7: Advanced Gaming Telemetry

### Frame Analysis & Stutter Detection
```rust
// File: src/gaming_telemetry.rs
pub struct GamingTelemetryCollector {
    pub fn analyze_frame_performance(&self) -> FrameAnalysis {
        FrameAnalysis {
            // Frame timing
            avg_fps: self.calculate_avg_fps(),
            frame_time_p1: self.calculate_percentile(0.01),
            frame_time_p99: self.calculate_percentile(0.99),

            // Stutter detection
            stutter_count: self.detect_stutters(),
            stutter_severity: self.calculate_stutter_severity(),

            // GPU bottlenecks
            gpu_bound_percent: self.calculate_gpu_bound_time(),
            vram_pressure_events: self.detect_vram_pressure(),
            thermal_throttle_events: self.detect_thermal_throttling(),

            // Recommendations
            recommended_settings: self.generate_recommendations(),
        }
    }
}
```

**GhostForge Dashboard**:
- Real-time FPS graph
- Frame time distribution histogram
- Stutter event markers
- GPU utilization heatmap
- Automatic quality recommendations

---

## 🎯 Implementation Priority

### **Must-Have (for Bolt/GhostForge adoption)**
1. ✅ Real-time GPU metrics WebSocket API (GhostForge GUI)
2. ✅ AMD GPU support (many Linux gamers use AMD)
3. ✅ One-click gaming profiles (user experience)
4. ✅ Hot-reload GPU configs (no restart needed)

### **Should-Have (competitive advantage)**
5. ✅ Snapshot-aware GPU state (Bolt snapshots)
6. ✅ Gaming telemetry & stutter detection
7. ✅ QUIC-aware GPU scheduling

### **Nice-to-Have (future enhancements)**
8. Intel GPU support (iGPU passthrough)
9. Multi-GPU load balancing
10. Cloud gaming optimizations (Moonlight/Sunshine)

---

## 🚀 Quick Wins for Immediate Adoption

### Week 1: Real-Time Metrics API
```bash
# Add to nvbind
cargo add tokio-tungstenite  # WebSocket
cargo add sysinfo             # System metrics
cargo add nvml-wrapper        # NVIDIA metrics
```

Create `src/ghostforge_api.rs` with WebSocket server

### Week 2: AMD GPU Detection
```bash
# Add ROCm support
cargo add rocm-smi           # AMD GPU metrics
```

Add AMD detection to `src/gpu.rs`

### Week 3: Gaming Profiles
Create `gaming_profiles/` directory with TOML configs:
```
gaming_profiles/
  ├── cyberpunk2077.toml
  ├── eldenring.toml
  ├── cs2.toml
  └── default_competitive.toml
```

### Week 4: Hot-Reload
Implement NVML-based runtime configuration changes

---

## 📈 Success Metrics

**For Bolt Adoption**:
- [ ] GPU passthrough latency < 50μs (current: ~100μs)
- [ ] Zero-copy GPU memory with QUIC networking
- [ ] GPU state snapshot/restore < 1 second
- [ ] Support for 100+ concurrent gaming capsules

**For GhostForge Adoption**:
- [ ] Real-time metrics update < 16ms (60 FPS)
- [ ] One-click profile application < 100ms
- [ ] AMD + NVIDIA GPU detection 100%
- [ ] Hot-reload config changes < 50ms

---

## 🎮 Competitive Advantages

### vs NVIDIA Container Toolkit
- ✅ Gaming-first design (not just AI/ML)
- ✅ AMD GPU support
- ✅ Real-time GUI integration
- ✅ Snapshot-aware GPU state

### vs Docker GPU Passthrough
- ✅ 100x faster (Bolt's claim validated)
- ✅ Hot-reload configurations
- ✅ Gaming-specific telemetry
- ✅ QUIC network optimization

---

## 🔮 Future Vision

**nvbind becomes**:
- The default GPU runtime for Bolt (like containerd for Kubernetes)
- The GPU backend for GhostForge (powering all gaming containers)
- The reference implementation for gaming GPU passthrough on Linux

**Integration Flow**:
```
User clicks "Launch Game" in GhostForge
    ↓
GhostForge creates Bolt container with nvbind GPU spec
    ↓
nvbind applies gaming profile + AMD/NVIDIA optimizations
    ↓
Real-time metrics streamed to GhostForge GUI
    ↓
User adjusts performance → hot-reloaded without restart
```

---

## 📝 Next Steps

1. **Validate with Bolt team**: Confirm QUIC integration points
2. **Validate with GhostForge team**: Confirm GUI metrics requirements
3. **Implement real-time metrics API** (highest priority)
4. **Add AMD GPU support** (second highest priority)
5. **Create gaming profile library**
6. **Test with real gaming workloads**

**Target**: Make nvbind the obvious choice for Bolt + GhostForge by end of Q4 2025
