//! GhostForge Real-Time GPU Metrics API
//!
//! Provides WebSocket and REST APIs for real-time GPU monitoring in GhostForge GUI

use anyhow::Result;
use axum::{
    extract::{Path, State, WebSocketUpgrade},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use axum::extract::ws::{Message, WebSocket};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};
use tracing::{debug, info, warn};

/// Real-time GPU metrics for GhostForge GUI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeGpuMetrics {
    pub timestamp: u64,
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
    pub vram_pressure: f32, // 0.0-1.0

    // Power metrics
    pub power_draw_w: f32,
    pub power_limit_w: f32,
    pub thermal_throttling: bool,

    // Gaming-specific (NVIDIA)
    pub rtx_utilization: Option<f32>,
    pub tensor_core_utilization: Option<f32>,
    pub dlss_active: bool,
    pub reflex_enabled: bool,
}

/// GPU configuration for hot-reload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfiguration {
    pub power_limit: Option<u32>,        // Watts
    pub gpu_clock_offset: Option<i32>,   // MHz offset
    pub memory_clock_offset: Option<i32>, // MHz offset
    pub fan_speed: Option<u32>,          // Percentage
    pub performance_mode: PerformanceMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMode {
    PowerSave,
    Balanced,
    Performance,
    Maximum,
}

/// Container GPU status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerGpuStatus {
    pub container_id: String,
    pub gpu_allocated: bool,
    pub gpu_index: Vec<u32>,
    pub gpu_model: String,
    pub current_config: GpuConfiguration,
    pub uptime_secs: u64,
}

/// GhostForge Metrics Server
pub struct GhostForgeMetricsServer {
    /// Active container monitors
    monitors: Arc<RwLock<HashMap<String, ContainerGpuMonitor>>>,
    /// NVML handle
    nvml: Arc<RwLock<Option<nvml_wrapper::Nvml>>>,
}

/// Per-container GPU monitor
struct ContainerGpuMonitor {
    container_id: String,
    gpu_indices: Vec<u32>,
    last_metrics: RealtimeGpuMetrics,
    config: GpuConfiguration,
}

impl GhostForgeMetricsServer {
    /// Create new metrics server
    pub fn new() -> Result<Self> {
        // Initialize NVML for NVIDIA GPUs
        let nvml = match nvml_wrapper::Nvml::init() {
            Ok(n) => {
                info!("NVML initialized successfully");
                Some(n)
            }
            Err(e) => {
                warn!("NVML initialization failed: {}. GPU metrics will be limited.", e);
                None
            }
        };

        Ok(Self {
            monitors: Arc::new(RwLock::new(HashMap::new())),
            nvml: Arc::new(RwLock::new(nvml)),
        })
    }

    /// Start HTTP + WebSocket server for GhostForge
    pub async fn start_server(self: Arc<Self>, port: u16) -> Result<()> {
        let app = Router::new()
            // WebSocket endpoint for real-time metrics
            .route("/ws/metrics/:container_id", get(ws_handler))
            // REST endpoints
            .route("/api/gpu/status", get(gpu_status_handler))
            .route("/api/containers/:id/gpu", get(container_gpu_handler))
            .route("/api/containers/:id/gpu/profile", post(update_gpu_profile_handler))
            .with_state(self);

        let addr = format!("0.0.0.0:{}", port);
        info!("GhostForge metrics API listening on {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Collect metrics for a specific container
    async fn collect_metrics(&self, container_id: &str) -> Result<RealtimeGpuMetrics> {
        let nvml_guard = self.nvml.read().await;

        if let Some(nvml) = nvml_guard.as_ref() {
            // Get GPU device
            let device = nvml.device_by_index(0)?; // TODO: get actual GPU index from container

            let utilization = device.utilization_rates()?;
            let memory_info = device.memory_info()?;
            let temperature = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)?;
            let power_usage = device.power_usage()?;
            let power_limit = device.power_management_limit()?;
            let clock_info = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Graphics)?;
            let mem_clock = device.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory)?;

            // Check thermal throttling
            let thermal_throttling = false; // TODO: Properly check throttle reasons with NVML bindings

            Ok(RealtimeGpuMetrics {
                timestamp: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)?
                    .as_secs(),
                container_id: container_id.to_string(),
                fps: 0.0, // TODO: Calculate from frame timestamps
                frame_time_ms: 0.0,
                frame_time_p99: 0.0,
                gpu_utilization: utilization.gpu as f32,
                gpu_temp_c: temperature as f32,
                gpu_clock_mhz: clock_info,
                memory_clock_mhz: mem_clock,
                vram_used_mb: memory_info.used / (1024 * 1024),
                vram_total_mb: memory_info.total / (1024 * 1024),
                vram_pressure: memory_info.used as f32 / memory_info.total as f32,
                power_draw_w: power_usage as f32 / 1000.0,
                power_limit_w: power_limit as f32 / 1000.0,
                thermal_throttling,
                rtx_utilization: None, // TODO: Get from NVML if available
                tensor_core_utilization: None,
                dlss_active: false,   // TODO: Detect from process
                reflex_enabled: false, // TODO: Detect from process
            })
        } else {
            // Fallback metrics when NVML not available
            Ok(RealtimeGpuMetrics {
                timestamp: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)?
                    .as_secs(),
                container_id: container_id.to_string(),
                fps: 0.0,
                frame_time_ms: 0.0,
                frame_time_p99: 0.0,
                gpu_utilization: 0.0,
                gpu_temp_c: 0.0,
                gpu_clock_mhz: 0,
                memory_clock_mhz: 0,
                vram_used_mb: 0,
                vram_total_mb: 0,
                vram_pressure: 0.0,
                power_draw_w: 0.0,
                power_limit_w: 0.0,
                thermal_throttling: false,
                rtx_utilization: None,
                tensor_core_utilization: None,
                dlss_active: false,
                reflex_enabled: false,
            })
        }
    }

    /// Update GPU configuration (hot-reload)
    pub async fn update_gpu_config(
        &self,
        container_id: &str,
        config: GpuConfiguration,
    ) -> Result<()> {
        info!("Updating GPU config for container: {}", container_id);

        let nvml_guard = self.nvml.read().await;
        if let Some(nvml) = nvml_guard.as_ref() {
            let mut device = nvml.device_by_index(0)?; // TODO: get actual GPU index

            // Apply power limit
            if let Some(power_limit) = config.power_limit {
                device.set_power_management_limit(power_limit * 1000)?; // Convert W to mW
                debug!("Set power limit to {}W", power_limit);
            }

            // Note: Clock offsets and fan speed require root/NVML advanced features
            // These would typically be set via nvidia-smi or nvidia-settings

            info!("GPU configuration updated successfully");
        }

        // Update monitor config
        let mut monitors = self.monitors.write().await;
        if let Some(monitor) = monitors.get_mut(container_id) {
            monitor.config = config;
        }

        Ok(())
    }
}

impl Default for GhostForgeMetricsServer {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// WebSocket handler for real-time metrics streaming
async fn ws_handler(
    ws: WebSocketUpgrade,
    Path(container_id): Path<String>,
    State(server): State<Arc<GhostForgeMetricsServer>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, container_id, server))
}

async fn handle_socket(
    mut socket: WebSocket,
    container_id: String,
    server: Arc<GhostForgeMetricsServer>,
) {
    // Send metrics every 16ms (60 FPS)
    let mut metrics_interval = interval(Duration::from_millis(16));

    loop {
        tokio::select! {
            _ = metrics_interval.tick() => {
                match server.collect_metrics(&container_id).await {
                    Ok(metrics) => {
                        let json = serde_json::to_string(&metrics).unwrap();
                        if socket.send(Message::Text(json)).await.is_err() {
                            debug!("WebSocket connection closed for {}", container_id);
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Error collecting metrics: {}", e);
                    }
                }
            }
            Some(msg) = socket.recv() => {
                if msg.is_err() {
                    debug!("WebSocket connection closed by client");
                    break;
                }
            }
        }
    }
}

/// REST handler: Get overall GPU status
async fn gpu_status_handler(
    State(server): State<Arc<GhostForgeMetricsServer>>,
) -> Json<Vec<GpuStatusInfo>> {
    let nvml_guard = server.nvml.read().await;

    if let Some(nvml) = nvml_guard.as_ref() {
        let mut status = Vec::new();

        if let Ok(device_count) = nvml.device_count() {
            for i in 0..device_count {
                if let Ok(device) = nvml.device_by_index(i) {
                    if let Ok(name) = device.name() {
                        status.push(GpuStatusInfo {
                            index: i,
                            name,
                            available: true,
                        });
                    }
                }
            }
        }

        Json(status)
    } else {
        Json(Vec::new())
    }
}

#[derive(Debug, Serialize)]
struct GpuStatusInfo {
    index: u32,
    name: String,
    available: bool,
}

/// REST handler: Get container GPU info
async fn container_gpu_handler(
    Path(container_id): Path<String>,
    State(server): State<Arc<GhostForgeMetricsServer>>,
) -> Json<Option<ContainerGpuStatus>> {
    let monitors = server.monitors.read().await;

    if let Some(monitor) = monitors.get(&container_id) {
        Json(Some(ContainerGpuStatus {
            container_id: monitor.container_id.clone(),
            gpu_allocated: !monitor.gpu_indices.is_empty(),
            gpu_index: monitor.gpu_indices.clone(),
            gpu_model: "NVIDIA GPU".to_string(), // TODO: Get actual model
            current_config: monitor.config.clone(),
            uptime_secs: 0, // TODO: Calculate uptime
        }))
    } else {
        Json(None)
    }
}

/// REST handler: Update GPU profile (hot-reload)
async fn update_gpu_profile_handler(
    Path(container_id): Path<String>,
    State(server): State<Arc<GhostForgeMetricsServer>>,
    Json(config): Json<GpuConfiguration>,
) -> Json<Result<(), String>> {
    match server.update_gpu_config(&container_id, config).await {
        Ok(()) => Json(Ok(())),
        Err(e) => Json(Err(format!("Failed to update GPU config: {}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_server_creation() {
        let server = GhostForgeMetricsServer::new();
        assert!(server.is_ok());
    }
}
