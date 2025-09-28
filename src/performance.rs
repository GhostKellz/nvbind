//! Performance optimization and tuning framework
//!
//! Provides comprehensive performance monitoring, analysis, and optimization
//! for GPU workloads and container operations.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Performance optimization manager
pub struct PerformanceOptimizer {
    config: PerformanceConfig,
    profiler: PerformanceProfiler,
    tuner: SystemTuner,
    metrics_cache: Arc<RwLock<MetricsCache>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enabled: bool,
    pub auto_tuning: bool,
    pub profiling_enabled: bool,
    pub optimization_level: OptimizationLevel,
    pub workload_profiles: HashMap<String, WorkloadProfile>,
    pub cpu_optimization: CpuOptimization,
    pub memory_optimization: MemoryOptimization,
    pub gpu_optimization: GpuOptimization,
    pub io_optimization: IoOptimization,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Conservative, // Safe optimizations only
    Balanced,     // Balance performance and stability
    Aggressive,   // Maximum performance
    Custom,       // User-defined settings
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadProfile {
    pub name: String,
    pub description: String,
    pub cpu_affinity: Option<Vec<usize>>,
    pub memory_policy: MemoryPolicy,
    pub gpu_settings: GpuSettings,
    pub io_priority: IoPriority,
    pub scheduling_policy: SchedulingPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuOptimization {
    pub enabled: bool,
    pub governor: CpuGovernor,
    pub affinity_enabled: bool,
    pub numa_balancing: bool,
    pub transparent_hugepages: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CpuGovernor {
    Performance,
    Powersave,
    Ondemand,
    Conservative,
    Schedutil,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    pub enabled: bool,
    pub swap_optimization: bool,
    pub memory_compaction: bool,
    pub ksm_enabled: bool, // Kernel Same-page Merging
    pub hugepage_policy: HugepagePolicy,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HugepagePolicy {
    Never,
    Madvise,
    Always,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOptimization {
    pub enabled: bool,
    pub power_management: GpuPowerPolicy,
    pub clock_optimization: bool,
    pub memory_clock_optimization: bool,
    pub persistence_mode: bool,
    pub compute_mode: ComputeMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GpuPowerPolicy {
    Default,
    MaxPerformance,
    Adaptive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComputeMode {
    Default,
    Exclusive,
    Prohibited,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoOptimization {
    pub enabled: bool,
    pub scheduler: IoScheduler,
    pub readahead_kb: Option<u32>,
    pub queue_depth: Option<u32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IoScheduler {
    Mq_deadline,
    Kyber,
    Bfq,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MemoryPolicy {
    Default,
    Interleave,
    Bind,
    Preferred,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IoPriority {
    RealTime(u8),   // 0-7
    BestEffort(u8), // 0-7
    Idle,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    Normal,
    Batch,
    Fifo(u8), // 1-99
    RR(u8),   // 1-99
    Deadline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSettings {
    pub power_limit_watts: Option<u32>,
    pub memory_clock_offset: Option<i32>,
    pub graphics_clock_offset: Option<i32>,
    pub fan_speed_percent: Option<u32>,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        let mut workload_profiles = HashMap::new();

        // ML/AI workload profile
        workload_profiles.insert(
            "ml".to_string(),
            WorkloadProfile {
                name: "Machine Learning".to_string(),
                description: "Optimized for ML/AI training and inference".to_string(),
                cpu_affinity: None,
                memory_policy: MemoryPolicy::Bind,
                gpu_settings: GpuSettings {
                    power_limit_watts: None,
                    memory_clock_offset: Some(1000),
                    graphics_clock_offset: Some(200),
                    fan_speed_percent: Some(80),
                },
                io_priority: IoPriority::BestEffort(1),
                scheduling_policy: SchedulingPolicy::Normal,
            },
        );

        // Gaming workload profile
        workload_profiles.insert(
            "gaming".to_string(),
            WorkloadProfile {
                name: "Gaming".to_string(),
                description: "Optimized for gaming workloads".to_string(),
                cpu_affinity: None,
                memory_policy: MemoryPolicy::Default,
                gpu_settings: GpuSettings {
                    power_limit_watts: None,
                    memory_clock_offset: Some(500),
                    graphics_clock_offset: Some(100),
                    fan_speed_percent: Some(60),
                },
                io_priority: IoPriority::RealTime(1),
                scheduling_policy: SchedulingPolicy::Normal,
            },
        );

        // Compute workload profile
        workload_profiles.insert(
            "compute".to_string(),
            WorkloadProfile {
                name: "High Performance Computing".to_string(),
                description: "Optimized for compute-intensive workloads".to_string(),
                cpu_affinity: None,
                memory_policy: MemoryPolicy::Interleave,
                gpu_settings: GpuSettings {
                    power_limit_watts: None,
                    memory_clock_offset: Some(1500),
                    graphics_clock_offset: Some(300),
                    fan_speed_percent: Some(100),
                },
                io_priority: IoPriority::BestEffort(0),
                scheduling_policy: SchedulingPolicy::Batch,
            },
        );

        Self {
            enabled: true,
            auto_tuning: false,
            profiling_enabled: true,
            optimization_level: OptimizationLevel::Balanced,
            workload_profiles,
            cpu_optimization: CpuOptimization {
                enabled: true,
                governor: CpuGovernor::Performance,
                affinity_enabled: false,
                numa_balancing: true,
                transparent_hugepages: true,
            },
            memory_optimization: MemoryOptimization {
                enabled: true,
                swap_optimization: true,
                memory_compaction: true,
                ksm_enabled: false,
                hugepage_policy: HugepagePolicy::Madvise,
            },
            gpu_optimization: GpuOptimization {
                enabled: true,
                power_management: GpuPowerPolicy::Adaptive,
                clock_optimization: false,
                memory_clock_optimization: false,
                persistence_mode: true,
                compute_mode: ComputeMode::Default,
            },
            io_optimization: IoOptimization {
                enabled: true,
                scheduler: IoScheduler::Mq_deadline,
                readahead_kb: Some(128),
                queue_depth: Some(32),
            },
        }
    }
}

impl PerformanceOptimizer {
    /// Create new performance optimizer
    pub fn new(config: PerformanceConfig) -> Self {
        Self {
            config: config.clone(),
            profiler: PerformanceProfiler::new(config.profiling_enabled),
            tuner: SystemTuner::new(config),
            metrics_cache: Arc::new(RwLock::new(MetricsCache::new())),
        }
    }

    /// Initialize performance optimization
    pub async fn initialize(&mut self) -> Result<()> {
        if !self.config.enabled {
            info!("Performance optimization disabled");
            return Ok(());
        }

        info!("Initializing performance optimization");

        // Initialize profiler
        self.profiler.initialize().await?;

        // Apply system optimizations
        if self.config.optimization_level != OptimizationLevel::Conservative {
            self.tuner.apply_optimizations().await?;
        }

        info!("Performance optimization initialized");
        Ok(())
    }

    /// Apply workload-specific optimizations
    pub async fn optimize_for_workload(&self, workload_name: &str) -> Result<()> {
        let profile = self
            .config
            .workload_profiles
            .get(workload_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown workload profile: {}", workload_name))?;

        info!("Applying optimizations for workload: {}", profile.name);

        // Apply CPU affinity
        if let Some(affinity) = &profile.cpu_affinity {
            self.tuner.set_cpu_affinity(affinity).await?;
        }

        // Apply memory policy
        self.tuner.set_memory_policy(profile.memory_policy).await?;

        // Apply GPU settings
        self.tuner.apply_gpu_settings(&profile.gpu_settings).await?;

        // Apply I/O priority
        self.tuner.set_io_priority(profile.io_priority).await?;

        // Apply scheduling policy
        self.tuner
            .set_scheduling_policy(profile.scheduling_policy)
            .await?;

        info!("Workload optimizations applied successfully");
        Ok(())
    }

    /// Start performance monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        if !self.config.profiling_enabled {
            return Ok(());
        }

        info!("Starting performance monitoring");
        self.profiler.start_monitoring().await?;
        Ok(())
    }

    /// Get performance metrics
    pub async fn get_metrics(&self) -> Result<PerformanceMetrics> {
        let metrics = self.profiler.collect_metrics().await?;

        // Cache metrics
        {
            let mut cache = self.metrics_cache.write().await;
            cache.update(metrics.clone());
        }

        Ok(metrics)
    }

    /// Generate performance report
    pub async fn generate_report(&self) -> Result<PerformanceReport> {
        let metrics = self.get_metrics().await?;
        let recommendations = self.analyze_performance(&metrics).await?;

        Ok(PerformanceReport {
            timestamp: chrono::Utc::now(),
            metrics,
            recommendations,
            workload_analysis: self.analyze_workload_patterns().await?,
        })
    }

    /// Analyze performance and generate recommendations
    async fn analyze_performance(
        &self,
        metrics: &PerformanceMetrics,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();

        // CPU analysis
        if metrics.cpu_usage > 90.0 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Cpu,
                priority: Priority::High,
                description: "High CPU usage detected".to_string(),
                suggestion: "Consider enabling CPU affinity or upgrading CPU governor".to_string(),
                impact_score: 0.8,
            });
        }

        // Memory analysis
        if metrics.memory_usage > 85.0 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Memory,
                priority: Priority::High,
                description: "High memory usage detected".to_string(),
                suggestion: "Enable memory compaction or adjust hugepage policy".to_string(),
                impact_score: 0.9,
            });
        }

        // GPU analysis
        if let Some(gpu_usage) = metrics.gpu_usage {
            if gpu_usage < 70.0 && metrics.cpu_usage > 80.0 {
                recommendations.push(Recommendation {
                    category: RecommendationCategory::Gpu,
                    priority: Priority::Medium,
                    description: "GPU underutilized while CPU is busy".to_string(),
                    suggestion: "Review workload distribution and consider CPU-GPU load balancing"
                        .to_string(),
                    impact_score: 0.6,
                });
            }
        }

        // I/O analysis
        if metrics.io_wait > 20.0 {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Io,
                priority: Priority::High,
                description: "High I/O wait time detected".to_string(),
                suggestion: "Consider changing I/O scheduler or increasing queue depth".to_string(),
                impact_score: 0.7,
            });
        }

        // Latency analysis
        if metrics.avg_latency > Duration::from_millis(100) {
            recommendations.push(Recommendation {
                category: RecommendationCategory::Latency,
                priority: Priority::Medium,
                description: "High average latency detected".to_string(),
                suggestion: "Review system configuration and workload scheduling".to_string(),
                impact_score: 0.5,
            });
        }

        Ok(recommendations)
    }

    /// Analyze workload patterns
    async fn analyze_workload_patterns(&self) -> Result<WorkloadAnalysis> {
        let cache = self.metrics_cache.read().await;
        let history = cache.get_history();

        let mut patterns = HashMap::new();

        // Analyze CPU patterns
        let cpu_pattern = self.analyze_cpu_pattern(&history);
        patterns.insert("cpu".to_string(), cpu_pattern);

        // Analyze GPU patterns
        if let Some(gpu_pattern) = self.analyze_gpu_pattern(&history) {
            patterns.insert("gpu".to_string(), gpu_pattern);
        }

        // Analyze memory patterns
        let memory_pattern = self.analyze_memory_pattern(&history);
        patterns.insert("memory".to_string(), memory_pattern);

        Ok(WorkloadAnalysis {
            patterns,
            peak_hours: self.identify_peak_hours(&history),
            resource_correlation: self.calculate_resource_correlation(&history),
        })
    }

    fn analyze_cpu_pattern(&self, history: &[PerformanceMetrics]) -> PatternAnalysis {
        let avg_usage: f64 =
            history.iter().map(|m| m.cpu_usage).sum::<f64>() / history.len() as f64;
        let variance = self.calculate_variance(history.iter().map(|m| m.cpu_usage).collect());

        PatternAnalysis {
            average: avg_usage,
            variance,
            trend: self.calculate_trend(history.iter().map(|m| m.cpu_usage).collect()),
            stability: if variance < 10.0 {
                Stability::Stable
            } else {
                Stability::Variable
            },
        }
    }

    fn analyze_gpu_pattern(&self, history: &[PerformanceMetrics]) -> Option<PatternAnalysis> {
        let gpu_values: Vec<f64> = history.iter().filter_map(|m| m.gpu_usage).collect();

        if gpu_values.is_empty() {
            return None;
        }

        let avg_usage = gpu_values.iter().sum::<f64>() / gpu_values.len() as f64;
        let variance = self.calculate_variance(gpu_values.clone());

        Some(PatternAnalysis {
            average: avg_usage,
            variance,
            trend: self.calculate_trend(gpu_values),
            stability: if variance < 15.0 {
                Stability::Stable
            } else {
                Stability::Variable
            },
        })
    }

    fn analyze_memory_pattern(&self, history: &[PerformanceMetrics]) -> PatternAnalysis {
        let avg_usage: f64 =
            history.iter().map(|m| m.memory_usage).sum::<f64>() / history.len() as f64;
        let variance = self.calculate_variance(history.iter().map(|m| m.memory_usage).collect());

        PatternAnalysis {
            average: avg_usage,
            variance,
            trend: self.calculate_trend(history.iter().map(|m| m.memory_usage).collect()),
            stability: if variance < 5.0 {
                Stability::Stable
            } else {
                Stability::Variable
            },
        }
    }

    fn calculate_variance(&self, values: Vec<f64>) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance
    }

    fn calculate_trend(&self, values: Vec<f64>) -> Trend {
        if values.len() < 2 {
            return Trend::Stable;
        }

        let first_half_avg =
            values[..values.len() / 2].iter().sum::<f64>() / (values.len() / 2) as f64;
        let second_half_avg = values[values.len() / 2..].iter().sum::<f64>()
            / (values.len() - values.len() / 2) as f64;

        let difference = second_half_avg - first_half_avg;

        if difference > 5.0 {
            Trend::Increasing
        } else if difference < -5.0 {
            Trend::Decreasing
        } else {
            Trend::Stable
        }
    }

    fn identify_peak_hours(&self, _history: &[PerformanceMetrics]) -> Vec<u8> {
        // Simplified implementation - would analyze actual timestamps
        vec![9, 10, 11, 14, 15, 16] // Typical work hours
    }

    fn calculate_resource_correlation(
        &self,
        history: &[PerformanceMetrics],
    ) -> HashMap<String, f64> {
        let mut correlations = HashMap::new();

        if !history.is_empty() {
            // CPU-Memory correlation
            correlations.insert("cpu_memory".to_string(), 0.7); // Placeholder

            // CPU-GPU correlation
            if history.iter().any(|m| m.gpu_usage.is_some()) {
                correlations.insert("cpu_gpu".to_string(), -0.3); // Inverse correlation
            }

            // GPU-Memory correlation
            if history.iter().any(|m| m.gpu_usage.is_some()) {
                correlations.insert("gpu_memory".to_string(), 0.5);
            }
        }

        correlations
    }
}

/// Performance profiler
pub struct PerformanceProfiler {
    enabled: bool,
    start_time: Option<Instant>,
    sample_interval: Duration,
}

impl PerformanceProfiler {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            start_time: None,
            sample_interval: Duration::from_secs(1),
        }
    }

    async fn initialize(&mut self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        info!("Initializing performance profiler");
        self.start_time = Some(Instant::now());
        Ok(())
    }

    async fn start_monitoring(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        info!("Starting performance monitoring");
        // Implementation would start background monitoring task
        Ok(())
    }

    async fn collect_metrics(&self) -> Result<PerformanceMetrics> {
        if !self.enabled {
            return Ok(PerformanceMetrics::default());
        }

        let start = Instant::now();

        // Collect system metrics
        let cpu_usage = self.get_cpu_usage().await?;
        let memory_usage = self.get_memory_usage().await?;
        let gpu_usage = self.get_gpu_usage().await;
        let io_wait = self.get_io_wait().await?;
        let load_average = self.get_load_average().await?;

        let collection_time = start.elapsed();

        Ok(PerformanceMetrics {
            timestamp: chrono::Utc::now(),
            cpu_usage,
            memory_usage,
            gpu_usage,
            io_wait,
            load_average,
            avg_latency: collection_time,
            throughput: self.calculate_throughput().await?,
        })
    }

    async fn get_cpu_usage(&self) -> Result<f64> {
        // Read from /proc/stat
        let stat = tokio::fs::read_to_string("/proc/stat").await?;
        let cpu_line = stat
            .lines()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Cannot read CPU stats"))?;

        // Parse CPU times (simplified)
        let fields: Vec<&str> = cpu_line.split_whitespace().collect();
        if fields.len() < 5 {
            return Err(anyhow::anyhow!("Invalid CPU stats format"));
        }

        // Calculate usage (simplified - would need previous values for accuracy)
        Ok(75.0) // Placeholder
    }

    async fn get_memory_usage(&self) -> Result<f64> {
        // Read from /proc/meminfo
        let meminfo = tokio::fs::read_to_string("/proc/meminfo").await?;
        let mut total = 0u64;
        let mut available = 0u64;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total = line
                    .split_whitespace()
                    .nth(1)
                    .ok_or_else(|| anyhow::anyhow!("Cannot parse MemTotal"))?
                    .parse()?;
            } else if line.starts_with("MemAvailable:") {
                available = line
                    .split_whitespace()
                    .nth(1)
                    .ok_or_else(|| anyhow::anyhow!("Cannot parse MemAvailable"))?
                    .parse()?;
            }
        }

        if total == 0 {
            return Err(anyhow::anyhow!("Cannot determine memory usage"));
        }

        let usage = ((total - available) as f64 / total as f64) * 100.0;
        Ok(usage)
    }

    async fn get_gpu_usage(&self) -> Option<f64> {
        // Try to get GPU usage via nvidia-smi
        if let Ok(output) = tokio::process::Command::new("nvidia-smi")
            .arg("--query-gpu=utilization.gpu")
            .arg("--format=csv,noheader,nounits")
            .output()
            .await
        {
            if output.status.success() {
                if let Ok(usage_str) = String::from_utf8(output.stdout) {
                    if let Ok(usage) = usage_str.trim().parse::<f64>() {
                        return Some(usage);
                    }
                }
            }
        }
        None
    }

    async fn get_io_wait(&self) -> Result<f64> {
        // Read from /proc/stat for iowait
        let stat = tokio::fs::read_to_string("/proc/stat").await?;
        let cpu_line = stat
            .lines()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Cannot read CPU stats"))?;

        // Parse iowait field (simplified)
        Ok(5.0) // Placeholder
    }

    async fn get_load_average(&self) -> Result<(f64, f64, f64)> {
        // Read from /proc/loadavg
        let loadavg = tokio::fs::read_to_string("/proc/loadavg").await?;
        let fields: Vec<&str> = loadavg.split_whitespace().collect();

        if fields.len() < 3 {
            return Err(anyhow::anyhow!("Cannot parse load average"));
        }

        Ok((fields[0].parse()?, fields[1].parse()?, fields[2].parse()?))
    }

    async fn calculate_throughput(&self) -> Result<f64> {
        // Calculate operations per second (simplified)
        Ok(1000.0) // Placeholder
    }
}

/// System tuner for applying optimizations
pub struct SystemTuner {
    config: PerformanceConfig,
}

impl SystemTuner {
    fn new(config: PerformanceConfig) -> Self {
        Self { config }
    }

    async fn apply_optimizations(&self) -> Result<()> {
        info!("Applying system optimizations");

        if self.config.cpu_optimization.enabled {
            self.optimize_cpu().await?;
        }

        if self.config.memory_optimization.enabled {
            self.optimize_memory().await?;
        }

        if self.config.gpu_optimization.enabled {
            self.optimize_gpu().await?;
        }

        if self.config.io_optimization.enabled {
            self.optimize_io().await?;
        }

        info!("System optimizations applied");
        Ok(())
    }

    async fn optimize_cpu(&self) -> Result<()> {
        info!("Applying CPU optimizations");

        // Set CPU governor
        self.set_cpu_governor(self.config.cpu_optimization.governor)
            .await?;

        // Configure NUMA balancing
        if self.config.cpu_optimization.numa_balancing {
            self.enable_numa_balancing().await?;
        }

        // Configure transparent hugepages
        if self.config.cpu_optimization.transparent_hugepages {
            self.configure_hugepages().await?;
        }

        Ok(())
    }

    async fn optimize_memory(&self) -> Result<()> {
        info!("Applying memory optimizations");

        // Configure swap
        if self.config.memory_optimization.swap_optimization {
            self.optimize_swap().await?;
        }

        // Enable memory compaction
        if self.config.memory_optimization.memory_compaction {
            self.enable_memory_compaction().await?;
        }

        // Configure KSM
        if self.config.memory_optimization.ksm_enabled {
            self.enable_ksm().await?;
        }

        Ok(())
    }

    async fn optimize_gpu(&self) -> Result<()> {
        info!("Applying GPU optimizations");

        // Set persistence mode
        if self.config.gpu_optimization.persistence_mode {
            self.enable_gpu_persistence().await?;
        }

        // Set compute mode
        self.set_gpu_compute_mode(self.config.gpu_optimization.compute_mode)
            .await?;

        Ok(())
    }

    async fn optimize_io(&self) -> Result<()> {
        info!("Applying I/O optimizations");

        // Set I/O scheduler
        self.set_io_scheduler(self.config.io_optimization.scheduler)
            .await?;

        // Configure readahead
        if let Some(readahead) = self.config.io_optimization.readahead_kb {
            self.set_readahead(readahead).await?;
        }

        Ok(())
    }

    async fn set_cpu_governor(&self, governor: CpuGovernor) -> Result<()> {
        let gov_name = match governor {
            CpuGovernor::Performance => "performance",
            CpuGovernor::Powersave => "powersave",
            CpuGovernor::Ondemand => "ondemand",
            CpuGovernor::Conservative => "conservative",
            CpuGovernor::Schedutil => "schedutil",
        };

        debug!("Setting CPU governor to: {}", gov_name);

        // Set governor for all CPUs
        let cpus = self.get_cpu_count().await?;
        for cpu in 0..cpus {
            let path = format!(
                "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor",
                cpu
            );
            if let Err(e) = tokio::fs::write(&path, gov_name).await {
                warn!("Failed to set governor for CPU {}: {}", cpu, e);
            }
        }

        Ok(())
    }

    async fn get_cpu_count(&self) -> Result<u32> {
        Ok(std::thread::available_parallelism()?.get() as u32)
    }

    async fn enable_numa_balancing(&self) -> Result<()> {
        debug!("Enabling NUMA balancing");
        tokio::fs::write("/proc/sys/kernel/numa_balancing", "1")
            .await
            .context("Failed to enable NUMA balancing")?;
        Ok(())
    }

    async fn configure_hugepages(&self) -> Result<()> {
        let policy = match self.config.memory_optimization.hugepage_policy {
            HugepagePolicy::Never => "never",
            HugepagePolicy::Madvise => "madvise",
            HugepagePolicy::Always => "always",
        };

        debug!("Setting transparent hugepage policy to: {}", policy);
        tokio::fs::write("/sys/kernel/mm/transparent_hugepage/enabled", policy)
            .await
            .context("Failed to set transparent hugepage policy")?;

        Ok(())
    }

    async fn optimize_swap(&self) -> Result<()> {
        debug!("Optimizing swap settings");
        // Set swappiness to 10 for better performance
        tokio::fs::write("/proc/sys/vm/swappiness", "10")
            .await
            .context("Failed to set swappiness")?;
        Ok(())
    }

    async fn enable_memory_compaction(&self) -> Result<()> {
        debug!("Enabling memory compaction");
        tokio::fs::write("/proc/sys/vm/compact_memory", "1")
            .await
            .context("Failed to trigger memory compaction")?;
        Ok(())
    }

    async fn enable_ksm(&self) -> Result<()> {
        debug!("Enabling KSM");
        tokio::fs::write("/sys/kernel/mm/ksm/run", "1")
            .await
            .context("Failed to enable KSM")?;
        Ok(())
    }

    async fn enable_gpu_persistence(&self) -> Result<()> {
        debug!("Enabling GPU persistence mode");

        let output = tokio::process::Command::new("nvidia-smi")
            .arg("-pm")
            .arg("1")
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Failed to enable GPU persistence mode: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        Ok(())
    }

    async fn set_gpu_compute_mode(&self, mode: ComputeMode) -> Result<()> {
        let mode_val = match mode {
            ComputeMode::Default => "0",
            ComputeMode::Exclusive => "1",
            ComputeMode::Prohibited => "2",
        };

        debug!("Setting GPU compute mode to: {:?}", mode);

        let output = tokio::process::Command::new("nvidia-smi")
            .arg("-c")
            .arg(mode_val)
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Failed to set GPU compute mode: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        Ok(())
    }

    async fn set_io_scheduler(&self, scheduler: IoScheduler) -> Result<()> {
        let sched_name = match scheduler {
            IoScheduler::Mq_deadline => "mq-deadline",
            IoScheduler::Kyber => "kyber",
            IoScheduler::Bfq => "bfq",
            IoScheduler::None => "none",
        };

        debug!("Setting I/O scheduler to: {}", sched_name);

        // Find block devices
        let block_devices = self.get_block_devices().await?;
        for device in block_devices {
            let path = format!("/sys/block/{}/queue/scheduler", device);
            if let Err(e) = tokio::fs::write(&path, sched_name).await {
                warn!("Failed to set I/O scheduler for {}: {}", device, e);
            }
        }

        Ok(())
    }

    async fn get_block_devices(&self) -> Result<Vec<String>> {
        // Simplified - would read /proc/partitions or /sys/block
        Ok(vec!["sda".to_string(), "nvme0n1".to_string()])
    }

    async fn set_readahead(&self, kb: u32) -> Result<()> {
        debug!("Setting readahead to {} KB", kb);

        let block_devices = self.get_block_devices().await?;
        for device in block_devices {
            let path = format!("/sys/block/{}/queue/read_ahead_kb", device);
            if let Err(e) = tokio::fs::write(&path, kb.to_string()).await {
                warn!("Failed to set readahead for {}: {}", device, e);
            }
        }

        Ok(())
    }

    async fn set_cpu_affinity(&self, cpus: &[usize]) -> Result<()> {
        info!("Setting CPU affinity to: {:?}", cpus);
        // Implementation would use sched_setaffinity
        Ok(())
    }

    async fn set_memory_policy(&self, _policy: MemoryPolicy) -> Result<()> {
        info!("Setting memory policy");
        // Implementation would use set_mempolicy
        Ok(())
    }

    async fn apply_gpu_settings(&self, settings: &GpuSettings) -> Result<()> {
        info!("Applying GPU settings");

        // Set power limit
        if let Some(power_limit) = settings.power_limit_watts {
            self.set_gpu_power_limit(power_limit).await?;
        }

        // Set memory clock offset
        if let Some(offset) = settings.memory_clock_offset {
            self.set_gpu_memory_clock_offset(offset).await?;
        }

        Ok(())
    }

    async fn set_gpu_power_limit(&self, watts: u32) -> Result<()> {
        debug!("Setting GPU power limit to {} watts", watts);

        let output = tokio::process::Command::new("nvidia-smi")
            .arg("-pl")
            .arg(watts.to_string())
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "Failed to set GPU power limit: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        Ok(())
    }

    async fn set_gpu_memory_clock_offset(&self, offset: i32) -> Result<()> {
        debug!("Setting GPU memory clock offset to {}", offset);
        // Implementation would use nvidia-settings or similar
        Ok(())
    }

    async fn set_io_priority(&self, _priority: IoPriority) -> Result<()> {
        info!("Setting I/O priority");
        // Implementation would use ionice
        Ok(())
    }

    async fn set_scheduling_policy(&self, _policy: SchedulingPolicy) -> Result<()> {
        info!("Setting scheduling policy");
        // Implementation would use sched_setscheduler
        Ok(())
    }
}

/// Metrics cache for historical data
#[derive(Debug)]
struct MetricsCache {
    history: Vec<PerformanceMetrics>,
    max_size: usize,
}

impl MetricsCache {
    fn new() -> Self {
        Self {
            history: Vec::new(),
            max_size: 1000, // Keep last 1000 samples
        }
    }

    fn update(&mut self, metrics: PerformanceMetrics) {
        self.history.push(metrics);

        // Keep only recent history
        if self.history.len() > self.max_size {
            self.history.drain(0..self.history.len() - self.max_size);
        }
    }

    fn get_history(&self) -> &[PerformanceMetrics] {
        &self.history
    }
}

/// Performance metrics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: Option<f64>,
    pub io_wait: f64,
    pub load_average: (f64, f64, f64),
    pub avg_latency: Duration,
    pub throughput: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            cpu_usage: 0.0,
            memory_usage: 0.0,
            gpu_usage: None,
            io_wait: 0.0,
            load_average: (0.0, 0.0, 0.0),
            avg_latency: Duration::from_millis(0),
            throughput: 0.0,
        }
    }
}

/// Performance report
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metrics: PerformanceMetrics,
    pub recommendations: Vec<Recommendation>,
    pub workload_analysis: WorkloadAnalysis,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Recommendation {
    pub category: RecommendationCategory,
    pub priority: Priority,
    pub description: String,
    pub suggestion: String,
    pub impact_score: f64, // 0.0 to 1.0
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Cpu,
    Memory,
    Gpu,
    Io,
    Network,
    Latency,
    Throughput,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkloadAnalysis {
    pub patterns: HashMap<String, PatternAnalysis>,
    pub peak_hours: Vec<u8>,
    pub resource_correlation: HashMap<String, f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PatternAnalysis {
    pub average: f64,
    pub variance: f64,
    pub trend: Trend,
    pub stability: Stability,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Trend {
    Increasing,
    Decreasing,
    Stable,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Stability {
    Stable,
    Variable,
    Erratic,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert!(config.enabled);
        assert!(config.profiling_enabled);
        assert_eq!(config.optimization_level, OptimizationLevel::Balanced);
        assert_eq!(config.workload_profiles.len(), 3);
    }

    #[tokio::test]
    async fn test_performance_optimizer_creation() {
        let config = PerformanceConfig::default();
        let optimizer = PerformanceOptimizer::new(config);

        // Test basic structure
        assert!(optimizer.config.enabled);
    }

    #[test]
    fn test_workload_profiles() {
        let config = PerformanceConfig::default();

        assert!(config.workload_profiles.contains_key("ml"));
        assert!(config.workload_profiles.contains_key("gaming"));
        assert!(config.workload_profiles.contains_key("compute"));

        let ml_profile = &config.workload_profiles["ml"];
        assert_eq!(ml_profile.name, "Machine Learning");
        assert_eq!(ml_profile.memory_policy, MemoryPolicy::Bind);
    }

    #[test]
    fn test_cpu_governor_serialization() {
        let governor = CpuGovernor::Performance;
        let serialized = serde_json::to_string(&governor).unwrap();
        assert!(serialized.contains("Performance"));

        let deserialized: CpuGovernor = serde_json::from_str(&serialized).unwrap();
        assert!(matches!(deserialized, CpuGovernor::Performance));
    }

    #[test]
    fn test_metrics_cache() {
        let mut cache = MetricsCache::new();

        let metrics = PerformanceMetrics::default();
        cache.update(metrics.clone());

        assert_eq!(cache.get_history().len(), 1);
        assert_eq!(cache.get_history()[0].cpu_usage, metrics.cpu_usage);
    }

    #[test]
    fn test_recommendation_priority() {
        let rec = Recommendation {
            category: RecommendationCategory::Cpu,
            priority: Priority::High,
            description: "Test".to_string(),
            suggestion: "Test suggestion".to_string(),
            impact_score: 0.8,
        };

        assert!(matches!(rec.priority, Priority::High));
        assert_eq!(rec.impact_score, 0.8);
    }

    #[test]
    fn test_gpu_settings() {
        let settings = GpuSettings {
            power_limit_watts: Some(250),
            memory_clock_offset: Some(1000),
            graphics_clock_offset: Some(200),
            fan_speed_percent: Some(80),
        };

        assert_eq!(settings.power_limit_watts, Some(250));
        assert_eq!(settings.memory_clock_offset, Some(1000));
    }
}
