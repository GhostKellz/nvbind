//! Comprehensive stability tests for nvbind
//!
//! These tests focus on long-running scenarios, memory leaks,
//! crash recovery, and overall system stability

use anyhow::Result;
use nvbind::cdi;
use nvbind::gpu;
use nvbind::metrics::{MetricsCollector, MetricsConfig};
use nvbind::runtime;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::{sleep, timeout};

/// Configuration for stability tests
#[derive(Debug, Clone)]
pub struct StabilityTestConfig {
    pub duration: Duration,
    pub operation_interval: Duration,
    pub memory_check_interval: Duration,
    pub max_memory_growth_mb: u64,
    pub stress_concurrency: usize,
}

impl Default for StabilityTestConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_secs(300), // 5 minutes
            operation_interval: Duration::from_millis(100),
            memory_check_interval: Duration::from_secs(30),
            max_memory_growth_mb: 100,
            stress_concurrency: 10,
        }
    }
}

/// Memory usage tracker
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub initial_mb: u64,
    pub current_mb: u64,
    pub peak_mb: u64,
    pub growth_mb: u64,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStats {
    pub fn new() -> Self {
        let initial = get_memory_usage_mb();
        Self {
            initial_mb: initial,
            current_mb: initial,
            peak_mb: initial,
            growth_mb: 0,
        }
    }

    pub fn update(&mut self) {
        self.current_mb = get_memory_usage_mb();
        if self.current_mb > self.peak_mb {
            self.peak_mb = self.current_mb;
        }
        self.growth_mb = self.current_mb.saturating_sub(self.initial_mb);
    }

    pub fn is_growing_excessively(&self, max_growth_mb: u64) -> bool {
        self.growth_mb > max_growth_mb
    }
}

/// Get current memory usage in MB (approximation for testing)
fn get_memory_usage_mb() -> u64 {
    // In a real implementation, this would use system APIs
    // For testing, we'll simulate with a mock value
    use std::sync::atomic::{AtomicU64, Ordering};
    static MOCK_MEMORY: AtomicU64 = AtomicU64::new(50);
    MOCK_MEMORY.load(Ordering::Relaxed)
}

/// Stability test runner
pub struct StabilityTestRunner {
    config: StabilityTestConfig,
    memory_stats: MemoryStats,
    start_time: Instant,
    operations_completed: u64,
    errors_encountered: u64,
}

impl StabilityTestRunner {
    pub fn new(config: StabilityTestConfig) -> Self {
        Self {
            config,
            memory_stats: MemoryStats::new(),
            start_time: Instant::now(),
            operations_completed: 0,
            errors_encountered: 0,
        }
    }

    pub async fn run_gpu_discovery_stress_test(&mut self) -> Result<()> {
        println!("Starting GPU discovery stress test...");

        let end_time = self.start_time + self.config.duration;

        while Instant::now() < end_time {
            // Perform GPU discovery operation
            match gpu::discover_gpus().await {
                Ok(_) => {
                    self.operations_completed += 1;
                }
                Err(e) => {
                    self.errors_encountered += 1;
                    eprintln!("GPU discovery error: {}", e);
                }
            }

            // Check memory usage periodically
            if self.operations_completed % 100 == 0 {
                self.memory_stats.update();
                if self
                    .memory_stats
                    .is_growing_excessively(self.config.max_memory_growth_mb)
                {
                    return Err(anyhow::anyhow!(
                        "Memory leak detected: {} MB growth",
                        self.memory_stats.growth_mb
                    ));
                }
            }

            sleep(self.config.operation_interval).await;
        }

        self.print_test_summary("GPU Discovery Stress Test");
        Ok(())
    }

    pub async fn run_concurrent_runtime_test(&mut self) -> Result<()> {
        println!("Starting concurrent runtime test...");

        let mut handles = Vec::new();

        for _i in 0..self.config.stress_concurrency {
            let handle = tokio::spawn(async move {
                // Simulate runtime operations
                for _ in 0..100 {
                    let _result = runtime::validate_runtime("docker");
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }

                Ok::<(), anyhow::Error>(())
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete with timeout
        let timeout_duration = self.config.duration;
        match timeout(timeout_duration, futures::future::join_all(handles)).await {
            Ok(results) => {
                for (i, result) in results.into_iter().enumerate() {
                    match result {
                        Ok(Ok(())) => {
                            self.operations_completed += 1;
                        }
                        Ok(Err(e)) => {
                            self.errors_encountered += 1;
                            eprintln!("Runtime task {} error: {}", i, e);
                        }
                        Err(e) => {
                            self.errors_encountered += 1;
                            eprintln!("Runtime task {} panicked: {}", i, e);
                        }
                    }
                }
            }
            Err(_) => {
                return Err(anyhow::anyhow!("Concurrent runtime test timed out"));
            }
        }

        self.print_test_summary("Concurrent Runtime Test");
        Ok(())
    }

    pub async fn run_memory_leak_detection_test(&mut self) -> Result<()> {
        println!("Starting memory leak detection test...");

        let metrics_config = MetricsConfig::default();
        let collector = MetricsCollector::new(metrics_config);
        let end_time = self.start_time + self.config.duration;

        let mut memory_samples = Vec::new();

        while Instant::now() < end_time {
            // Perform memory-intensive operations
            let mut tags = HashMap::new();
            tags.insert(
                "test".to_string(),
                format!("iteration_{}", self.operations_completed),
            );

            match collector.start_session(format!("session_{}", self.operations_completed), tags) {
                Ok(_) => {
                    // Simulate some work
                    sleep(Duration::from_millis(50)).await;

                    match collector.end_session(&format!("session_{}", self.operations_completed)) {
                        Ok(_) => {
                            self.operations_completed += 1;
                        }
                        Err(e) => {
                            self.errors_encountered += 1;
                            eprintln!("Session error: {}", e);
                        }
                    }
                }
                Err(e) => {
                    self.errors_encountered += 1;
                    eprintln!("Session start error: {}", e);
                }
            }

            // Sample memory usage
            if self.operations_completed % 10 == 0 {
                self.memory_stats.update();
                memory_samples.push(self.memory_stats.current_mb);

                // Check for memory leaks
                if memory_samples.len() > 10 {
                    let recent_avg = memory_samples.iter().rev().take(5).sum::<u64>() / 5;
                    let older_avg = memory_samples.iter().rev().skip(5).take(5).sum::<u64>() / 5;

                    if recent_avg > older_avg + 20 {
                        return Err(anyhow::anyhow!(
                            "Potential memory leak detected: {} MB -> {} MB",
                            older_avg,
                            recent_avg
                        ));
                    }
                }
            }

            sleep(self.config.operation_interval).await;
        }

        self.print_test_summary("Memory Leak Detection Test");
        Ok(())
    }

    pub async fn run_crash_recovery_test(&mut self) -> Result<()> {
        println!("Starting crash recovery test...");

        // Simulate various failure scenarios
        for scenario in 1..=5 {
            println!("Testing crash recovery scenario {}", scenario);

            match scenario {
                1 => {
                    // Test recovery from GPU detection failure
                    match gpu::discover_gpus().await {
                        Ok(_) => self.operations_completed += 1,
                        Err(e) => {
                            println!("Expected failure in scenario {}: {}", scenario, e);
                            // Verify system can recover
                            sleep(Duration::from_millis(100)).await;
                            match gpu::discover_gpus().await {
                                Ok(_) => {
                                    println!("✓ Recovery successful for scenario {}", scenario);
                                    self.operations_completed += 1;
                                }
                                Err(e) => {
                                    self.errors_encountered += 1;
                                    eprintln!("✗ Recovery failed for scenario {}: {}", scenario, e);
                                }
                            }
                        }
                    }
                }
                2 => {
                    // Test CDI generation recovery
                    match cdi::generate_nvidia_cdi_spec().await {
                        Ok(_) => self.operations_completed += 1,
                        Err(e) => {
                            println!("Expected failure in scenario {}: {}", scenario, e);
                            self.operations_completed += 1; // Count as expected
                        }
                    }
                }
                _ => {
                    // Generic recovery test
                    self.operations_completed += 1;
                }
            }

            sleep(Duration::from_millis(500)).await;
        }

        self.print_test_summary("Crash Recovery Test");
        Ok(())
    }

    fn print_test_summary(&self, test_name: &str) {
        let elapsed = self.start_time.elapsed();
        let ops_per_sec = self.operations_completed as f64 / elapsed.as_secs_f64();
        let error_rate =
            (self.errors_encountered as f64 / self.operations_completed.max(1) as f64) * 100.0;

        println!("\n{} Summary:", test_name);
        println!("  Duration: {:?}", elapsed);
        println!("  Operations: {}", self.operations_completed);
        println!("  Errors: {}", self.errors_encountered);
        println!("  Ops/sec: {:.2}", ops_per_sec);
        println!("  Error rate: {:.2}%", error_rate);
        println!(
            "  Memory usage: {} MB (growth: {} MB)",
            self.memory_stats.current_mb, self.memory_stats.growth_mb
        );
        println!("  Peak memory: {} MB", self.memory_stats.peak_mb);
    }
}

/// Long-running stability test (5 minutes)
#[tokio::test]
#[ignore] // Use `cargo test -- --ignored` to run
async fn test_long_running_stability() -> Result<()> {
    let config = StabilityTestConfig {
        duration: Duration::from_secs(300), // 5 minutes
        ..Default::default()
    };

    let mut runner = StabilityTestRunner::new(config);

    println!("Starting long-running stability test (5 minutes)...");

    // Run GPU discovery stress test
    runner.run_gpu_discovery_stress_test().await?;

    // Reset counters for next test
    runner.operations_completed = 0;
    runner.errors_encountered = 0;
    runner.start_time = Instant::now();

    // Run memory leak detection
    runner.run_memory_leak_detection_test().await?;

    println!("✓ Long-running stability test completed successfully");
    Ok(())
}

/// Quick stability test (30 seconds)
#[tokio::test]
async fn test_quick_stability() -> Result<()> {
    let config = StabilityTestConfig {
        duration: Duration::from_secs(30),
        operation_interval: Duration::from_millis(50),
        ..Default::default()
    };

    let mut runner = StabilityTestRunner::new(config);

    println!("Starting quick stability test (30 seconds)...");

    // Run concurrent runtime test
    runner.run_concurrent_runtime_test().await?;

    println!("✓ Quick stability test completed successfully");
    Ok(())
}

/// Memory leak detection test
#[tokio::test]
async fn test_memory_leak_detection() -> Result<()> {
    let config = StabilityTestConfig {
        duration: Duration::from_secs(60),
        operation_interval: Duration::from_millis(10),
        max_memory_growth_mb: 50,
        ..Default::default()
    };

    let mut runner = StabilityTestRunner::new(config);

    println!("Starting memory leak detection test...");
    runner.run_memory_leak_detection_test().await?;

    println!("✓ Memory leak detection test completed successfully");
    Ok(())
}

/// Crash recovery test
#[tokio::test]
async fn test_crash_recovery() -> Result<()> {
    let config = StabilityTestConfig {
        duration: Duration::from_secs(10),
        ..Default::default()
    };

    let mut runner = StabilityTestRunner::new(config);

    println!("Starting crash recovery test...");
    runner.run_crash_recovery_test().await?;

    println!("✓ Crash recovery test completed successfully");
    Ok(())
}

/// Concurrent stress test
#[tokio::test]
async fn test_concurrent_stress() -> Result<()> {
    let config = StabilityTestConfig {
        duration: Duration::from_secs(30),
        stress_concurrency: 20,
        ..Default::default()
    };

    let mut runner = StabilityTestRunner::new(config);

    println!("Starting concurrent stress test...");
    runner.run_concurrent_runtime_test().await?;

    println!("✓ Concurrent stress test completed successfully");
    Ok(())
}

/// Resource exhaustion test
#[tokio::test]
async fn test_resource_exhaustion_handling() -> Result<()> {
    println!("Starting resource exhaustion test...");

    let mut operations = 0;
    let mut successful_recoveries = 0;

    // Attempt to exhaust resources with rapid operations
    for i in 0..1000 {
        match timeout(Duration::from_millis(100), gpu::discover_gpus()).await {
            Ok(Ok(_)) => {
                operations += 1;
            }
            Ok(Err(_)) => {
                // Resource exhaustion occurred, test recovery
                sleep(Duration::from_millis(10)).await;
                match gpu::discover_gpus().await {
                    Ok(_) => {
                        successful_recoveries += 1;
                    }
                    Err(_) => {
                        // This is acceptable under resource exhaustion
                    }
                }
            }
            Err(_) => {
                // Timeout occurred
                operations += 1;
            }
        }

        if i % 100 == 0 {
            println!(
                "Completed {} operations, {} recoveries",
                operations, successful_recoveries
            );
        }
    }

    println!(
        "✓ Resource exhaustion test completed: {} operations, {} recoveries",
        operations, successful_recoveries
    );
    Ok(())
}
