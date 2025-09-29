//! Stress tests for nvbind edge cases and error conditions
//!
//! These tests push the system to its limits to identify potential
//! failure modes and ensure graceful degradation

use anyhow::Result;
use futures::future;
use nvbind::cdi;
use nvbind::config::Config;
use nvbind::gpu;
use nvbind::runtime;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::{sleep, timeout};

/// Type aliases for complex types
type OperationResult = Result<(Duration, String), (Duration, String)>;
type OperationHandle = tokio::task::JoinHandle<OperationResult>;
type OperationHandles = Vec<OperationHandle>;
type BatchResults = Vec<Result<OperationResult, tokio::task::JoinError>>;

/// Stress test configuration
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    pub max_concurrent_operations: usize,
    pub operation_timeout: Duration,
    pub total_operations: usize,
    pub failure_injection_rate: f64, // 0.0 to 1.0
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            max_concurrent_operations: 100,
            operation_timeout: Duration::from_secs(30),
            total_operations: 1000,
            failure_injection_rate: 0.1, // 10% failure rate
        }
    }
}

/// Test result tracking
#[derive(Debug, Default)]
pub struct StressTestResults {
    pub total_operations: usize,
    pub successful_operations: usize,
    pub failed_operations: usize,
    pub timeout_operations: usize,
    pub average_latency_ms: f64,
    pub max_latency_ms: u64,
    pub min_latency_ms: u64,
    pub errors: HashMap<String, usize>,
}

impl StressTestResults {
    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            return 0.0;
        }
        (self.successful_operations as f64 / self.total_operations as f64) * 100.0
    }

    pub fn print_summary(&self, test_name: &str) {
        println!("\n{} Results:", test_name);
        println!("  Total operations: {}", self.total_operations);
        println!("  Successful: {}", self.successful_operations);
        println!("  Failed: {}", self.failed_operations);
        println!("  Timeouts: {}", self.timeout_operations);
        println!("  Success rate: {:.2}%", self.success_rate());
        println!("  Average latency: {:.2} ms", self.average_latency_ms);
        println!("  Min latency: {} ms", self.min_latency_ms);
        println!("  Max latency: {} ms", self.max_latency_ms);

        if !self.errors.is_empty() {
            println!("  Error breakdown:");
            for (error_type, count) in &self.errors {
                println!("    {}: {}", error_type, count);
            }
        }
    }
}

/// Stress test runner
pub struct StressTestRunner {
    config: StressTestConfig,
}

impl StressTestRunner {
    pub fn new(config: StressTestConfig) -> Self {
        Self { config }
    }

    /// Run GPU discovery under extreme load
    pub async fn run_gpu_discovery_stress(&self) -> Result<StressTestResults> {
        println!("Running GPU discovery stress test...");

        let mut results = StressTestResults::default();
        let mut handles: OperationHandles = Vec::new();
        let mut latencies = Vec::new();

        let operations_per_batch = self.config.max_concurrent_operations;
        let total_batches = self.config.total_operations.div_ceil(operations_per_batch);

        for batch in 0..total_batches {
            let batch_size = std::cmp::min(
                operations_per_batch,
                self.config.total_operations - batch * operations_per_batch,
            );

            for _ in 0..batch_size {
                let timeout_duration = self.config.operation_timeout;

                let handle = tokio::spawn(async move {
                    let start = Instant::now();

                    let result = timeout(timeout_duration, gpu::discover_gpus()).await;

                    let latency = start.elapsed();

                    match result {
                        Ok(Ok(_)) => Ok((latency, "success".to_string())),
                        Ok(Err(e)) => Ok((latency, format!("error: {}", e))),
                        Err(_) => Ok((latency, "timeout".to_string())),
                    }
                });

                handles.push(handle);
            }

            // Wait for batch to complete
            let batch_results = future::join_all(handles.drain(..)).await;

            for batch_result in batch_results {
                results.total_operations += 1;

                match batch_result {
                    Ok(Ok((latency, status))) => {
                        let latency_ms = latency.as_millis() as u64;
                        latencies.push(latency_ms);

                        if status == "success" {
                            results.successful_operations += 1;
                        } else if status == "timeout" {
                            results.timeout_operations += 1;
                            *results.errors.entry("timeout".to_string()).or_insert(0) += 1;
                        } else {
                            results.failed_operations += 1;
                            *results.errors.entry(status).or_insert(0) += 1;
                        }
                    }
                    Ok(Err((latency, error))) => {
                        results.failed_operations += 1;
                        let latency_ms = latency.as_millis() as u64;
                        latencies.push(latency_ms);
                        *results.errors.entry(error).or_insert(0) += 1;
                    }
                    Err(e) => {
                        results.failed_operations += 1;
                        *results.errors.entry(format!("panic: {}", e)).or_insert(0) += 1;
                    }
                }
            }

            // Small delay between batches to prevent overwhelming the system
            sleep(Duration::from_millis(10)).await;
        }

        // Calculate latency statistics
        if !latencies.is_empty() {
            results.average_latency_ms =
                latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
            results.min_latency_ms = *latencies.iter().min().unwrap();
            results.max_latency_ms = *latencies.iter().max().unwrap();
        }

        Ok(results)
    }

    /// Run CDI generation stress test
    pub async fn run_cdi_generation_stress(&self) -> Result<StressTestResults> {
        println!("Running CDI generation stress test...");

        let mut results = StressTestResults::default();
        let mut handles: OperationHandles = Vec::new();
        let mut latencies = Vec::new();

        for _ in 0..self.config.total_operations {
            let timeout_duration = self.config.operation_timeout;

            let handle = tokio::spawn(async move {
                let start = Instant::now();

                let result = timeout(timeout_duration, cdi::generate_nvidia_cdi_spec()).await;

                let latency = start.elapsed();

                match result {
                    Ok(Ok(_)) => Ok((latency, "success".to_string())),
                    Ok(Err(e)) => Ok((latency, format!("error: {}", e))),
                    Err(_) => Ok((latency, "timeout".to_string())),
                }
            });

            handles.push(handle);

            // Process in smaller batches to avoid overwhelming
            if handles.len() >= 50 {
                let batch_results = future::join_all(handles.drain(..)).await;
                self.process_batch_results(&mut results, &mut latencies, batch_results);
            }
        }

        // Process remaining handles
        if !handles.is_empty() {
            let batch_results = future::join_all(handles).await;
            self.process_batch_results(&mut results, &mut latencies, batch_results);
        }

        // Calculate latency statistics
        if !latencies.is_empty() {
            results.average_latency_ms =
                latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
            results.min_latency_ms = *latencies.iter().min().unwrap();
            results.max_latency_ms = *latencies.iter().max().unwrap();
        }

        Ok(results)
    }

    /// Run configuration management stress test
    pub async fn run_config_stress(&self) -> Result<StressTestResults> {
        println!("Running configuration management stress test...");

        let mut results = StressTestResults::default();
        let mut handles: OperationHandles = Vec::new();
        let mut latencies = Vec::new();

        for _ in 0..self.config.total_operations {
            let handle = tokio::spawn(async move {
                let start = Instant::now();

                // Perform rapid config operations
                let test_config = Config::default();

                // Simple validation by checking if config can be serialized
                let result = serde_json::to_string(&test_config);

                let latency = start.elapsed();

                match result {
                    Ok(_) => Ok((latency, "success".to_string())),
                    Err(e) => Ok((latency, format!("error: {}", e))),
                }
            });

            handles.push(handle);

            // Process in batches
            if handles.len() >= self.config.max_concurrent_operations {
                let batch_results = future::join_all(handles.drain(..)).await;
                self.process_batch_results(&mut results, &mut latencies, batch_results);
            }
        }

        // Process remaining handles
        if !handles.is_empty() {
            let batch_results = future::join_all(handles).await;
            self.process_batch_results(&mut results, &mut latencies, batch_results);
        }

        // Calculate latency statistics
        if !latencies.is_empty() {
            results.average_latency_ms =
                latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
            results.min_latency_ms = *latencies.iter().min().unwrap();
            results.max_latency_ms = *latencies.iter().max().unwrap();
        }

        Ok(results)
    }

    /// Run runtime validation stress test
    pub async fn run_runtime_validation_stress(&self) -> Result<StressTestResults> {
        println!("Running runtime validation stress test...");

        let mut results = StressTestResults::default();
        let mut handles: OperationHandles = Vec::new();
        let mut latencies = Vec::new();

        // Test various runtime scenarios
        let runtime_scenarios = vec![
            "docker",
            "podman",
            "containerd",
            "crio",
            "invalid_runtime",
            "", // Empty runtime
        ];

        let operations_per_scenario = self.config.total_operations / runtime_scenarios.len();

        for runtime in runtime_scenarios {
            for _ in 0..operations_per_scenario {
                let runtime_clone = runtime.to_string();
                let timeout_duration = self.config.operation_timeout;

                let handle = tokio::spawn(async move {
                    let start = Instant::now();

                    let result = timeout(timeout_duration, async {
                        runtime::validate_runtime(&runtime_clone)
                    })
                    .await;

                    let latency = start.elapsed();

                    match result {
                        Ok(Ok(_)) => Ok((latency, "success".to_string())),
                        Ok(Err(e)) => Ok((latency, format!("error: {}", e))),
                        Err(_) => Ok((latency, "timeout".to_string())),
                    }
                });

                handles.push(handle);

                // Process in batches
                if handles.len() >= 50 {
                    let batch_results = future::join_all(handles.drain(..)).await;
                    self.process_batch_results(&mut results, &mut latencies, batch_results);
                }
            }
        }

        // Process remaining handles
        if !handles.is_empty() {
            let batch_results = future::join_all(handles).await;
            self.process_batch_results(&mut results, &mut latencies, batch_results);
        }

        // Calculate latency statistics
        if !latencies.is_empty() {
            results.average_latency_ms =
                latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
            results.min_latency_ms = *latencies.iter().min().unwrap();
            results.max_latency_ms = *latencies.iter().max().unwrap();
        }

        Ok(results)
    }

    /// Helper to process batch results
    fn process_batch_results(
        &self,
        results: &mut StressTestResults,
        latencies: &mut Vec<u64>,
        batch_results: BatchResults,
    ) {
        for batch_result in batch_results {
            results.total_operations += 1;

            match batch_result {
                Ok(Ok((latency, status))) => {
                    let latency_ms = latency.as_millis() as u64;
                    latencies.push(latency_ms);

                    if status == "success" {
                        results.successful_operations += 1;
                    } else if status == "timeout" {
                        results.timeout_operations += 1;
                        *results.errors.entry("timeout".to_string()).or_insert(0) += 1;
                    } else {
                        results.failed_operations += 1;
                        *results.errors.entry(status).or_insert(0) += 1;
                    }
                }
                Ok(Err((latency, error))) => {
                    results.failed_operations += 1;
                    let latency_ms = latency.as_millis() as u64;
                    latencies.push(latency_ms);
                    *results.errors.entry(error).or_insert(0) += 1;
                }
                Err(e) => {
                    results.failed_operations += 1;
                    *results.errors.entry(format!("panic: {}", e)).or_insert(0) += 1;
                }
            }
        }
    }
}

/// GPU discovery stress test
#[tokio::test]
async fn test_gpu_discovery_stress() -> Result<()> {
    let config = StressTestConfig {
        max_concurrent_operations: 50,
        total_operations: 500,
        ..Default::default()
    };

    let runner = StressTestRunner::new(config);
    let results = runner.run_gpu_discovery_stress().await?;

    results.print_summary("GPU Discovery Stress Test");

    // Assert reasonable success rate (should handle failures gracefully)
    assert!(
        results.success_rate() >= 50.0,
        "Success rate too low: {:.2}%",
        results.success_rate()
    );

    println!("✓ GPU discovery stress test completed");
    Ok(())
}

/// CDI generation stress test
#[tokio::test]
async fn test_cdi_generation_stress() -> Result<()> {
    let config = StressTestConfig {
        max_concurrent_operations: 30,
        total_operations: 200,
        ..Default::default()
    };

    let runner = StressTestRunner::new(config);
    let results = runner.run_cdi_generation_stress().await?;

    results.print_summary("CDI Generation Stress Test");

    // CDI generation should be more reliable
    assert!(
        results.success_rate() >= 70.0,
        "Success rate too low: {:.2}%",
        results.success_rate()
    );

    println!("✓ CDI generation stress test completed");
    Ok(())
}

/// Configuration management stress test
#[tokio::test]
async fn test_config_stress() -> Result<()> {
    let config = StressTestConfig {
        max_concurrent_operations: 100,
        total_operations: 1000,
        operation_timeout: Duration::from_secs(5),
        ..Default::default()
    };

    let runner = StressTestRunner::new(config);
    let results = runner.run_config_stress().await?;

    results.print_summary("Configuration Management Stress Test");

    // Config operations should be very reliable
    assert!(
        results.success_rate() >= 90.0,
        "Success rate too low: {:.2}%",
        results.success_rate()
    );

    println!("✓ Configuration management stress test completed");
    Ok(())
}

/// Runtime validation stress test
#[tokio::test]
async fn test_runtime_validation_stress() -> Result<()> {
    let config = StressTestConfig {
        max_concurrent_operations: 40,
        total_operations: 300,
        ..Default::default()
    };

    let runner = StressTestRunner::new(config);
    let results = runner.run_runtime_validation_stress().await?;

    results.print_summary("Runtime Validation Stress Test");

    // Note: This test includes invalid runtimes, so lower success rate is expected
    assert!(
        results.success_rate() >= 30.0,
        "Success rate too low: {:.2}%",
        results.success_rate()
    );

    println!("✓ Runtime validation stress test completed");
    Ok(())
}

/// Edge case test for extreme scenarios
#[tokio::test]
async fn test_extreme_edge_cases() -> Result<()> {
    println!("Testing extreme edge cases...");

    // Test with very long runtime names
    let long_runtime_name = "a".repeat(1000);

    match runtime::validate_runtime(&long_runtime_name) {
        Ok(_) => println!("✓ Long runtime name handled"),
        Err(e) => println!("✓ Long runtime name rejected: {}", e),
    }

    // Test with special characters in runtime names
    let special_runtime_names = vec![
        "runtime\0with\0nulls",
        "runtime with spaces",
        "runtime/with/slashes",
        "runtime\\with\\backslashes",
        "runtime'with'quotes",
        "runtime\"with\"doublequotes",
        "runtime\nwith\nnewlines",
    ];

    for runtime_name in special_runtime_names {
        match runtime::validate_runtime(runtime_name) {
            Ok(_) => println!(
                "✓ Special runtime name '{}' handled",
                runtime_name.replace('\n', "\\n").replace('\0', "\\0")
            ),
            Err(e) => println!(
                "✓ Special runtime name '{}' rejected: {}",
                runtime_name.replace('\n', "\\n").replace('\0', "\\0"),
                e
            ),
        }
    }

    println!("✓ Extreme edge cases test completed");
    Ok(())
}
