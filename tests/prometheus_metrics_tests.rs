//! Prometheus metrics tests for nvbind
//!
//! Tests the metrics collection functionality

use anyhow::Result;
use nvbind::metrics::{MetricsCollector, MetricsConfig};
use nvbind::monitoring::MonitoringConfig;
use nvbind::observability::ObservabilityConfig;
use std::collections::HashMap;

/// Test basic metrics collection setup
#[test]
fn test_metrics_collector_initialization() -> Result<()> {
    let config = MetricsConfig::default();
    let _collector = MetricsCollector::new(config);

    println!("✓ Metrics collector initialized successfully");
    Ok(())
}

/// Test session management
#[test]
fn test_metrics_session() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config);

    let mut tags = HashMap::new();
    tags.insert("test".to_string(), "value".to_string());

    // Start a measurement session
    collector.start_session("test-session".to_string(), tags)?;

    // End the session
    let _results = collector.end_session("test-session")?;

    println!("✓ Metrics session test completed");
    Ok(())
}

/// Test GPU discovery measurement
#[tokio::test]
async fn test_gpu_discovery_measurement() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config);

    // Measure a mock GPU discovery operation
    let (result, latency) = collector.measure_gpu_discovery(|| {
        // Simulate GPU discovery work
        std::thread::sleep(std::time::Duration::from_millis(1));
        Ok("mock_result")
    }).await?;

    println!("GPU discovery took {} nanoseconds", latency);
    assert_eq!(result, "mock_result");

    Ok(())
}

/// Test container creation measurement
#[tokio::test]
async fn test_container_creation_measurement() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config);

    // Measure a mock container creation operation
    let (result, startup_metric) = collector.measure_container_creation("test-container", "docker", || {
        // Simulate container creation work
        std::thread::sleep(std::time::Duration::from_millis(2));
        Ok("container_id")
    }).await?;

    println!("Container creation took {:?}", startup_metric);
    assert_eq!(result, "container_id");

    Ok(())
}

/// Test GPU utilization collection
#[tokio::test]
async fn test_gpu_utilization_collection() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config);

    // Collect GPU utilization metrics (may fail without actual GPU)
    match collector.collect_gpu_utilization().await {
        Ok(metrics) => {
            println!("Collected {} GPU utilization metrics", metrics.len());
        }
        Err(e) => {
            println!("GPU utilization collection failed (expected without GPU): {}", e);
        }
    }

    Ok(())
}

/// Test metrics export
#[tokio::test]
async fn test_metrics_export() -> Result<()> {
    use tempfile::TempDir;

    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config);

    let temp_dir = TempDir::new()?;
    let export_path = temp_dir.path().join("metrics.json");

    // Export metrics to file
    collector.export_metrics(export_path.to_str().unwrap()).await?;

    // Verify file exists
    assert!(export_path.exists());

    println!("✓ Metrics export test completed");
    Ok(())
}

/// Test performance summary
#[tokio::test]
async fn test_performance_summary() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config);

    // Get performance summary
    let summary = collector.get_performance_summary().await?;

    println!("Performance summary:");
    println!("  Total containers: {}", summary.total_containers_created);
    println!("  Average GPU latency: {} ns", summary.average_gpu_latency_ns);
    println!("  Sub-microsecond achieved: {}", summary.sub_microsecond_achieved);

    Ok(())
}

/// Test monitoring configuration
#[test]
fn test_monitoring_config() -> Result<()> {
    let _config = MonitoringConfig::default();
    println!("✓ Monitoring configuration test completed");
    Ok(())
}

/// Test observability configuration
#[test]
fn test_observability_config() -> Result<()> {
    let _config = ObservabilityConfig::default();
    println!("✓ Observability configuration test completed");
    Ok(())
}

/// Test default metrics collector creation
#[test]
fn test_default_metrics_collector() {
    let _collector = nvbind::metrics::create_default_metrics_collector();
    println!("✓ Default metrics collector creation test completed");
}

/// Test metrics configuration defaults
#[test]
fn test_metrics_config_defaults() {
    let config = MetricsConfig::default();

    // Verify default values
    assert!(config.max_metrics_history > 0);
    assert!(config.collection_interval.as_millis() > 0);

    println!("✓ Metrics configuration defaults test completed");
}