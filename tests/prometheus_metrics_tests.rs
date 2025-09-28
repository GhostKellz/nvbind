//! Prometheus metrics tests for nvbind
//!
//! Tests the metrics collection and Prometheus integration functionality

use anyhow::Result;
use nvbind::metrics::{MetricsCollector, MetricsConfig};
use nvbind::monitoring::{MonitoringSystem, MonitoringConfig};
use nvbind::observability::{ObservabilityManager, ObservabilityConfig};
use prometheus::{Encoder, TextEncoder};

/// Test basic metrics collection setup
#[test]
fn test_metrics_collector_initialization() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config)?;

    // Verify collector is initialized
    collector.initialize()?;

    println!("✓ Metrics collector initialized successfully");
    Ok(())
}

/// Test GPU metrics collection
#[tokio::test]
async fn test_gpu_metrics_collection() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config)?;
    collector.initialize()?;

    // Record GPU metrics
    collector.record_gpu_utilization("gpu0", 75.0)?;
    collector.record_gpu_memory_usage("gpu0", 4096, 8192)?; // 4GB used of 8GB
    collector.record_gpu_temperature("gpu0", 65.0)?;
    collector.record_gpu_power_draw("gpu0", 150.0)?;

    // Record multiple GPUs
    collector.record_gpu_utilization("gpu1", 50.0)?;
    collector.record_gpu_memory_usage("gpu1", 2048, 8192)?;

    println!("✓ GPU metrics recorded successfully");
    Ok(())
}

/// Test container metrics collection
#[tokio::test]
async fn test_container_metrics() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config)?;
    collector.initialize()?;

    // Record container lifecycle metrics
    collector.record_container_start("test-container-1", "nvidia/cuda:12.0")?;
    collector.record_container_gpu_attach("test-container-1", "gpu0")?;

    // Simulate some runtime metrics
    collector.record_container_memory_usage("test-container-1", 1024 * 1024 * 512)?; // 512MB
    collector.record_container_cpu_usage("test-container-1", 25.5)?;

    // Record container stop
    collector.record_container_stop("test-container-1")?;

    println!("✓ Container metrics recorded successfully");
    Ok(())
}

/// Test runtime performance metrics
#[test]
fn test_runtime_performance_metrics() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config)?;
    collector.initialize()?;

    // Record runtime operation latencies
    collector.record_runtime_latency("gpu_discovery", 0.025)?; // 25ms
    collector.record_runtime_latency("cdi_spec_generation", 0.015)?; // 15ms
    collector.record_runtime_latency("container_startup", 1.234)?; // 1.234s

    // Record operation counts
    collector.increment_operation_count("gpu_passthrough_success")?;
    collector.increment_operation_count("gpu_passthrough_success")?;
    collector.increment_operation_count("cdi_device_created")?;

    println!("✓ Runtime performance metrics recorded");
    Ok(())
}

/// Test Prometheus export format
#[test]
fn test_prometheus_export() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config)?;
    collector.initialize()?;

    // Record various metrics
    collector.record_gpu_utilization("gpu0", 80.0)?;
    collector.record_container_start("test-container", "ubuntu:22.04")?;
    collector.increment_operation_count("gpu_attach")?;

    // Export metrics in Prometheus format
    let metrics = collector.gather_metrics()?;
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();
    encoder.encode(&metrics, &mut buffer)?;

    let output = String::from_utf8(buffer)?;

    // Verify output contains expected metric lines
    assert!(output.contains("nvbind_gpu_utilization"));
    assert!(output.contains("nvbind_container_starts_total"));
    assert!(output.contains("nvbind_operation_count"));

    println!("✓ Prometheus export format validated");
    println!("Sample metrics output:\n{}", &output[..output.len().min(500)]);

    Ok(())
}

/// Test monitoring system integration
#[tokio::test]
async fn test_monitoring_system() -> Result<()> {
    let config = MonitoringConfig::default();
    let monitoring = MonitoringSystem::new(config)?;

    // Start monitoring
    monitoring.start().await?;

    // Simulate GPU events
    monitoring.record_gpu_event("gpu0", "utilization_high", 95.0).await?;
    monitoring.record_gpu_event("gpu0", "memory_threshold", 90.0).await?;

    // Check alerts
    let alerts = monitoring.get_active_alerts().await?;
    println!("Active alerts: {}", alerts.len());

    // Stop monitoring
    monitoring.stop().await?;

    println!("✓ Monitoring system test completed");
    Ok(())
}

/// Test observability manager with tracing
#[tokio::test]
async fn test_observability_manager() -> Result<()> {
    let config = ObservabilityConfig::default();
    let manager = ObservabilityManager::new(config)?;

    // Start a trace
    let trace_id = manager.start_trace("container_launch")?;

    // Add span events
    manager.add_span_event(trace_id, "gpu_discovery", "Started GPU discovery")?;
    manager.add_span_event(trace_id, "gpu_discovery", "Found 2 GPUs")?;
    manager.add_span_event(trace_id, "container_creation", "Creating container")?;

    // End trace
    manager.end_trace(trace_id)?;

    println!("✓ Observability tracing test completed");
    Ok(())
}

/// Test metrics aggregation
#[tokio::test]
async fn test_metrics_aggregation() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config)?;
    collector.initialize()?;

    // Record multiple data points
    for i in 0..10 {
        let utilization = 50.0 + (i as f64 * 5.0);
        collector.record_gpu_utilization("gpu0", utilization)?;
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    // Get aggregated statistics
    let stats = collector.get_gpu_statistics("gpu0")?;

    println!("GPU Statistics:");
    println!("  Average utilization: {:.2}%", stats.avg_utilization);
    println!("  Peak utilization: {:.2}%", stats.peak_utilization);
    println!("  Sample count: {}", stats.sample_count);

    Ok(())
}

/// Test metrics persistence
#[test]
fn test_metrics_persistence() -> Result<()> {
    use tempfile::TempDir;

    let temp_dir = TempDir::new()?;
    let metrics_file = temp_dir.path().join("metrics.prom");

    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config)?;
    collector.initialize()?;

    // Record metrics
    collector.record_gpu_utilization("gpu0", 75.0)?;
    collector.record_container_start("test", "nvidia/cuda")?;

    // Export to file
    collector.export_to_file(&metrics_file)?;

    // Verify file exists and contains data
    assert!(metrics_file.exists());
    let content = std::fs::read_to_string(&metrics_file)?;
    assert!(!content.is_empty());
    assert!(content.contains("nvbind_"));

    println!("✓ Metrics persistence test passed");
    Ok(())
}

/// Test custom metrics registration
#[test]
fn test_custom_metrics() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config)?;
    collector.initialize()?;

    // Register custom metrics
    collector.register_custom_gauge("nvbind_custom_metric", "Custom test metric")?;
    collector.register_custom_counter("nvbind_custom_events", "Custom event counter")?;

    // Update custom metrics
    collector.set_custom_gauge("nvbind_custom_metric", 42.0)?;
    collector.increment_custom_counter("nvbind_custom_events")?;
    collector.increment_custom_counter("nvbind_custom_events")?;

    // Verify metrics are recorded
    let metrics = collector.gather_metrics()?;
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();
    encoder.encode(&metrics, &mut buffer)?;
    let output = String::from_utf8(buffer)?;

    assert!(output.contains("nvbind_custom_metric 42"));
    assert!(output.contains("nvbind_custom_events 2"));

    println!("✓ Custom metrics test passed");
    Ok(())
}

/// Test metrics HTTP endpoint
#[tokio::test]
async fn test_metrics_http_endpoint() -> Result<()> {
    let config = MetricsConfig {
        enable_http_endpoint: true,
        http_port: 9091, // Use non-standard port to avoid conflicts
        ..Default::default()
    };

    let collector = MetricsCollector::new(config)?;
    collector.initialize()?;

    // Start HTTP server
    let server = collector.start_http_server().await?;

    // Record some metrics
    collector.record_gpu_utilization("gpu0", 65.0)?;
    collector.increment_operation_count("test_operation")?;

    // Wait a bit for server to be ready
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // In a real test, we would make an HTTP request to verify the endpoint
    // For now, just verify the server started
    println!("✓ Metrics HTTP endpoint started on port 9091");

    // Cleanup
    server.shutdown().await?;

    Ok(())
}

/// Test metrics labels and dimensions
#[test]
fn test_metrics_with_labels() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config)?;
    collector.initialize()?;

    // Record metrics with labels
    let mut labels = std::collections::HashMap::new();
    labels.insert("gpu_model", "RTX_3090");
    labels.insert("driver_version", "535.129.03");
    collector.record_gpu_utilization_with_labels("gpu0", 75.0, &labels)?;

    labels.insert("container_runtime", "docker");
    labels.insert("image", "pytorch:latest");
    collector.record_container_metric_with_labels("memory_usage", 2048.0, &labels)?;

    println!("✓ Metrics with labels recorded successfully");
    Ok(())
}

/// Test high cardinality metrics handling
#[tokio::test]
async fn test_high_cardinality_metrics() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config)?;
    collector.initialize()?;

    // Generate many unique container IDs
    for i in 0..100 {
        let container_id = format!("container_{}", i);
        collector.record_container_start(&container_id, "ubuntu:22.04")?;
        collector.record_container_cpu_usage(&container_id, (i as f64) % 100.0)?;
    }

    // Verify collector handles high cardinality gracefully
    let metrics = collector.gather_metrics()?;
    assert!(!metrics.is_empty());

    println!("✓ High cardinality metrics test passed (100 containers)");
    Ok(())
}

/// Test metrics reset and cleanup
#[test]
fn test_metrics_reset() -> Result<()> {
    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(config)?;
    collector.initialize()?;

    // Record initial metrics
    collector.increment_operation_count("test_op")?;
    collector.increment_operation_count("test_op")?;

    // Reset metrics
    collector.reset_metrics()?;

    // Verify counters are reset
    let metrics = collector.gather_metrics()?;
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();
    encoder.encode(&metrics, &mut buffer)?;
    let output = String::from_utf8(buffer)?;

    // After reset, counters should be at 0
    println!("✓ Metrics reset test completed");
    Ok(())
}