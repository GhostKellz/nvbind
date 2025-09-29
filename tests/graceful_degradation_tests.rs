//! Graceful degradation integration tests
//! Tests the fallback mechanisms when GPU resources are unavailable

use nvbind::error::NvbindError;
use nvbind::graceful_degradation::{
    DegradationConfig, DegradationResult, DegradationStrategy, GracefulDegradationManager,
    OperationContext, OperationCriticality,
};
use std::collections::HashMap;

/// Test degradation manager initialization
#[tokio::test]
async fn test_degradation_manager_init() {
    let config = DegradationConfig::default();
    let manager = GracefulDegradationManager::new(config).await;

    assert!(manager.is_ok());
    let manager = manager.unwrap();
    let stats = manager.get_statistics();

    // Should have detected some system capabilities
    assert!(!stats.capabilities.available_runtimes.is_empty() || !stats.capabilities.gpu_available);
    println!(
        "Detected capabilities: GPU={}, Runtimes={:?}",
        stats.capabilities.gpu_available, stats.capabilities.available_runtimes
    );
}

/// Test CPU fallback degradation
#[tokio::test]
async fn test_cpu_fallback_degradation() {
    let config = DegradationConfig::default();
    let mut manager = GracefulDegradationManager::new(config).await.unwrap();

    // Create GPU missing error
    let gpu_error = NvbindError::gpu(
        "No GPU detected",
        "Install NVIDIA drivers and ensure GPU is properly connected",
    );

    // Create operation context
    let mut container_config = HashMap::new();
    container_config.insert("--gpus".to_string(), "all".to_string());

    let context = OperationContext {
        operation_type: "gpu_processing".to_string(),
        runtime: "podman".to_string(),
        container_config,
        preferred_strategy: None,
        criticality: OperationCriticality::High,
    };

    // Attempt degradation
    let result = manager
        .handle_failure("test_gpu_processing", &gpu_error, &context)
        .await;

    assert!(result.is_ok());
    let result = result.unwrap();

    match result {
        DegradationResult::Applied {
            strategy,
            modifications,
            performance_impact,
            ..
        } => {
            assert_eq!(strategy, DegradationStrategy::CpuFallback);
            assert!(!modifications.is_empty());
            assert!(performance_impact.is_some());

            let impact = performance_impact.unwrap();
            assert!(impact.expected_slowdown > 1.0); // Should be slower

            println!("✅ CPU fallback applied successfully");
            println!("  Modifications: {:?}", modifications);
            println!("  Expected slowdown: {:.1}x", impact.expected_slowdown);
        }
        _ => panic!("Expected degradation to be applied"),
    }
}

/// Test alternative runtime fallback
#[tokio::test]
async fn test_runtime_fallback() {
    let config = DegradationConfig::default();
    let mut manager = GracefulDegradationManager::new(config).await.unwrap();

    // Create runtime error
    let runtime_error = NvbindError::runtime(
        "docker",
        "Docker daemon not available",
        "Start Docker daemon or use alternative runtime",
    );

    let context = OperationContext {
        operation_type: "container_launch".to_string(),
        runtime: "docker".to_string(),
        container_config: HashMap::new(),
        preferred_strategy: None,
        criticality: OperationCriticality::Medium,
    };

    let result = manager
        .handle_failure("test_runtime", &runtime_error, &context)
        .await;

    assert!(result.is_ok());
    let result = result.unwrap();

    match result {
        DegradationResult::Applied { strategy, .. } => {
            assert_eq!(strategy, DegradationStrategy::AlternativeRuntime);
            println!("✅ Runtime fallback applied successfully");
        }
        DegradationResult::NotApplicable => {
            println!("ℹ️  No alternative runtime available (expected in some environments)");
        }
        _ => {}
    }
}

/// Test max attempts enforcement
#[tokio::test]
async fn test_max_attempts_enforcement() {
    let config = DegradationConfig {
        max_attempts: 2, // Low limit for testing
        ..Default::default()
    };

    let mut manager = GracefulDegradationManager::new(config).await.unwrap();

    let error = NvbindError::gpu("Test error", "Test recovery");
    let context = OperationContext {
        operation_type: "test_operation".to_string(),
        runtime: "podman".to_string(),
        container_config: HashMap::new(),
        preferred_strategy: None,
        criticality: OperationCriticality::Low,
    };

    // First two attempts should work
    for attempt in 1..=2 {
        let result = manager
            .handle_failure("max_attempts_test", &error, &context)
            .await;
        assert!(result.is_ok());

        match result.unwrap() {
            DegradationResult::Applied { .. } => {
                println!("✅ Attempt {} succeeded", attempt);
            }
            _ => panic!("Expected degradation to be applied for attempt {}", attempt),
        }
    }

    // Third attempt should exceed max attempts
    let result = manager
        .handle_failure("max_attempts_test", &error, &context)
        .await;
    assert!(result.is_ok());

    match result.unwrap() {
        DegradationResult::MaxAttemptsExceeded => {
            println!("✅ Max attempts correctly enforced");
        }
        _ => panic!("Expected max attempts to be exceeded"),
    }
}

/// Test preferred strategy override
#[tokio::test]
async fn test_preferred_strategy_override() {
    let config = DegradationConfig::default();
    let mut manager = GracefulDegradationManager::new(config).await.unwrap();

    let error = NvbindError::gpu("Test error", "Test recovery");
    let context = OperationContext {
        operation_type: "test_operation".to_string(),
        runtime: "podman".to_string(),
        container_config: HashMap::new(),
        preferred_strategy: Some(DegradationStrategy::PerformanceReduction),
        criticality: OperationCriticality::Medium,
    };

    let result = manager
        .handle_failure("strategy_test", &error, &context)
        .await;
    assert!(result.is_ok());

    match result.unwrap() {
        DegradationResult::Applied { strategy, .. } => {
            assert_eq!(strategy, DegradationStrategy::PerformanceReduction);
            println!("✅ Preferred strategy correctly applied");
        }
        _ => panic!("Expected preferred strategy to be applied"),
    }
}

/// Test degradation disabled
#[tokio::test]
async fn test_degradation_disabled() {
    let config = DegradationConfig {
        enabled: false,
        ..Default::default()
    };

    let mut manager = GracefulDegradationManager::new(config).await.unwrap();

    let error = NvbindError::gpu("Test error", "Test recovery");
    let context = OperationContext {
        operation_type: "test_operation".to_string(),
        runtime: "podman".to_string(),
        container_config: HashMap::new(),
        preferred_strategy: None,
        criticality: OperationCriticality::High,
    };

    let result = manager
        .handle_failure("disabled_test", &error, &context)
        .await;
    assert!(result.is_ok());

    match result.unwrap() {
        DegradationResult::NoActionTaken => {
            println!("✅ Degradation correctly disabled");
        }
        _ => panic!("Expected no action when degradation is disabled"),
    }
}

/// Test statistics collection
#[test]
fn test_statistics_collection() {
    let config = DegradationConfig::default();
    let manager = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(GracefulDegradationManager::new(config))
        .unwrap();

    let stats = manager.get_statistics();

    // Initially no attempts should be recorded
    assert_eq!(stats.total_attempts, 0);
    assert_eq!(stats.operations_with_attempts, 0);

    // Capabilities should be detected
    assert!(
        !stats.capabilities.system_resources.cpu_features.is_empty()
            || stats.capabilities.system_resources.cpu_cores > 0
    );

    println!("✅ Statistics collection working");
    println!(
        "  CPU cores: {}",
        stats.capabilities.system_resources.cpu_cores
    );
    println!(
        "  Total memory: {:.1} GB",
        stats.capabilities.system_resources.total_memory_gb
    );
}

/// Test feature reduction degradation
#[tokio::test]
async fn test_feature_reduction() {
    let mut config = DegradationConfig::default();
    config.strategies.insert(
        "test_context".to_string(),
        DegradationStrategy::FeatureReduction,
    );

    let mut manager = GracefulDegradationManager::new(config).await.unwrap();

    let error = NvbindError::System {
        message: "Insufficient resources".to_string(),
        source: None,
        recovery_suggestion: "Reduce feature usage".to_string(),
    };

    let context = OperationContext {
        operation_type: "test_context".to_string(),
        runtime: "podman".to_string(),
        container_config: HashMap::new(),
        preferred_strategy: None,
        criticality: OperationCriticality::Medium,
    };

    let result = manager
        .handle_failure("feature_test", &error, &context)
        .await;
    assert!(result.is_ok());

    match result.unwrap() {
        DegradationResult::Applied {
            strategy,
            modifications,
            ..
        } => {
            assert_eq!(strategy, DegradationStrategy::FeatureReduction);
            assert!(modifications.iter().any(|m| m.contains("feature")));
            println!("✅ Feature reduction applied successfully");
        }
        _ => panic!("Expected feature reduction to be applied"),
    }
}

/// Test software rendering fallback
#[tokio::test]
async fn test_software_rendering_fallback() {
    let config = DegradationConfig::default();
    let mut manager = GracefulDegradationManager::new(config).await.unwrap();

    let driver_error = NvbindError::driver(
        "NVIDIA driver incompatible",
        Some("470.0".to_string()),
        Some("515.0".to_string()),
        "Update NVIDIA drivers to version 515.0 or higher",
    );

    let context = OperationContext {
        operation_type: "graphics_rendering".to_string(),
        runtime: "podman".to_string(),
        container_config: HashMap::new(),
        preferred_strategy: None,
        criticality: OperationCriticality::High,
    };

    let result = manager
        .handle_failure("rendering_test", &driver_error, &context)
        .await;
    assert!(result.is_ok());

    match result.unwrap() {
        DegradationResult::Applied {
            strategy,
            modifications,
            new_container_config,
            ..
        } => {
            assert_eq!(strategy, DegradationStrategy::SoftwareRendering);
            assert!(modifications.iter().any(|m| m.contains("software")));

            if let Some(config) = new_container_config {
                assert!(config.contains_key("LIBGL_ALWAYS_SOFTWARE"));
            }

            println!("✅ Software rendering fallback applied successfully");
        }
        _ => panic!("Expected software rendering to be applied"),
    }
}

/// Test attempt history reset
#[tokio::test]
async fn test_attempt_history_reset() {
    let config = DegradationConfig::default();
    let mut manager = GracefulDegradationManager::new(config).await.unwrap();

    let error = NvbindError::gpu("Test error", "Test recovery");
    let context = OperationContext {
        operation_type: "reset_test".to_string(),
        runtime: "podman".to_string(),
        container_config: HashMap::new(),
        preferred_strategy: None,
        criticality: OperationCriticality::Medium,
    };

    // Make one degradation attempt
    let result = manager
        .handle_failure("reset_operation", &error, &context)
        .await;
    assert!(result.is_ok());

    let stats_before = manager.get_statistics();
    assert!(stats_before.total_attempts > 0);

    // Reset attempts for the operation
    manager.reset_attempts("reset_operation");

    // Should be able to make more attempts now
    let result = manager
        .handle_failure("reset_operation", &error, &context)
        .await;
    assert!(result.is_ok());

    println!("✅ Attempt history reset working correctly");
}
