//! Plugin architecture for extensible runtime adapters
//!
//! This module provides a plugin system that allows nvbind to integrate with different
//! container runtimes (Docker, Podman, Bolt) through a unified interface.

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

/// Runtime adapter trait that all container runtime plugins must implement
#[async_trait]
pub trait RuntimeAdapter: Send + Sync {
    /// Unique identifier for this runtime adapter
    fn runtime_name(&self) -> &str;

    /// Initialize the runtime adapter with configuration
    async fn initialize(&mut self, config: RuntimeConfig) -> Result<()>;

    /// Validate that the runtime is available and functional
    async fn validate_runtime(&self) -> Result<RuntimeInfo>;
}

/// Container specification for runtime adapters
#[derive(Debug, Clone)]
pub struct ContainerSpec {
    pub image: String,
    pub name: Option<String>,
    pub command: Vec<String>,
    pub environment: HashMap<String, String>,
}

/// Runtime configuration for adapters
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub default_args: Vec<String>,
    pub environment: HashMap<String, String>,
    pub security_opts: Vec<String>,
}

/// Runtime information and status
#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    pub name: String,
    pub version: String,
    pub available: bool,
    pub gpu_support: bool,
}

/// Plugin registry for managing runtime adapters
pub struct PluginRegistry {
    adapters: HashMap<String, Arc<dyn RuntimeAdapter>>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }

    /// Register a runtime adapter plugin
    pub fn register_adapter(&mut self, adapter: Arc<dyn RuntimeAdapter>) {
        let name = adapter.runtime_name().to_string();
        info!("Registering runtime adapter: {}", name);
        self.adapters.insert(name, adapter);
    }

    /// Get a runtime adapter by name
    pub fn get_adapter(&self, name: &str) -> Option<Arc<dyn RuntimeAdapter>> {
        self.adapters.get(name).cloned()
    }

    /// List all registered adapters
    pub fn list_adapters(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Bolt runtime adapter implementation
#[cfg(feature = "bolt")]
pub mod bolt_adapter {
    use super::*;
    use crate::config::BoltConfig;

    pub struct BoltRuntimeAdapter {
        config: Option<RuntimeConfig>,
        bolt_config: BoltConfig,
    }

    impl BoltRuntimeAdapter {
        pub fn new(bolt_config: BoltConfig) -> Self {
            Self {
                config: None,
                bolt_config,
            }
        }
    }

    #[async_trait]
    impl RuntimeAdapter for BoltRuntimeAdapter {
        fn runtime_name(&self) -> &str {
            "bolt"
        }

        async fn initialize(&mut self, config: RuntimeConfig) -> Result<()> {
            info!("Initializing Bolt runtime adapter");
            self.config = Some(config);
            Ok(())
        }

        async fn validate_runtime(&self) -> Result<RuntimeInfo> {
            let validation = crate::runtime::validate_runtime("bolt");

            Ok(RuntimeInfo {
                name: "bolt".to_string(),
                version: "unknown".to_string(),
                available: validation.is_ok(),
                gpu_support: true,
            })
        }
    }
}

/// Docker runtime adapter implementation
pub mod docker_adapter {
    use super::*;

    pub struct DockerRuntimeAdapter {
        config: Option<RuntimeConfig>,
    }

    impl DockerRuntimeAdapter {
        pub fn new() -> Self {
            Self { config: None }
        }
    }

    impl Default for DockerRuntimeAdapter {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl RuntimeAdapter for DockerRuntimeAdapter {
        fn runtime_name(&self) -> &str {
            "docker"
        }

        async fn initialize(&mut self, config: RuntimeConfig) -> Result<()> {
            info!("Initializing Docker runtime adapter");
            self.config = Some(config);
            Ok(())
        }

        async fn validate_runtime(&self) -> Result<RuntimeInfo> {
            let validation = crate::runtime::validate_runtime("docker");

            Ok(RuntimeInfo {
                name: "docker".to_string(),
                version: "unknown".to_string(),
                available: validation.is_ok(),
                gpu_support: true,
            })
        }
    }
}

/// Create a default plugin registry with all available adapters
pub fn create_default_registry() -> PluginRegistry {
    let mut registry = PluginRegistry::new();

    // Register Docker adapter
    registry.register_adapter(Arc::new(docker_adapter::DockerRuntimeAdapter::new()));

    // Register Bolt adapter if feature is enabled
    #[cfg(feature = "bolt")]
    {
        let bolt_config = crate::config::BoltConfig::default();
        registry.register_adapter(Arc::new(bolt_adapter::BoltRuntimeAdapter::new(bolt_config)));
    }

    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_plugin_registry() {
        let registry = create_default_registry();
        let adapters = registry.list_adapters();

        assert!(!adapters.is_empty());
        assert!(adapters.contains(&"docker".to_string()));

        #[cfg(feature = "bolt")]
        assert!(adapters.contains(&"bolt".to_string()));
    }

    #[tokio::test]
    async fn test_docker_adapter() {
        let mut adapter = docker_adapter::DockerRuntimeAdapter::new();

        let config = RuntimeConfig {
            default_args: vec![],
            environment: HashMap::new(),
            security_opts: vec![],
        };

        adapter.initialize(config).await.unwrap();
    }
}
