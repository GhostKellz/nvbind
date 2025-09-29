// Library interface for nvbind
// This allows the modules to be used as a library and enables benchmarking

pub mod cdi;
pub mod cloud;
pub mod compat;
pub mod config;
pub mod config_validation;
pub mod distro;
pub mod distributed_training;
pub mod docs;
pub mod error;
pub mod gaming;
pub mod gaming_optimization;
pub mod gpu;
pub mod gpu_advanced;
pub mod gpu_scheduling_optimization;
pub mod graceful_degradation;
pub mod ha;
pub mod isolation;
pub mod k8s;
pub mod kubernetes_device_plugin;
pub mod mesh;
pub mod metrics;
pub mod mlflow_integration;
pub mod monitoring;
pub mod observability;
pub mod ollama;
pub mod performance;
pub mod performance_optimization;
pub mod plugin;
pub mod pytorch_optimization;
pub mod raytracing_acceleration;
pub mod rbac;
pub mod runtime;
pub mod security;
pub mod security_audit;
pub mod snapshot;
pub mod tensorflow_optimization;
pub mod user_error;
pub mod wine;
pub mod wsl2;

#[cfg(feature = "bolt")]
pub mod bolt;
