// Library interface for nvbind
// This allows the modules to be used as a library and enables benchmarking

pub mod cdi;
pub mod cloud;
pub mod compat;
pub mod config;
pub mod config_validation;
pub mod distro;
pub mod docs;
pub mod error;
pub mod gaming;
pub mod gpu;
pub mod gpu_advanced;
pub mod graceful_degradation;
pub mod ha;
pub mod isolation;
pub mod k8s;
pub mod mesh;
pub mod metrics;
pub mod monitoring;
pub mod observability;
pub mod ollama;
pub mod performance;
pub mod plugin;
pub mod rbac;
pub mod runtime;
pub mod security;
pub mod snapshot;
pub mod user_error;
pub mod wine;
pub mod wsl2;

#[cfg(feature = "bolt")]
pub mod bolt;
