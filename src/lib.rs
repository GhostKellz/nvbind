// Library interface for nvbind
// This allows the modules to be used as a library and enables benchmarking

pub mod cdi;
pub mod compat;
pub mod config;
pub mod distro;
pub mod gaming;
pub mod gpu;
pub mod isolation;
pub mod ollama;
pub mod runtime;
pub mod wsl2;
pub mod plugin;
pub mod snapshot;
pub mod metrics;
pub mod wine;
pub mod error;
pub mod graceful_degradation;
pub mod rbac;
pub mod monitoring;
pub mod config_validation;
pub mod k8s;
pub mod performance;
pub mod gpu_advanced;
pub mod docs;
pub mod ha;
pub mod security;
pub mod observability;
pub mod cloud;
pub mod mesh;

#[cfg(feature = "bolt")]
pub mod bolt;
