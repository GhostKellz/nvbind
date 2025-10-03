//! Library interface for nvbind
//! This allows the modules to be used as a library and enables benchmarking

#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::field_reassign_with_default)]
#![allow(unused_imports)]

pub mod cdi;
pub mod cloud;
pub mod compat;
pub mod config;
pub mod config_validation;
#[cfg(feature = "experimental-distributed")]
pub mod distributed_training;
pub mod distro;
pub mod docs;
pub mod error;
pub mod gaming;
pub mod gaming_optimization;
pub mod gpu;
pub mod gpu_advanced;
#[cfg(feature = "experimental-scheduling")]
pub mod gpu_scheduling_optimization;
pub mod graceful_degradation;
pub mod ha;
pub mod isolation;
#[cfg(feature = "experimental-k8s")]
pub mod k8s;
#[cfg(feature = "experimental-k8s")]
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
#[cfg(feature = "ml-optimizations")]
pub mod pytorch_optimization;
#[cfg(feature = "experimental-raytracing")]
pub mod raytracing_acceleration;
pub mod rbac;
pub mod runtime;
pub mod security;
pub mod security_audit;
pub mod snapshot;
#[cfg(feature = "ml-optimizations")]
pub mod tensorflow_optimization;
pub mod user_error;
pub mod wine;
pub mod wsl2;

#[cfg(feature = "bolt")]
pub mod bolt;

// GhostForge integration
pub mod ghostforge_api;
pub mod gaming_profiles;
