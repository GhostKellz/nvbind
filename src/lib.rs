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

#[cfg(feature = "bolt")]
pub mod bolt;
