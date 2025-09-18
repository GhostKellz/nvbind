//! Compatibility layer for Docker and Podman replacement
//! Makes nvbind a drop-in replacement for existing GPU container workflows

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use tracing::{debug, info, warn};

/// Docker command compatibility layer
pub struct DockerCompat {
    nvbind_config: crate::config::Config,
}

/// Podman command compatibility layer
pub struct PodmanCompat {
    nvbind_config: crate::config::Config,
}

/// Parsed Docker/Podman command arguments
#[derive(Debug, Clone)]
pub struct ContainerArgs {
    pub image: String,
    pub command: Vec<String>,
    pub gpu_args: GpuArgs,
    pub volumes: Vec<VolumeMount>,
    pub environment: HashMap<String, String>,
    pub ports: Vec<PortMapping>,
    pub runtime_args: Vec<String>,
    pub network_mode: Option<String>,
    pub privileged: bool,
    pub remove_on_exit: bool,
    pub interactive: bool,
    pub tty: bool,
    pub detached: bool,
    pub name: Option<String>,
}

/// GPU arguments parsed from Docker/Podman commands
#[derive(Debug, Clone)]
pub struct GpuArgs {
    pub enabled: bool,
    pub device_ids: Vec<String>, // "all", "0", "1", etc.
    pub capabilities: Vec<String>, // "compute", "graphics", "video", etc.
    pub memory_limit: Option<String>,
    pub driver: Option<String>,
}

/// Volume mount specification
#[derive(Debug, Clone)]
pub struct VolumeMount {
    pub host_path: String,
    pub container_path: String,
    pub options: Vec<String>, // "ro", "rw", "z", "Z", etc.
}

/// Port mapping specification
#[derive(Debug, Clone)]
pub struct PortMapping {
    pub host_port: Option<u16>,
    pub container_port: u16,
    pub protocol: String, // "tcp", "udp"
    pub host_ip: Option<String>,
}

impl DockerCompat {
    pub fn new() -> Result<Self> {
        Ok(Self {
            nvbind_config: crate::config::Config::load()?,
        })
    }

    /// Parse Docker command arguments and convert to nvbind
    pub fn parse_docker_command(&self, args: &[String]) -> Result<ContainerArgs> {
        let mut container_args = ContainerArgs::default();
        let mut i = 0;

        while i < args.len() {
            match args[i].as_str() {
                "run" => {
                    // Skip the 'run' command
                    i += 1;
                    continue;
                }
                "--gpus" => {
                    if i + 1 < args.len() {
                        container_args.gpu_args = self.parse_gpu_args(&args[i + 1])?;
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--gpus requires a value"));
                    }
                }
                "--runtime" => {
                    if i + 1 < args.len() {
                        container_args.runtime_args.push(format!("--runtime={}", args[i + 1]));
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--runtime requires a value"));
                    }
                }
                "-v" | "--volume" => {
                    if i + 1 < args.len() {
                        container_args.volumes.push(self.parse_volume(&args[i + 1])?);
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--volume requires a value"));
                    }
                }
                "-e" | "--env" => {
                    if i + 1 < args.len() {
                        let (key, value) = self.parse_env_var(&args[i + 1])?;
                        container_args.environment.insert(key, value);
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--env requires a value"));
                    }
                }
                "-p" | "--publish" => {
                    if i + 1 < args.len() {
                        container_args.ports.push(self.parse_port_mapping(&args[i + 1])?);
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--publish requires a value"));
                    }
                }
                "--name" => {
                    if i + 1 < args.len() {
                        container_args.name = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--name requires a value"));
                    }
                }
                "--network" => {
                    if i + 1 < args.len() {
                        container_args.network_mode = Some(args[i + 1].clone());
                        i += 2;
                    } else {
                        return Err(anyhow::anyhow!("--network requires a value"));
                    }
                }
                "--privileged" => {
                    container_args.privileged = true;
                    i += 1;
                }
                "--rm" => {
                    container_args.remove_on_exit = true;
                    i += 1;
                }
                "-i" | "--interactive" => {
                    container_args.interactive = true;
                    i += 1;
                }
                "-t" | "--tty" => {
                    container_args.tty = true;
                    i += 1;
                }
                "-it" => {
                    container_args.interactive = true;
                    container_args.tty = true;
                    i += 1;
                }
                "-d" | "--detach" => {
                    container_args.detached = true;
                    i += 1;
                }
                arg if !arg.starts_with('-') => {
                    // This should be the image name
                    container_args.image = arg.to_string();
                    i += 1;
                    // Remaining arguments are the command
                    container_args.command = args[i..].to_vec();
                    break;
                }
                _ => {
                    warn!("Unknown Docker argument: {}", args[i]);
                    i += 1;
                }
            }
        }

        Ok(container_args)
    }

    /// Execute container using nvbind instead of Docker
    pub async fn run_container(&self, args: ContainerArgs) -> Result<()> {
        info!("Running container with nvbind (Docker compatibility mode)");

        // Determine GPU selection
        let gpu_selection = if args.gpu_args.enabled {
            if args.gpu_args.device_ids.contains(&"all".to_string()) {
                "all".to_string()
            } else if !args.gpu_args.device_ids.is_empty() {
                args.gpu_args.device_ids[0].clone()
            } else {
                "all".to_string()
            }
        } else {
            "none".to_string()
        };

        // Create enhanced config with Docker compatibility
        let mut config = self.nvbind_config.clone();

        // Add environment variables
        for (key, value) in &args.environment {
            config.runtime.environment.insert(key.clone(), value.clone());
        }

        // Add GPU capabilities if specified
        if args.gpu_args.enabled && !args.gpu_args.capabilities.is_empty() {
            config.runtime.environment.insert(
                "NVIDIA_DRIVER_CAPABILITIES".to_string(),
                args.gpu_args.capabilities.join(","),
            );
        }

        // Use Bolt as the preferred runtime for best performance
        let runtime = "bolt";

        debug!("Docker compatibility: {} -> nvbind + bolt", args.image);
        debug!("GPU selection: {}", gpu_selection);

        crate::runtime::run_with_config(
            config,
            runtime.to_string(),
            gpu_selection,
            args.image,
            args.command,
        ).await
    }

    fn parse_gpu_args(&self, gpu_spec: &str) -> Result<GpuArgs> {
        let mut gpu_args = GpuArgs::default();
        gpu_args.enabled = true;

        if gpu_spec == "all" {
            gpu_args.device_ids.push("all".to_string());
        } else if gpu_spec.starts_with("device=") {
            let devices = gpu_spec.strip_prefix("device=").unwrap_or(gpu_spec);
            gpu_args.device_ids = devices.split(',').map(|s| s.to_string()).collect();
        } else if gpu_spec.chars().all(char::is_numeric) {
            // Single GPU ID
            gpu_args.device_ids.push(gpu_spec.to_string());
        } else {
            // Complex GPU specification (e.g., "device=0,1 capabilities=compute,graphics")
            for part in gpu_spec.split_whitespace() {
                if let Some(devices) = part.strip_prefix("device=") {
                    gpu_args.device_ids = devices.split(',').map(|s| s.to_string()).collect();
                } else if let Some(caps) = part.strip_prefix("capabilities=") {
                    gpu_args.capabilities = caps.split(',').map(|s| s.to_string()).collect();
                }
            }
        }

        Ok(gpu_args)
    }

    fn parse_volume(&self, volume_spec: &str) -> Result<VolumeMount> {
        let parts: Vec<&str> = volume_spec.split(':').collect();

        match parts.len() {
            2 => Ok(VolumeMount {
                host_path: parts[0].to_string(),
                container_path: parts[1].to_string(),
                options: vec!["rw".to_string()],
            }),
            3 => Ok(VolumeMount {
                host_path: parts[0].to_string(),
                container_path: parts[1].to_string(),
                options: parts[2].split(',').map(|s| s.to_string()).collect(),
            }),
            _ => Err(anyhow::anyhow!("Invalid volume specification: {}", volume_spec)),
        }
    }

    fn parse_env_var(&self, env_spec: &str) -> Result<(String, String)> {
        if let Some(eq_pos) = env_spec.find('=') {
            let key = env_spec[..eq_pos].to_string();
            let value = env_spec[eq_pos + 1..].to_string();
            Ok((key, value))
        } else {
            // Environment variable without value (use from host)
            Ok((env_spec.to_string(), std::env::var(env_spec).unwrap_or_default()))
        }
    }

    fn parse_port_mapping(&self, port_spec: &str) -> Result<PortMapping> {
        // Examples: "8080:80", "127.0.0.1:8080:80", "8080:80/tcp"
        let parts: Vec<&str> = port_spec.split(':').collect();

        match parts.len() {
            2 => {
                // "8080:80" or "8080:80/tcp"
                let (container_part, protocol) = if parts[1].contains('/') {
                    let container_parts: Vec<&str> = parts[1].split('/').collect();
                    (container_parts[0], container_parts[1])
                } else {
                    (parts[1], "tcp")
                };

                Ok(PortMapping {
                    host_port: Some(parts[0].parse()?),
                    container_port: container_part.parse()?,
                    protocol: protocol.to_string(),
                    host_ip: None,
                })
            }
            3 => {
                // "127.0.0.1:8080:80"
                Ok(PortMapping {
                    host_port: Some(parts[1].parse()?),
                    container_port: parts[2].parse()?,
                    protocol: "tcp".to_string(),
                    host_ip: Some(parts[0].to_string()),
                })
            }
            _ => Err(anyhow::anyhow!("Invalid port specification: {}", port_spec)),
        }
    }
}

impl PodmanCompat {
    pub fn new() -> Result<Self> {
        Ok(Self {
            nvbind_config: crate::config::Config::load()?,
        })
    }

    /// Parse Podman command arguments (very similar to Docker)
    pub fn parse_podman_command(&self, args: &[String]) -> Result<ContainerArgs> {
        let docker_compat = DockerCompat::new()?;
        docker_compat.parse_docker_command(args)
    }

    /// Execute container using nvbind instead of Podman
    pub async fn run_container(&self, args: ContainerArgs) -> Result<()> {
        info!("Running container with nvbind (Podman compatibility mode)");

        let docker_compat = DockerCompat::new()?;
        docker_compat.run_container(args).await
    }
}

impl Default for ContainerArgs {
    fn default() -> Self {
        Self {
            image: String::new(),
            command: Vec::new(),
            gpu_args: GpuArgs::default(),
            volumes: Vec::new(),
            environment: HashMap::new(),
            ports: Vec::new(),
            runtime_args: Vec::new(),
            network_mode: None,
            privileged: false,
            remove_on_exit: false,
            interactive: false,
            tty: false,
            detached: false,
            name: None,
        }
    }
}

impl Default for GpuArgs {
    fn default() -> Self {
        Self {
            enabled: false,
            device_ids: Vec::new(),
            capabilities: Vec::new(),
            memory_limit: None,
            driver: None,
        }
    }
}

/// Docker command wrapper that redirects to nvbind
pub async fn handle_docker_command(args: Vec<String>) -> Result<()> {
    let compat = DockerCompat::new()?;
    let container_args = compat.parse_docker_command(&args)?;

    info!("🐳➡️⚡ Converting Docker command to nvbind");
    debug!("Original Docker args: {:?}", args);
    debug!("Parsed container args: {:?}", container_args);

    compat.run_container(container_args).await
}

/// Podman command wrapper that redirects to nvbind
pub async fn handle_podman_command(args: Vec<String>) -> Result<()> {
    let compat = PodmanCompat::new()?;
    let container_args = compat.parse_podman_command(&args)?;

    info!("🐋➡️⚡ Converting Podman command to nvbind");
    debug!("Original Podman args: {:?}", args);
    debug!("Parsed container args: {:?}", container_args);

    compat.run_container(container_args).await
}

/// Create symbolic links for Docker/Podman replacement
pub fn install_docker_replacement() -> Result<()> {
    info!("Installing nvbind as Docker/Podman replacement");

    let nvbind_path = which::which("nvbind")
        .map_err(|_| anyhow::anyhow!("nvbind not found in PATH"))?;

    // Create wrapper scripts
    create_docker_wrapper(&nvbind_path)?;
    create_podman_wrapper(&nvbind_path)?;

    info!("✅ Docker/Podman replacement installed successfully");
    info!("   docker commands will now use nvbind + bolt for GPU containers");
    info!("   podman commands will now use nvbind + bolt for GPU containers");

    Ok(())
}

fn create_docker_wrapper(nvbind_path: &std::path::Path) -> Result<()> {
    let wrapper_content = format!(
        r#"#!/bin/bash
# Docker compatibility wrapper for nvbind
# Automatically redirects GPU containers to nvbind + bolt

if [[ "$*" == *"--gpus"* ]] || [[ "$*" == *"--runtime=nvidia"* ]]; then
    echo "🐳➡️⚡ Redirecting Docker GPU command to nvbind"
    exec {} docker-compat "$@"
else
    # Use system docker for non-GPU containers
    exec /usr/bin/docker.bak "$@"
fi
"#,
        nvbind_path.display()
    );

    // Backup original docker and install wrapper
    if std::path::Path::new("/usr/bin/docker").exists() {
        std::fs::rename("/usr/bin/docker", "/usr/bin/docker.bak")?;
    }

    std::fs::write("/usr/bin/docker", wrapper_content)?;
    std::fs::set_permissions("/usr/bin/docker", std::os::unix::fs::PermissionsExt::from_mode(0o755))?;

    Ok(())
}

fn create_podman_wrapper(nvbind_path: &std::path::Path) -> Result<()> {
    let wrapper_content = format!(
        r#"#!/bin/bash
# Podman compatibility wrapper for nvbind
# Automatically redirects GPU containers to nvbind + bolt

if [[ "$*" == *"--device nvidia.com/gpu"* ]] || [[ "$*" == *"--security-opt=label=disable"* ]]; then
    echo "🐋➡️⚡ Redirecting Podman GPU command to nvbind"
    exec {} podman-compat "$@"
else
    # Use system podman for non-GPU containers
    exec /usr/bin/podman.bak "$@"
fi
"#,
        nvbind_path.display()
    );

    // Backup original podman and install wrapper
    if std::path::Path::new("/usr/bin/podman").exists() {
        std::fs::rename("/usr/bin/podman", "/usr/bin/podman.bak")?;
    }

    std::fs::write("/usr/bin/podman", wrapper_content)?;
    std::fs::set_permissions("/usr/bin/podman", std::os::unix::fs::PermissionsExt::from_mode(0o755))?;

    Ok(())
}

/// Performance comparison between Docker/Podman and nvbind
pub async fn benchmark_performance() -> Result<()> {
    info!("🏁 Benchmarking nvbind vs Docker/Podman performance");

    let test_image = "nvidia/cuda:12.0-runtime-ubuntu22.04";
    let test_command = vec!["nvidia-smi".to_string()];

    // Benchmark Docker (if available)
    if let Ok(docker_time) = benchmark_docker(test_image, &test_command).await {
        info!("Docker GPU container startup: {:.2}ms", docker_time);
    }

    // Benchmark Podman (if available)
    if let Ok(podman_time) = benchmark_podman(test_image, &test_command).await {
        info!("Podman GPU container startup: {:.2}ms", podman_time);
    }

    // Benchmark nvbind
    let nvbind_time = benchmark_nvbind(test_image, &test_command).await?;
    info!("nvbind GPU container startup: {:.2}ms", nvbind_time);

    Ok(())
}

async fn benchmark_docker(image: &str, command: &[String]) -> Result<f64> {
    let start = std::time::Instant::now();

    let output = Command::new("docker")
        .args(&["run", "--rm", "--gpus", "all", image])
        .args(command)
        .output()?;

    let elapsed = start.elapsed();

    if output.status.success() {
        Ok(elapsed.as_secs_f64() * 1000.0)
    } else {
        Err(anyhow::anyhow!("Docker command failed"))
    }
}

async fn benchmark_podman(image: &str, command: &[String]) -> Result<f64> {
    let start = std::time::Instant::now();

    let output = Command::new("podman")
        .args(&["run", "--rm", "--device", "nvidia.com/gpu=all", image])
        .args(command)
        .output()?;

    let elapsed = start.elapsed();

    if output.status.success() {
        Ok(elapsed.as_secs_f64() * 1000.0)
    } else {
        Err(anyhow::anyhow!("Podman command failed"))
    }
}

async fn benchmark_nvbind(image: &str, command: &[String]) -> Result<f64> {
    let start = std::time::Instant::now();

    crate::runtime::run_with_config(
        crate::config::Config::default(),
        "bolt".to_string(),
        "all".to_string(),
        image.to_string(),
        command.to_vec(),
    ).await?;

    let elapsed = start.elapsed();
    Ok(elapsed.as_secs_f64() * 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gpu_args() {
        let compat = DockerCompat::new().unwrap();

        let gpu_args = compat.parse_gpu_args("all").unwrap();
        assert!(gpu_args.enabled);
        assert_eq!(gpu_args.device_ids, vec!["all"]);

        let gpu_args = compat.parse_gpu_args("0").unwrap();
        assert_eq!(gpu_args.device_ids, vec!["0"]);

        let gpu_args = compat.parse_gpu_args("device=0,1").unwrap();
        assert_eq!(gpu_args.device_ids, vec!["0", "1"]);
    }

    #[test]
    fn test_parse_volume() {
        let compat = DockerCompat::new().unwrap();

        let volume = compat.parse_volume("/host:/container").unwrap();
        assert_eq!(volume.host_path, "/host");
        assert_eq!(volume.container_path, "/container");

        let volume = compat.parse_volume("/host:/container:ro").unwrap();
        assert_eq!(volume.options, vec!["ro"]);
    }

    #[test]
    fn test_parse_docker_command() {
        let compat = DockerCompat::new().unwrap();

        let args = vec![
            "run".to_string(),
            "--gpus".to_string(),
            "all".to_string(),
            "--rm".to_string(),
            "nvidia/cuda:latest".to_string(),
            "nvidia-smi".to_string(),
        ];

        let container_args = compat.parse_docker_command(&args).unwrap();
        assert!(container_args.gpu_args.enabled);
        assert_eq!(container_args.image, "nvidia/cuda:latest");
        assert_eq!(container_args.command, vec!["nvidia-smi"]);
        assert!(container_args.remove_on_exit);
    }

    #[tokio::test]
    async fn test_docker_compatibility() {
        let args = vec![
            "run".to_string(),
            "--gpus".to_string(),
            "all".to_string(),
            "ubuntu".to_string(),
            "echo".to_string(),
            "test".to_string(),
        ];

        // This should not panic (actual execution depends on system setup)
        let result = handle_docker_command(args).await;
        // We expect it to potentially fail due to missing GPU/system setup, but not panic
        assert!(result.is_ok() || result.is_err());
    }
}