use crate::config::Config;
use crate::gpu::check_nvidia_requirements;
use anyhow::{Context, Result};
use std::process::Command;
use tracing::{debug, info};

/// Run container with CDI devices
pub async fn run_with_cdi_devices(
    config: Config,
    runtime: String,
    gpu: String,
    image: String,
    args: Vec<String>,
) -> Result<()> {
    info!("Setting up container with CDI devices");

    // Check NVIDIA requirements
    check_nvidia_requirements()?;

    // Validate runtime exists
    validate_runtime(&runtime)?;

    // Load CDI devices
    let mut cdi_registry = crate::cdi::CdiRegistry::new();
    cdi_registry.load_specs()?;

    // Determine CDI device names based on GPU selection
    let cdi_devices = match gpu.as_str() {
        "all" => vec!["nvidia.com/gpu=all".to_string()],
        "none" => vec![],
        gpu_id => {
            // Try to find specific GPU device
            let device_name = format!("nvidia.com/gpu=gpu{}", gpu_id);
            if cdi_registry.get_device(&device_name).is_some() {
                vec![device_name]
            } else {
                return Err(anyhow::anyhow!("CDI device not found: {}", device_name));
            }
        }
    };

    match runtime.as_str() {
        "podman" => run_podman_with_cdi(&config, image, args, cdi_devices).await,
        "docker" => run_docker_with_cdi(&config, image, args, cdi_devices).await,
        "bolt" => run_bolt_with_cdi(&config, image, args, cdi_devices).await,
        _ => Err(anyhow::anyhow!("Unsupported runtime: {}", runtime)),
    }
}

async fn run_podman_with_cdi(
    config: &Config,
    image: String,
    args: Vec<String>,
    cdi_devices: Vec<String>,
) -> Result<()> {
    let mut cmd = Command::new("podman");
    cmd.arg("run");

    // Add default runtime args from config
    for arg in &config.runtime.default_args {
        cmd.arg(arg);
    }

    // Add CDI devices
    for cdi_device in &cdi_devices {
        cmd.arg("--device").arg(cdi_device);
    }

    // Add security options from config
    for opt in &config.security.security_opts {
        cmd.arg("--security-opt").arg(opt);
    }

    // Add environment variables from config
    for (key, value) in &config.runtime.environment {
        cmd.env(key, value);
    }

    cmd.arg(&image);
    cmd.args(&args);

    info!("Executing: podman run with CDI devices");
    debug!("Command: {:?}", cmd);
    debug!("CDI devices: {:?}", cdi_devices);

    let status = cmd.status().context("Failed to execute podman")?;

    if !status.success() {
        return Err(anyhow::anyhow!("Podman exited with status: {}", status));
    }

    Ok(())
}

async fn run_docker_with_cdi(
    config: &Config,
    image: String,
    args: Vec<String>,
    cdi_devices: Vec<String>,
) -> Result<()> {
    let mut cmd = Command::new("docker");
    cmd.arg("run");

    // Add default runtime args from config
    for arg in &config.runtime.default_args {
        cmd.arg(arg);
    }

    // Add CDI devices
    for cdi_device in &cdi_devices {
        cmd.arg("--device").arg(cdi_device);
    }

    // Add security options from config
    for opt in &config.security.security_opts {
        cmd.arg("--security-opt").arg(opt);
    }

    // Add environment variables from config
    for (key, value) in &config.runtime.environment {
        cmd.env(key, value);
    }

    cmd.arg(&image);
    cmd.args(&args);

    info!("Executing: docker run with CDI devices");
    debug!("Command: {:?}", cmd);
    debug!("CDI devices: {:?}", cdi_devices);

    let status = cmd.status().context("Failed to execute docker")?;

    if !status.success() {
        return Err(anyhow::anyhow!("Docker exited with status: {}", status));
    }

    Ok(())
}

// Legacy function kept for API compatibility
#[allow(dead_code)]
pub async fn run(runtime: String, gpu: String, image: String, args: Vec<String>) -> Result<()> {
    let config = Config::default();
    run_with_config(config, runtime, gpu, image, args).await
}

pub async fn run_with_config(
    config: Config,
    runtime: String,
    gpu: String,
    image: String,
    args: Vec<String>,
) -> Result<()> {
    info!("Setting up GPU passthrough for runtime: {}", runtime);

    // Check NVIDIA requirements
    check_nvidia_requirements()?;

    // Validate runtime exists
    validate_runtime(&runtime)?;

    let devices = config.get_all_devices();
    let libraries = config.get_all_libraries()?;

    debug!("Required devices: {:?}", devices);
    debug!("Required libraries: {} found", libraries.len());

    match runtime.as_str() {
        "podman" => run_podman_with_config(&config, gpu, image, args, devices, libraries).await,
        "docker" => run_docker_with_config(&config, gpu, image, args, devices, libraries).await,
        "bolt" => run_bolt_with_config(&config, gpu, image, args, devices, libraries).await,
        _ => Err(anyhow::anyhow!("Unsupported runtime: {}", runtime)),
    }
}

async fn run_podman_with_config(
    config: &Config,
    gpu: String,
    image: String,
    args: Vec<String>,
    devices: Vec<String>,
    libraries: Vec<String>,
) -> Result<()> {
    let mut cmd = Command::new("podman");
    cmd.arg("run");

    // Add default runtime args from config
    for arg in &config.runtime.default_args {
        cmd.arg(arg);
    }

    // Add device mounts
    for device in &devices {
        cmd.arg("--device").arg(format!("{}:{}", device, device));
    }

    // Add library bind mounts
    for lib in &libraries {
        cmd.arg("-v").arg(format!("{}:{}:ro", lib, lib));
    }

    // Add security options from config
    for opt in &config.security.security_opts {
        cmd.arg("--security-opt").arg(opt);
    }

    // Add environment variables from config
    for (key, value) in &config.runtime.environment {
        cmd.env(key, value);
    }

    // Override with GPU-specific env vars
    cmd.env("NVIDIA_VISIBLE_DEVICES", &gpu);

    cmd.arg(&image);
    cmd.args(&args);

    info!("Executing: podman run with GPU passthrough");
    debug!("Command: {:?}", cmd);

    let status = cmd.status().context("Failed to execute podman")?;

    if !status.success() {
        return Err(anyhow::anyhow!("Podman exited with status: {}", status));
    }

    Ok(())
}

async fn run_docker_with_config(
    config: &Config,
    gpu: String,
    image: String,
    args: Vec<String>,
    devices: Vec<String>,
    libraries: Vec<String>,
) -> Result<()> {
    let mut cmd = Command::new("docker");
    cmd.arg("run");

    // Add default runtime args from config
    for arg in &config.runtime.default_args {
        cmd.arg(arg);
    }

    // Add device mounts
    for device in &devices {
        cmd.arg("--device").arg(format!("{}:{}", device, device));
    }

    // Add library bind mounts
    for lib in &libraries {
        cmd.arg("-v").arg(format!("{}:{}:ro", lib, lib));
    }

    // Add security options from config
    for opt in &config.security.security_opts {
        cmd.arg("--security-opt").arg(opt);
    }

    // Add environment variables from config
    for (key, value) in &config.runtime.environment {
        cmd.env(key, value);
    }

    // Override with GPU-specific env vars
    cmd.env("NVIDIA_VISIBLE_DEVICES", &gpu);

    cmd.arg(&image);
    cmd.args(&args);

    info!("Executing: docker run with GPU passthrough");
    debug!("Command: {:?}", cmd);

    let status = cmd.status().context("Failed to execute docker")?;

    if !status.success() {
        return Err(anyhow::anyhow!("Docker exited with status: {}", status));
    }

    Ok(())
}

async fn run_bolt_with_config(
    config: &Config,
    gpu: String,
    image: String,
    args: Vec<String>,
    devices: Vec<String>,
    libraries: Vec<String>,
) -> Result<()> {
    let mut cmd = Command::new("bolt");
    cmd.arg("surge").arg("run");

    // Add default runtime args from config
    for arg in &config.runtime.default_args {
        cmd.arg(arg);
    }

    // Add device mounts in Bolt format
    for device in &devices {
        cmd.arg("--device").arg(format!("{}:{}", device, device));
    }

    // Add library bind mounts
    for lib in &libraries {
        cmd.arg("--volume").arg(format!("{}:{}:ro", lib, lib));
    }

    // Add security options from config (adapted for Bolt's capsule security model)
    for opt in &config.security.security_opts {
        // Convert Docker/Podman security opts to Bolt capsule format
        if let Some(seccomp_val) = opt.strip_prefix("seccomp=") {
            cmd.arg("--seccomp").arg(seccomp_val);
        } else if let Some(apparmor_val) = opt.strip_prefix("apparmor=") {
            cmd.arg("--apparmor").arg(apparmor_val);
        }
    }

    // Add environment variables from config
    for (key, value) in &config.runtime.environment {
        cmd.arg("--env").arg(format!("{}={}", key, value));
    }

    // Override with GPU-specific env vars
    cmd.arg("--env")
        .arg(format!("NVIDIA_VISIBLE_DEVICES={}", gpu));

    // Add Bolt-specific GPU optimizations
    cmd.arg("--gpu").arg(&gpu);
    cmd.arg("--runtime-optimization").arg("gpu-passthrough");

    cmd.arg(&image);
    cmd.args(&args);

    info!("Executing: bolt surge run with GPU passthrough");
    debug!("Command: {:?}", cmd);

    let status = cmd.status().context("Failed to execute bolt")?;

    if !status.success() {
        return Err(anyhow::anyhow!("Bolt exited with status: {}", status));
    }

    Ok(())
}

async fn run_bolt_with_cdi(
    config: &Config,
    image: String,
    args: Vec<String>,
    cdi_devices: Vec<String>,
) -> Result<()> {
    let mut cmd = Command::new("bolt");
    cmd.arg("surge").arg("run");

    // Add default runtime args from config
    for arg in &config.runtime.default_args {
        cmd.arg(arg);
    }

    // Add CDI devices in Bolt format
    for cdi_device in &cdi_devices {
        cmd.arg("--cdi-device").arg(cdi_device);
    }

    // Add security options from config (adapted for Bolt's capsule security model)
    for opt in &config.security.security_opts {
        if let Some(seccomp_val) = opt.strip_prefix("seccomp=") {
            cmd.arg("--seccomp").arg(seccomp_val);
        } else if let Some(apparmor_val) = opt.strip_prefix("apparmor=") {
            cmd.arg("--apparmor").arg(apparmor_val);
        }
    }

    // Add environment variables from config
    for (key, value) in &config.runtime.environment {
        cmd.arg("--env").arg(format!("{}={}", key, value));
    }

    // Add Bolt-specific optimizations for CDI
    cmd.arg("--capsule-isolation").arg("gpu-exclusive");
    cmd.arg("--runtime-optimization").arg("cdi-passthrough");

    cmd.arg(&image);
    cmd.args(&args);

    info!("Executing: bolt surge run with CDI devices");
    debug!("Command: {:?}", cmd);
    debug!("CDI devices: {:?}", cdi_devices);

    let status = cmd.status().context("Failed to execute bolt")?;

    if !status.success() {
        return Err(anyhow::anyhow!("Bolt exited with status: {}", status));
    }

    Ok(())
}

pub fn validate_runtime(runtime: &str) -> Result<()> {
    let output = match runtime {
        "bolt" => Command::new("bolt")
            .arg("--version")
            .output()
            .context("Failed to check bolt availability")?,
        _ => Command::new(runtime)
            .arg("--version")
            .output()
            .context(format!("Failed to check {} availability", runtime))?,
    };

    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "{} is not available or not working",
            runtime
        ));
    }

    info!("{} runtime validated successfully", runtime);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validate_runtime_success() {
        // Test with a known command that should exist
        let result = validate_runtime("echo");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_runtime_failure() {
        // Test with a command that shouldn't exist
        let result = validate_runtime("nonexistent_command_12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_integration() {
        let config = Config::default();

        // Test runtime command resolution
        assert_eq!(config.get_runtime_command(Some("docker")), "docker");
        assert_eq!(config.get_runtime_command(None), "podman");

        // Test GPU selection resolution
        assert_eq!(config.get_gpu_selection(Some("0")), "0");
        assert_eq!(config.get_gpu_selection(None), "all");

        // Test device and library collection
        let devices = config.get_all_devices();
        assert!(!devices.is_empty());

        let libraries_result = config.get_all_libraries();
        assert!(libraries_result.is_ok());
    }

    #[tokio::test]
    async fn test_run_with_config_validation() {
        let config = Config::default();

        // This will fail due to NVIDIA requirements or runtime validation
        // but we're testing that it fails gracefully
        let result = run_with_config(
            config,
            "nonexistent_runtime".to_string(),
            "all".to_string(),
            "ubuntu".to_string(),
            vec!["echo".to_string()],
        )
        .await;

        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("nonexistent_runtime") || error_msg.contains("NVIDIA"));
    }

    #[test]
    fn test_config_device_library_integration() {
        let mut config = Config::default();

        // Add custom devices and libraries
        config
            .gpu
            .additional_devices
            .push("/dev/test-device".to_string());
        config
            .gpu
            .additional_libraries
            .push("/lib/test-lib.so".to_string());

        // Test that they're included in the results
        let devices = config.get_all_devices();
        assert!(devices.contains(&"/dev/test-device".to_string()));

        let libraries = config.get_all_libraries().unwrap();
        assert!(libraries.contains(&"/lib/test-lib.so".to_string()));
    }

    #[test]
    fn test_environment_variables() {
        let mut config = Config::default();
        config
            .runtime
            .environment
            .insert("TEST_VAR".to_string(), "test_value".to_string());

        // Verify environment variables are properly stored
        assert_eq!(
            config.runtime.environment.get("TEST_VAR"),
            Some(&"test_value".to_string())
        );
        assert_eq!(
            config.runtime.environment.get("NVIDIA_DRIVER_CAPABILITIES"),
            Some(&"all".to_string())
        );
    }
}
