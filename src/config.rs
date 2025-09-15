use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub gpu: GpuConfig,
    pub runtime: RuntimeConfig,
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Default GPU selection (all, none, or specific IDs)
    pub default_selection: String,
    /// Additional device paths to mount
    pub additional_devices: Vec<String>,
    /// Additional library paths to mount
    pub additional_libraries: Vec<String>,
    /// Enable GPU isolation/sandboxing
    pub enable_isolation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Default container runtime (podman, docker)
    pub default_runtime: String,
    /// Additional runtime arguments
    pub default_args: Vec<String>,
    /// Environment variables to set
    pub environment: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Allow rootless containers
    pub allow_rootless: bool,
    /// Restrict device access
    pub restrict_devices: bool,
    /// Additional security options
    pub security_opts: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        let mut environment = std::collections::HashMap::new();
        environment.insert("NVIDIA_DRIVER_CAPABILITIES".to_string(), "all".to_string());

        Self {
            gpu: GpuConfig {
                default_selection: "all".to_string(),
                additional_devices: vec![],
                additional_libraries: vec![],
                enable_isolation: false,
            },
            runtime: RuntimeConfig {
                default_runtime: "podman".to_string(),
                default_args: vec!["--rm".to_string()],
                environment,
            },
            security: SecurityConfig {
                allow_rootless: true,
                restrict_devices: false,
                security_opts: vec![],
            },
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let config_path = Self::find_config_file()?;

        if let Some(path) = config_path {
            info!("Loading config from: {}", path.display());
            Self::load_from_file(&path)
        } else {
            info!("No config file found, using defaults");
            Ok(Self::default())
        }
    }

    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .context(format!("Failed to read config file: {}", path.display()))?;

        let config: Config = toml::from_str(&content).context("Failed to parse TOML config")?;

        debug!("Loaded config: {:?}", config);
        Ok(config)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self).context("Failed to serialize config to TOML")?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).context("Failed to create config directory")?;
        }

        fs::write(path, content)
            .context(format!("Failed to write config file: {}", path.display()))?;

        info!("Config saved to: {}", path.display());
        Ok(())
    }

    fn find_config_file() -> Result<Option<PathBuf>> {
        let mut config_paths = vec![
            // Current directory
            Some(PathBuf::from("nvbind.toml")),
            Some(PathBuf::from(".nvbind.toml")),
            // System config directory
            Some(PathBuf::from("/etc/nvbind/config.toml")),
        ];

        // Add user config paths if available
        if let Some(config_dir) = dirs::config_dir() {
            config_paths.push(Some(config_dir.join("nvbind").join("config.toml")));
        }
        if let Some(home_dir) = dirs::home_dir() {
            config_paths.push(Some(
                home_dir.join(".config").join("nvbind").join("config.toml"),
            ));
        }

        for path in config_paths.iter().flatten() {
            if path.exists() {
                return Ok(Some(path.clone()));
            }
        }

        Ok(None)
    }

    pub fn get_runtime_command(&self, runtime: Option<&str>) -> String {
        runtime.unwrap_or(&self.runtime.default_runtime).to_string()
    }

    pub fn get_gpu_selection(&self, gpu: Option<&str>) -> String {
        gpu.unwrap_or(&self.gpu.default_selection).to_string()
    }

    pub fn get_all_devices(&self) -> Vec<String> {
        let mut devices = crate::gpu::get_required_devices();
        devices.extend(self.gpu.additional_devices.clone());
        devices
    }

    pub fn get_all_libraries(&self) -> Result<Vec<String>> {
        let mut libraries = crate::gpu::get_required_libraries()?;
        libraries.extend(self.gpu.additional_libraries.clone());
        Ok(libraries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.gpu.default_selection, "all");
        assert_eq!(config.runtime.default_runtime, "podman");
        assert!(config.runtime.default_args.contains(&"--rm".to_string()));
        assert_eq!(
            config.runtime.environment.get("NVIDIA_DRIVER_CAPABILITIES"),
            Some(&"all".to_string())
        );
        assert!(config.security.allow_rootless);
        assert!(!config.security.restrict_devices);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();

        assert_eq!(config.gpu.default_selection, parsed.gpu.default_selection);
        assert_eq!(
            config.runtime.default_runtime,
            parsed.runtime.default_runtime
        );
        assert_eq!(config.runtime.environment, parsed.runtime.environment);
        assert_eq!(
            config.security.allow_rootless,
            parsed.security.allow_rootless
        );
    }

    #[test]
    fn test_config_file_roundtrip() {
        let config = Config::default();

        let mut temp_file = NamedTempFile::new().unwrap();
        let content = toml::to_string_pretty(&config).unwrap();
        temp_file.write_all(content.as_bytes()).unwrap();

        let loaded = Config::load_from_file(temp_file.path()).unwrap();
        assert_eq!(config.gpu.default_selection, loaded.gpu.default_selection);
        assert_eq!(
            config.runtime.default_runtime,
            loaded.runtime.default_runtime
        );
        assert_eq!(config.runtime.environment, loaded.runtime.environment);
    }

    #[test]
    fn test_config_get_runtime_command() {
        let config = Config::default();

        // Test with explicit runtime
        assert_eq!(config.get_runtime_command(Some("docker")), "docker");
        assert_eq!(config.get_runtime_command(Some("podman")), "podman");

        // Test with default runtime
        assert_eq!(config.get_runtime_command(None), "podman");
    }

    #[test]
    fn test_config_get_gpu_selection() {
        let config = Config::default();

        // Test with explicit GPU selection
        assert_eq!(config.get_gpu_selection(Some("0")), "0");
        assert_eq!(config.get_gpu_selection(Some("none")), "none");

        // Test with default GPU selection
        assert_eq!(config.get_gpu_selection(None), "all");
    }

    #[test]
    fn test_config_get_all_devices() {
        let mut config = Config::default();
        config.gpu.additional_devices = vec!["/dev/custom-device".to_string()];

        let devices = config.get_all_devices();
        assert!(!devices.is_empty());
        assert!(devices.contains(&"/dev/nvidiactl".to_string()));
        assert!(devices.contains(&"/dev/nvidia-uvm".to_string()));
        assert!(devices.contains(&"/dev/custom-device".to_string()));
    }

    #[test]
    fn test_config_get_all_libraries() {
        let mut config = Config::default();
        config.gpu.additional_libraries = vec!["/usr/lib/custom-lib.so".to_string()];

        let libraries = config.get_all_libraries().unwrap();
        assert!(libraries.contains(&"/usr/lib/custom-lib.so".to_string()));
    }

    #[test]
    fn test_config_load_nonexistent_file() {
        let result = Config::load_from_file(Path::new("/nonexistent/path/config.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_config_save_to_file() {
        let config = Config::default();
        let temp_file = NamedTempFile::new().unwrap();

        let result = config.save_to_file(temp_file.path());
        assert!(result.is_ok());

        // Verify file was created and contains expected content
        let content = std::fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("[gpu]"));
        assert!(content.contains("[runtime]"));
        assert!(content.contains("[security]"));
    }

    #[test]
    fn test_invalid_toml_config() {
        let invalid_toml = "[gpu]\ninvalid = ";
        let result: Result<Config, _> = toml::from_str(invalid_toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_config_fields() {
        let mut gpu_config = GpuConfig {
            default_selection: "0,1".to_string(),
            additional_devices: vec!["/dev/custom".to_string()],
            additional_libraries: vec!["/lib/custom.so".to_string()],
            enable_isolation: true,
        };

        assert_eq!(gpu_config.default_selection, "0,1");
        assert!(gpu_config.enable_isolation);
        assert_eq!(gpu_config.additional_devices.len(), 1);
        assert_eq!(gpu_config.additional_libraries.len(), 1);

        gpu_config.enable_isolation = false;
        assert!(!gpu_config.enable_isolation);
    }

    #[test]
    fn test_runtime_config_fields() {
        let mut env = std::collections::HashMap::new();
        env.insert("TEST_VAR".to_string(), "test_value".to_string());

        let runtime_config = RuntimeConfig {
            default_runtime: "docker".to_string(),
            default_args: vec!["--rm".to_string(), "--interactive".to_string()],
            environment: env,
        };

        assert_eq!(runtime_config.default_runtime, "docker");
        assert_eq!(runtime_config.default_args.len(), 2);
        assert_eq!(
            runtime_config.environment.get("TEST_VAR"),
            Some(&"test_value".to_string())
        );
    }

    #[test]
    fn test_security_config_fields() {
        let security_config = SecurityConfig {
            allow_rootless: false,
            restrict_devices: true,
            security_opts: vec!["no-new-privileges".to_string()],
        };

        assert!(!security_config.allow_rootless);
        assert!(security_config.restrict_devices);
        assert_eq!(security_config.security_opts.len(), 1);
    }
}
