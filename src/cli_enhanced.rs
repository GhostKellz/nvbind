//! Enhanced CLI with interactive setup wizard and diagnostic tools
//!
//! Provides user-friendly command-line interface enhancements

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use dialoguer::{theme::ColorfulTheme, Confirm, Input, Select, MultiSelect};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Enhanced CLI interface
#[derive(Parser)]
#[command(name = "nvbind")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Quiet mode
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Configuration file path
    #[arg(short, long, global = true)]
    pub config: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Interactive setup wizard
    Setup {
        /// Skip confirmation prompts
        #[arg(long)]
        yes: bool,
    },

    /// Run system diagnostics
    Diagnose {
        /// Output format
        #[arg(short, long, default_value = "human")]
        format: OutputFormat,

        /// Include detailed GPU information
        #[arg(long)]
        gpu_details: bool,
    },

    /// Performance tuning wizard
    Tune {
        /// Workload profile
        #[arg(short, long)]
        profile: Option<String>,

        /// Benchmark after tuning
        #[arg(long)]
        benchmark: bool,
    },

    /// Validate configuration
    Validate {
        /// Configuration file to validate
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Strict validation mode
        #[arg(long)]
        strict: bool,
    },

    /// Monitor system resources
    Monitor {
        /// Update interval in seconds
        #[arg(short, long, default_value = "1")]
        interval: u64,

        /// Export metrics to file
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Migrate configuration to latest version
    Migrate {
        /// Configuration file to migrate
        #[arg(short, long)]
        file: PathBuf,

        /// Backup original file
        #[arg(long, default_value = "true")]
        backup: bool,
    },

    /// Generate shell completions
    Completions {
        /// Shell type
        #[arg(value_enum)]
        shell: Shell,
    },

    /// Service mesh management
    Mesh {
        #[command(subcommand)]
        action: MeshAction,
    },
}

#[derive(Subcommand)]
pub enum MeshAction {
    /// Start the service mesh
    Start {
        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
    /// Stop the service mesh
    Stop,
    /// Register a service
    Register {
        /// Service name
        #[arg(short, long)]
        name: String,
        /// Service address
        #[arg(short, long)]
        address: String,
        /// Service port
        #[arg(short, long)]
        port: u16,
    },
    /// Deregister a service
    Deregister {
        /// Service name
        #[arg(short, long)]
        name: String,
    },
    /// List services
    List,
    /// Show mesh status
    Status,
    /// Add routing rule
    Route {
        /// Rule in JSON format
        #[arg(short, long)]
        rule: String,
    },
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum Shell {
    Bash,
    Zsh,
    Fish,
    PowerShell,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OutputFormat {
    #[serde(rename = "human")]
    Human,
    #[serde(rename = "json")]
    Json,
    #[serde(rename = "yaml")]
    Yaml,
    #[serde(rename = "table")]
    Table,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "human" => Ok(OutputFormat::Human),
            "json" => Ok(OutputFormat::Json),
            "yaml" => Ok(OutputFormat::Yaml),
            "table" => Ok(OutputFormat::Table),
            _ => Err(format!("Unknown format: {}", s)),
        }
    }
}

/// Interactive setup wizard
pub struct SetupWizard {
    theme: ColorfulTheme,
}

impl SetupWizard {
    pub fn new() -> Self {
        Self {
            theme: ColorfulTheme::default(),
        }
    }

    /// Run interactive setup
    pub async fn run(&self) -> Result<()> {
        println!("{}", "Welcome to nvbind Setup Wizard!".bold().green());
        println!("This wizard will help you configure nvbind for your system.\n");

        // Step 1: System check
        self.system_check().await?;

        // Step 2: Runtime selection
        let runtime = self.select_runtime()?;

        // Step 3: GPU configuration
        let gpu_config = self.configure_gpu().await?;

        // Step 4: Security settings
        let security_config = self.configure_security()?;

        // Step 5: Monitoring setup
        let monitoring_config = self.configure_monitoring()?;

        // Step 6: Generate configuration
        let config = self.generate_config(runtime, gpu_config, security_config, monitoring_config)?;

        // Step 7: Save configuration
        self.save_config(config)?;

        println!("\n{}", "Setup completed successfully!".bold().green());
        println!("Configuration saved to: {}", "/etc/nvbind/config.toml".cyan());
        println!("\nYou can now run: {}", "nvbind --help".yellow());

        Ok(())
    }

    /// System compatibility check
    async fn system_check(&self) -> Result<()> {
        println!("{}", "Checking system compatibility...".bold());

        let pb = ProgressBar::new(4);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")?
            .progress_chars("##-"));

        // Check GPU
        pb.set_message("Checking GPU...");
        let gpu_available = crate::gpu::is_nvidia_driver_available();
        pb.inc(1);

        // Check runtime
        pb.set_message("Checking container runtimes...");
        let podman_available = crate::runtime::validate_runtime("podman").is_ok();
        let docker_available = crate::runtime::validate_runtime("docker").is_ok();
        pb.inc(1);

        // Check permissions
        pb.set_message("Checking permissions...");
        let is_root = unsafe { libc::getuid() } == 0;
        pb.inc(1);

        // Check kernel modules
        pb.set_message("Checking kernel modules...");
        let nvidia_loaded = Path::new("/dev/nvidia0").exists();
        pb.inc(1);

        pb.finish_with_message("System check complete");

        // Display results
        println!("\n{}", "System Check Results:".bold().underline());
        self.print_check("NVIDIA GPU", gpu_available);
        self.print_check("NVIDIA Driver", gpu_available);
        self.print_check("Podman", podman_available);
        self.print_check("Docker", docker_available);
        self.print_check("Root Access", is_root);
        self.print_check("Kernel Modules", nvidia_loaded);

        if !gpu_available {
            println!("\n{}", "Warning: No NVIDIA GPU detected. Some features will be limited.".yellow());
        }

        Ok(())
    }

    fn print_check(&self, name: &str, status: bool) {
        let icon = if status { "✓".green() } else { "✗".red() };
        let status_text = if status { "Available".green() } else { "Not Available".red() };
        println!("  {} {:<20} {}", icon, name, status_text);
    }

    /// Select container runtime
    fn select_runtime(&self) -> Result<String> {
        let runtimes = vec!["podman", "docker", "containerd"];

        let selection = Select::with_theme(&self.theme)
            .with_prompt("Select container runtime")
            .default(0)
            .items(&runtimes)
            .interact()?;

        Ok(runtimes[selection].to_string())
    }

    /// Configure GPU settings
    async fn configure_gpu(&self) -> Result<GpuConfiguration> {
        println!("\n{}", "GPU Configuration".bold());

        let gpu_mode = Select::with_theme(&self.theme)
            .with_prompt("GPU access mode")
            .default(0)
            .items(&["All GPUs", "Specific GPUs", "No GPU"])
            .interact()?;

        let selection = match gpu_mode {
            0 => "all".to_string(),
            1 => {
                // List available GPUs
                if let Ok(gpus) = crate::gpu::discover_gpus().await {
                    let gpu_names: Vec<String> = gpus.iter()
                        .enumerate()
                        .map(|(i, g)| format!("GPU {}: {}", i, g.name))
                        .collect();

                    let selections = MultiSelect::with_theme(&self.theme)
                        .with_prompt("Select GPUs to use")
                        .items(&gpu_names)
                        .interact()?;

                    selections.iter().map(|&i| i.to_string()).collect::<Vec<_>>().join(",")
                } else {
                    "0".to_string()
                }
            }
            _ => "none".to_string(),
        };

        let mig_enabled = Confirm::with_theme(&self.theme)
            .with_prompt("Enable Multi-Instance GPU (MIG) support?")
            .default(false)
            .interact()?;

        Ok(GpuConfiguration {
            selection,
            mig_enabled,
        })
    }

    /// Configure security settings
    fn configure_security(&self) -> Result<SecurityConfiguration> {
        println!("\n{}", "Security Configuration".bold());

        let enable_rbac = Confirm::with_theme(&self.theme)
            .with_prompt("Enable Role-Based Access Control (RBAC)?")
            .default(false)
            .interact()?;

        let allow_privileged = Confirm::with_theme(&self.theme)
            .with_prompt("Allow privileged containers?")
            .default(false)
            .interact()?;

        let audit_logging = Confirm::with_theme(&self.theme)
            .with_prompt("Enable audit logging?")
            .default(true)
            .interact()?;

        Ok(SecurityConfiguration {
            enable_rbac,
            allow_privileged,
            audit_logging,
        })
    }

    /// Configure monitoring
    fn configure_monitoring(&self) -> Result<MonitoringConfiguration> {
        println!("\n{}", "Monitoring Configuration".bold());

        let enable_monitoring = Confirm::with_theme(&self.theme)
            .with_prompt("Enable performance monitoring?")
            .default(true)
            .interact()?;

        let prometheus_port = if enable_monitoring {
            Some(Input::<u16>::with_theme(&self.theme)
                .with_prompt("Prometheus metrics port")
                .default(9090)
                .interact()?)
        } else {
            None
        };

        Ok(MonitoringConfiguration {
            enabled: enable_monitoring,
            prometheus_port,
        })
    }

    /// Generate configuration
    fn generate_config(
        &self,
        runtime: String,
        gpu: GpuConfiguration,
        security: SecurityConfiguration,
        monitoring: MonitoringConfiguration,
    ) -> Result<String> {
        let config = format!(r#"# nvbind configuration
# Generated by setup wizard

schema_version = "1.0.0"

[runtime]
default = "{}"
timeout_seconds = 300

[gpu]
default_selection = "{}"
mig_enabled = {}

[security]
enable_rbac = {}
allow_privileged = {}
audit_logging = {}

[monitoring]
enabled = {}
prometheus_port = {}

[logging]
level = "info"
format = "json"

[cdi]
spec_dir = "/etc/cdi"
auto_generate = true
"#,
            runtime,
            gpu.selection,
            gpu.mig_enabled,
            security.enable_rbac,
            security.allow_privileged,
            security.audit_logging,
            monitoring.enabled,
            monitoring.prometheus_port.unwrap_or(9090),
        );

        Ok(config)
    }

    /// Save configuration file
    fn save_config(&self, config: String) -> Result<()> {
        let path = PathBuf::from("/etc/nvbind/config.toml");

        if path.exists() {
            let backup = Confirm::with_theme(&self.theme)
                .with_prompt("Configuration file exists. Create backup?")
                .default(true)
                .interact()?;

            if backup {
                let backup_path = path.with_extension("toml.bak");
                fs::copy(&path, backup_path)?;
                println!("Backup created: {}", path.with_extension("toml.bak").display());
            }
        }

        // Create directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&path, config)?;
        Ok(())
    }
}

#[derive(Debug)]
struct GpuConfiguration {
    selection: String,
    mig_enabled: bool,
}

#[derive(Debug)]
struct SecurityConfiguration {
    enable_rbac: bool,
    allow_privileged: bool,
    audit_logging: bool,
}

#[derive(Debug)]
struct MonitoringConfiguration {
    enabled: bool,
    prometheus_port: Option<u16>,
}

/// System diagnostics
pub struct DiagnosticTool;

impl DiagnosticTool {
    /// Run comprehensive diagnostics
    pub async fn run(format: OutputFormat, gpu_details: bool) -> Result<()> {
        let mut report = DiagnosticReport::new();

        // System information
        report.system = Self::collect_system_info()?;

        // GPU information
        report.gpu = Self::collect_gpu_info(gpu_details).await?;

        // Container runtime
        report.runtime = Self::collect_runtime_info()?;

        // Configuration
        report.configuration = Self::collect_config_info()?;

        // Health checks
        report.health = crate::monitoring::perform_health_checks().await;

        // Output report
        Self::output_report(report, format)?;

        Ok(())
    }

    fn collect_system_info() -> Result<SystemInfo> {
        Ok(SystemInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            hostname: hostname::get()?.to_string_lossy().to_string(),
            kernel: Self::get_kernel_version()?,
            memory_gb: Self::get_total_memory_gb(),
            cpu_cores: std::thread::available_parallelism()?.get(),
        })
    }

    fn collect_gpu_info(detailed: bool) -> async Result<GpuInfo> {
        let available = crate::gpu::is_nvidia_driver_available();
        let driver_info = if available {
            crate::gpu::get_driver_info().await.ok()
        } else {
            None
        };

        let devices = if available && detailed {
            crate::gpu::discover_gpus().await.ok()
        } else {
            None
        };

        Ok(GpuInfo {
            available,
            driver_info,
            devices,
        })
    }

    fn collect_runtime_info() -> Result<RuntimeInfo> {
        Ok(RuntimeInfo {
            podman: crate::runtime::validate_runtime("podman").is_ok(),
            docker: crate::runtime::validate_runtime("docker").is_ok(),
            containerd: crate::runtime::validate_runtime("containerd").is_ok(),
        })
    }

    fn collect_config_info() -> Result<ConfigInfo> {
        let path = PathBuf::from("/etc/nvbind/config.toml");
        let exists = path.exists();
        let valid = if exists {
            let validator = crate::config_validation::ConfigValidator::new(false);
            validator.validate_file(&path).map(|r| r.valid).unwrap_or(false)
        } else {
            false
        };

        Ok(ConfigInfo {
            path: path.to_string_lossy().to_string(),
            exists,
            valid,
        })
    }

    fn get_kernel_version() -> Result<String> {
        let uname = std::process::Command::new("uname")
            .arg("-r")
            .output()?;
        Ok(String::from_utf8_lossy(&uname.stdout).trim().to_string())
    }

    fn get_total_memory_gb() -> f64 {
        16.0 // Placeholder
    }

    fn output_report(report: DiagnosticReport, format: OutputFormat) -> Result<()> {
        match format {
            OutputFormat::Human => Self::output_human(report),
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&report)?);
            }
            OutputFormat::Yaml => {
                println!("{}", serde_yaml::to_string(&report)?);
            }
            OutputFormat::Table => Self::output_table(report),
        }
        Ok(())
    }

    fn output_human(report: DiagnosticReport) {
        println!("{}", "nvbind System Diagnostics Report".bold().underline());
        println!("{}", "=================================".dim());

        println!("\n{}", "System Information:".bold());
        println!("  OS: {}", report.system.os);
        println!("  Architecture: {}", report.system.arch);
        println!("  Hostname: {}", report.system.hostname);
        println!("  Kernel: {}", report.system.kernel);
        println!("  Memory: {} GB", report.system.memory_gb);
        println!("  CPU Cores: {}", report.system.cpu_cores);

        println!("\n{}", "GPU Information:".bold());
        println!("  Available: {}", if report.gpu.available { "Yes".green() } else { "No".red() });
        if let Some(info) = report.gpu.driver_info {
            println!("  Driver: {} ({})", info.version, format!("{:?}", info.driver_type));
        }

        println!("\n{}", "Container Runtimes:".bold());
        println!("  Podman: {}", if report.runtime.podman { "✓".green() } else { "✗".red() });
        println!("  Docker: {}", if report.runtime.docker { "✓".green() } else { "✗".red() });
        println!("  Containerd: {}", if report.runtime.containerd { "✓".green() } else { "✗".red() });

        println!("\n{}", "Configuration:".bold());
        println!("  Path: {}", report.configuration.path);
        println!("  Exists: {}", if report.configuration.exists { "Yes".green() } else { "No".yellow() });
        println!("  Valid: {}", if report.configuration.valid { "Yes".green() } else { "No".red() });

        println!("\n{}", "Health Status:".bold());
        let health_icon = if report.health.healthy { "✓".green() } else { "✗".red() };
        println!("  Overall: {} {}",
            health_icon,
            if report.health.healthy { "Healthy".green() } else { "Issues Detected".red() }
        );

        for check in &report.health.checks {
            let icon = match check.status {
                crate::monitoring::CheckStatus::Ok => "✓".green(),
                crate::monitoring::CheckStatus::Warning => "⚠".yellow(),
                crate::monitoring::CheckStatus::Critical => "✗".red(),
            };
            println!("  {} {}: {} ({}ms)", icon, check.name, check.message, check.duration_ms);
        }
    }

    fn output_table(report: DiagnosticReport) {
        use prettytable::{Table, row, cell};

        let mut table = Table::new();
        table.add_row(row!["Component", "Status", "Details"]);

        table.add_row(row![
            "System",
            "OK",
            format!("{} {} ({})", report.system.os, report.system.arch, report.system.kernel)
        ]);

        table.add_row(row![
            "GPU",
            if report.gpu.available { "Available" } else { "Not Available" },
            report.gpu.driver_info
                .map(|i| format!("Driver {}", i.version))
                .unwrap_or_else(|| "N/A".to_string())
        ]);

        table.add_row(row![
            "Runtime",
            "Varies",
            format!("Podman: {}, Docker: {}",
                if report.runtime.podman { "✓" } else { "✗" },
                if report.runtime.docker { "✓" } else { "✗" }
            )
        ]);

        table.add_row(row![
            "Health",
            if report.health.healthy { "Healthy" } else { "Issues" },
            format!("{} checks performed", report.health.checks.len())
        ]);

        table.printstd();
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct DiagnosticReport {
    system: SystemInfo,
    gpu: GpuInfo,
    runtime: RuntimeInfo,
    configuration: ConfigInfo,
    health: crate::monitoring::HealthStatus,
}

impl DiagnosticReport {
    fn new() -> Self {
        Self {
            system: SystemInfo::default(),
            gpu: GpuInfo::default(),
            runtime: RuntimeInfo::default(),
            configuration: ConfigInfo::default(),
            health: crate::monitoring::HealthStatus {
                healthy: false,
                checks: vec![],
                timestamp: chrono::Utc::now(),
            },
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct SystemInfo {
    os: String,
    arch: String,
    hostname: String,
    kernel: String,
    memory_gb: f64,
    cpu_cores: usize,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct GpuInfo {
    available: bool,
    driver_info: Option<crate::gpu::DriverInfo>,
    devices: Option<Vec<crate::gpu::GpuDevice>>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct RuntimeInfo {
    podman: bool,
    docker: bool,
    containerd: bool,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ConfigInfo {
    path: String,
    exists: bool,
    valid: bool,
}