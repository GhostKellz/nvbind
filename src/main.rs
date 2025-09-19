use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{debug, info};

mod cdi;
mod config;
mod distro;
mod gpu;
mod isolation;
mod runtime;
mod wsl2;

use config::Config;

#[derive(Parser)]
#[command(name = "nvbind")]
#[command(about = "A lightweight, Rust-based alternative to NVIDIA Container Toolkit")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show detected GPUs and driver information
    Info {
        /// Show detailed system compatibility report
        #[arg(long)]
        detailed: bool,
        /// Show WSL2 specific information
        #[arg(long)]
        wsl2: bool,
    },
    /// Run a container with GPU passthrough
    Run {
        /// Container runtime to use (podman, docker, bolt)
        #[arg(long)]
        runtime: Option<String>,
        /// GPU selection (all, none, or device ID)
        #[arg(long)]
        gpu: Option<String>,
        /// Container image
        image: String,
        /// Command to run in container
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
        /// Workload profile (gaming, ai-ml, shared)
        #[arg(long)]
        profile: Option<String>,
        /// Enable GPU isolation
        #[arg(long)]
        isolate: bool,
        /// Use CDI devices instead of direct GPU access
        #[arg(long)]
        cdi: bool,
    },
    /// Generate default configuration file
    Config {
        /// Path to save config file
        #[arg(long)]
        output: Option<String>,
        /// Show current config without saving
        #[arg(long)]
        show: bool,
    },
    /// CDI (Container Device Interface) commands
    Cdi {
        #[command(subcommand)]
        command: CdiCommands,
    },
    /// System diagnostics and compatibility check
    Doctor {
        /// Generate installation instructions
        #[arg(long)]
        install: bool,
    },
    /// WSL2 specific commands
    Wsl2 {
        #[command(subcommand)]
        command: Wsl2Commands,
    },
}

#[derive(Subcommand)]
enum CdiCommands {
    /// Generate CDI specification for NVIDIA GPUs
    Generate {
        /// Output directory for CDI spec
        #[arg(long)]
        output: Option<String>,
    },
    /// List available CDI devices
    List,
    /// Validate CDI specifications
    Validate {
        /// Path to CDI spec file
        spec_file: Option<String>,
    },
}

#[derive(Subcommand)]
enum Wsl2Commands {
    /// Check WSL2 GPU support
    Check,
    /// Setup WSL2 GPU environment
    Setup {
        /// Workload type (gaming, ai-ml, development)
        #[arg(long)]
        workload: Option<String>,
    },
    /// Show WSL2 diagnostics
    Diagnostics,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Info { detailed, wsl2 } => {
            if wsl2 {
                handle_wsl2_info().await?;
            } else if detailed {
                handle_detailed_info().await?;
            } else {
                info!("Scanning for NVIDIA GPUs...");
                gpu::info().await?;
            }
        }
        Commands::Run {
            runtime,
            gpu,
            image,
            args,
            profile,
            isolate,
            cdi,
        } => {
            handle_run_command(runtime, gpu, image, args, profile, isolate, cdi).await?;
        }
        Commands::Config { output, show } => {
            handle_config_command(output, show).await?;
        }
        Commands::Cdi { command } => {
            handle_cdi_command(command).await?;
        }
        Commands::Doctor { install } => {
            handle_doctor_command(install).await?;
        }
        Commands::Wsl2 { command } => {
            handle_wsl2_command(command).await?;
        }
    }

    Ok(())
}

async fn handle_run_command(
    runtime: Option<String>,
    gpu: Option<String>,
    image: String,
    args: Vec<String>,
    profile: Option<String>,
    isolate: bool,
    cdi: bool,
) -> Result<()> {
    info!("Starting container with GPU passthrough");
    let mut config = Config::load()?;

    // Apply WSL2 optimizations if detected
    if wsl2::Wsl2Manager::detect_wsl2() {
        info!("WSL2 detected, applying optimizations");
        let _wsl2_manager = wsl2::Wsl2Manager::new()?;

        // Apply workload-specific optimizations
        if let Some(ref profile_name) = profile {
            match profile_name.as_str() {
                "gaming" => {
                    let gaming_env = wsl2::gaming::setup_gaming_optimizations()?;
                    for (key, value) in gaming_env {
                        config.runtime.environment.insert(key, value);
                    }
                }
                "ai-ml" => {
                    let ai_ml_env = wsl2::ai_ml::setup_ai_ml_optimizations()?;
                    for (key, value) in ai_ml_env {
                        config.runtime.environment.insert(key, value);
                    }
                }
                _ => {}
            }
        }
    }

    // Setup GPU isolation if requested
    if isolate {
        info!("Enabling GPU isolation");
        let isolation_config = if wsl2::Wsl2Manager::detect_wsl2() {
            isolation::wsl2::get_wsl2_gpu_config()?
        } else {
            isolation::IsolationConfig::default()
        };

        let isolation_manager = isolation::IsolationManager::new(isolation_config);
        isolation_manager.initialize()?;

        // Create isolated container
        let container_profile = profile.as_deref();
        let gpu_devices = vec![gpu.clone().unwrap_or_else(|| "all".to_string())];
        let _isolated_container = isolation_manager.create_isolated_container(
            &uuid::Uuid::new_v4().to_string(),
            container_profile,
            gpu_devices,
        )?;
    }

    let runtime_cmd = config.get_runtime_command(runtime.as_deref());
    let gpu_selection = config.get_gpu_selection(gpu.as_deref());

    debug!(
        "Runtime: {}, GPU: {}, Image: {}, Profile: {:?}, Isolate: {}, CDI: {}",
        runtime_cmd, gpu_selection, image, profile, isolate, cdi
    );

    if cdi {
        // Use CDI devices
        runtime::run_with_cdi_devices(config, runtime_cmd, gpu_selection, image, args).await?;
    } else {
        // Traditional GPU passthrough
        runtime::run_with_config(config, runtime_cmd, gpu_selection, image, args).await?;
    }

    Ok(())
}

async fn handle_config_command(output: Option<String>, show: bool) -> Result<()> {
    let config = Config::load()?;

    if show {
        let toml_str = toml::to_string_pretty(&config)?;
        println!("Current nvbind configuration:");
        println!("{}", toml_str);
        return Ok(());
    }

    let output_path = if let Some(path) = output {
        std::path::PathBuf::from(path)
    } else {
        // Default to current directory
        std::path::PathBuf::from("nvbind.toml")
    };

    config.save_to_file(&output_path)?;
    println!("Configuration saved to: {}", output_path.display());
    Ok(())
}

async fn handle_cdi_command(command: CdiCommands) -> Result<()> {
    match command {
        CdiCommands::Generate { output } => {
            info!("Generating NVIDIA CDI specification");
            let spec = cdi::generate_nvidia_cdi_spec().await?;
            let registry = cdi::CdiRegistry::new();

            let output_dir = output.as_ref().map(std::path::Path::new);
            let spec_path = registry.save_spec(&spec, output_dir)?;

            println!("CDI specification generated: {}", spec_path.display());
            println!("Devices available:");
            for device in &spec.devices {
                println!("  nvidia.com/gpu={}", device.name);
            }
        }
        CdiCommands::List => {
            info!("Listing available CDI devices");
            let devices = cdi::list_cdi_devices()?;

            if devices.is_empty() {
                println!("No CDI devices found. Generate specifications with 'nvbind cdi generate'");
            } else {
                println!("Available CDI devices:");
                for device in devices {
                    println!("  {}", device);
                }
            }
        }
        CdiCommands::Validate { spec_file } => {
            info!("Validating CDI specifications");

            if let Some(file_path) = spec_file {
                // Validate specific file
                let registry = cdi::CdiRegistry::new();
                let spec = registry.load_spec_file(std::path::Path::new(&file_path))?;
                println!("CDI specification is valid: {}", spec.kind);
            } else {
                // Validate all specs
                let mut registry = cdi::CdiRegistry::new();
                registry.load_specs()?;
                println!("All CDI specifications validated successfully");
            }
        }
    }
    Ok(())
}

async fn handle_doctor_command(install: bool) -> Result<()> {
    info!("Running system diagnostics");

    let distro_manager = distro::DistroManager::new()?;
    let compatibility = distro_manager.check_compatibility()?;

    println!("=== System Diagnostics ===");
    println!("Distribution: {} {}", compatibility.distribution.name(), compatibility.version);
    println!("System Ready: {}", if compatibility.is_ready() { "✅ Yes" } else { "❌ No" });
    println!();

    // Container runtime status
    println!("Container Runtimes:");
    for (runtime, available) in &compatibility.container_runtime_available {
        println!("  {}: {}", runtime, if *available { "✅ Available" } else { "❌ Missing" });
    }
    println!();

    // NVIDIA packages status
    println!("NVIDIA Packages:");
    for (package, installed) in &compatibility.packages_installed {
        println!("  {}: {}", package, if *installed { "✅ Installed" } else { "❌ Missing" });
    }
    println!();

    // Kernel modules status
    println!("Kernel Modules:");
    for (module, loaded) in &compatibility.kernel_modules_loaded {
        println!("  {}: {}", module, if *loaded { "✅ Loaded" } else { "❌ Not loaded" });
    }
    println!();

    // Library paths status
    println!("Library Paths:");
    for (path, exists) in &compatibility.library_paths_available {
        println!("  {}: {}", path, if *exists { "✅ Exists" } else { "❌ Missing" });
    }
    println!();

    // Recommendations
    let recommendations = compatibility.get_recommendations();
    if !recommendations.is_empty() {
        println!("Recommendations:");
        for recommendation in &recommendations {
            println!("  • {}", recommendation);
        }
        println!();
    }

    if install {
        println!("=== Installation Instructions ===");
        let instructions = distro_manager.generate_install_guide();
        for instruction in instructions {
            println!("{}", instruction);
        }
    }

    Ok(())
}

async fn handle_wsl2_command(command: Wsl2Commands) -> Result<()> {
    match command {
        Wsl2Commands::Check => {
            let manager = wsl2::Wsl2Manager::new()?;
            let gpu_support = manager.check_gpu_support()?;

            println!("=== WSL2 GPU Support ===");
            match gpu_support {
                wsl2::Wsl2GpuSupport::NotWsl2 => {
                    println!("❌ Not running under WSL2");
                }
                wsl2::Wsl2GpuSupport::Available { cuda, opencl, directx, opengl, vulkan } => {
                    println!("✅ WSL2 detected");
                    println!("CUDA Support: {}", if cuda { "✅ Available" } else { "❌ Missing" });
                    println!("OpenCL Support: {}", if opencl { "✅ Available" } else { "❌ Missing" });
                    println!("DirectX Support: {}", if directx { "✅ Available" } else { "❌ Missing" });
                    println!("OpenGL Support: {}", if opengl { "✅ Available" } else { "❌ Missing" });
                    println!("Vulkan Support: {}", if vulkan { "✅ Available" } else { "❌ Missing" });
                }
            }
        }
        Wsl2Commands::Setup { workload } => {
            if !wsl2::Wsl2Manager::detect_wsl2() {
                return Err(anyhow::anyhow!("Not running under WSL2"));
            }

            let manager = wsl2::Wsl2Manager::new()?;
            let env = manager.setup_gpu_environment()?;

            println!("=== WSL2 GPU Environment Setup ===");

            // Apply workload-specific optimizations
            match workload.as_deref() {
                Some("gaming") => {
                    let gaming_env = wsl2::gaming::setup_gaming_optimizations()?;
                    println!("Gaming optimizations applied:");
                    for (key, value) in gaming_env {
                        println!("  export {}={}", key, value);
                    }
                }
                Some("ai-ml") => {
                    let ai_ml_env = wsl2::ai_ml::setup_ai_ml_optimizations()?;
                    println!("AI/ML optimizations applied:");
                    for (key, value) in ai_ml_env {
                        println!("  export {}={}", key, value);
                    }
                }
                _ => {
                    println!("Base WSL2 environment:");
                    for (key, value) in env {
                        println!("  export {}={}", key, value);
                    }
                }
            }
        }
        Wsl2Commands::Diagnostics => {
            let manager = wsl2::Wsl2Manager::new()?;
            let diagnostics = manager.generate_diagnostics()?;

            println!("=== WSL2 Diagnostics ===");
            println!("WSL2 Detected: {}", diagnostics.is_wsl2);
            if let Some(ref version) = diagnostics.wsl_version {
                println!("WSL Version: {}", version);
            }
            if let Some(build) = diagnostics.windows_build {
                println!("Windows Build: {}", build);
            }
            println!("Windows GPU Support: {}", diagnostics.windows_gpu_support);
            println!("Driver Path Exists: {}", diagnostics.driver_path_exists);
        }
    }
    Ok(())
}

async fn handle_detailed_info() -> Result<()> {
    info!("Gathering detailed system information");

    // GPU information
    gpu::info().await?;

    // Distribution information
    let distro_manager = distro::DistroManager::new()?;
    let compatibility = distro_manager.check_compatibility()?;

    println!("\n=== System Information ===");
    println!("Distribution: {} {}", compatibility.distribution.name(), compatibility.version);

    // WSL2 information if applicable
    if wsl2::Wsl2Manager::detect_wsl2() {
        let wsl2_manager = wsl2::Wsl2Manager::new()?;
        let diagnostics = wsl2_manager.generate_diagnostics()?;

        println!("\n=== WSL2 Information ===");
        if let Some(ref version) = diagnostics.wsl_version {
            println!("WSL Version: {}", version);
        }
        if let Some(build) = diagnostics.windows_build {
            println!("Windows Build: {}", build);
        }
    }

    Ok(())
}

async fn handle_wsl2_info() -> Result<()> {
    if !wsl2::Wsl2Manager::detect_wsl2() {
        println!("❌ Not running under WSL2");
        return Ok(());
    }

    let wsl2_manager = wsl2::Wsl2Manager::new()?;
    let diagnostics = wsl2_manager.generate_diagnostics()?;

    println!("=== WSL2 GPU Information ===");
    if let Some(ref version) = diagnostics.wsl_version {
        println!("WSL Version: {}", version);
    }
    if let Some(build) = diagnostics.windows_build {
        println!("Windows Build: {}", build);
    }
    println!("Windows GPU Support: {}", if diagnostics.windows_gpu_support { "✅ Yes" } else { "❌ No" });

    match diagnostics.gpu_support {
        wsl2::Wsl2GpuSupport::Available { cuda, opencl, directx, opengl, vulkan } => {
            println!("\nGPU API Support:");
            println!("  CUDA: {}", if cuda { "✅" } else { "❌" });
            println!("  OpenCL: {}", if opencl { "✅" } else { "❌" });
            println!("  DirectX: {}", if directx { "✅" } else { "❌" });
            println!("  OpenGL: {}", if opengl { "✅" } else { "❌" });
            println!("  Vulkan: {}", if vulkan { "✅" } else { "❌" });
        }
        wsl2::Wsl2GpuSupport::NotWsl2 => {
            println!("❌ WSL2 GPU support not detected");
        }
    }

    Ok(())
}
