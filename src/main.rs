use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{debug, info};

mod config;
mod gpu;
mod runtime;

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
    Info,
    /// Run a container with GPU passthrough
    Run {
        /// Container runtime to use (podman, docker)
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
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Info => {
            info!("Scanning for NVIDIA GPUs...");
            gpu::info().await?;
        }
        Commands::Run {
            runtime,
            gpu,
            image,
            args,
        } => {
            info!("Starting container with GPU passthrough");
            let config = Config::load()?;

            let runtime_cmd = config.get_runtime_command(runtime.as_deref());
            let gpu_selection = config.get_gpu_selection(gpu.as_deref());

            debug!(
                "Runtime: {}, GPU: {}, Image: {}",
                runtime_cmd, gpu_selection, image
            );
            runtime::run_with_config(config, runtime_cmd, gpu_selection, image, args).await?;
        }
        Commands::Config { output, show } => {
            handle_config_command(output, show).await?;
        }
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
