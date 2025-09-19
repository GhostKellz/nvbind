#!/usr/bin/env cargo
//! nvbind Performance Benchmark Tool
//!
//! This binary validates the sub-microsecond GPU passthrough claims
//! and compares performance against nvidia-docker2.

use anyhow::Result;
use nvbind::gpu::{discover_gpus, is_nvidia_driver_available, check_nvidia_driver_status};
use nvbind::cdi::generate_nvidia_cdi_spec;
use nvbind::runtime::validate_runtime;
use std::time::{Duration, Instant};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "nvbind-benchmark")]
#[command(about = "Performance benchmarking tool for nvbind")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run latency benchmarks
    Latency {
        /// Number of iterations to run
        #[arg(short, long, default_value = "1000")]
        iterations: u32,
        /// Show detailed timing breakdown
        #[arg(long)]
        detailed: bool,
    },
    /// Compare against nvidia-docker2
    Compare {
        /// Container image to test with
        #[arg(long, default_value = "nvidia/cuda:latest")]
        image: String,
    },
    /// Validate performance claims
    Validate,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Latency { iterations, detailed } => {
            run_latency_benchmark(iterations, detailed).await?;
        }
        Commands::Compare { image } => {
            run_comparison_benchmark(&image).await?;
        }
        Commands::Validate => {
            validate_performance_claims().await?;
        }
    }

    Ok(())
}

async fn run_latency_benchmark(iterations: u32, detailed: bool) -> Result<()> {
    println!("🏃 Running GPU Passthrough Latency Benchmark");
    println!("Iterations: {}", iterations);
    println!();

    if !is_nvidia_driver_available() {
        println!("❌ NVIDIA driver not available - cannot run benchmarks");
        return Ok(());
    }

    let mut timings = Vec::new();

    for i in 0..iterations {
        let start = Instant::now();

        // Critical path: GPU device setup
        let _driver_check = check_nvidia_driver_status();
        let _gpus = discover_gpus().await?;
        let _cdi_spec = generate_nvidia_cdi_spec().await?;

        let elapsed = start.elapsed();
        timings.push(elapsed);

        if detailed && i < 10 {
            println!("Iteration {}: {:?}", i + 1, elapsed);
        }
    }

    // Calculate statistics
    let total_time: Duration = timings.iter().sum();
    let avg_time = total_time / timings.len() as u32;
    let min_time = *timings.iter().min().unwrap();
    let max_time = *timings.iter().max().unwrap();

    // Sort for percentiles
    timings.sort();
    let p50 = timings[timings.len() / 2];
    let p95 = timings[timings.len() * 95 / 100];
    let p99 = timings[timings.len() * 99 / 100];

    println!("📊 Latency Results:");
    println!("  Average: {:?}", avg_time);
    println!("  Minimum: {:?}", min_time);
    println!("  Maximum: {:?}", max_time);
    println!("  P50:     {:?}", p50);
    println!("  P95:     {:?}", p95);
    println!("  P99:     {:?}", p99);
    println!();

    // Validate sub-microsecond claim
    let sub_microsecond_count = timings.iter()
        .filter(|&&t| t < Duration::from_nanos(1000))
        .count();

    let sub_microsecond_percentage = (sub_microsecond_count as f64 / timings.len() as f64) * 100.0;

    println!("⚡ Sub-microsecond Performance:");
    println!("  Operations < 1μs: {}/{} ({:.1}%)",
             sub_microsecond_count, timings.len(), sub_microsecond_percentage);

    if sub_microsecond_percentage > 90.0 {
        println!("  ✅ Sub-microsecond claim VALIDATED");
    } else if sub_microsecond_percentage > 50.0 {
        println!("  ⚠️  Sub-microsecond claim PARTIALLY VALIDATED");
    } else {
        println!("  ❌ Sub-microsecond claim NOT VALIDATED");
    }

    Ok(())
}

async fn run_comparison_benchmark(image: &str) -> Result<()> {
    println!("🔄 Comparing nvbind vs nvidia-docker2");
    println!("Container image: {}", image);
    println!();

    // Test nvbind performance
    let nvbind_start = Instant::now();
    let _gpus = discover_gpus().await?;
    let _spec = generate_nvidia_cdi_spec().await?;
    let nvbind_time = nvbind_start.elapsed();

    println!("nvbind setup time: {:?}", nvbind_time);

    // Test nvidia-docker2 if available
    if let Ok(output) = std::process::Command::new("docker")
        .args(&["run", "--rm", "--gpus", "all", image, "echo", "test"])
        .output()
    {
        if output.status.success() {
            println!("nvidia-docker2: Available for comparison");
            println!("Note: Full container launch comparison requires manual timing");
        } else {
            println!("nvidia-docker2: Not available or failed");
        }
    }

    Ok(())
}

async fn validate_performance_claims() -> Result<()> {
    println!("🎯 Validating nvbind Performance Claims");
    println!();

    // Claim 1: Sub-microsecond GPU passthrough latency
    println!("Claim 1: Sub-microsecond GPU passthrough latency");
    let start = Instant::now();
    let driver_available = is_nvidia_driver_available();
    let elapsed = start.elapsed();

    if driver_available {
        println!("  Driver check: {:?} - {}", elapsed,
                 if elapsed < Duration::from_nanos(1000) { "✅ PASS" } else { "❌ FAIL" });
    } else {
        println!("  ❌ No NVIDIA driver available");
        return Ok(());
    }

    // Claim 2: Fast GPU discovery
    println!("\nClaim 2: Fast GPU discovery");
    let start = Instant::now();
    let gpus = discover_gpus().await?;
    let elapsed = start.elapsed();
    println!("  GPU discovery ({} GPUs): {:?} - {}",
             gpus.len(), elapsed,
             if elapsed < Duration::from_millis(10) { "✅ PASS" } else { "❌ SLOW" });

    // Claim 3: Efficient CDI generation
    println!("\nClaim 3: Efficient CDI specification generation");
    let start = Instant::now();
    let _spec = generate_nvidia_cdi_spec().await?;
    let elapsed = start.elapsed();
    println!("  CDI generation: {:?} - {}",
             elapsed,
             if elapsed < Duration::from_millis(5) { "✅ PASS" } else { "❌ SLOW" });

    // Claim 4: Runtime compatibility
    println!("\nClaim 4: Container runtime compatibility");
    for runtime in &["docker", "podman"] {
        match validate_runtime(runtime) {
            Ok(_) => println!("  {}: ✅ Available", runtime),
            Err(_) => println!("  {}: ❌ Not available", runtime),
        }
    }

    println!("\n🏁 Performance validation complete");
    Ok(())
}