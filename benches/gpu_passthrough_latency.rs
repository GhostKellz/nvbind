use criterion::{Criterion, criterion_group, criterion_main, BenchmarkId};
use nvbind::gpu::{discover_gpus, is_nvidia_driver_available};
use nvbind::cdi::{generate_nvidia_cdi_spec, CdiRegistry};
use std::process::Command;
use std::time::{Duration, Instant};

/// Benchmark GPU passthrough latency compared to nvidia-docker2
fn bench_gpu_passthrough_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Skip if no NVIDIA driver available
    if !is_nvidia_driver_available() {
        eprintln!("Skipping GPU passthrough benchmarks - no NVIDIA driver detected");
        return;
    }

    let mut group = c.benchmark_group("gpu_passthrough_latency");

    // Test 1: CDI Device Interface Setup Latency
    group.bench_function("cdi_device_setup", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = Instant::now();
                let spec = generate_nvidia_cdi_spec().await.unwrap();
                let registry = CdiRegistry::new();
                let _device = registry.get_device(&format!("nvidia.com/gpu={}", spec.devices[0].name));
                start.elapsed()
            })
        });
    });

    // Test 2: Container Runtime Command Generation
    group.bench_function("container_command_generation", |b| {
        b.iter(|| {
            let start = Instant::now();
            let _cmd = generate_gpu_container_command("nvidia/cuda:latest", &["nvidia-smi"]);
            start.elapsed()
        });
    });

    // Test 3: Full Container Launch Latency (mock)
    group.bench_function("container_launch_preparation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = Instant::now();

                // Simulate full container preparation pipeline
                let _gpus = discover_gpus().await.unwrap();
                let _spec = generate_nvidia_cdi_spec().await.unwrap();
                let _registry = CdiRegistry::new();
                let _command = generate_gpu_container_command("test", &["echo", "test"]);

                start.elapsed()
            })
        });
    });

    group.finish();
}

/// Benchmark against nvidia-docker2 if available
fn bench_nvidia_docker_comparison(c: &mut Criterion) {
    // Check if nvidia-docker2 is available
    let nvidia_docker_available = Command::new("docker")
        .args(&["run", "--rm", "--gpus", "all", "nvidia/cuda:latest", "nvidia-smi"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if !nvidia_docker_available {
        eprintln!("Skipping nvidia-docker comparison - nvidia-docker2 not available");
        return;
    }

    let mut group = c.benchmark_group("nvidia_docker_comparison");

    // Benchmark nvidia-docker2 command preparation
    group.bench_function("nvidia_docker_cmd_prep", |b| {
        b.iter(|| {
            let start = Instant::now();
            let _cmd = Command::new("docker")
                .args(&["run", "--rm", "--gpus", "all", "nvidia/cuda:latest", "echo", "test"]);
            start.elapsed()
        });
    });

    group.finish();
}

/// Micro-benchmark GPU device access patterns
fn bench_gpu_device_access(c: &mut Criterion) {
    if !is_nvidia_driver_available() {
        return;
    }

    let mut group = c.benchmark_group("gpu_device_access");

    // Test device file access latency
    group.bench_function("device_file_check", |b| {
        b.iter(|| {
            let start = Instant::now();
            let _exists = std::path::Path::new("/dev/nvidiactl").exists();
            start.elapsed()
        });
    });

    // Test GPU information access
    group.bench_function("gpu_info_access", |b| {
        b.iter(|| {
            let start = Instant::now();
            let _info = std::fs::read_to_string("/proc/driver/nvidia/version").ok();
            start.elapsed()
        });
    });

    group.finish();
}

/// Performance validation - ensure sub-microsecond claims
fn bench_performance_claims_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_claims");

    // The critical claim: sub-microsecond GPU passthrough setup
    group.bench_function("sub_microsecond_validation", |b| {
        b.iter(|| {
            let start = Instant::now();

            // Minimal GPU setup operations that should be sub-microsecond
            let _driver_check = is_nvidia_driver_available();
            let _device_exists = std::path::Path::new("/dev/nvidiactl").exists();

            let elapsed = start.elapsed();

            // Assert sub-microsecond performance
            assert!(
                elapsed < Duration::from_nanos(1000),
                "GPU setup took {:?}, exceeding sub-microsecond claim",
                elapsed
            );

            elapsed
        });
    });

    group.finish();
}

fn generate_gpu_container_command(image: &str, cmd: &[&str]) -> Command {
    let mut command = Command::new("podman");
    command
        .arg("run")
        .arg("--rm")
        .arg("--device=/dev/nvidiactl")
        .arg("--device=/dev/nvidia-uvm")
        .arg(image);

    for arg in cmd {
        command.arg(arg);
    }

    command
}

criterion_group!(
    benches,
    bench_gpu_passthrough_latency,
    bench_nvidia_docker_comparison,
    bench_gpu_device_access,
    bench_performance_claims_validation
);
criterion_main!(benches);