use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nvbind::cdi::{CdiRegistry, generate_nvidia_cdi_spec};
use nvbind::gpu::{discover_gpus, is_nvidia_driver_available};
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
                let _device =
                    registry.get_device(&format!("nvidia.com/gpu={}", spec.devices[0].name));
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
        .args(&[
            "run",
            "--rm",
            "--gpus",
            "all",
            "nvidia/cuda:latest",
            "nvidia-smi",
        ])
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
            let _cmd = Command::new("docker").args(&[
                "run",
                "--rm",
                "--gpus",
                "all",
                "nvidia/cuda:latest",
                "echo",
                "test",
            ]);
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

    // Configure for high-precision measurements
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10000);

    // Test 1: Basic GPU detection (should be sub-microsecond)
    group.bench_function("gpu_detection_speed", |b| {
        b.iter(|| {
            let start = Instant::now();
            let _available = is_nvidia_driver_available();
            let elapsed = start.elapsed();

            // Log if we exceed sub-microsecond
            if elapsed >= Duration::from_nanos(1000) {
                eprintln!("Warning: GPU detection took {:?} (> 1μs)", elapsed);
            }

            elapsed
        });
    });

    // Test 2: Device file access (filesystem cache should make this sub-microsecond)
    group.bench_function("device_file_access", |b| {
        b.iter(|| {
            let start = Instant::now();
            let _exists = std::path::Path::new("/dev/nvidiactl").exists();
            let elapsed = start.elapsed();

            if elapsed >= Duration::from_nanos(1000) {
                eprintln!("Warning: Device file access took {:?} (> 1μs)", elapsed);
            }

            elapsed
        });
    });

    // Test 3: Memory allocation performance (critical for zero-overhead)
    group.bench_function("memory_allocation_speed", |b| {
        b.iter(|| {
            let start = Instant::now();
            let _registry = CdiRegistry::new();
            let elapsed = start.elapsed();

            if elapsed >= Duration::from_nanos(1000) {
                eprintln!("Warning: CDI registry creation took {:?} (> 1μs)", elapsed);
            }

            elapsed
        });
    });

    // Test 4: String operations (should be very fast)
    group.bench_function("string_operations", |b| {
        b.iter(|| {
            let start = Instant::now();
            let device_name = format!("nvidia.com/gpu=gpu0");
            let _processed = device_name.contains("gpu");
            let elapsed = start.elapsed();

            if elapsed >= Duration::from_nanos(500) {
                eprintln!("Warning: String operations took {:?} (> 500ns)", elapsed);
            }

            elapsed
        });
    });

    // Test 5: Critical path validation - combined operations
    group.bench_function("critical_path_combined", |b| {
        b.iter(|| {
            let start = Instant::now();

            // Simulate the critical path for GPU setup
            let _driver_available = is_nvidia_driver_available();
            let _device_exists = std::path::Path::new("/dev/nvidiactl").exists();
            let _registry = CdiRegistry::new();
            let device_name = "nvidia.com/gpu=all";
            let _contains_gpu = device_name.contains("gpu");

            let elapsed = start.elapsed();

            // This is the key test - entire critical path should be sub-microsecond
            if elapsed >= Duration::from_nanos(1000) {
                eprintln!("CRITICAL: Combined operations took {:?} (> 1μs)", elapsed);
            }

            elapsed
        });
    });

    // Test 6: High-frequency operations (simulate container lifecycle)
    group.bench_function("high_frequency_simulation", |b| {
        let mut counter = 0u32;
        b.iter(|| {
            let start = Instant::now();

            counter = counter.wrapping_add(1);
            let _registry = CdiRegistry::new();
            let _check = counter % 2 == 0;

            let elapsed = start.elapsed();

            if elapsed >= Duration::from_nanos(800) {
                eprintln!("Warning: High-frequency op took {:?} (> 800ns)", elapsed);
            }

            elapsed
        });
    });

    group.finish();
}

/// Comprehensive latency measurement with percentile analysis
fn bench_latency_percentiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_percentiles");

    // Configure for statistical analysis
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(50000);

    group.bench_function("gpu_passthrough_latency_distribution", |b| {
        b.iter(|| {
            let start = Instant::now();

            // Core GPU passthrough operations
            let _driver_check = is_nvidia_driver_available();
            let _device_access = std::path::Path::new("/dev/nvidiactl").exists();
            let _registry = CdiRegistry::new();

            start.elapsed()
        });
    });

    group.finish();
}

/// Memory overhead and allocation benchmarks
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");

    group.bench_function("cdi_registry_memory_usage", |b| {
        b.iter(|| {
            let start = Instant::now();

            // Measure memory allocation overhead
            let registries: Vec<CdiRegistry> = (0..100).map(|_| CdiRegistry::new()).collect();
            let creation_time = start.elapsed();

            // Force use of registries
            let _count = registries.len();

            creation_time
        });
    });

    group.bench_function("string_allocation_overhead", |b| {
        b.iter(|| {
            let start = Instant::now();

            let device_names: Vec<String> = (0..1000)
                .map(|i| format!("nvidia.com/gpu=gpu{}", i))
                .collect();

            let creation_time = start.elapsed();
            let _count = device_names.len();

            creation_time
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
    bench_performance_claims_validation,
    bench_latency_percentiles,
    bench_memory_overhead
);
criterion_main!(benches);
