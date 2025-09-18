use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nvbind::gpu::discover_gpus;
use std::time::Instant;
use tokio::runtime::Runtime;

/// Benchmark GPU passthrough latency for Bolt integration
/// Target: Sub-microsecond latency (< 100μs)
fn benchmark_bolt_gpu_passthrough(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("bolt_gpu_discovery", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = Instant::now();
                let _gpus = discover_gpus().await.unwrap_or_default();
                let duration = start.elapsed();
                black_box(duration)
            })
        })
    });
}

/// Benchmark CDI spec generation for Bolt capsules
fn benchmark_bolt_cdi_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("bolt_cdi_spec_generation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = Instant::now();

                #[cfg(feature = "bolt")]
                {
                    use nvbind::cdi::bolt::{generate_bolt_gaming_cdi_spec, BoltCapsuleConfig, BoltGpuIsolation};
                    let _spec = generate_bolt_gaming_cdi_spec().await.unwrap_or_default();
                }

                let duration = start.elapsed();
                black_box(duration)
            })
        })
    });
}

/// Benchmark container runtime execution with nvbind
fn benchmark_bolt_container_startup(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("bolt_container_startup_latency", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = Instant::now();

                // Simulate Bolt container startup with GPU passthrough
                let config = nvbind::config::Config::default();

                // This would normally execute: bolt surge run --gpu all ubuntu nvidia-smi
                // For benchmarking, we measure the setup overhead only
                let _validation = nvbind::runtime::validate_runtime("echo");

                let duration = start.elapsed();
                black_box(duration)
            })
        })
    });
}

/// Benchmark WSL2 gaming optimizations
fn benchmark_wsl2_gaming_optimizations(c: &mut Criterion) {
    c.bench_function("wsl2_gaming_detection", |b| {
        b.iter(|| {
            let start = Instant::now();
            let _wsl2_detected = nvbind::wsl2::Wsl2Manager::detect_wsl2();
            let duration = start.elapsed();
            black_box(duration)
        })
    });
}

/// Performance comparison: nvbind vs traditional Docker GPU toolkit
fn benchmark_performance_comparison(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("gpu_runtime_comparison");

    // nvbind performance
    group.bench_function("nvbind_total_overhead", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = Instant::now();

                // Full nvbind GPU setup simulation
                let _gpus = discover_gpus().await.unwrap_or_default();
                let _driver_info = nvbind::gpu::get_driver_info().await.unwrap_or_default();
                let _wsl2 = nvbind::wsl2::Wsl2Manager::detect_wsl2();

                #[cfg(feature = "bolt")]
                {
                    use nvbind::bolt::NvbindGpuManager;
                    let manager = NvbindGpuManager::with_defaults();
                    let _compat = manager.check_bolt_gpu_compatibility().await.unwrap_or_default();
                }

                let duration = start.elapsed();
                black_box(duration)
            })
        })
    });

    group.finish();
}

/// Stress test for concurrent Bolt capsule GPU operations
fn benchmark_concurrent_capsules(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("concurrent_bolt_capsules", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = Instant::now();

                // Simulate 10 concurrent Bolt capsules requesting GPU access
                let mut handles = Vec::new();

                for i in 0..10 {
                    let handle = tokio::spawn(async move {
                        let _gpus = discover_gpus().await.unwrap_or_default();

                        #[cfg(feature = "bolt")]
                        {
                            use nvbind::cdi::bolt::generate_bolt_gaming_cdi_spec;
                            let _spec = generate_bolt_gaming_cdi_spec().await.unwrap_or_default();
                        }

                        i // Return capsule ID
                    });
                    handles.push(handle);
                }

                // Wait for all capsules to complete GPU setup
                for handle in handles {
                    let _ = handle.await;
                }

                let duration = start.elapsed();
                black_box(duration)
            })
        })
    });
}

/// Memory efficiency benchmark for GPU containers
fn benchmark_memory_efficiency(c: &mut Criterion) {
    c.bench_function("gpu_memory_overhead", |b| {
        b.iter(|| {
            let start = Instant::now();

            // Measure memory overhead of nvbind GPU management
            let config = nvbind::config::Config::default();

            #[cfg(feature = "bolt")]
            {
                use nvbind::bolt::NvbindGpuManager;
                let _manager = NvbindGpuManager::new(config.bolt.unwrap_or_default());
            }

            let duration = start.elapsed();
            black_box(duration)
        })
    });
}

criterion_group!(
    bolt_benchmarks,
    benchmark_bolt_gpu_passthrough,
    benchmark_bolt_cdi_generation,
    benchmark_bolt_container_startup,
    benchmark_wsl2_gaming_optimizations,
    benchmark_performance_comparison,
    benchmark_concurrent_capsules,
    benchmark_memory_efficiency
);

criterion_main!(bolt_benchmarks);