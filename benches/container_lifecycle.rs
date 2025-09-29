use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use nvbind::cdi::{CdiRegistry, generate_nvidia_cdi_spec};
use nvbind::config::Config;
use nvbind::gpu::{discover_gpus, is_nvidia_driver_available};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// High-frequency container lifecycle benchmarks
fn bench_high_frequency_container_lifecycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("container_lifecycle");

    // Configure for high-frequency testing
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(1000);

    // Test 1: Rapid container creation/destruction simulation
    group.bench_function("rapid_container_cycles", |b| {
        b.iter_batched(
            || {
                // Setup: prepare shared resources
                Arc::new(CdiRegistry::new())
            },
            |registry| {
                let start = Instant::now();

                // Simulate 100 rapid container lifecycle operations
                for i in 0..100 {
                    let container_id = format!("container-{}", i);

                    // Container creation simulation
                    let _device_name = format!("nvidia.com/gpu=gpu{}", i % 4);
                    let _device = registry.get_device("nvidia.com/gpu=all");

                    // Container destruction cleanup simulation
                    drop(container_id);
                }

                start.elapsed()
            },
            BatchSize::SmallInput,
        );
    });

    // Test 2: Concurrent container access patterns
    group.bench_function("concurrent_container_simulation", |b| {
        b.iter_batched(
            || Arc::new(CdiRegistry::new()),
            |registry| {
                let start = Instant::now();

                // Simulate concurrent container requests
                let handles: Vec<_> = (0..10)
                    .map(|i| {
                        let registry_clone = Arc::clone(&registry);
                        std::thread::spawn(move || {
                            for j in 0..10 {
                                let _device = registry_clone.get_device("nvidia.com/gpu=all");
                                let _container_id = format!("concurrent-{}-{}", i, j);
                            }
                        })
                    })
                    .collect();

                // Wait for all threads
                for handle in handles {
                    let _ = handle.join();
                }

                start.elapsed()
            },
            BatchSize::SmallInput,
        );
    });

    // Test 3: Memory pressure under high-frequency operations
    group.bench_function("memory_pressure_simulation", |b| {
        b.iter(|| {
            let start = Instant::now();

            let mut registries = Vec::with_capacity(1000);

            // Create many registries rapidly (simulating container pressure)
            for _ in 0..1000 {
                registries.push(CdiRegistry::new());
            }

            let creation_time = start.elapsed();

            // Force cleanup
            drop(registries);

            creation_time
        });
    });

    group.finish();
}

/// Benchmark container startup latency components
fn bench_container_startup_components(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("container_startup_components");

    // Skip if no NVIDIA driver available
    if !is_nvidia_driver_available() {
        eprintln!("Skipping container startup benchmarks - no NVIDIA driver detected");
        return;
    }

    // Test 1: GPU discovery latency
    group.bench_function("gpu_discovery_latency", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = Instant::now();
                let _gpus = discover_gpus().await.ok();
                start.elapsed()
            })
        });
    });

    // Test 2: CDI spec generation latency
    group.bench_function("cdi_spec_generation_latency", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = Instant::now();
                let _spec = generate_nvidia_cdi_spec().await.ok();
                start.elapsed()
            })
        });
    });

    // Test 3: Configuration loading latency
    group.bench_function("config_loading_latency", |b| {
        b.iter(|| {
            let start = Instant::now();
            let _config = Config::default();
            start.elapsed()
        });
    });

    // Test 4: Full container preparation pipeline
    group.bench_function("full_container_preparation", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = Instant::now();

                // Full pipeline simulation
                let _config = Config::default();
                let _gpus = discover_gpus().await.ok();
                let _spec = generate_nvidia_cdi_spec().await.ok();
                let _registry = CdiRegistry::new();

                start.elapsed()
            })
        });
    });

    group.finish();
}

/// Memory leak detection benchmarks
fn bench_memory_leak_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_leak_detection");

    // Configure for sustained load testing
    group.measurement_time(Duration::from_secs(30));

    // Test 1: Sustained registry creation/destruction
    group.bench_function("sustained_registry_lifecycle", |b| {
        b.iter(|| {
            let start = Instant::now();

            // Create and drop many registries to detect leaks
            for _ in 0..10000 {
                let registry = CdiRegistry::new();
                let _device = registry.get_device("nvidia.com/gpu=all");
                drop(registry);
            }

            start.elapsed()
        });
    });

    // Test 2: String allocation patterns (GPU device names)
    group.bench_function("string_allocation_patterns", |b| {
        b.iter(|| {
            let start = Instant::now();

            let mut device_names = Vec::new();
            for i in 0..10000 {
                device_names.push(format!("nvidia.com/gpu=gpu{}", i));
            }

            let creation_time = start.elapsed();

            // Force cleanup and measure
            drop(device_names);

            creation_time
        });
    });

    // Test 3: Nested resource allocation
    group.bench_function("nested_resource_allocation", |b| {
        b.iter(|| {
            let start = Instant::now();

            let registries: Vec<_> = (0..1000)
                .map(|_| {
                    let registry = CdiRegistry::new();
                    let _device_names: Vec<String> = (0..10)
                        .map(|i| format!("nvidia.com/gpu=gpu{}", i))
                        .collect();
                    registry
                })
                .collect();

            let creation_time = start.elapsed();
            drop(registries);

            creation_time
        });
    });

    group.finish();
}

/// Multi-GPU concurrent access validation
fn bench_multi_gpu_concurrent_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_gpu_concurrent");

    if !is_nvidia_driver_available() {
        eprintln!("Skipping multi-GPU benchmarks - no NVIDIA driver detected");
        return;
    }

    // Test 1: Concurrent access to different GPU devices
    group.bench_function("concurrent_multi_gpu_access", |b| {
        b.iter_batched(
            || Arc::new(CdiRegistry::new()),
            |registry| {
                let start = Instant::now();

                // Simulate concurrent access to multiple GPUs
                let handles: Vec<_> = (0..8)
                    .map(|gpu_id| {
                        let registry_clone = Arc::clone(&registry);
                        std::thread::spawn(move || {
                            for _ in 0..100 {
                                let device_name = format!("nvidia.com/gpu=gpu{}", gpu_id);
                                let _device = registry_clone.get_device(&device_name);
                            }
                        })
                    })
                    .collect();

                for handle in handles {
                    let _ = handle.join();
                }

                start.elapsed()
            },
            BatchSize::SmallInput,
        );
    });

    // Test 2: GPU contention simulation
    group.bench_function("gpu_contention_simulation", |b| {
        b.iter_batched(
            || Arc::new(CdiRegistry::new()),
            |registry| {
                let start = Instant::now();

                // Multiple threads trying to access the same GPU
                let handles: Vec<_> = (0..16)
                    .map(|_| {
                        let registry_clone = Arc::clone(&registry);
                        std::thread::spawn(move || {
                            for _ in 0..50 {
                                let _device = registry_clone.get_device("nvidia.com/gpu=all");
                            }
                        })
                    })
                    .collect();

                for handle in handles {
                    let _ = handle.join();
                }

                start.elapsed()
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_high_frequency_container_lifecycle,
    bench_container_startup_components,
    bench_memory_leak_detection,
    bench_multi_gpu_concurrent_access
);
criterion_main!(benches);
