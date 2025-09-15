use criterion::{Criterion, criterion_group, criterion_main};
use nvbind::gpu::{
    check_nvidia_requirements, discover_gpus, get_driver_info, is_nvidia_driver_available,
};

fn bench_gpu_discovery(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("discover_gpus", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = discover_gpus().await;
            });
        });
    });
}

fn bench_driver_info(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("get_driver_info", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = get_driver_info().await;
            });
        });
    });
}

fn bench_driver_availability(c: &mut Criterion) {
    c.bench_function("is_nvidia_driver_available", |b| {
        b.iter(|| {
            let _ = is_nvidia_driver_available();
        });
    });
}

fn bench_requirements_check(c: &mut Criterion) {
    c.bench_function("check_nvidia_requirements", |b| {
        b.iter(|| {
            let _ = check_nvidia_requirements();
        });
    });
}

criterion_group!(
    benches,
    bench_gpu_discovery,
    bench_driver_info,
    bench_driver_availability,
    bench_requirements_check
);
criterion_main!(benches);
