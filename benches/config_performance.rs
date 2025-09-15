use criterion::{Criterion, criterion_group, criterion_main};
use nvbind::config::Config;
use std::io::Write;
use tempfile::NamedTempFile;

fn bench_config_creation(c: &mut Criterion) {
    c.bench_function("config_default", |b| {
        b.iter(|| Config::default());
    });
}

fn bench_config_serialization(c: &mut Criterion) {
    let config = Config::default();

    c.bench_function("config_to_toml", |b| {
        b.iter(|| toml::to_string_pretty(&config).unwrap());
    });
}

fn bench_config_deserialization(c: &mut Criterion) {
    let config = Config::default();
    let toml_str = toml::to_string_pretty(&config).unwrap();

    c.bench_function("config_from_toml", |b| {
        b.iter(|| {
            let _: Config = toml::from_str(&toml_str).unwrap();
        });
    });
}

fn bench_config_file_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_file_ops");

    let config = Config::default();

    group.bench_function("save_to_file", |b| {
        b.iter(|| {
            let temp_file = NamedTempFile::new().unwrap();
            config.save_to_file(temp_file.path()).unwrap();
        });
    });

    group.bench_function("load_from_file", |b| {
        // Setup: create a file with config
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = toml::to_string_pretty(&config).unwrap();
        temp_file.write_all(content.as_bytes()).unwrap();

        b.iter(|| {
            Config::load_from_file(temp_file.path()).unwrap();
        });
    });

    group.finish();
}

fn bench_config_methods(c: &mut Criterion) {
    let config = Config::default();

    c.bench_function("get_runtime_command", |b| {
        b.iter(|| config.get_runtime_command(Some("docker")));
    });

    c.bench_function("get_gpu_selection", |b| {
        b.iter(|| config.get_gpu_selection(Some("all")));
    });

    c.bench_function("get_all_devices", |b| {
        b.iter(|| config.get_all_devices());
    });
}

criterion_group!(
    benches,
    bench_config_creation,
    bench_config_serialization,
    bench_config_deserialization,
    bench_config_file_operations,
    bench_config_methods
);
criterion_main!(benches);
