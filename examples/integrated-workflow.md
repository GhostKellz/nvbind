# nvbind Integrated Workflow Example
## Cloud + ML Frameworks + Bolt Runtime

This guide demonstrates using nvbind's cloud provider optimizations, ML framework integrations, and Bolt runtime together.

## 🌩️ Multi-Cloud GPU Deployment

### 1. AWS GPU Instance with PyTorch
```rust
use nvbind::cloud::{CloudManager, CloudConfig, CloudWorkload, ResourceRequirements};
use nvbind::pytorch_optimization::PyTorchCudaManager;

// Initialize cloud manager with AWS
let cloud_config = CloudConfig::from_file("examples/cloud-config.toml")?;
let cloud_manager = CloudManager::new(cloud_config);
cloud_manager.initialize().await?;

// Define PyTorch training workload
let workload = CloudWorkload {
    id: Uuid::new_v4(),
    name: "distributed-training".to_string(),
    requirements: ResourceRequirements {
        gpu_count: 4,
        gpu_type_preference: Some(GpuType::A100),
        vcpus: 96,
        memory_gb: 1152,
        storage_gb: 1000,
        network_bandwidth_mbps: Some(100000),
    },
    // Schedule on AWS p4d.24xlarge
};

// Schedule workload - CloudManager picks best provider (AWS/GCP/Azure)
let result = cloud_manager.schedule_workload(workload).await?;
println!("Deployed on {:?} in region {}", result.provider, result.region);
```

### 2. GCP A100 Instance with TensorFlow
```rust
use nvbind::tensorflow_optimization::TensorFlowGpuManager;

let tf_manager = TensorFlowGpuManager::new();

// Create TensorFlow serving session on GCP
let session_id = tf_manager.create_session(
    SessionType::Serving,
    Some(model_info),
    resource_limits,
).await?;

// Generate TensorFlow config for GCP A100
let tf_config = tf_manager.generate_tf_config(&session)?;
```

### 3. Azure H100 Instance with MLflow Tracking
```rust
use nvbind::mlflow_integration::MlflowIntegrationManager;

let mlflow_manager = MlflowIntegrationManager::new(mlflow_config);

// Track experiment on Azure VM
let experiment_id = mlflow_manager.create_experiment(
    "azure-h100-training",
    "Training on Azure H100 GPUs"
).await?;

mlflow_manager.log_gpu_metrics(&experiment_id).await?;
```

## 🎮 Bolt Runtime Integration

### Gaming Container with GPU
```rust
use nvbind::bolt::{NvbindGpuManager, BoltGpuCompatibility};

let gpu_manager = NvbindGpuManager::with_defaults();

// Check Bolt GPU compatibility
let compatibility = gpu_manager.check_bolt_gpu_compatibility().await?;
println!("GPUs available: {}", compatibility.gpu_count);
println!("NVIDIA Open driver: {}", compatibility.nvidia_open_driver);
println!("WSL2 mode: {}", compatibility.wsl2_mode);

// Generate Bolt gaming CDI spec
let gaming_cdi = gpu_manager.generate_gaming_cdi_spec().await?;

// Run gaming container with Bolt
gpu_manager.run_with_bolt_runtime(
    "steam-game:latest".to_string(),
    vec!["--game-id".to_string(), "730".to_string()],
    Some("gpu0".to_string()),
).await?;
```

### AI/ML Container with Bolt
```rust
// Generate AI/ML optimized CDI spec
let aiml_cdi = gpu_manager.generate_aiml_cdi_spec().await?;

// Run Ollama with Bolt
gpu_manager.run_with_bolt_runtime(
    "ollama/ollama:latest".to_string(),
    vec!["run".to_string(), "llama2:70b".to_string()],
    Some("all".to_string()),
).await?;
```

## 🔄 Hybrid Cloud + Bolt Workflow

### Example: Train on Cloud, Deploy with Bolt

```rust
// 1. Train on AWS P4 (8x A100)
let cloud_training = cloud_manager.schedule_workload(CloudWorkload {
    name: "model-training".to_string(),
    requirements: ResourceRequirements {
        gpu_count: 8,
        gpu_type_preference: Some(GpuType::A100),
        //...
    },
    //...
}).await?;

// 2. Track with MLflow
mlflow_manager.log_run(
    &experiment_id,
    &cloud_training.instance_id,
    gpu_metrics
).await?;

// 3. Deploy locally with Bolt
let local_inference = gpu_manager.run_with_bolt_runtime(
    "custom-model:latest".to_string(),
    vec!["serve".to_string(), "--port".to_string(), "8080".to_string()],
    Some("gpu0".to_string()),
).await?;
```

## 📊 Cost Optimization Across Clouds

```rust
// Get cost analysis across all providers
let cost_analysis = cloud_manager.get_cost_analysis(time_range).await?;

println!("Total cloud cost: ${:.2}", cost_analysis.total_cost);
println!("Spot instance savings: ${:.2}", cost_analysis.spot_savings);

for (provider, cost) in cost_analysis.provider_breakdown {
    println!("{:?}: ${:.2}/month", provider, cost);
}

// Get recommendations
for rec in cost_analysis.recommendations {
    println!("{}: save ${:.2}", rec.description, rec.potential_savings);
}
```

## 🎯 Complete Integration Example

```bash
# 1. Configure cloud providers
nvbind cloud configure --providers aws,gcp,azure

# 2. Test cloud connectivity
nvbind cloud test --provider aws

# 3. Deploy ML training on best available cloud instance
nvbind cloud deploy \
  --workload pytorch-training \
  --gpus 4 \
  --gpu-type A100 \
  --strategy cost-optimized

# 4. Run local inference with Bolt
nvbind run \
  --runtime bolt \
  --gpu all \
  --image model-serving:latest

# 5. Monitor costs
nvbind cloud costs --period 30d
```

## 📋 Configuration Files

### `~/.nvbind/config.toml`
```toml
[cloud]
enabled = true
default_provider = "AWS"

[bolt]
enabled = true
capsule.snapshot_gpu_state = true
capsule.isolation_level = "exclusive"

[ml_frameworks]
tensorflow.enabled = true
pytorch.enabled = true
mlflow.enabled = true
```

## 🚀 Quick Start

```bash
# Enable all features
cargo build --release --all-features

# Run with cloud + Bolt
./target/release/nvbind cloud deploy \
  --workload ai-training \
  --bolt-capsule \
  --gpus 8 \
  --provider auto
```

## 💡 Best Practices

1. **Cloud Selection**: Use `strategy = "cost-optimized"` for batch jobs, `"latency-optimized"` for interactive
2. **Bolt Isolation**: Use `exclusive` for gaming, `shared` for AI inference
3. **Spot Instances**: Enable with `fallback_to_ondemand = true` for resilience
4. **GPU Types**: Match workload - A100 for training, T4 for inference
5. **MLflow Tracking**: Always enable for experiment reproducibility

## 📈 Performance Targets

- **Cloud Scheduling**: < 2 minutes to launch GPU instance
- **Bolt GPU Passthrough**: < 100ms overhead vs bare metal
- **TensorFlow/PyTorch**: 95%+ GPU utilization
- **Multi-cloud failover**: < 5 minutes recovery time
