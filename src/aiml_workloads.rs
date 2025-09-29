//! AI/ML Workload Optimization for nvbind
//!
//! This module provides comprehensive support for AI/ML workloads including
//! Ollama integration, model-specific GPU configurations, multi-model resource
//! scheduling, and LLM container optimization profiles.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Model size categories for optimal resource allocation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelSize {
    /// Small models (3B-7B parameters) - Phi-3, Llama 3.2, Mistral 7B
    Small,
    /// Medium models (13B-14B parameters) - Llama 3.1 13B, Vicuna 13B
    Medium,
    /// Large models (30B-40B parameters) - CodeLlama 34B, Falcon 40B
    Large,
    /// Extra large models (65B-70B parameters) - Llama 3.1 70B, Llama 2 70B
    XLarge,
    /// Massive models (175B+ parameters) - GPT-3 scale, Falcon 180B
    Massive,
}

impl ModelSize {
    /// Get recommended GPU memory allocation for model size
    pub fn recommended_gpu_memory_gb(&self) -> u32 {
        match self {
            ModelSize::Small => 6,      // 6GB for 7B models
            ModelSize::Medium => 12,     // 12GB for 13B models
            ModelSize::Large => 24,      // 24GB for 34B models
            ModelSize::XLarge => 48,     // 48GB for 70B models
            ModelSize::Massive => 80,    // 80GB+ for 175B+ models
        }
    }

    /// Get optimal CUDA cache size in MB
    pub fn cuda_cache_size_mb(&self) -> u32 {
        match self {
            ModelSize::Small => 1024,    // 1GB cache
            ModelSize::Medium => 2048,   // 2GB cache
            ModelSize::Large => 4096,    // 4GB cache
            ModelSize::XLarge => 8192,   // 8GB cache
            ModelSize::Massive => 16384, // 16GB cache
        }
    }

    /// Get recommended batch size for inference
    pub fn default_batch_size(&self) -> u32 {
        match self {
            ModelSize::Small => 8,
            ModelSize::Medium => 4,
            ModelSize::Large => 2,
            ModelSize::XLarge => 1,
            ModelSize::Massive => 1,
        }
    }

    /// Get maximum context length
    pub fn max_context_length(&self) -> u32 {
        match self {
            ModelSize::Small => 8192,
            ModelSize::Medium => 16384,
            ModelSize::Large => 32768,
            ModelSize::XLarge => 65536,
            ModelSize::Massive => 131072,
        }
    }
}

/// Precision modes for model inference
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Precision {
    /// Full precision (32-bit floating point)
    FP32,
    /// Half precision (16-bit floating point)
    FP16,
    /// Mixed precision (automatic)
    Mixed,
    /// 8-bit quantization
    INT8,
    /// 4-bit quantization
    INT4,
}

impl Precision {
    /// Get memory multiplier for precision mode
    pub fn memory_multiplier(&self) -> f32 {
        match self {
            Precision::FP32 => 1.0,
            Precision::FP16 => 0.5,
            Precision::Mixed => 0.6,
            Precision::INT8 => 0.25,
            Precision::INT4 => 0.125,
        }
    }

    /// Get performance multiplier for precision mode
    pub fn performance_multiplier(&self) -> f32 {
        match self {
            Precision::FP32 => 1.0,
            Precision::FP16 => 2.0,
            Precision::Mixed => 1.8,
            Precision::INT8 => 4.0,
            Precision::INT4 => 8.0,
        }
    }
}

/// Ollama-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub model_name: String,
    pub model_size: ModelSize,
    pub precision: Precision,
    pub batch_size: u32,
    pub context_length: u32,
    pub num_gpu_layers: Option<u32>,
    pub num_threads: Option<u32>,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub gpu_memory_limit: Option<String>,
    pub enable_flash_attention: bool,
}

impl OllamaConfig {
    /// Create optimized config for specific model
    pub fn for_model(model_name: &str) -> Result<Self> {
        let (model_size, precision) = Self::detect_model_characteristics(model_name)?;

        Ok(Self {
            model_name: model_name.to_string(),
            model_size: model_size.clone(),
            precision: precision.clone(),
            batch_size: model_size.default_batch_size(),
            context_length: model_size.max_context_length(),
            num_gpu_layers: None, // Auto-detect
            num_threads: None, // Use system default
            use_mmap: true,
            use_mlock: model_size == ModelSize::Small || model_size == ModelSize::Medium,
            gpu_memory_limit: Some(format!("{}GB", model_size.recommended_gpu_memory_gb())),
            enable_flash_attention: true,
        })
    }

    /// Detect model characteristics from name
    fn detect_model_characteristics(model_name: &str) -> Result<(ModelSize, Precision)> {
        let name_lower = model_name.to_lowercase();

        // Detect model size
        let model_size = if name_lower.contains("3b") || name_lower.contains("phi") {
            ModelSize::Small
        } else if name_lower.contains("7b") || name_lower.contains("mistral") {
            ModelSize::Small
        } else if name_lower.contains("13b") || name_lower.contains("vicuna") {
            ModelSize::Medium
        } else if name_lower.contains("34b") || name_lower.contains("codellama") {
            ModelSize::Large
        } else if name_lower.contains("70b") {
            ModelSize::XLarge
        } else if name_lower.contains("175b") || name_lower.contains("180b") {
            ModelSize::Massive
        } else {
            ModelSize::Small // Default
        };

        // Detect precision preference
        let precision = if name_lower.contains("q4") || name_lower.contains("int4") {
            Precision::INT4
        } else if name_lower.contains("q8") || name_lower.contains("int8") {
            Precision::INT8
        } else if name_lower.contains("fp16") {
            Precision::FP16
        } else if model_size == ModelSize::XLarge || model_size == ModelSize::Massive {
            Precision::Mixed // Use mixed precision for large models
        } else {
            Precision::FP16 // Default to FP16
        };

        Ok((model_size, precision))
    }

    /// Generate environment variables for Ollama
    pub fn to_env_vars(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();

        // Ollama-specific environment variables
        env.insert("OLLAMA_NUM_PARALLEL".to_string(), self.batch_size.to_string());
        env.insert("OLLAMA_MAX_LOADED_MODELS".to_string(), "1".to_string());
        env.insert("OLLAMA_HOST".to_string(), "0.0.0.0:11434".to_string());

        // GPU memory configuration
        if let Some(ref limit) = self.gpu_memory_limit {
            env.insert("OLLAMA_GPU_MEMORY".to_string(), limit.clone());
        }

        // Context length
        env.insert("OLLAMA_NUM_CTX".to_string(), self.context_length.to_string());

        // GPU layers (for llama.cpp backend)
        if let Some(layers) = self.num_gpu_layers {
            env.insert("OLLAMA_GPU_LAYERS".to_string(), layers.to_string());
        }

        // Thread configuration
        if let Some(threads) = self.num_threads {
            env.insert("OLLAMA_NUM_THREAD".to_string(), threads.to_string());
        }

        // Memory mapping
        env.insert("OLLAMA_USE_MMAP".to_string(), self.use_mmap.to_string());
        env.insert("OLLAMA_USE_MLOCK".to_string(), self.use_mlock.to_string());

        // CUDA optimizations
        env.insert("CUDA_VISIBLE_DEVICES".to_string(), "all".to_string());
        env.insert("CUDA_CACHE_MAXSIZE".to_string(),
                  (self.model_size.cuda_cache_size_mb() * 1024 * 1024).to_string());

        // Precision settings
        match self.precision {
            Precision::FP32 => {
                env.insert("OLLAMA_PRECISION".to_string(), "fp32".to_string());
            }
            Precision::FP16 => {
                env.insert("OLLAMA_PRECISION".to_string(), "fp16".to_string());
                env.insert("NVIDIA_TF32_OVERRIDE".to_string(), "0".to_string());
            }
            Precision::Mixed => {
                env.insert("OLLAMA_PRECISION".to_string(), "mixed".to_string());
                env.insert("NVIDIA_TF32_OVERRIDE".to_string(), "1".to_string());
            }
            Precision::INT8 => {
                env.insert("OLLAMA_PRECISION".to_string(), "int8".to_string());
            }
            Precision::INT4 => {
                env.insert("OLLAMA_PRECISION".to_string(), "int4".to_string());
            }
        }

        // Flash attention optimization
        if self.enable_flash_attention {
            env.insert("OLLAMA_FLASH_ATTENTION".to_string(), "1".to_string());
        }

        // Additional optimizations for large models
        if self.model_size == ModelSize::XLarge || self.model_size == ModelSize::Massive {
            env.insert("PYTORCH_CUDA_ALLOC_CONF".to_string(),
                      "max_split_size_mb:512".to_string());
            env.insert("CUDA_LAUNCH_BLOCKING".to_string(), "0".to_string());
        }

        env
    }

    /// Generate CDI device specification for Ollama
    pub fn generate_cdi_devices(&self) -> Vec<String> {
        let mut devices = vec![
            "nvidia.com/gpu=all".to_string(),
        ];

        // Add MIG devices for massive models
        if self.model_size == ModelSize::Massive {
            devices.push("nvidia.com/mig=all".to_string());
        }

        devices
    }
}

/// Multi-model resource scheduler
pub struct MultiModelScheduler {
    active_models: Arc<RwLock<HashMap<String, ModelAllocation>>>,
    total_gpu_memory_gb: u32,
    reserved_memory_gb: u32,
}

/// Model allocation information
#[derive(Debug, Clone)]
struct ModelAllocation {
    model_name: String,
    config: OllamaConfig,
    allocated_memory_gb: u32,
    gpu_devices: Vec<u32>,
    container_id: Option<String>,
    last_accessed: std::time::Instant,
}

impl MultiModelScheduler {
    /// Create new multi-model scheduler
    pub fn new(total_gpu_memory_gb: u32) -> Self {
        Self {
            active_models: Arc::new(RwLock::new(HashMap::new())),
            total_gpu_memory_gb,
            reserved_memory_gb: 2, // Reserve 2GB for system
        }
    }

    /// Schedule a new model for loading
    pub async fn schedule_model(&self, config: &OllamaConfig) -> Result<Vec<u32>> {
        let mut allocations = self.active_models.write().await;

        let required_memory = config.model_size.recommended_gpu_memory_gb();
        let available_memory = self.get_available_memory(&allocations);

        if required_memory > available_memory {
            // Try to evict least recently used models
            self.evict_lru_models(&mut allocations, required_memory - available_memory).await?;
        }

        // Allocate GPU devices for model
        let gpu_devices = self.allocate_gpu_devices(required_memory)?;

        let allocation = ModelAllocation {
            model_name: config.model_name.clone(),
            config: config.clone(),
            allocated_memory_gb: required_memory,
            gpu_devices: gpu_devices.clone(),
            container_id: None,
            last_accessed: std::time::Instant::now(),
        };

        allocations.insert(config.model_name.clone(), allocation);

        info!("Scheduled model {} with {}GB memory on GPUs {:?}",
              config.model_name, required_memory, gpu_devices);

        Ok(gpu_devices)
    }

    /// Get available GPU memory
    fn get_available_memory(&self, allocations: &HashMap<String, ModelAllocation>) -> u32 {
        let used_memory: u32 = allocations.values()
            .map(|a| a.allocated_memory_gb)
            .sum();

        self.total_gpu_memory_gb
            .saturating_sub(used_memory)
            .saturating_sub(self.reserved_memory_gb)
    }

    /// Evict least recently used models
    async fn evict_lru_models(
        &self,
        allocations: &mut HashMap<String, ModelAllocation>,
        required_memory_gb: u32
    ) -> Result<()> {
        let mut freed_memory = 0u32;

        // Sort models by last accessed time
        let mut models: Vec<_> = allocations.values().cloned().collect();
        models.sort_by_key(|m| m.last_accessed);

        for model in models {
            if freed_memory >= required_memory_gb {
                break;
            }

            info!("Evicting model {} to free {}GB", model.model_name, model.allocated_memory_gb);

            // Stop container if running
            if let Some(container_id) = &model.container_id {
                self.stop_model_container(container_id).await?;
            }

            freed_memory += model.allocated_memory_gb;
            allocations.remove(&model.model_name);
        }

        if freed_memory < required_memory_gb {
            return Err(anyhow::anyhow!(
                "Cannot free enough memory. Required: {}GB, Freed: {}GB",
                required_memory_gb, freed_memory
            ));
        }

        Ok(())
    }

    /// Allocate GPU devices for model
    fn allocate_gpu_devices(&self, memory_gb: u32) -> Result<Vec<u32>> {
        // Simple allocation strategy: use first available GPUs
        // In production, this would consider GPU topology and NUMA
        let gpus_needed = (memory_gb + 23) / 24; // Assume 24GB per GPU

        Ok((0..gpus_needed).collect())
    }

    /// Stop model container
    async fn stop_model_container(&self, container_id: &str) -> Result<()> {
        // This would interact with the container runtime
        info!("Stopping container {}", container_id);
        Ok(())
    }

    /// Update model access time
    pub async fn touch_model(&self, model_name: &str) {
        let mut allocations = self.active_models.write().await;
        if let Some(allocation) = allocations.get_mut(model_name) {
            allocation.last_accessed = std::time::Instant::now();
        }
    }

    /// Get current allocations
    pub async fn get_allocations(&self) -> HashMap<String, (String, u32)> {
        let allocations = self.active_models.read().await;
        allocations.iter()
            .map(|(k, v)| (k.clone(), (v.model_name.clone(), v.allocated_memory_gb)))
            .collect()
    }
}

/// LLM container optimization profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMOptimizationProfile {
    pub name: String,
    pub description: String,
    pub model_patterns: Vec<String>,
    pub cpu_optimization: CpuOptimization,
    pub memory_optimization: MemoryOptimization,
    pub network_optimization: NetworkOptimization,
    pub storage_optimization: StorageOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuOptimization {
    pub cpu_shares: u32,
    pub cpu_quota: Option<u32>,
    pub cpu_period: Option<u32>,
    pub cpuset_cpus: Option<String>,
    pub numa_node: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    pub memory_limit: String,
    pub memory_swap: Option<String>,
    pub memory_swappiness: Option<u32>,
    pub kernel_memory: Option<String>,
    pub oom_kill_disable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOptimization {
    pub network_mode: String,
    pub enable_ipv6: bool,
    pub dns_servers: Vec<String>,
    pub extra_hosts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageOptimization {
    pub storage_driver: String,
    pub storage_opts: HashMap<String, String>,
    pub tmpfs_mounts: Vec<String>,
    pub volume_driver: Option<String>,
}

impl LLMOptimizationProfile {
    /// Create profile for inference workload
    pub fn inference_profile() -> Self {
        Self {
            name: "inference".to_string(),
            description: "Optimized for model inference".to_string(),
            model_patterns: vec!["*".to_string()],
            cpu_optimization: CpuOptimization {
                cpu_shares: 2048,
                cpu_quota: None,
                cpu_period: None,
                cpuset_cpus: None,
                numa_node: Some(0),
            },
            memory_optimization: MemoryOptimization {
                memory_limit: "64g".to_string(),
                memory_swap: Some("64g".to_string()),
                memory_swappiness: Some(10),
                kernel_memory: None,
                oom_kill_disable: true,
            },
            network_optimization: NetworkOptimization {
                network_mode: "host".to_string(),
                enable_ipv6: false,
                dns_servers: vec!["8.8.8.8".to_string()],
                extra_hosts: vec![],
            },
            storage_optimization: StorageOptimization {
                storage_driver: "overlay2".to_string(),
                storage_opts: HashMap::new(),
                tmpfs_mounts: vec!["/tmp".to_string()],
                volume_driver: None,
            },
        }
    }

    /// Create profile for training workload
    pub fn training_profile() -> Self {
        Self {
            name: "training".to_string(),
            description: "Optimized for model training".to_string(),
            model_patterns: vec!["*train*".to_string()],
            cpu_optimization: CpuOptimization {
                cpu_shares: 4096,
                cpu_quota: None,
                cpu_period: None,
                cpuset_cpus: None,
                numa_node: None,
            },
            memory_optimization: MemoryOptimization {
                memory_limit: "128g".to_string(),
                memory_swap: None,
                memory_swappiness: Some(0),
                kernel_memory: None,
                oom_kill_disable: false,
            },
            network_optimization: NetworkOptimization {
                network_mode: "bridge".to_string(),
                enable_ipv6: true,
                dns_servers: vec![],
                extra_hosts: vec![],
            },
            storage_optimization: StorageOptimization {
                storage_driver: "overlay2".to_string(),
                storage_opts: HashMap::from([
                    ("size".to_string(), "100G".to_string()),
                ]),
                tmpfs_mounts: vec!["/tmp".to_string(), "/cache".to_string()],
                volume_driver: Some("local".to_string()),
            },
        }
    }

    /// Generate container runtime arguments
    pub fn to_container_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        // CPU optimization
        args.push(format!("--cpu-shares={}", self.cpu_optimization.cpu_shares));
        if let Some(quota) = self.cpu_optimization.cpu_quota {
            args.push(format!("--cpu-quota={}", quota));
        }
        if let Some(cpus) = &self.cpu_optimization.cpuset_cpus {
            args.push(format!("--cpuset-cpus={}", cpus));
        }

        // Memory optimization
        args.push(format!("--memory={}", self.memory_optimization.memory_limit));
        if let Some(swap) = &self.memory_optimization.memory_swap {
            args.push(format!("--memory-swap={}", swap));
        }
        if let Some(swappiness) = self.memory_optimization.memory_swappiness {
            args.push(format!("--memory-swappiness={}", swappiness));
        }
        if self.memory_optimization.oom_kill_disable {
            args.push("--oom-kill-disable".to_string());
        }

        // Network optimization
        args.push(format!("--network={}", self.network_optimization.network_mode));
        if self.network_optimization.enable_ipv6 {
            args.push("--ipv6".to_string());
        }

        // Storage optimization
        args.push(format!("--storage-driver={}", self.storage_optimization.storage_driver));
        for mount in &self.storage_optimization.tmpfs_mounts {
            args.push(format!("--tmpfs={}", mount));
        }

        args
    }
}

/// Jarvis backend GPU acceleration support
pub struct JarvisGpuBackend {
    config: JarvisConfig,
    scheduler: Arc<MultiModelScheduler>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JarvisConfig {
    pub api_endpoint: String,
    pub model_backend: String,
    pub gpu_acceleration: bool,
    pub batch_processing: bool,
    pub cache_responses: bool,
    pub max_batch_size: u32,
}

impl JarvisGpuBackend {
    /// Create new Jarvis GPU backend
    pub fn new(config: JarvisConfig, total_gpu_memory_gb: u32) -> Self {
        Self {
            config,
            scheduler: Arc::new(MultiModelScheduler::new(total_gpu_memory_gb)),
        }
    }

    /// Accelerate Jarvis inference with GPU
    pub async fn accelerate_inference(
        &self,
        prompt: &str,
        model_name: &str,
    ) -> Result<String> {
        if !self.config.gpu_acceleration {
            return Err(anyhow::anyhow!("GPU acceleration not enabled"));
        }

        // Create optimized Ollama config for model
        let ollama_config = OllamaConfig::for_model(model_name)?;

        // Schedule model with multi-model scheduler
        let gpu_devices = self.scheduler.schedule_model(&ollama_config).await?;

        info!("Accelerating Jarvis inference for {} on GPUs {:?}",
              model_name, gpu_devices);

        // Update access time
        self.scheduler.touch_model(model_name).await;

        // In production, this would actually run inference
        Ok(format!("Accelerated response for: {}", prompt))
    }

    /// Get GPU acceleration status
    pub async fn get_acceleration_status(&self) -> HashMap<String, (String, u32)> {
        self.scheduler.get_allocations().await
    }
}

/// Pre-configured AI/ML profiles
pub fn get_predefined_profiles() -> HashMap<String, OllamaConfig> {
    let mut profiles = HashMap::new();

    // Phi-3 Mini (3.8B)
    profiles.insert("phi-3-mini".to_string(), OllamaConfig {
        model_name: "phi3:mini".to_string(),
        model_size: ModelSize::Small,
        precision: Precision::FP16,
        batch_size: 16,
        context_length: 4096,
        num_gpu_layers: Some(32),
        num_threads: Some(8),
        use_mmap: true,
        use_mlock: true,
        gpu_memory_limit: Some("4GB".to_string()),
        enable_flash_attention: true,
    });

    // Llama 3.2 (7B)
    profiles.insert("llama-3.2-7b".to_string(), OllamaConfig {
        model_name: "llama3.2:7b".to_string(),
        model_size: ModelSize::Small,
        precision: Precision::FP16,
        batch_size: 8,
        context_length: 8192,
        num_gpu_layers: Some(35),
        num_threads: Some(8),
        use_mmap: true,
        use_mlock: true,
        gpu_memory_limit: Some("6GB".to_string()),
        enable_flash_attention: true,
    });

    // Mistral 7B
    profiles.insert("mistral-7b".to_string(), OllamaConfig {
        model_name: "mistral:7b".to_string(),
        model_size: ModelSize::Small,
        precision: Precision::FP16,
        batch_size: 8,
        context_length: 8192,
        num_gpu_layers: Some(35),
        num_threads: Some(8),
        use_mmap: true,
        use_mlock: true,
        gpu_memory_limit: Some("6GB".to_string()),
        enable_flash_attention: true,
    });

    // Llama 3.1 13B
    profiles.insert("llama-3.1-13b".to_string(), OllamaConfig {
        model_name: "llama3.1:13b".to_string(),
        model_size: ModelSize::Medium,
        precision: Precision::Mixed,
        batch_size: 4,
        context_length: 16384,
        num_gpu_layers: Some(43),
        num_threads: Some(12),
        use_mmap: true,
        use_mlock: false,
        gpu_memory_limit: Some("12GB".to_string()),
        enable_flash_attention: true,
    });

    // CodeLlama 34B
    profiles.insert("codellama-34b".to_string(), OllamaConfig {
        model_name: "codellama:34b".to_string(),
        model_size: ModelSize::Large,
        precision: Precision::Mixed,
        batch_size: 2,
        context_length: 32768,
        num_gpu_layers: Some(60),
        num_threads: Some(16),
        use_mmap: true,
        use_mlock: false,
        gpu_memory_limit: Some("24GB".to_string()),
        enable_flash_attention: true,
    });

    // Llama 3.1 70B
    profiles.insert("llama-3.1-70b".to_string(), OllamaConfig {
        model_name: "llama3.1:70b".to_string(),
        model_size: ModelSize::XLarge,
        precision: Precision::INT8,
        batch_size: 1,
        context_length: 65536,
        num_gpu_layers: Some(80),
        num_threads: Some(24),
        use_mmap: true,
        use_mlock: false,
        gpu_memory_limit: Some("48GB".to_string()),
        enable_flash_attention: true,
    });

    // Mixtral 8x7B (MoE)
    profiles.insert("mixtral-8x7b".to_string(), OllamaConfig {
        model_name: "mixtral:8x7b".to_string(),
        model_size: ModelSize::Large,
        precision: Precision::Mixed,
        batch_size: 2,
        context_length: 32768,
        num_gpu_layers: Some(48),
        num_threads: Some(16),
        use_mmap: true,
        use_mlock: false,
        gpu_memory_limit: Some("32GB".to_string()),
        enable_flash_attention: true,
    });

    profiles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_size_detection() {
        let config = OllamaConfig::for_model("llama3.1:70b").unwrap();
        assert_eq!(config.model_size, ModelSize::XLarge);
        assert_eq!(config.model_size.recommended_gpu_memory_gb(), 48);
    }

    #[test]
    fn test_ollama_env_generation() {
        let config = OllamaConfig::for_model("phi3:mini").unwrap();
        let env = config.to_env_vars();

        assert!(env.contains_key("OLLAMA_NUM_PARALLEL"));
        assert!(env.contains_key("OLLAMA_GPU_MEMORY"));
        assert!(env.contains_key("CUDA_VISIBLE_DEVICES"));
    }

    #[tokio::test]
    async fn test_multi_model_scheduling() {
        let scheduler = MultiModelScheduler::new(48); // 48GB total

        let config1 = OllamaConfig::for_model("llama3.2:7b").unwrap();
        let config2 = OllamaConfig::for_model("mistral:7b").unwrap();

        let devices1 = scheduler.schedule_model(&config1).await.unwrap();
        let devices2 = scheduler.schedule_model(&config2).await.unwrap();

        assert!(!devices1.is_empty());
        assert!(!devices2.is_empty());

        let allocations = scheduler.get_allocations().await;
        assert_eq!(allocations.len(), 2);
    }

    #[test]
    fn test_llm_optimization_profiles() {
        let inference = LLMOptimizationProfile::inference_profile();
        let training = LLMOptimizationProfile::training_profile();

        assert_eq!(inference.cpu_optimization.cpu_shares, 2048);
        assert_eq!(training.cpu_optimization.cpu_shares, 4096);

        let args = inference.to_container_args();
        assert!(args.contains(&"--cpu-shares=2048".to_string()));
        assert!(args.contains(&"--memory=64g".to_string()));
    }
}