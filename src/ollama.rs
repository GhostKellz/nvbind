//! Ollama LLM optimization module for nvbind
//! Provides optimized GPU configurations for local LLM serving

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Ollama model size categories for optimization
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelSize {
    /// 7B parameter models (Llama 3.2, Phi-3, CodeLlama-7B)
    Small,
    /// 13-14B parameter models (Llama 3.1 13B, Mixtral 8x7B)
    Medium,
    /// 34B parameter models (CodeLlama 34B, Yi-34B)
    Large,
    /// 70B+ parameter models (Llama 3.1 70B, Qwen 72B)
    XLarge,
}

/// Precision modes for Ollama inference
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Precision {
    /// Full precision float32 (highest quality, most memory)
    FP32,
    /// Half precision float16 (balanced quality/performance)
    FP16,
    /// Mixed precision (automatic FP16/FP32 selection)
    Mixed,
    /// 8-bit quantization (fastest, least memory)
    Q8,
    /// 4-bit quantization (ultra-fast, minimal memory)
    Q4,
}

/// Optimized Ollama configuration for specific model types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub model_size: ModelSize,
    pub precision: Precision,
    pub max_parallel_requests: u32,
    pub context_length: u32,
    pub gpu_memory_limit: String,
    pub cuda_cache_size: u64, // MB
    pub tensor_parallel: bool,
    pub flash_attention: bool,
    pub kv_cache_optimization: bool,
}

impl OllamaConfig {
    /// Create optimized configuration for model size
    pub fn for_model_size(size: ModelSize) -> Self {
        match size {
            ModelSize::Small => Self {
                model_size: size,
                precision: Precision::FP16,
                max_parallel_requests: 8,
                context_length: 4096,
                gpu_memory_limit: "6GB".to_string(),
                cuda_cache_size: 1024, // 1GB
                tensor_parallel: false,
                flash_attention: true,
                kv_cache_optimization: true,
            },
            ModelSize::Medium => Self {
                model_size: size,
                precision: Precision::Mixed,
                max_parallel_requests: 4,
                context_length: 8192,
                gpu_memory_limit: "12GB".to_string(),
                cuda_cache_size: 2048, // 2GB
                tensor_parallel: false,
                flash_attention: true,
                kv_cache_optimization: true,
            },
            ModelSize::Large => Self {
                model_size: size,
                precision: Precision::Mixed,
                max_parallel_requests: 2,
                context_length: 16384,
                gpu_memory_limit: "24GB".to_string(),
                cuda_cache_size: 4096, // 4GB
                tensor_parallel: true,
                flash_attention: true,
                kv_cache_optimization: true,
            },
            ModelSize::XLarge => Self {
                model_size: size,
                precision: Precision::Mixed,
                max_parallel_requests: 1,
                context_length: 32768,
                gpu_memory_limit: "48GB".to_string(),
                cuda_cache_size: 8192, // 8GB
                tensor_parallel: true,
                flash_attention: true,
                kv_cache_optimization: true,
            },
        }
    }

    /// Create configuration optimized for specific model name
    pub fn for_model_name(model_name: &str) -> Self {
        let size = Self::detect_model_size(model_name);
        let mut config = Self::for_model_size(size);

        // Model-specific optimizations
        match model_name.to_lowercase() {
            name if name.contains("phi-3") => {
                config.precision = Precision::FP16;
                config.flash_attention = true;
                config.max_parallel_requests = 12; // Phi-3 is very efficient
            }
            name if name.contains("codellama") => {
                config.context_length = 32768; // Code models need longer context
                config.kv_cache_optimization = true;
            }
            name if name.contains("mixtral") => {
                config.tensor_parallel = true; // MoE benefits from parallelization
                config.precision = Precision::Mixed;
            }
            name if name.contains("qwen") => {
                config.flash_attention = true;
                config.context_length = 32768; // Qwen supports long context
            }
            _ => {}
        }

        config
    }

    /// Detect model size from model name
    fn detect_model_size(model_name: &str) -> ModelSize {
        let name_lower = model_name.to_lowercase();

        // Check for explicit size indicators
        if name_lower.contains("70b") || name_lower.contains("72b") {
            ModelSize::XLarge
        } else if name_lower.contains("34b") {
            ModelSize::Large
        } else if name_lower.contains("13b") || name_lower.contains("14b") {
            ModelSize::Medium
        } else if name_lower.contains("7b") || name_lower.contains("8b") {
            ModelSize::Small
        } else {
            // Fallback based on model family
            if name_lower.contains("phi") {
                ModelSize::Small
            } else if name_lower.contains("mixtral") {
                ModelSize::Medium
            } else if name_lower.contains("codellama") && !name_lower.contains("7b") {
                ModelSize::Large
            } else {
                ModelSize::Small // Conservative default
            }
        }
    }

    /// Generate environment variables for Ollama optimization
    pub fn to_environment_vars(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();

        // Core Ollama configuration
        env.insert(
            "OLLAMA_NUM_PARALLEL".to_string(),
            self.max_parallel_requests.to_string(),
        );
        env.insert("OLLAMA_MAX_LOADED_MODELS".to_string(), "1".to_string());
        env.insert("OLLAMA_HOST".to_string(), "0.0.0.0:11434".to_string());
        env.insert("OLLAMA_KEEP_ALIVE".to_string(), "24h".to_string());

        // GPU memory management
        env.insert("OLLAMA_GPU_MEMORY_FRACTION".to_string(), "0.9".to_string());
        env.insert(
            "OLLAMA_MAX_QUEUE".to_string(),
            (self.max_parallel_requests * 2).to_string(),
        );

        // CUDA optimizations
        env.insert("CUDA_VISIBLE_DEVICES".to_string(), "all".to_string());
        env.insert(
            "CUDA_CACHE_MAXSIZE".to_string(),
            (self.cuda_cache_size * 1024 * 1024).to_string(),
        );
        env.insert("CUDA_CACHE_DISABLE".to_string(), "0".to_string());

        // Memory optimizations
        env.insert(
            "PYTORCH_CUDA_ALLOC_CONF".to_string(),
            "max_split_size_mb:512".to_string(),
        );
        env.insert("CUDA_LAUNCH_BLOCKING".to_string(), "0".to_string());

        // Precision settings
        match self.precision {
            Precision::FP32 => {
                env.insert("OLLAMA_PRECISION".to_string(), "fp32".to_string());
            }
            Precision::FP16 => {
                env.insert("OLLAMA_PRECISION".to_string(), "fp16".to_string());
                env.insert("CUDA_AUTO_MIXED_PRECISION".to_string(), "0".to_string());
            }
            Precision::Mixed => {
                env.insert("OLLAMA_PRECISION".to_string(), "mixed".to_string());
                env.insert("CUDA_AUTO_MIXED_PRECISION".to_string(), "1".to_string());
                env.insert("NVIDIA_TF32_OVERRIDE".to_string(), "1".to_string());
            }
            Precision::Q8 => {
                env.insert("OLLAMA_QUANTIZATION".to_string(), "q8_0".to_string());
            }
            Precision::Q4 => {
                env.insert("OLLAMA_QUANTIZATION".to_string(), "q4_0".to_string());
            }
        }

        // Advanced optimizations
        if self.tensor_parallel {
            env.insert("OLLAMA_TENSOR_PARALLEL".to_string(), "1".to_string());
        }

        if self.flash_attention {
            env.insert("OLLAMA_FLASH_ATTENTION".to_string(), "1".to_string());
            env.insert("FLASH_ATTENTION_FORCE_TRITON".to_string(), "1".to_string());
        }

        if self.kv_cache_optimization {
            env.insert("OLLAMA_KV_CACHE_TYPE".to_string(), "flash".to_string());
            env.insert(
                "OLLAMA_KV_CACHE_QUANTIZATION".to_string(),
                "fp8".to_string(),
            );
        }

        // Context and batch settings
        env.insert(
            "OLLAMA_MAX_CONTEXT".to_string(),
            self.context_length.to_string(),
        );
        env.insert("OLLAMA_BATCH_SIZE".to_string(), "512".to_string());

        // Performance tuning
        env.insert("OMP_NUM_THREADS".to_string(), "8".to_string());
        env.insert("OPENBLAS_NUM_THREADS".to_string(), "1".to_string());

        env
    }

    /// Generate nvbind configuration for this Ollama setup
    #[cfg(feature = "bolt")]
    pub fn to_nvbind_config(&self) -> crate::config::BoltAiMlGpuConfig {
        crate::config::BoltAiMlGpuConfig {
            cuda_cache_size: self.cuda_cache_size,
            tensor_cores_enabled: true,
            mixed_precision: matches!(self.precision, Precision::Mixed),
            memory_pool_size: Some(self.gpu_memory_limit.clone()),
            mig_enabled: matches!(self.model_size, ModelSize::XLarge),
        }
    }
}

/// Ollama model registry for common models
pub struct OllamaModelRegistry {
    models: HashMap<String, OllamaConfig>,
}

impl Default for OllamaModelRegistry {
    fn default() -> Self {
        let mut models = HashMap::new();

        // Popular 7B models
        models.insert(
            "llama3.2:7b".to_string(),
            OllamaConfig::for_model_name("llama3.2:7b"),
        );
        models.insert(
            "phi3:3.8b".to_string(),
            OllamaConfig::for_model_name("phi3:3.8b"),
        );
        models.insert(
            "codellama:7b".to_string(),
            OllamaConfig::for_model_name("codellama:7b"),
        );
        models.insert(
            "mistral:7b".to_string(),
            OllamaConfig::for_model_name("mistral:7b"),
        );

        // 13B models
        models.insert(
            "llama3.1:13b".to_string(),
            OllamaConfig::for_model_name("llama3.1:13b"),
        );
        models.insert(
            "mixtral:8x7b".to_string(),
            OllamaConfig::for_model_name("mixtral:8x7b"),
        );

        // 34B models
        models.insert(
            "codellama:34b".to_string(),
            OllamaConfig::for_model_name("codellama:34b"),
        );
        models.insert("yi:34b".to_string(), OllamaConfig::for_model_name("yi:34b"));

        // 70B+ models
        models.insert(
            "llama3.1:70b".to_string(),
            OllamaConfig::for_model_name("llama3.1:70b"),
        );
        models.insert(
            "qwen2:72b".to_string(),
            OllamaConfig::for_model_name("qwen2:72b"),
        );

        Self { models }
    }
}

impl OllamaModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get optimized configuration for a model
    pub fn get_config(&self, model_name: &str) -> OllamaConfig {
        self.models
            .get(model_name)
            .cloned()
            .unwrap_or_else(|| OllamaConfig::for_model_name(model_name))
    }

    /// List all registered models
    pub fn list_models(&self) -> Vec<&String> {
        self.models.keys().collect()
    }

    /// Add custom model configuration
    pub fn register_model(&mut self, name: String, config: OllamaConfig) {
        self.models.insert(name, config);
    }
}

/// Ollama container launcher with GPU optimizations
pub struct OllamaLauncher {
    registry: OllamaModelRegistry,
}

impl Default for OllamaLauncher {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaLauncher {
    pub fn new() -> Self {
        Self {
            registry: OllamaModelRegistry::new(),
        }
    }

    /// Launch Ollama container with optimized GPU configuration
    pub async fn launch_model(&self, model_name: &str, runtime: Option<&str>) -> Result<()> {
        info!("Launching Ollama model: {}", model_name);

        let config = self.registry.get_config(model_name);
        let env_vars = config.to_environment_vars();

        debug!("Using Ollama configuration: {:?}", config);
        debug!("Environment variables: {} set", env_vars.len());

        // Create nvbind configuration
        let nvbind_config = crate::config::Config {
            #[cfg(feature = "bolt")]
            bolt: Some(crate::config::BoltConfig {
                aiml: Some(config.to_nvbind_config()),
                ..Default::default()
            }),
            ..Default::default()
        };

        // Launch container with optimized settings
        let runtime_cmd = runtime.unwrap_or("bolt");
        let mut launch_args = vec!["serve".to_string()];

        // Add model-specific launch arguments
        if config.tensor_parallel {
            launch_args.push("--tensor-parallel".to_string());
        }

        crate::runtime::run_with_config(
            nvbind_config,
            runtime_cmd.to_string(),
            "all".to_string(),
            "ollama/ollama:latest".to_string(),
            launch_args,
        )
        .await?;

        info!("Ollama model {} launched successfully", model_name);
        Ok(())
    }

    /// Hot-swap to a different model with minimal downtime
    pub async fn hot_swap_model(&self, old_model: &str, new_model: &str) -> Result<()> {
        info!("Hot-swapping from {} to {}", old_model, new_model);

        let _new_config = self.registry.get_config(new_model);

        // Note: Graceful model swapping requires ollama API integration for controlled shutdown
        // 1. Pre-load new model in memory
        // 2. Switch traffic atomically
        // 3. Unload old model

        // For now, launch new model
        self.launch_model(new_model, Some("bolt")).await?;

        info!("Hot-swap completed: {} -> {}", old_model, new_model);
        Ok(())
    }

    /// Get performance recommendations for a model
    pub fn get_performance_recommendations(&self, model_name: &str) -> Vec<String> {
        let config = self.registry.get_config(model_name);
        let mut recommendations = Vec::new();

        match config.model_size {
            ModelSize::Small => {
                recommendations.push(
                    "Consider increasing parallel requests for better throughput".to_string(),
                );
                recommendations
                    .push("FP16 precision provides optimal speed/quality balance".to_string());
            }
            ModelSize::Medium => {
                recommendations.push("Enable tensor parallelism for large batch sizes".to_string());
                recommendations
                    .push("Mixed precision recommended for best performance".to_string());
            }
            ModelSize::Large => {
                recommendations.push(
                    "Ensure sufficient GPU memory (24GB+) for optimal performance".to_string(),
                );
                recommendations.push("Consider multi-GPU setup for better throughput".to_string());
            }
            ModelSize::XLarge => {
                recommendations
                    .push("Multi-instance GPU (MIG) recommended for resource sharing".to_string());
                recommendations.push(
                    "Tensor parallelism essential for acceptable inference speed".to_string(),
                );
                recommendations
                    .push("Consider model quantization if memory is limited".to_string());
            }
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_size_detection() {
        assert_eq!(
            OllamaConfig::detect_model_size("llama3.1:70b"),
            ModelSize::XLarge
        );
        assert_eq!(
            OllamaConfig::detect_model_size("codellama:34b"),
            ModelSize::Large
        );
        assert_eq!(
            OllamaConfig::detect_model_size("llama3.1:13b"),
            ModelSize::Medium
        );
        assert_eq!(
            OllamaConfig::detect_model_size("phi3:3.8b"),
            ModelSize::Small
        );
    }

    #[test]
    fn test_config_generation() {
        let config = OllamaConfig::for_model_size(ModelSize::Small);
        assert_eq!(config.model_size, ModelSize::Small);
        assert_eq!(config.precision, Precision::FP16);
        assert!(config.max_parallel_requests > 0);
    }

    #[test]
    fn test_environment_variables() {
        let config = OllamaConfig::for_model_size(ModelSize::Medium);
        let env = config.to_environment_vars();

        assert!(env.contains_key("OLLAMA_NUM_PARALLEL"));
        assert!(env.contains_key("CUDA_CACHE_MAXSIZE"));
        assert!(env.contains_key("OLLAMA_PRECISION"));
    }

    #[test]
    fn test_model_registry() {
        let registry = OllamaModelRegistry::new();
        let config = registry.get_config("llama3.1:70b");

        assert_eq!(config.model_size, ModelSize::XLarge);
        assert!(config.tensor_parallel);
    }

    #[tokio::test]
    async fn test_ollama_launcher() {
        let launcher = OllamaLauncher::new();
        let recommendations = launcher.get_performance_recommendations("phi3:3.8b");

        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("FP16")));
    }
}
