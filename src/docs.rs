//! Documentation generation and API reference
//!
//! Provides comprehensive API documentation, usage examples,
//! and interactive help system for nvbind.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use tracing::info;

/// Documentation generator
pub struct DocGenerator {
    output_dir: String,
    api_docs: Vec<ApiDoc>,
    examples: Vec<Example>,
    guides: Vec<Guide>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiDoc {
    pub module: String,
    pub name: String,
    pub description: String,
    pub functions: Vec<FunctionDoc>,
    pub structs: Vec<StructDoc>,
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDoc {
    pub name: String,
    pub description: String,
    pub parameters: Vec<Parameter>,
    pub returns: String,
    pub example: Option<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub required: bool,
    pub default: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructDoc {
    pub name: String,
    pub description: String,
    pub fields: Vec<FieldDoc>,
    pub example: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDoc {
    pub name: String,
    pub field_type: String,
    pub description: String,
    pub required: bool,
    pub default: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub name: String,
    pub description: String,
    pub category: String,
    pub code: String,
    pub language: String,
    pub output: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Guide {
    pub title: String,
    pub description: String,
    pub sections: Vec<GuideSection>,
    pub difficulty: Difficulty,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Difficulty {
    Beginner,
    Intermediate,
    Advanced,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuideSection {
    pub title: String,
    pub content: String,
    pub code_examples: Vec<String>,
}

impl DocGenerator {
    /// Create new documentation generator
    pub fn new(output_dir: String) -> Self {
        Self {
            output_dir,
            api_docs: Vec::new(),
            examples: Vec::new(),
            guides: Vec::new(),
        }
    }

    /// Initialize documentation system
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing documentation generator");

        // Create output directories
        self.create_directories().await?;

        // Generate API documentation
        self.generate_api_docs().await?;

        // Generate examples
        self.generate_examples().await?;

        // Generate user guides
        self.generate_guides().await?;

        info!("Documentation generation complete");
        Ok(())
    }

    async fn create_directories(&self) -> Result<()> {
        let paths = [
            &self.output_dir,
            &format!("{}/api", self.output_dir),
            &format!("{}/examples", self.output_dir),
            &format!("{}/guides", self.output_dir),
            &format!("{}/reference", self.output_dir),
        ];

        for path in paths {
            fs::create_dir_all(path)?;
        }

        Ok(())
    }

    async fn generate_api_docs(&mut self) -> Result<()> {
        info!("Generating API documentation");

        // GPU Module Documentation
        self.api_docs.push(ApiDoc {
            module: "gpu".to_string(),
            name: "GPU Management".to_string(),
            description: "Core GPU discovery and management functionality".to_string(),
            functions: vec![
                FunctionDoc {
                    name: "discover_gpus".to_string(),
                    description: "Discover all available NVIDIA GPUs on the system".to_string(),
                    parameters: vec![],
                    returns: "Result<Vec<GpuDevice>>".to_string(),
                    example: Some(
                        r#"
use nvbind::gpu;

let gpus = gpu::discover_gpus().await?;
for gpu in gpus {
    println!("Found GPU: {} ({}MB)", gpu.name, gpu.memory_mb);
}
                    "#
                        .to_string(),
                    ),
                    errors: vec![
                        "No NVIDIA GPUs found".to_string(),
                        "NVIDIA driver not available".to_string(),
                    ],
                },
                FunctionDoc {
                    name: "get_driver_info".to_string(),
                    description: "Get NVIDIA driver information and version".to_string(),
                    parameters: vec![],
                    returns: "Result<DriverInfo>".to_string(),
                    example: Some(
                        r#"
use nvbind::gpu;

let driver = gpu::get_driver_info().await?;
println!("Driver: {} ({})", driver.version, driver.driver_type);
                    "#
                        .to_string(),
                    ),
                    errors: vec!["Driver information unavailable".to_string()],
                },
            ],
            structs: vec![StructDoc {
                name: "GpuDevice".to_string(),
                description: "Represents a single GPU device".to_string(),
                fields: vec![
                    FieldDoc {
                        name: "index".to_string(),
                        field_type: "u32".to_string(),
                        description: "GPU index (0-based)".to_string(),
                        required: true,
                        default: None,
                    },
                    FieldDoc {
                        name: "name".to_string(),
                        field_type: "String".to_string(),
                        description: "GPU model name".to_string(),
                        required: true,
                        default: None,
                    },
                    FieldDoc {
                        name: "memory_mb".to_string(),
                        field_type: "u64".to_string(),
                        description: "GPU memory in megabytes".to_string(),
                        required: true,
                        default: None,
                    },
                    FieldDoc {
                        name: "uuid".to_string(),
                        field_type: "String".to_string(),
                        description: "Unique GPU identifier".to_string(),
                        required: true,
                        default: None,
                    },
                ],
                example: Some(
                    r#"
{
    "index": 0,
    "name": "NVIDIA GeForce RTX 4090",
    "memory_mb": 24576,
    "uuid": "GPU-12345678-1234-1234-1234-123456789abc"
}
                    "#
                    .to_string(),
                ),
            }],
            examples: vec![
                "basic_gpu_discovery".to_string(),
                "gpu_monitoring".to_string(),
            ],
        });

        // RBAC Module Documentation
        self.api_docs.push(ApiDoc {
            module: "rbac".to_string(),
            name: "Role-Based Access Control".to_string(),
            description: "User and group permission management for GPU resources".to_string(),
            functions: vec![FunctionDoc {
                name: "check_permission".to_string(),
                description: "Check if user has permission for specific action".to_string(),
                parameters: vec![
                    Parameter {
                        name: "user".to_string(),
                        param_type: "User".to_string(),
                        description: "User to check permissions for".to_string(),
                        required: true,
                        default: None,
                    },
                    Parameter {
                        name: "action".to_string(),
                        param_type: "String".to_string(),
                        description: "Action to check (e.g., 'gpu.use', 'container.create')"
                            .to_string(),
                        required: true,
                        default: None,
                    },
                ],
                returns: "Result<bool>".to_string(),
                example: Some(
                    r#"
use nvbind::rbac::{RbacManager, User};

let rbac = RbacManager::new(config);
let user = User::from_uid(1000)?;
let has_permission = rbac.check_permission(user, "gpu.use").await?;
                    "#
                    .to_string(),
                ),
                errors: vec!["User not found".to_string(), "Invalid action".to_string()],
            }],
            structs: vec![StructDoc {
                name: "User".to_string(),
                description: "Represents a system user".to_string(),
                fields: vec![
                    FieldDoc {
                        name: "uid".to_string(),
                        field_type: "u32".to_string(),
                        description: "User ID".to_string(),
                        required: true,
                        default: None,
                    },
                    FieldDoc {
                        name: "username".to_string(),
                        field_type: "String".to_string(),
                        description: "Username".to_string(),
                        required: true,
                        default: None,
                    },
                ],
                example: None,
            }],
            examples: vec!["rbac_setup".to_string(), "permission_checking".to_string()],
        });

        // Write API documentation to files
        for api_doc in &self.api_docs {
            let content = self.generate_api_doc_content(api_doc)?;
            let filename = format!("{}/api/{}.md", self.output_dir, api_doc.module);
            fs::write(filename, content)?;
        }

        Ok(())
    }

    fn generate_api_doc_content(&self, doc: &ApiDoc) -> Result<String> {
        let mut content = String::new();

        content.push_str(&format!("# {} API Reference\n\n", doc.name));
        content.push_str(&format!("{}\n\n", doc.description));

        // Table of Contents
        content.push_str("## Table of Contents\n\n");
        content.push_str("- [Functions](#functions)\n");
        content.push_str("- [Structs](#structs)\n");
        content.push_str("- [Examples](#examples)\n\n");

        // Functions
        if !doc.functions.is_empty() {
            content.push_str("## Functions\n\n");
            for func in &doc.functions {
                content.push_str(&self.generate_function_doc(func)?);
            }
        }

        // Structs
        if !doc.structs.is_empty() {
            content.push_str("## Structs\n\n");
            for struct_doc in &doc.structs {
                content.push_str(&self.generate_struct_doc(struct_doc)?);
            }
        }

        // Examples
        if !doc.examples.is_empty() {
            content.push_str("## Examples\n\n");
            for example_name in &doc.examples {
                content.push_str(&format!(
                    "- [{}](../examples/{}.md)\n",
                    example_name, example_name
                ));
            }
        }

        Ok(content)
    }

    fn generate_function_doc(&self, func: &FunctionDoc) -> Result<String> {
        let mut content = String::new();

        content.push_str(&format!("### {}\n\n", func.name));
        content.push_str(&format!("{}\n\n", func.description));

        // Parameters
        if !func.parameters.is_empty() {
            content.push_str("**Parameters:**\n\n");
            for param in &func.parameters {
                let required = if param.required {
                    " (required)"
                } else {
                    " (optional)"
                };
                content.push_str(&format!(
                    "- `{}`: `{}` - {}{}\n",
                    param.name, param.param_type, param.description, required
                ));
                if let Some(default) = &param.default {
                    content.push_str(&format!("  - Default: `{}`\n", default));
                }
            }
            content.push('\n');
        }

        // Returns
        content.push_str(&format!("**Returns:** `{}`\n\n", func.returns));

        // Errors
        if !func.errors.is_empty() {
            content.push_str("**Errors:**\n\n");
            for error in &func.errors {
                content.push_str(&format!("- {}\n", error));
            }
            content.push('\n');
        }

        // Example
        if let Some(example) = &func.example {
            content.push_str("**Example:**\n\n");
            content.push_str("```rust\n");
            content.push_str(example.trim());
            content.push_str("\n```\n\n");
        }

        Ok(content)
    }

    fn generate_struct_doc(&self, struct_doc: &StructDoc) -> Result<String> {
        let mut content = String::new();

        content.push_str(&format!("### {}\n\n", struct_doc.name));
        content.push_str(&format!("{}\n\n", struct_doc.description));

        // Fields
        if !struct_doc.fields.is_empty() {
            content.push_str("**Fields:**\n\n");
            for field in &struct_doc.fields {
                let required = if field.required {
                    " (required)"
                } else {
                    " (optional)"
                };
                content.push_str(&format!(
                    "- `{}`: `{}` - {}{}\n",
                    field.name, field.field_type, field.description, required
                ));
                if let Some(default) = &field.default {
                    content.push_str(&format!("  - Default: `{}`\n", default));
                }
            }
            content.push('\n');
        }

        // Example
        if let Some(example) = &struct_doc.example {
            content.push_str("**Example:**\n\n");
            content.push_str("```json\n");
            content.push_str(example.trim());
            content.push_str("\n```\n\n");
        }

        Ok(content)
    }

    async fn generate_examples(&mut self) -> Result<()> {
        info!("Generating code examples");

        // Basic GPU Discovery Example
        self.examples.push(Example {
            name: "basic_gpu_discovery".to_string(),
            description: "Discover and list all available GPUs".to_string(),
            category: "gpu".to_string(),
            language: "rust".to_string(),
            code: r#"use nvbind::gpu;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    // Discover GPUs
    let gpus = gpu::discover_gpus().await?;

    println!("Found {} GPU(s):", gpus.len());
    for (i, gpu) in gpus.iter().enumerate() {
        println!("  GPU {}: {}", i, gpu.name);
        println!("    Memory: {} MB", gpu.memory_mb);
        println!("    UUID: {}", gpu.uuid);
        println!("    PCI Bus ID: {}", gpu.pci_bus_id);
    }

    Ok(())
}"#
            .to_string(),
            output: Some(
                r#"Found 1 GPU(s):
  GPU 0: NVIDIA GeForce RTX 4090
    Memory: 24576 MB
    UUID: GPU-12345678-1234-1234-1234-123456789abc
    PCI Bus ID: 0000:01:00.0"#
                    .to_string(),
            ),
        });

        // Container Runtime Example
        self.examples.push(Example {
            name: "container_with_gpu".to_string(),
            description: "Run a container with GPU access".to_string(),
            category: "runtime".to_string(),
            language: "rust".to_string(),
            code: r#"use nvbind::{runtime, config::Config};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration
    let config = Config::load("/etc/nvbind/config.toml")?;

    // Create container specification
    let container_spec = runtime::ContainerSpec {
        image: "nvidia/cuda:12.0-runtime-ubuntu22.04".to_string(),
        command: vec!["nvidia-smi".to_string()],
        gpu_request: Some(runtime::GpuRequest {
            count: 1,
            memory_mb: Some(4096),
            capabilities: vec!["compute".to_string(), "utility".to_string()],
        }),
        ..Default::default()
    };

    // Run container
    let runtime = runtime::create_runtime("podman", &config)?;
    let result = runtime.run_container(container_spec).await?;

    println!("Container output:\n{}", result.stdout);

    Ok(())
}"#
            .to_string(),
            output: Some(
                r#"Container output:
Wed Sep 25 10:30:00 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.42                 Driver Version: 580.42         CUDA Version: 12.0 |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M.     |
|                               |                      |               MIG M.     |
|=======================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   35C    P8    25W / 450W |      0MiB / 24576MiB |      0%      Default    |
|                               |                      |                  N/A     |
+---------------------------------------------------------------------------------------+"#
                    .to_string(),
            ),
        });

        // RBAC Example
        self.examples.push(Example {
            name: "rbac_setup".to_string(),
            description: "Set up role-based access control".to_string(),
            category: "rbac".to_string(),
            language: "rust".to_string(),
            code: r#"use nvbind::rbac::{RbacManager, RbacConfig, Role, Permission, User};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Create RBAC configuration
    let mut config = RbacConfig::default();

    // Define roles
    let admin_role = Role {
        name: "admin".to_string(),
        permissions: vec![
            Permission::new("gpu.*", "Allow all GPU operations"),
            Permission::new("container.*", "Allow all container operations"),
            Permission::new("system.*", "Allow all system operations"),
        ],
        resource_limits: None,
    };

    let user_role = Role {
        name: "gpu_user".to_string(),
        permissions: vec![
            Permission::new("gpu.use", "Allow GPU usage"),
            Permission::new("container.create", "Allow container creation"),
        ],
        resource_limits: Some(ResourceLimits {
            max_gpus: Some(1),
            max_memory_gb: Some(8),
            max_containers: Some(5),
        }),
    };

    config.roles.push(admin_role);
    config.roles.push(user_role);

    // Create RBAC manager
    let rbac = RbacManager::new(config);
    rbac.initialize().await?;

    // Assign role to user
    let user = User::from_uid(1000)?;
    rbac.assign_role(&user, "gpu_user").await?;

    // Check permission
    let can_use_gpu = rbac.check_permission(&user, "gpu.use").await?;
    println!("User can use GPU: {}", can_use_gpu);

    Ok(())
}"#
            .to_string(),
            output: Some("User can use GPU: true".to_string()),
        });

        // MIG Configuration Example
        self.examples.push(Example {
            name: "mig_configuration".to_string(),
            description: "Configure Multi-Instance GPU (MIG)".to_string(),
            category: "gpu_advanced".to_string(),
            language: "rust".to_string(),
            code: r#"use nvbind::gpu_advanced::{AdvancedGpuManager, AdvancedGpuConfig, MigProfile};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Create configuration with MIG enabled
    let mut config = AdvancedGpuConfig::default();
    config.mig_enabled = true;
    config.auto_mig_config = true;

    // Create GPU manager
    let mut gpu_manager = AdvancedGpuManager::new(config);
    gpu_manager.initialize().await?;

    // Define MIG profiles
    let profiles = vec![
        MigProfile {
            name: "1g.5gb".to_string(),
            profile_id: 0,
            compute_slices: 1,
            memory_slices: 1,
            memory_mb: 5120,
            max_instances: 7,
        },
        MigProfile {
            name: "2g.10gb".to_string(),
            profile_id: 1,
            compute_slices: 2,
            memory_slices: 2,
            memory_mb: 10240,
            max_instances: 3,
        },
    ];

    // Configure MIG
    gpu_manager.configure_mig(profiles).await?;

    // Get MIG metrics
    let metrics = gpu_manager.get_advanced_metrics().await?;
    if let Some(mig_metrics) = metrics.mig_metrics {
        println!("MIG Instances: {} active / {} total",
                 mig_metrics.active_instances,
                 mig_metrics.total_instances);
    }

    Ok(())
}"#
            .to_string(),
            output: Some("MIG Instances: 7 active / 7 total".to_string()),
        });

        // Write examples to files
        for example in &self.examples {
            let content = self.generate_example_content(example)?;
            let filename = format!("{}/examples/{}.md", self.output_dir, example.name);
            fs::write(filename, content)?;
        }

        Ok(())
    }

    fn generate_example_content(&self, example: &Example) -> Result<String> {
        let mut content = String::new();

        content.push_str(&format!(
            "# {}\n\n",
            example.name.replace("_", " ").to_uppercase()
        ));
        content.push_str(&format!("{}\n\n", example.description));
        content.push_str(&format!("**Category:** {}\n\n", example.category));

        content.push_str("## Code\n\n");
        content.push_str(&format!("```{}\n", example.language));
        content.push_str(&example.code);
        content.push_str("\n```\n\n");

        if let Some(output) = &example.output {
            content.push_str("## Expected Output\n\n");
            content.push_str("```\n");
            content.push_str(output);
            content.push_str("\n```\n\n");
        }

        content.push_str("## Running the Example\n\n");
        content.push_str("1. Ensure you have NVIDIA GPU and drivers installed\n");
        content.push_str("2. Install nvbind: `cargo install nvbind`\n");
        content.push_str("3. Copy the code above to a file (e.g., `example.rs`)\n");
        content.push_str("4. Run: `cargo run --bin example`\n\n");

        Ok(content)
    }

    async fn generate_guides(&mut self) -> Result<()> {
        info!("Generating user guides");

        // Getting Started Guide
        self.guides.push(Guide {
            title: "Getting Started with nvbind".to_string(),
            description: "Complete guide to installing and using nvbind".to_string(),
            difficulty: Difficulty::Beginner,
            sections: vec![
                GuideSection {
                    title: "Installation".to_string(),
                    content: r#"nvbind can be installed in several ways:

## From Source
```bash
git clone https://github.com/ghostkellz/nvbind.git
cd nvbind
cargo build --release
sudo cp target/release/nvbind /usr/local/bin/
```

## Using Cargo
```bash
cargo install nvbind
```

## Package Managers
### Arch Linux (AUR)
```bash
yay -S nvbind
```

### Ubuntu/Debian
```bash
# Add repository
curl -s https://packagecloud.io/install/repositories/nvbind/stable/script.deb.sh | sudo bash
sudo apt-get install nvbind
```

### Fedora
```bash
sudo dnf copr enable ghostkellz/nvbind
sudo dnf install nvbind
```"#
                        .to_string(),
                    code_examples: vec!["cargo install nvbind".to_string()],
                },
                GuideSection {
                    title: "First Steps".to_string(),
                    content: r#"After installation, verify nvbind is working:

## Check Installation
```bash
nvbind --version
```

## Run System Diagnostics
```bash
nvbind diagnose
```

This will check for:
- NVIDIA GPU availability
- Driver installation
- Container runtimes (Docker, Podman)
- System compatibility

## Interactive Setup
```bash
nvbind setup
```

This launches an interactive wizard to configure nvbind for your system."#
                        .to_string(),
                    code_examples: vec![
                        "nvbind --version".to_string(),
                        "nvbind diagnose".to_string(),
                    ],
                },
            ],
        });

        // Advanced Configuration Guide
        self.guides.push(Guide {
            title: "Advanced Configuration".to_string(),
            description: "Detailed guide for advanced nvbind configuration".to_string(),
            difficulty: Difficulty::Advanced,
            sections: vec![
                GuideSection {
                    title: "Configuration File Structure".to_string(),
                    content:
                        r#"The main configuration file is located at `/etc/nvbind/config.toml`:

```toml
schema_version = "1.0.0"

[runtime]
default = "podman"
timeout_seconds = 300

[gpu]
default_selection = "all"
mig_enabled = false

[security]
enable_rbac = true
allow_privileged = false
audit_logging = true

[monitoring]
enabled = true
prometheus_port = 9090

[logging]
level = "info"
format = "json"

[cdi]
spec_dir = "/etc/cdi"
auto_generate = true
```"#
                            .to_string(),
                    code_examples: vec!["/etc/nvbind/config.toml".to_string()],
                },
                GuideSection {
                    title: "Role-Based Access Control".to_string(),
                    content: r#"RBAC allows fine-grained control over GPU access:

## Enable RBAC
```toml
[security]
enable_rbac = true
```

## Define Roles
```toml
[[security.roles]]
name = "admin"
permissions = ["gpu.*", "container.*", "system.*"]

[[security.roles]]
name = "gpu_user"
permissions = ["gpu.use", "container.create"]
max_gpus = 1
max_memory_gb = 8

[[security.roles]]
name = "ml_researcher"
permissions = ["gpu.use", "container.create", "mig.configure"]
max_gpus = 4
max_memory_gb = 32
```

## Assign Users to Roles
```bash
nvbind rbac assign-role --user alice --role gpu_user
nvbind rbac assign-role --user bob --role ml_researcher
```"#
                        .to_string(),
                    code_examples: vec![
                        "nvbind rbac assign-role --user alice --role gpu_user".to_string(),
                    ],
                },
            ],
        });

        // Write guides to files
        for guide in &self.guides {
            let content = self.generate_guide_content(guide)?;
            let filename = format!(
                "{}/guides/{}.md",
                self.output_dir,
                guide.title.to_lowercase().replace(" ", "_")
            );
            fs::write(filename, content)?;
        }

        // Generate main documentation index
        self.generate_main_index().await?;

        Ok(())
    }

    fn generate_guide_content(&self, guide: &Guide) -> Result<String> {
        let mut content = String::new();

        content.push_str(&format!("# {}\n\n", guide.title));
        content.push_str(&format!("{}\n\n", guide.description));
        content.push_str(&format!("**Difficulty:** {:?}\n\n", guide.difficulty));

        // Table of contents
        content.push_str("## Table of Contents\n\n");
        for (i, section) in guide.sections.iter().enumerate() {
            content.push_str(&format!(
                "{}. [{}](#{})\n",
                i + 1,
                section.title,
                section.title.to_lowercase().replace(" ", "-")
            ));
        }
        content.push('\n');

        // Sections
        for (i, section) in guide.sections.iter().enumerate() {
            content.push_str(&format!("## {}. {}\n\n", i + 1, section.title));
            content.push_str(&section.content);
            content.push_str("\n\n");
        }

        Ok(content)
    }

    async fn generate_main_index(&self) -> Result<()> {
        let mut content = String::new();

        content.push_str("# nvbind Documentation\n\n");
        content.push_str("Welcome to the nvbind documentation! This guide will help you get started with using nvbind, a lightweight Rust-based alternative to NVIDIA Container Toolkit.\n\n");

        // Quick Links
        content.push_str("## Quick Links\n\n");
        content.push_str(
            "- [Installation Guide](guides/getting_started_with_nvbind.md#installation)\n",
        );
        content.push_str("- [API Reference](api/)\n");
        content.push_str("- [Code Examples](examples/)\n");
        content.push_str("- [Advanced Configuration](guides/advanced_configuration.md)\n\n");

        // API Reference
        content.push_str("## API Reference\n\n");
        for api_doc in &self.api_docs {
            content.push_str(&format!(
                "- [{}](api/{}.md) - {}\n",
                api_doc.name, api_doc.module, api_doc.description
            ));
        }
        content.push('\n');

        // Examples
        content.push_str("## Examples\n\n");
        let mut examples_by_category: HashMap<String, Vec<&Example>> = HashMap::new();
        for example in &self.examples {
            examples_by_category
                .entry(example.category.clone())
                .or_default()
                .push(example);
        }

        for (category, examples) in examples_by_category {
            content.push_str(&format!("### {}\n\n", category.to_uppercase()));
            for example in examples {
                content.push_str(&format!(
                    "- [{}](examples/{}.md) - {}\n",
                    example.name.replace("_", " "),
                    example.name,
                    example.description
                ));
            }
            content.push('\n');
        }

        // Guides
        content.push_str("## User Guides\n\n");
        for guide in &self.guides {
            let difficulty_badge = match guide.difficulty {
                Difficulty::Beginner => "🟢 Beginner",
                Difficulty::Intermediate => "🟡 Intermediate",
                Difficulty::Advanced => "🔴 Advanced",
            };
            content.push_str(&format!(
                "- [{}](guides/{}.md) - {} ({})\n",
                guide.title,
                guide.title.to_lowercase().replace(" ", "_"),
                guide.description,
                difficulty_badge
            ));
        }
        content.push('\n');

        // System Requirements
        content.push_str("## System Requirements\n\n");
        content.push_str("- Linux (Arch, Ubuntu, Debian, Fedora, PopOS)\n");
        content.push_str("- NVIDIA GPU with CUDA support\n");
        content.push_str("- NVIDIA Driver (Open or Proprietary)\n");
        content.push_str("- Container runtime (Docker, Podman, or Containerd)\n");
        content.push_str("- Rust 1.70+ (for building from source)\n\n");

        // Support
        content.push_str("## Support\n\n");
        content.push_str(
            "- **Issues:** [GitHub Issues](https://github.com/ghostkellz/nvbind/issues)\n",
        );
        content.push_str("- **Discussions:** [GitHub Discussions](https://github.com/ghostkellz/nvbind/discussions)\n");
        content.push_str("- **Documentation:** This site\n\n");

        // Write main index
        let filename = format!("{}/README.md", self.output_dir);
        fs::write(filename, content)?;

        Ok(())
    }

    /// Generate OpenAPI specification
    pub async fn generate_openapi_spec(&self) -> Result<()> {
        info!("Generating OpenAPI specification");

        let openapi_spec = r##"{"
  "openapi": "3.0.3",
  "info": {
    "title": "nvbind API",
    "description": "REST API for nvbind GPU container runtime",
    "version": "0.1.0",
    "license": {
      "name": "MIT",
      "url": "https://opensource.org/licenses/MIT"
    }
  },
  "servers": [
    {
      "url": "http://localhost:8080/api/v1",
      "description": "Local development server"
    }
  ],
  "paths": {
    "/gpus": {
      "get": {
        "summary": "List all GPUs",
        "description": "Get information about all available GPUs",
        "responses": {
          "200": {
            "description": "List of GPU devices",
            "content": {
              "application/json": {
                "schema": {
                  "type": "array",
                  "items": {
                    "$ref": "#/components/schemas/GpuDevice"
                  }
                }
              }
            }
          }
        }
      }
    },
    "/containers": {
      "post": {
        "summary": "Create container with GPU",
        "description": "Create and run a container with GPU access",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ContainerSpec"
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "Container created successfully",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ContainerResult"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "GpuDevice": {
        "type": "object",
        "properties": {
          "index": {
            "type": "integer",
            "description": "GPU index"
          },
          "name": {
            "type": "string",
            "description": "GPU model name"
          },
          "memory_mb": {
            "type": "integer",
            "description": "GPU memory in megabytes"
          },
          "uuid": {
            "type": "string",
            "description": "Unique GPU identifier"
          }
        }
      },
      "ContainerSpec": {
        "type": "object",
        "required": ["image"],
        "properties": {
          "image": {
            "type": "string",
            "description": "Container image name"
          },
          "command": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "Command to run in container"
          },
          "gpu_request": {
            "$ref": "#/components/schemas/GpuRequest"
          }
        }
      },
      "GpuRequest": {
        "type": "object",
        "properties": {
          "count": {
            "type": "integer",
            "description": "Number of GPUs requested"
          },
          "memory_mb": {
            "type": "integer",
            "description": "GPU memory requirement in MB"
          },
          "capabilities": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "Required GPU capabilities"
          }
        }
      },
      "ContainerResult": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string",
            "description": "Container ID"
          },
          "status": {
            "type": "string",
            "description": "Container status"
          },
          "stdout": {
            "type": "string",
            "description": "Container output"
          },
          "stderr": {
            "type": "string",
            "description": "Container errors"
          }
        }
      }
    }
  }
}"##;

        let filename = format!("{}/reference/openapi.json", self.output_dir);
        fs::write(filename, openapi_spec)?;

        Ok(())
    }

    /// Generate CLI reference
    pub async fn generate_cli_reference(&self) -> Result<()> {
        info!("Generating CLI reference");

        let cli_reference = r#"# nvbind CLI Reference

## Overview

nvbind provides a comprehensive command-line interface for managing GPU containers.

## Global Options

- `-v, --verbose`: Enable verbose output
- `-q, --quiet`: Suppress non-error output
- `-c, --config <PATH>`: Configuration file path

## Commands

### `nvbind setup`

Interactive setup wizard to configure nvbind.

**Usage:** `nvbind setup [OPTIONS]`

**Options:**
- `--yes`: Skip confirmation prompts

**Example:**
```bash
nvbind setup
```

### `nvbind diagnose`

Run system diagnostics and compatibility checks.

**Usage:** `nvbind diagnose [OPTIONS]`

**Options:**
- `-f, --format <FORMAT>`: Output format (human, json, yaml, table)
- `--gpu-details`: Include detailed GPU information

**Example:**
```bash
nvbind diagnose --format json --gpu-details
```

### `nvbind tune`

Performance tuning wizard for GPU workloads.

**Usage:** `nvbind tune [OPTIONS]`

**Options:**
- `-p, --profile <PROFILE>`: Workload profile (ml, gaming, compute)
- `--benchmark`: Run benchmarks after tuning

**Example:**
```bash
nvbind tune --profile ml --benchmark
```

### `nvbind validate`

Validate configuration files.

**Usage:** `nvbind validate [OPTIONS]`

**Options:**
- `-f, --file <PATH>`: Configuration file to validate
- `--strict`: Enable strict validation mode

**Example:**
```bash
nvbind validate --file /etc/nvbind/config.toml --strict
```

### `nvbind monitor`

Monitor system resources and GPU usage.

**Usage:** `nvbind monitor [OPTIONS]`

**Options:**
- `-i, --interval <SECONDS>`: Update interval (default: 1)
- `--export <PATH>`: Export metrics to file

**Example:**
```bash
nvbind monitor --interval 5 --export metrics.json
```

### `nvbind migrate`

Migrate configuration to latest version.

**Usage:** `nvbind migrate [OPTIONS]`

**Options:**
- `-f, --file <PATH>`: Configuration file to migrate
- `--backup`: Create backup of original file

**Example:**
```bash
nvbind migrate --file config.toml --backup
```

### `nvbind completions`

Generate shell completions.

**Usage:** `nvbind completions <SHELL>`

**Arguments:**
- `SHELL`: Shell type (bash, zsh, fish, powershell)

**Example:**
```bash
nvbind completions bash > /etc/bash_completion.d/nvbind
```

## Configuration File

The configuration file uses TOML format and is typically located at `/etc/nvbind/config.toml`.

See the [Advanced Configuration Guide](../guides/advanced_configuration.md) for details.

## Environment Variables

- `NVBIND_CONFIG`: Override config file path
- `NVBIND_LOG_LEVEL`: Set logging level (trace, debug, info, warn, error)
- `NVBIND_NO_COLOR`: Disable colored output

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Configuration error
- `3`: Permission error
- `4`: Resource not found
- `5`: System incompatible
"#;

        let filename = format!("{}/reference/cli.md", self.output_dir);
        fs::write(filename, cli_reference)?;

        Ok(())
    }
}

/// Interactive help system
pub struct HelpSystem {
    topics: HashMap<String, HelpTopic>,
}

#[derive(Debug, Clone)]
pub struct HelpTopic {
    pub title: String,
    pub content: String,
    pub examples: Vec<String>,
    pub related: Vec<String>,
}

impl Default for HelpSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl HelpSystem {
    /// Create new help system
    pub fn new() -> Self {
        let mut system = Self {
            topics: HashMap::new(),
        };
        system.initialize_topics();
        system
    }

    fn initialize_topics(&mut self) {
        // GPU Management
        self.topics.insert(
            "gpu".to_string(),
            HelpTopic {
                title: "GPU Management".to_string(),
                content: r#"nvbind provides comprehensive GPU management capabilities:

- Automatic GPU discovery
- Driver compatibility checking
- Memory usage monitoring
- Multi-Instance GPU (MIG) support
- Performance optimization"#
                    .to_string(),
                examples: vec![
                    "nvbind diagnose --gpu-details".to_string(),
                    "nvbind tune --profile ml".to_string(),
                ],
                related: vec!["mig".to_string(), "performance".to_string()],
            },
        );

        // Container Management
        self.topics.insert(
            "containers".to_string(),
            HelpTopic {
                title: "Container Management".to_string(),
                content: r#"Run containers with GPU access:

- Support for Docker, Podman, Containerd
- Automatic CDI specification generation
- Resource isolation and limits
- Security controls"#
                    .to_string(),
                examples: vec![
                    "podman run --device nvidia.com/gpu=all nvidia/cuda:12.0-runtime nvidia-smi"
                        .to_string(),
                ],
                related: vec!["cdi".to_string(), "security".to_string()],
            },
        );

        // More topics...
    }

    /// Get help for specific topic
    pub fn get_help(&self, topic: &str) -> Option<&HelpTopic> {
        self.topics.get(topic)
    }

    /// List all available topics
    pub fn list_topics(&self) -> Vec<&str> {
        self.topics.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_doc_generator_creation() {
        let generator = DocGenerator::new("test_docs".to_string());
        assert_eq!(generator.output_dir, "test_docs");
        assert!(generator.api_docs.is_empty());
    }

    #[test]
    fn test_help_system() {
        let help = HelpSystem::new();
        assert!(help.get_help("gpu").is_some());
        assert!(!help.list_topics().is_empty());
    }

    #[test]
    fn test_example_creation() {
        let example = Example {
            name: "test".to_string(),
            description: "Test example".to_string(),
            category: "test".to_string(),
            code: "println!(\"Hello\");".to_string(),
            language: "rust".to_string(),
            output: None,
        };

        assert_eq!(example.name, "test");
        assert_eq!(example.language, "rust");
    }
}
