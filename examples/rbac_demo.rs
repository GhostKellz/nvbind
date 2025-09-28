//! RBAC Demo Example
//!
//! This example demonstrates role-based access control (RBAC)
//! functionality in nvbind.

use anyhow::Result;
use nvbind::rbac::{Permission, RbacConfig, RbacManager, ResourceLimits, Role, User};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    println!("🔐 nvbind RBAC Demo\n");

    // Create RBAC configuration
    let mut config = RbacConfig::default();
    config.enabled = true;
    config.audit_logging = true;

    // Define admin role with full permissions
    let admin_role = Role {
        name: "admin".to_string(),
        description: Some("System administrator with full access".to_string()),
        permissions: vec![
            Permission::new("gpu.*", "Allow all GPU operations"),
            Permission::new("container.*", "Allow all container operations"),
            Permission::new("system.*", "Allow all system operations"),
            Permission::new("rbac.*", "Allow RBAC management"),
        ],
        resource_limits: None, // No limits for admin
    };

    // Define ML researcher role with GPU access
    let ml_role = Role {
        name: "ml_researcher".to_string(),
        description: Some("Machine learning researcher".to_string()),
        permissions: vec![
            Permission::new("gpu.use", "Allow GPU usage"),
            Permission::new("container.create", "Allow container creation"),
            Permission::new("container.run", "Allow container execution"),
            Permission::new("mig.configure", "Allow MIG configuration"),
            Permission::new("metrics.read", "Allow reading metrics"),
        ],
        resource_limits: Some(ResourceLimits {
            max_gpus: Some(4),
            max_memory_gb: Some(32),
            max_containers: Some(10),
            max_cpu_cores: Some(16),
            time_limits: None,
        }),
    };

    // Define regular user role with limited access
    let user_role = Role {
        name: "gpu_user".to_string(),
        description: Some("Regular user with basic GPU access".to_string()),
        permissions: vec![
            Permission::new("gpu.use", "Allow basic GPU usage"),
            Permission::new("container.create", "Allow container creation"),
            Permission::new("container.run", "Allow container execution"),
        ],
        resource_limits: Some(ResourceLimits {
            max_gpus: Some(1),
            max_memory_gb: Some(8),
            max_containers: Some(3),
            max_cpu_cores: Some(4),
            time_limits: None,
        }),
    };

    // Add roles to configuration
    config.roles.push(admin_role);
    config.roles.push(ml_role);
    config.roles.push(user_role);

    // Create RBAC manager
    println!("🚀 Initializing RBAC manager...");
    let rbac = RbacManager::new(config);
    rbac.initialize().await?;

    println!(
        "✅ RBAC manager initialized with {} roles\n",
        rbac.get_roles().len()
    );

    // Simulate different users
    let admin_user = User {
        uid: 0,
        username: "root".to_string(),
        groups: vec!["admin".to_string()],
    };

    let ml_user = User {
        uid: 1001,
        username: "alice".to_string(),
        groups: vec!["researchers".to_string()],
    };

    let regular_user = User {
        uid: 1002,
        username: "bob".to_string(),
        groups: vec!["users".to_string()],
    };

    // Assign roles
    println!("👥 Assigning roles to users...");
    rbac.assign_role(&admin_user, "admin").await?;
    rbac.assign_role(&ml_user, "ml_researcher").await?;
    rbac.assign_role(&regular_user, "gpu_user").await?;
    println!("✅ Roles assigned\n");

    // Test permissions
    println!("🧪 Testing permissions:\n");

    let test_actions = vec![
        "gpu.use",
        "gpu.configure",
        "container.create",
        "container.run",
        "mig.configure",
        "system.shutdown",
        "rbac.assign_role",
    ];

    let users = vec![
        ("admin", &admin_user),
        ("alice (ML researcher)", &ml_user),
        ("bob (regular user)", &regular_user),
    ];

    for (user_desc, user) in users {
        println!("👤 Testing permissions for {}:", user_desc);

        for action in &test_actions {
            let has_permission = rbac.check_permission(user, action).await?;
            let icon = if has_permission { "✅" } else { "❌" };
            println!("   {} {}: {}", icon, action, has_permission);
        }

        // Check resource limits
        if let Ok(limits) = rbac.get_user_limits(user).await {
            if let Some(limits) = limits {
                println!("   📊 Resource Limits:");
                if let Some(max_gpus) = limits.max_gpus {
                    println!("      Max GPUs: {}", max_gpus);
                }
                if let Some(max_memory) = limits.max_memory_gb {
                    println!("      Max Memory: {} GB", max_memory);
                }
                if let Some(max_containers) = limits.max_containers {
                    println!("      Max Containers: {}", max_containers);
                }
            } else {
                println!("   📊 Resource Limits: None (unlimited)");
            }
        }

        println!();
    }

    // Demonstrate audit logging
    println!("📝 Recent audit log entries:");
    if let Ok(entries) = rbac.get_audit_log(5).await {
        for entry in entries {
            println!(
                "   [{}] {}: {} -> {} ({})",
                entry.timestamp.format("%H:%M:%S"),
                entry.user,
                entry.action,
                if entry.allowed { "ALLOWED" } else { "DENIED" },
                entry.details.unwrap_or_else(|| "No details".to_string())
            );
        }
    }

    println!("\n🎉 RBAC demo completed successfully!");

    Ok(())
}
