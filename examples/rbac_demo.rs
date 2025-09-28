//! RBAC Demo Example
//!
//! This example demonstrates role-based access control (RBAC)
//! functionality in nvbind.

use anyhow::Result;
use nvbind::rbac::{Permission, RbacConfig, RbacManager, ResourceLimits, Role, User};
use std::collections::HashSet;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🔐 nvbind RBAC Demo\n");

    // Create RBAC configuration
    let config = RbacConfig::default();

    // Define admin role with full permissions
    let mut admin_permissions = HashSet::new();
    admin_permissions.insert(Permission::GpuAccessAll);
    admin_permissions.insert(Permission::ContainerGpu);
    admin_permissions.insert(Permission::ContainerPrivileged);
    admin_permissions.insert(Permission::MetricsRead);
    admin_permissions.insert(Permission::ConfigWrite);
    admin_permissions.insert(Permission::ManageUsers);
    admin_permissions.insert(Permission::OverrideLimits);

    let admin_role = Role {
        name: "admin".to_string(),
        description: "System administrator with full access".to_string(),
        permissions: admin_permissions,
        resource_limits: ResourceLimits::default(), // No limits for admin
    };

    // Define ML researcher role with GPU access
    let mut ml_permissions = HashSet::new();
    ml_permissions.insert(Permission::GpuAccess(0));
    ml_permissions.insert(Permission::GpuAccess(1));
    ml_permissions.insert(Permission::GpuAccess(2));
    ml_permissions.insert(Permission::GpuAccess(3));
    ml_permissions.insert(Permission::ContainerGpu);
    ml_permissions.insert(Permission::MetricsRead);

    let ml_role = Role {
        name: "ml_researcher".to_string(),
        description: "Machine learning researcher".to_string(),
        permissions: ml_permissions,
        resource_limits: ResourceLimits {
            max_gpus: Some(4),
            max_gpu_memory: Some(32 * 1024 * 1024 * 1024), // 32GB
            max_containers: Some(10),
            max_cpu_cores: Some(16),
            max_system_memory: Some(64 * 1024 * 1024 * 1024), // 64GB
            time_quotas: None,
        },
    };

    // Define regular user role with limited access
    let mut user_permissions = HashSet::new();
    user_permissions.insert(Permission::GpuAccess(0));
    user_permissions.insert(Permission::ContainerGpu);
    user_permissions.insert(Permission::MetricsRead);

    let user_role = Role {
        name: "gpu_user".to_string(),
        description: "Regular user with basic GPU access".to_string(),
        permissions: user_permissions,
        resource_limits: ResourceLimits {
            max_gpus: Some(1),
            max_gpu_memory: Some(8 * 1024 * 1024 * 1024), // 8GB
            max_containers: Some(3),
            max_cpu_cores: Some(4),
            max_system_memory: Some(16 * 1024 * 1024 * 1024), // 16GB
            time_quotas: None,
        },
    };

    // Create RBAC manager
    println!("🚀 Initializing RBAC manager...");
    let mut rbac = RbacManager::new(config)?;

    println!("✅ RBAC manager initialized\n");

    // Simulate different users
    let admin_user = User {
        uid: 0,
        username: "root".to_string(),
        gid: 0,
        groups: vec![0, 1], // root and admin groups
    };

    let ml_user = User {
        uid: 1001,
        username: "alice".to_string(),
        gid: 1001,
        groups: vec![1001, 1002], // alice and researchers groups
    };

    let regular_user = User {
        uid: 1002,
        username: "bob".to_string(),
        gid: 1002,
        groups: vec![1002, 1003], // bob and users groups
    };

    // Test permissions using the actual API
    println!("🧪 Testing permissions:\n");

    let test_permissions = vec![
        ("GPU Access (GPU 0)", Permission::GpuAccess(0)),
        ("GPU Access (All)", Permission::GpuAccessAll),
        ("Container GPU", Permission::ContainerGpu),
        ("Container Privileged", Permission::ContainerPrivileged),
        ("Metrics Read", Permission::MetricsRead),
        ("Config Write", Permission::ConfigWrite),
        ("Manage Users", Permission::ManageUsers),
        ("Override Limits", Permission::OverrideLimits),
    ];

    let users = vec![
        ("admin", &admin_user),
        ("alice (ML researcher)", &ml_user),
        ("bob (regular user)", &regular_user),
    ];

    for (user_desc, user) in users {
        println!("👤 Testing permissions for {}:", user_desc);

        for (desc, permission) in &test_permissions {
            let decision = rbac.check_permission(user, "gpu", &format!("{:?}", permission));
            let icon = match &decision {
                Ok(nvbind::rbac::PolicyDecision::Allow) => "✅",
                Ok(nvbind::rbac::PolicyDecision::AllowWithLimits) => "⚠️",
                Ok(nvbind::rbac::PolicyDecision::Deny) => "❌",
                Err(_) => "⚠️",
            };
            println!("   {} {}: {:?}", icon, desc, decision);
        }

        // Check resource limits
        let limits = rbac.get_user_limits(user);
        println!("   📊 Resource Limits:");
        if let Some(max_gpus) = limits.max_gpus {
            println!("      Max GPUs: {}", max_gpus);
        } else {
            println!("      Max GPUs: unlimited");
        }
        if let Some(max_memory) = limits.max_gpu_memory {
            println!("      Max GPU Memory: {} bytes", max_memory);
        } else {
            println!("      Max GPU Memory: unlimited");
        }
        if let Some(max_containers) = limits.max_containers {
            println!("      Max Containers: {}", max_containers);
        } else {
            println!("      Max Containers: unlimited");
        }

        println!();
    }

    println!("\n🎉 RBAC demo completed successfully!");

    Ok(())
}
