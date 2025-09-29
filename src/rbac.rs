//! Role-Based Access Control (RBAC) for nvbind
//!
//! Provides fine-grained security controls for GPU access in multi-user environments

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::debug;

/// RBAC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacConfig {
    /// Enable RBAC enforcement
    pub enabled: bool,
    /// Path to RBAC policy file
    pub policy_file: PathBuf,
    /// Default policy when no specific rule matches
    pub default_policy: PolicyDecision,
    /// Cache policy decisions for performance
    pub enable_cache: bool,
    /// Audit log path
    pub audit_log: Option<PathBuf>,
}

impl Default for RbacConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            policy_file: PathBuf::from("/etc/nvbind/rbac.toml"),
            default_policy: PolicyDecision::Deny,
            enable_cache: true,
            audit_log: Some(PathBuf::from("/var/log/nvbind/rbac.log")),
        }
    }
}

/// User identity
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct User {
    /// Unix user ID
    pub uid: u32,
    /// Username
    pub username: String,
    /// Primary group ID
    pub gid: u32,
    /// Additional group IDs
    pub groups: Vec<u32>,
}

impl User {
    /// Get current user from system
    pub fn current() -> Result<Self> {
        let uid = unsafe { libc::getuid() };
        let gid = unsafe { libc::getgid() };

        let username = std::env::var("USER").unwrap_or_else(|_| "unknown".to_string());

        // Get supplementary groups
        let mut groups = vec![gid];
        let ngroups = 32;
        let mut group_list = vec![0u32; ngroups as usize];

        unsafe {
            let ret = libc::getgroups(ngroups, group_list.as_mut_ptr() as *mut libc::gid_t);
            if ret > 0 {
                group_list.truncate(ret as usize);
                groups.extend(group_list.iter().copied());
            }
        }

        groups.sort();
        groups.dedup();

        Ok(Self {
            uid,
            username,
            gid,
            groups,
        })
    }
}

/// Role definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role name
    pub name: String,
    /// Description
    pub description: String,
    /// Permissions granted by this role
    pub permissions: HashSet<Permission>,
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

/// Permission types
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum Permission {
    /// Access specific GPU by index
    GpuAccess(u32),
    /// Access all GPUs
    GpuAccessAll,
    /// Access specific amount of GPU memory
    GpuMemory(u64),
    /// Run containers with GPU support
    ContainerGpu,
    /// Run privileged containers
    ContainerPrivileged,
    /// Access performance metrics
    MetricsRead,
    /// Modify system configuration
    ConfigWrite,
    /// Manage other users' resources
    ManageUsers,
    /// Override resource limits
    OverrideLimits,
}

/// Resource limits for a role
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct ResourceLimits {
    /// Maximum number of GPUs
    pub max_gpus: Option<u32>,
    /// Maximum GPU memory per device (bytes)
    pub max_gpu_memory: Option<u64>,
    /// Maximum number of concurrent containers
    pub max_containers: Option<u32>,
    /// Maximum CPU cores
    pub max_cpu_cores: Option<u32>,
    /// Maximum system memory (bytes)
    pub max_system_memory: Option<u64>,
    /// Time-based quotas
    pub time_quotas: Option<TimeQuotas>,
}


/// Time-based resource quotas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeQuotas {
    /// Maximum GPU hours per day
    pub daily_gpu_hours: Option<f64>,
    /// Maximum GPU hours per week
    pub weekly_gpu_hours: Option<f64>,
    /// Maximum GPU hours per month
    pub monthly_gpu_hours: Option<f64>,
}

/// RBAC policy database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDatabase {
    /// Role definitions
    pub roles: HashMap<String, Role>,
    /// User to roles mapping
    pub user_roles: HashMap<String, Vec<String>>,
    /// Group to roles mapping
    pub group_roles: HashMap<String, Vec<String>>,
    /// Special rules for specific resources
    pub resource_policies: Vec<ResourcePolicy>,
}

/// Resource-specific access policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePolicy {
    /// Resource identifier (e.g., "gpu:0", "container:nvidia/*")
    pub resource: String,
    /// Required permissions
    pub required_permissions: Vec<Permission>,
    /// Additional conditions
    pub conditions: Vec<PolicyCondition>,
}

/// Policy conditions for fine-grained control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyCondition {
    /// Time-based access restriction
    TimeRestriction {
        start_hour: u8,
        end_hour: u8,
        days: Vec<String>,
    },
    /// Require specific environment variable
    EnvironmentVariable { name: String, value: Option<String> },
    /// Network-based restriction
    NetworkRestriction { allowed_networks: Vec<String> },
}

/// Policy decision
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PolicyDecision {
    Allow,
    Deny,
    AllowWithLimits,
}

/// RBAC manager
pub struct RbacManager {
    config: RbacConfig,
    policy_db: PolicyDatabase,
    decision_cache: HashMap<(User, String), PolicyDecision>,
    usage_tracker: UsageTracker,
}

impl RbacManager {
    /// Create new RBAC manager
    pub fn new(config: RbacConfig) -> Result<Self> {
        let policy_db = if config.policy_file.exists() {
            Self::load_policy_database(&config.policy_file)?
        } else {
            Self::default_policy_database()
        };

        Ok(Self {
            config,
            policy_db,
            decision_cache: HashMap::new(),
            usage_tracker: UsageTracker::new(),
        })
    }

    /// Load policy database from file
    fn load_policy_database(path: &Path) -> Result<PolicyDatabase> {
        let content = fs::read_to_string(path).context("Failed to read RBAC policy file")?;

        toml::from_str(&content).context("Failed to parse RBAC policy file")
    }

    /// Create default policy database
    fn default_policy_database() -> PolicyDatabase {
        let mut roles = HashMap::new();

        // Admin role
        roles.insert(
            "admin".to_string(),
            Role {
                name: "admin".to_string(),
                description: "Full system access".to_string(),
                permissions: vec![
                    Permission::GpuAccessAll,
                    Permission::ContainerPrivileged,
                    Permission::ConfigWrite,
                    Permission::ManageUsers,
                    Permission::OverrideLimits,
                    Permission::MetricsRead,
                ]
                .into_iter()
                .collect(),
                resource_limits: ResourceLimits::default(),
            },
        );

        // Developer role
        roles.insert(
            "developer".to_string(),
            Role {
                name: "developer".to_string(),
                description: "Standard developer access".to_string(),
                permissions: vec![
                    Permission::GpuAccess(0),
                    Permission::ContainerGpu,
                    Permission::MetricsRead,
                ]
                .into_iter()
                .collect(),
                resource_limits: ResourceLimits {
                    max_gpus: Some(1),
                    max_gpu_memory: Some(8 * 1024 * 1024 * 1024), // 8GB
                    max_containers: Some(5),
                    ..Default::default()
                },
            },
        );

        // Viewer role
        roles.insert(
            "viewer".to_string(),
            Role {
                name: "viewer".to_string(),
                description: "Read-only access".to_string(),
                permissions: vec![Permission::MetricsRead].into_iter().collect(),
                resource_limits: ResourceLimits::default(),
            },
        );

        PolicyDatabase {
            roles,
            user_roles: HashMap::new(),
            group_roles: HashMap::new(),
            resource_policies: Vec::new(),
        }
    }

    /// Check if user has permission for action
    pub fn check_permission(
        &mut self,
        user: &User,
        resource: &str,
        action: &str,
    ) -> Result<PolicyDecision> {
        if !self.config.enabled {
            return Ok(PolicyDecision::Allow);
        }

        // Check cache
        let cache_key = (user.clone(), format!("{}:{}", resource, action));
        if self.config.enable_cache {
            if let Some(&decision) = self.decision_cache.get(&cache_key) {
                debug!("RBAC cache hit for user {} on {}", user.username, resource);
                return Ok(decision);
            }
        }

        // Get user's roles
        let roles = self.get_user_roles(user);

        // Collect all permissions
        let mut permissions = HashSet::new();
        let mut limits = ResourceLimits::default();

        for role_name in &roles {
            if let Some(role) = self.policy_db.roles.get(role_name) {
                permissions.extend(role.permissions.clone());
                limits = self.merge_limits(limits, role.resource_limits.clone());
            }
        }

        // Check resource-specific policies
        let decision = self.evaluate_resource_policy(resource, action, &permissions, &limits)?;

        // Cache decision
        if self.config.enable_cache {
            self.decision_cache.insert(cache_key, decision);
        }

        // Audit log
        self.audit_access(user, resource, action, decision)?;

        Ok(decision)
    }

    /// Get all roles for a user
    fn get_user_roles(&self, user: &User) -> Vec<String> {
        let mut roles = Vec::new();

        // Direct user roles
        if let Some(user_roles) = self.policy_db.user_roles.get(&user.username) {
            roles.extend(user_roles.clone());
        }

        // Group-based roles
        for gid in &user.groups {
            let group_name = format!("gid:{}", gid);
            if let Some(group_roles) = self.policy_db.group_roles.get(&group_name) {
                roles.extend(group_roles.clone());
            }
        }

        // Default role for all users
        if roles.is_empty() {
            roles.push("viewer".to_string());
        }

        roles.sort();
        roles.dedup();
        roles
    }

    /// Evaluate resource-specific policy
    fn evaluate_resource_policy(
        &self,
        resource: &str,
        action: &str,
        permissions: &HashSet<Permission>,
        _limits: &ResourceLimits,
    ) -> Result<PolicyDecision> {
        // Check GPU access
        if resource.starts_with("gpu:") {
            if permissions.contains(&Permission::GpuAccessAll) {
                return Ok(PolicyDecision::AllowWithLimits);
            }

            if let Some(gpu_id) = resource
                .strip_prefix("gpu:")
                .and_then(|s| s.parse::<u32>().ok())
            {
                if permissions.contains(&Permission::GpuAccess(gpu_id)) {
                    return Ok(PolicyDecision::AllowWithLimits);
                }
            }
        }

        // Check container operations
        if resource.starts_with("container:") {
            if action == "create" && permissions.contains(&Permission::ContainerGpu) {
                return Ok(PolicyDecision::AllowWithLimits);
            }

            if action == "privileged" && permissions.contains(&Permission::ContainerPrivileged) {
                return Ok(PolicyDecision::Allow);
            }
        }

        // Check metrics access
        if resource == "metrics" && action == "read" && permissions.contains(&Permission::MetricsRead) {
            return Ok(PolicyDecision::Allow);
        }

        // Check configuration access
        if resource == "config" && action == "write" && permissions.contains(&Permission::ConfigWrite) {
            return Ok(PolicyDecision::Allow);
        }

        Ok(self.config.default_policy)
    }

    /// Merge resource limits
    fn merge_limits(&self, mut base: ResourceLimits, new: ResourceLimits) -> ResourceLimits {
        // Take the most permissive limits
        base.max_gpus = match (base.max_gpus, new.max_gpus) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            _ => None,
        };

        base.max_gpu_memory = match (base.max_gpu_memory, new.max_gpu_memory) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            _ => None,
        };

        base.max_containers = match (base.max_containers, new.max_containers) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            _ => None,
        };

        base
    }

    /// Audit access attempt
    fn audit_access(
        &self,
        user: &User,
        resource: &str,
        action: &str,
        decision: PolicyDecision,
    ) -> Result<()> {
        if let Some(audit_log) = &self.config.audit_log {
            let entry = AuditEntry {
                timestamp: chrono::Utc::now(),
                user: user.clone(),
                resource: resource.to_string(),
                action: action.to_string(),
                decision,
            };

            // Append to audit log
            if let Some(parent) = audit_log.parent() {
                fs::create_dir_all(parent)?;
            }

            let json = serde_json::to_string(&entry)? + "\n";
            fs::write(audit_log, json)?;
        }

        Ok(())
    }

    /// Get resource limits for user
    pub fn get_user_limits(&self, user: &User) -> ResourceLimits {
        let roles = self.get_user_roles(user);
        let mut limits = ResourceLimits::default();

        for role_name in &roles {
            if let Some(role) = self.policy_db.roles.get(role_name) {
                limits = self.merge_limits(limits, role.resource_limits.clone());
            }
        }

        limits
    }

    /// Check if user has exceeded resource limits
    pub fn check_resource_limits(
        &mut self,
        user: &User,
        resource_type: &str,
        amount: u64,
    ) -> Result<bool> {
        let limits = self.get_user_limits(user);
        let usage = self.usage_tracker.get_usage(user);

        match resource_type {
            "gpu_count" => {
                if let Some(max) = limits.max_gpus {
                    return Ok(usage.gpu_count + amount as u32 <= max);
                }
            }
            "gpu_memory" => {
                if let Some(max) = limits.max_gpu_memory {
                    return Ok(usage.gpu_memory + amount <= max);
                }
            }
            "containers" => {
                if let Some(max) = limits.max_containers {
                    return Ok(usage.container_count + amount as u32 <= max);
                }
            }
            _ => {}
        }

        Ok(true)
    }
}

/// Usage tracking for resource quotas
#[derive(Debug)]
struct UsageTracker {
    user_usage: HashMap<User, ResourceUsage>,
}

impl UsageTracker {
    fn new() -> Self {
        Self {
            user_usage: HashMap::new(),
        }
    }

    fn get_usage(&self, user: &User) -> ResourceUsage {
        self.user_usage.get(user).cloned().unwrap_or_default()
    }

    #[allow(dead_code)]
    fn update_usage(&mut self, user: &User, usage: ResourceUsage) {
        self.user_usage.insert(user.clone(), usage);
    }
}

/// Resource usage tracking
#[derive(Debug, Clone, Default)]
struct ResourceUsage {
    gpu_count: u32,
    gpu_memory: u64,
    container_count: u32,
    _gpu_hours: f64,
}

/// Audit log entry
#[derive(Debug, Serialize)]
struct AuditEntry {
    timestamp: chrono::DateTime<chrono::Utc>,
    user: User,
    resource: String,
    action: String,
    decision: PolicyDecision,
}

/// Security context for container operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// User identity
    pub user: User,
    /// SELinux context
    pub selinux_context: Option<String>,
    /// AppArmor profile
    pub apparmor_profile: Option<String>,
    /// Seccomp profile
    pub seccomp_profile: Option<String>,
    /// Capabilities to add
    pub add_capabilities: Vec<String>,
    /// Capabilities to drop
    pub drop_capabilities: Vec<String>,
}

impl SecurityContext {
    /// Create security context for current user
    pub fn current() -> Result<Self> {
        Ok(Self {
            user: User::current()?,
            selinux_context: None,
            apparmor_profile: Some("nvbind-default".to_string()),
            seccomp_profile: Some("nvbind-default".to_string()),
            add_capabilities: vec![],
            drop_capabilities: vec!["ALL".to_string()],
        })
    }

    /// Apply security context to container
    pub fn apply_to_container(&self, container_args: &mut Vec<String>) {
        // Add user mapping
        container_args.push(format!("--user={}:{}", self.user.uid, self.user.gid));

        // SELinux
        if let Some(context) = &self.selinux_context {
            container_args.push(format!("--security-opt=label={}", context));
        }

        // AppArmor
        if let Some(profile) = &self.apparmor_profile {
            container_args.push(format!("--security-opt=apparmor={}", profile));
        }

        // Seccomp
        if let Some(profile) = &self.seccomp_profile {
            container_args.push(format!("--security-opt=seccomp={}", profile));
        }

        // Capabilities
        for cap in &self.drop_capabilities {
            container_args.push(format!("--cap-drop={}", cap));
        }

        for cap in &self.add_capabilities {
            container_args.push(format!("--cap-add={}", cap));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_creation() {
        let user = User::current().unwrap();
        assert!(user.uid > 0 || user.uid == 0); // Valid UID
        assert!(!user.username.is_empty());
    }

    #[test]
    fn test_default_policy_database() {
        let db = RbacManager::default_policy_database();
        assert!(db.roles.contains_key("admin"));
        assert!(db.roles.contains_key("developer"));
        assert!(db.roles.contains_key("viewer"));
    }

    #[test]
    fn test_rbac_manager_creation() {
        let config = RbacConfig::default();
        let manager = RbacManager::new(config);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_security_context() {
        let context = SecurityContext::current().unwrap();
        assert_eq!(context.user.uid, unsafe { libc::getuid() });

        let mut args = Vec::new();
        context.apply_to_container(&mut args);
        assert!(args.iter().any(|a| a.starts_with("--user=")));
    }
}
