# Advanced Configuration

Detailed guide for advanced nvbind configuration

**Difficulty:** Advanced

## Table of Contents

1. [Configuration File Structure](#1-configuration-file-structure)
2. [Role-Based Access Control](#2-role-based-access-control)

## 1. Configuration File Structure

The main configuration file is located at `/etc/nvbind/config.toml`:

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
```

## 2. Role-Based Access Control

RBAC allows fine-grained control over GPU access:

### Enable RBAC
```toml
[security]
enable_rbac = true
```

### Define Roles
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

### Assign Users to Roles
```bash
nvbind rbac assign-role --user alice --role gpu_user
nvbind rbac assign-role --user bob --role ml_researcher
```

