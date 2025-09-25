# Role-Based Access Control API Reference

User and group permission management for GPU resources

## Table of Contents

- [Functions](#functions)
- [Structs](#structs)
- [Examples](#examples)

## Functions

### check_permission

Check if user has permission for specific action

**Parameters:**

- `user`: `User` - User to check permissions for (required)
- `action`: `String` - Action to check (e.g., 'gpu.use', 'container.create') (required)

**Returns:** `Result<bool>`

**Errors:**

- User not found
- Invalid action

**Example:**

```rust
use nvbind::rbac::{RbacManager, User};

let rbac = RbacManager::new(config);
let user = User::from_uid(1000)?;
let has_permission = rbac.check_permission(user, "gpu.use").await?;
```

## Structs

### User

Represents a system user

**Fields:**

- `uid`: `u32` - User ID (required)
- `username`: `String` - Username (required)

## Examples

- [rbac_setup](../examples/rbac_setup.md)
- [permission_checking](../examples/permission_checking.md)

