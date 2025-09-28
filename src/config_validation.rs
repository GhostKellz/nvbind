//! Configuration validation and schema enforcement for nvbind
//!
//! Provides comprehensive validation, migration tools, and schema management

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use toml::Value as TomlValue;
use tracing::info;

/// Schema version for configuration
const SCHEMA_VERSION: &str = "1.0.0";

/// Configuration validator
pub struct ConfigValidator {
    schema: ConfigSchema,
    strict_mode: bool,
}

impl ConfigValidator {
    /// Create new validator with default schema
    pub fn new(strict_mode: bool) -> Self {
        Self {
            schema: ConfigSchema::default(),
            strict_mode,
        }
    }

    /// Validate configuration file
    pub fn validate_file(&self, path: &Path) -> Result<ValidationResult> {
        let content = fs::read_to_string(path).context("Failed to read configuration file")?;

        self.validate_string(&content)
    }

    /// Validate configuration string
    pub fn validate_string(&self, content: &str) -> Result<ValidationResult> {
        let value: TomlValue = toml::from_str(content).context("Failed to parse TOML")?;

        self.validate_value(&value)
    }

    /// Validate parsed configuration
    pub fn validate_value(&self, value: &TomlValue) -> Result<ValidationResult> {
        let mut result = ValidationResult::default();

        // Check schema version
        if let Some(version) = value.get("schema_version").and_then(|v| v.as_str()) {
            if version != SCHEMA_VERSION {
                result.warnings.push(ValidationWarning {
                    field: "schema_version".to_string(),
                    message: format!(
                        "Schema version {} differs from expected {}",
                        version, SCHEMA_VERSION
                    ),
                    severity: WarningSeverity::Medium,
                });
            }
        } else {
            result.warnings.push(ValidationWarning {
                field: "schema_version".to_string(),
                message: "Missing schema version field".to_string(),
                severity: WarningSeverity::Low,
            });
        }

        // Validate each section
        for (section_name, section_schema) in &self.schema.sections {
            self.validate_section(section_name, value, section_schema, &mut result)?;
        }

        // Check for unknown sections in strict mode
        if self.strict_mode {
            if let TomlValue::Table(table) = value {
                for key in table.keys() {
                    if !self.schema.sections.contains_key(key) && key != "schema_version" {
                        result.errors.push(ValidationError {
                            field: key.clone(),
                            message: format!("Unknown configuration section: {}", key),
                            error_type: ErrorType::UnknownField,
                        });
                    }
                }
            }
        }

        result.valid = result.errors.is_empty();
        Ok(result)
    }

    /// Validate configuration section
    fn validate_section(
        &self,
        section_name: &str,
        root: &TomlValue,
        schema: &SectionSchema,
        result: &mut ValidationResult,
    ) -> Result<()> {
        let section = root.get(section_name);

        // Check if required section is missing
        if section.is_none() && schema.required {
            result.errors.push(ValidationError {
                field: section_name.to_string(),
                message: format!("Required section '{}' is missing", section_name),
                error_type: ErrorType::MissingField,
            });
            return Ok(());
        }

        // If section exists, validate its fields
        if let Some(TomlValue::Table(table)) = section {
            for (field_name, field_schema) in &schema.fields {
                self.validate_field(
                    &format!("{}.{}", section_name, field_name),
                    table.get(field_name),
                    field_schema,
                    result,
                )?;
            }

            // Check for unknown fields in strict mode
            if self.strict_mode {
                for key in table.keys() {
                    if !schema.fields.contains_key(key) {
                        result.warnings.push(ValidationWarning {
                            field: format!("{}.{}", section_name, key),
                            message: format!(
                                "Unknown field in section '{}': {}",
                                section_name, key
                            ),
                            severity: WarningSeverity::Low,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate individual field
    fn validate_field(
        &self,
        field_path: &str,
        value: Option<&TomlValue>,
        schema: &FieldSchema,
        result: &mut ValidationResult,
    ) -> Result<()> {
        // Check if required field is missing
        if value.is_none() {
            if schema.required {
                result.errors.push(ValidationError {
                    field: field_path.to_string(),
                    message: format!("Required field '{}' is missing", field_path),
                    error_type: ErrorType::MissingField,
                });
            }
            return Ok(());
        }

        let value = value.unwrap();

        // Type validation
        if !self.validate_type(value, &schema.field_type) {
            result.errors.push(ValidationError {
                field: field_path.to_string(),
                message: format!(
                    "Field '{}' has invalid type. Expected: {:?}",
                    field_path, schema.field_type
                ),
                error_type: ErrorType::InvalidType,
            });
            return Ok(());
        }

        // Value validation
        for validator in &schema.validators {
            if let Some(error) = self.apply_validator(field_path, value, validator) {
                result.errors.push(error);
            }
        }

        Ok(())
    }

    /// Validate value type
    fn validate_type(&self, value: &TomlValue, expected: &FieldType) -> bool {
        match expected {
            FieldType::String => value.is_str(),
            FieldType::Integer => value.is_integer(),
            FieldType::Float => value.is_float() || value.is_integer(),
            FieldType::Boolean => value.is_bool(),
            FieldType::Array(_) => value.is_array(),
            FieldType::Table => value.is_table(),
            FieldType::Path => value.is_str(),
            FieldType::Duration => value.is_str() || value.is_integer(),
        }
    }

    /// Apply validator to field
    fn apply_validator(
        &self,
        field_path: &str,
        value: &TomlValue,
        validator: &FieldValidator,
    ) -> Option<ValidationError> {
        match validator {
            FieldValidator::MinValue(min) => {
                if let Some(num) = value.as_integer() {
                    if num < *min {
                        return Some(ValidationError {
                            field: field_path.to_string(),
                            message: format!("Value {} is less than minimum {}", num, min),
                            error_type: ErrorType::InvalidValue,
                        });
                    }
                }
            }
            FieldValidator::MaxValue(max) => {
                if let Some(num) = value.as_integer() {
                    if num > *max {
                        return Some(ValidationError {
                            field: field_path.to_string(),
                            message: format!("Value {} is greater than maximum {}", num, max),
                            error_type: ErrorType::InvalidValue,
                        });
                    }
                }
            }
            FieldValidator::OneOf(choices) => {
                if let Some(s) = value.as_str() {
                    if !choices.contains(&s.to_string()) {
                        return Some(ValidationError {
                            field: field_path.to_string(),
                            message: format!("Value '{}' is not one of: {:?}", s, choices),
                            error_type: ErrorType::InvalidValue,
                        });
                    }
                }
            }
            FieldValidator::Regex(pattern) => {
                if let Some(s) = value.as_str() {
                    if let Ok(re) = regex::Regex::new(pattern) {
                        if !re.is_match(s) {
                            return Some(ValidationError {
                                field: field_path.to_string(),
                                message: format!(
                                    "Value '{}' does not match pattern '{}'",
                                    s, pattern
                                ),
                                error_type: ErrorType::InvalidValue,
                            });
                        }
                    }
                }
            }
            FieldValidator::PathExists => {
                if let Some(path_str) = value.as_str() {
                    if !Path::new(path_str).exists() {
                        return Some(ValidationError {
                            field: field_path.to_string(),
                            message: format!("Path '{}' does not exist", path_str),
                            error_type: ErrorType::InvalidValue,
                        });
                    }
                }
            }
        }
        None
    }
}

/// Configuration schema definition
#[derive(Debug, Clone)]
pub struct ConfigSchema {
    pub version: String,
    pub sections: HashMap<String, SectionSchema>,
}

impl Default for ConfigSchema {
    fn default() -> Self {
        let mut sections = HashMap::new();

        // GPU configuration schema
        sections.insert(
            "gpu".to_string(),
            SectionSchema {
                required: false,
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert(
                        "default_selection".to_string(),
                        FieldSchema {
                            field_type: FieldType::String,
                            required: false,
                            default: Some(TomlValue::String("all".to_string())),
                            description: "Default GPU selection mode".to_string(),
                            validators: vec![FieldValidator::OneOf(vec![
                                "all".to_string(),
                                "first".to_string(),
                                "none".to_string(),
                            ])],
                        },
                    );
                    fields.insert(
                        "allow_non_nvidia".to_string(),
                        FieldSchema {
                            field_type: FieldType::Boolean,
                            required: false,
                            default: Some(TomlValue::Boolean(false)),
                            description: "Allow non-NVIDIA GPUs".to_string(),
                            validators: vec![],
                        },
                    );
                    fields
                },
            },
        );

        // Runtime configuration schema
        sections.insert(
            "runtime".to_string(),
            SectionSchema {
                required: false,
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert(
                        "default".to_string(),
                        FieldSchema {
                            field_type: FieldType::String,
                            required: false,
                            default: Some(TomlValue::String("podman".to_string())),
                            description: "Default container runtime".to_string(),
                            validators: vec![FieldValidator::OneOf(vec![
                                "podman".to_string(),
                                "docker".to_string(),
                                "containerd".to_string(),
                            ])],
                        },
                    );
                    fields.insert(
                        "timeout_seconds".to_string(),
                        FieldSchema {
                            field_type: FieldType::Integer,
                            required: false,
                            default: Some(TomlValue::Integer(300)),
                            description: "Operation timeout".to_string(),
                            validators: vec![
                                FieldValidator::MinValue(1),
                                FieldValidator::MaxValue(3600),
                            ],
                        },
                    );
                    fields
                },
            },
        );

        // Security configuration schema
        sections.insert(
            "security".to_string(),
            SectionSchema {
                required: false,
                fields: {
                    let mut fields = HashMap::new();
                    fields.insert(
                        "enable_rbac".to_string(),
                        FieldSchema {
                            field_type: FieldType::Boolean,
                            required: false,
                            default: Some(TomlValue::Boolean(false)),
                            description: "Enable role-based access control".to_string(),
                            validators: vec![],
                        },
                    );
                    fields.insert(
                        "allow_privileged".to_string(),
                        FieldSchema {
                            field_type: FieldType::Boolean,
                            required: false,
                            default: Some(TomlValue::Boolean(false)),
                            description: "Allow privileged containers".to_string(),
                            validators: vec![],
                        },
                    );
                    fields
                },
            },
        );

        Self {
            version: SCHEMA_VERSION.to_string(),
            sections,
        }
    }
}

/// Section schema
#[derive(Debug, Clone)]
pub struct SectionSchema {
    pub required: bool,
    pub fields: HashMap<String, FieldSchema>,
}

/// Field schema
#[derive(Debug, Clone)]
pub struct FieldSchema {
    pub field_type: FieldType,
    pub required: bool,
    pub default: Option<TomlValue>,
    pub description: String,
    pub validators: Vec<FieldValidator>,
}

/// Field type
#[derive(Debug, Clone)]
pub enum FieldType {
    String,
    Integer,
    Float,
    Boolean,
    Array(Box<FieldType>),
    Table,
    Path,
    Duration,
}

/// Field validators
#[derive(Debug, Clone)]
pub enum FieldValidator {
    MinValue(i64),
    MaxValue(i64),
    OneOf(Vec<String>),
    Regex(String),
    PathExists,
}

/// Validation result
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

/// Validation error
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub error_type: ErrorType,
}

/// Validation warning
#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub field: String,
    pub message: String,
    pub severity: WarningSeverity,
}

/// Error type
#[derive(Debug, Serialize, Deserialize)]
pub enum ErrorType {
    MissingField,
    InvalidType,
    InvalidValue,
    UnknownField,
}

/// Warning severity
#[derive(Debug, Serialize, Deserialize)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
}

/// Configuration migration tool
pub struct ConfigMigrator {
    migrations: Vec<Box<dyn Migration>>,
}

impl ConfigMigrator {
    /// Create new migrator
    pub fn new() -> Self {
        Self {
            migrations: vec![Box::new(V0ToV1Migration)],
        }
    }

    /// Migrate configuration to latest version
    pub fn migrate(&self, config: &mut TomlValue) -> Result<()> {
        let current_version = config
            .get("schema_version")
            .and_then(|v| v.as_str())
            .unwrap_or("0.0.0")
            .to_string();

        for migration in &self.migrations {
            if migration.should_apply(&current_version) {
                info!(
                    "Applying migration from {} to {}",
                    migration.from_version(),
                    migration.to_version()
                );
                migration.apply(config)?;
            }
        }

        // Update schema version
        if let TomlValue::Table(table) = config {
            table.insert(
                "schema_version".to_string(),
                TomlValue::String(SCHEMA_VERSION.to_string()),
            );
        }

        Ok(())
    }
}

/// Migration trait
trait Migration {
    fn from_version(&self) -> &str;
    fn to_version(&self) -> &str;
    fn should_apply(&self, current_version: &str) -> bool;
    fn apply(&self, config: &mut TomlValue) -> Result<()>;
}

/// Example migration from v0 to v1
struct V0ToV1Migration;

impl Migration for V0ToV1Migration {
    fn from_version(&self) -> &str {
        "0.0.0"
    }

    fn to_version(&self) -> &str {
        "1.0.0"
    }

    fn should_apply(&self, current_version: &str) -> bool {
        current_version < "1.0.0"
    }

    fn apply(&self, config: &mut TomlValue) -> Result<()> {
        if let TomlValue::Table(table) = config {
            // Example: Rename old field names
            if let Some(value) = table.remove("old_field_name") {
                table.insert("new_field_name".to_string(), value);
            }

            // Example: Add new required fields with defaults
            table
                .entry("monitoring".to_string())
                .or_insert_with(|| TomlValue::Table(toml::map::Map::new()));
        }

        Ok(())
    }
}

/// Generate default configuration
pub fn generate_default_config() -> Result<String> {
    let config = toml::toml! {
        schema_version = SCHEMA_VERSION

        [gpu]
        default_selection = "all"
        allow_non_nvidia = false

        [runtime]
        default = "podman"
        timeout_seconds = 300

        [security]
        enable_rbac = false
        allow_privileged = false
        audit_log = "/var/log/nvbind/audit.log"

        [monitoring]
        enabled = true
        prometheus_port = 9090
        collection_interval_secs = 10

        [logging]
        level = "info"
        format = "json"
        output = "stdout"

        [cdi]
        spec_dir = "/etc/cdi"
        auto_generate = true

        [bolt]
        enabled = false
        capsule_path = "/opt/bolt/capsules"
    };

    toml::to_string_pretty(&config).context("Failed to serialize default config")
}

/// Write default configuration to file
pub fn write_default_config(path: &Path) -> Result<()> {
    let config = generate_default_config()?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(path, config)?;
    info!("Written default configuration to {:?}", path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = ConfigValidator::new(false);
        assert!(!validator.strict_mode);

        let strict_validator = ConfigValidator::new(true);
        assert!(strict_validator.strict_mode);
    }

    #[test]
    fn test_valid_config() {
        let config = r#"
            schema_version = "1.0.0"

            [gpu]
            default_selection = "all"

            [runtime]
            default = "podman"
            timeout_seconds = 120
        "#;

        let validator = ConfigValidator::new(false);
        let result = validator.validate_string(config).unwrap();

        assert!(result.valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_invalid_config() {
        let config = r#"
            [gpu]
            default_selection = "invalid_option"

            [runtime]
            timeout_seconds = 10000
        "#;

        let validator = ConfigValidator::new(false);
        let result = validator.validate_string(config).unwrap();

        assert!(!result.valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_default_config_generation() {
        let config = generate_default_config();
        assert!(config.is_ok());

        let config_str = config.unwrap();
        assert!(config_str.contains("schema_version"));
        assert!(config_str.contains("[gpu]"));
        assert!(config_str.contains("[runtime]"));
    }
}
