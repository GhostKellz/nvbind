# nvbind CLI Reference

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

