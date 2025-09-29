use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::{debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub id: String,
    pub name: String,
    pub pci_address: String,
    pub driver_version: Option<String>,
    pub memory: Option<u64>,
    pub device_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DriverType {
    /// NVIDIA Open GPU Kernel Modules (recommended)
    NvidiaOpen,
    /// NVIDIA Proprietary Driver
    NvidiaProprietary,
    /// Nouveau Open Source Driver
    Nouveau,
}

impl DriverType {
    pub fn name(&self) -> &'static str {
        match self {
            DriverType::NvidiaOpen => "NVIDIA Open",
            DriverType::NvidiaProprietary => "NVIDIA Proprietary",
            DriverType::Nouveau => "Nouveau",
        }
    }
}

/// GPU driver errors for better error handling
#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum GpuDriverError {
    #[error("NVIDIA driver not found - no GPU driver modules are loaded")]
    NoDriverFound,
    #[error("NVIDIA driver found but no GPUs detected - driver may be incompatible")]
    NoGpusDetected,
    #[error("NVIDIA driver permissions issue - try running with appropriate privileges")]
    PermissionDenied,
    #[error("Incompatible driver version: {version} - minimum CUDA {min_cuda} required")]
    IncompatibleVersion { version: String, min_cuda: String },
    #[error("Hardware not supported: {details}")]
    HardwareNotSupported { details: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverInfo {
    pub version: String,
    pub driver_type: DriverType,
    pub cuda_version: Option<String>,
    pub libraries: Vec<String>,
}

impl Default for DriverInfo {
    fn default() -> Self {
        Self {
            version: "unknown".to_string(),
            driver_type: DriverType::Nouveau,
            cuda_version: None,
            libraries: Vec::new(),
        }
    }
}

pub async fn info() -> Result<()> {
    // Enhanced driver availability check with diagnostics
    match check_nvidia_driver_status() {
        Ok(_) => {}
        Err(e) => {
            warn!("GPU driver issue detected: {}", e);
            print_driver_diagnostics();
            return Err(e.into());
        }
    }

    let gpus = discover_gpus().await?;
    let driver_info = get_driver_info().await?;

    println!("=== NVIDIA GPU Information ===");
    println!("Driver Version: {}", driver_info.version);
    println!(
        "Driver Type: {}",
        match driver_info.driver_type {
            DriverType::NvidiaOpen => "NVIDIA Open GPU Kernel Modules (recommended)",
            DriverType::NvidiaProprietary => "NVIDIA Proprietary Driver",
            DriverType::Nouveau => "Nouveau Open Source Driver",
        }
    );
    if let Some(cuda) = &driver_info.cuda_version {
        println!("CUDA Version: {}", cuda);
    }
    println!();

    if gpus.is_empty() {
        println!("No NVIDIA GPUs detected");
        return Ok(());
    }

    println!("Detected GPUs:");
    for gpu in &gpus {
        println!("  GPU {}: {}", gpu.id, gpu.name);
        println!("    PCI Address: {}", gpu.pci_address);
        println!("    Device Path: {}", gpu.device_path);
        if let Some(memory) = gpu.memory {
            println!("    Memory: {} MB", memory / 1024 / 1024);
        }
        println!();
    }

    println!("Driver Libraries:");
    for lib in &driver_info.libraries {
        println!("  {}", lib);
    }

    Ok(())
}

pub async fn discover_gpus() -> Result<Vec<GpuDevice>> {
    let mut gpus = Vec::new();

    let nvidia_devices = find_nvidia_devices_internal()?;
    debug!("Found {} NVIDIA device(s)", nvidia_devices.len());

    for (i, device_path) in nvidia_devices.iter().enumerate() {
        if let Ok(gpu) = create_gpu_device(i, device_path) {
            gpus.push(gpu);
        }
    }

    Ok(gpus)
}

fn find_nvidia_devices_internal() -> Result<Vec<String>> {
    let mut devices = Vec::new();

    // Check /dev for NVIDIA devices
    let dev_path = Path::new("/dev");
    if dev_path.exists() {
        if let Ok(entries) = fs::read_dir(dev_path) {
            for entry in entries.flatten() {
                let file_name = entry.file_name();
                let name = file_name.to_string_lossy();

                // Look for nvidia devices (nvidia0, nvidia1, etc.)
                if name.starts_with("nvidia") && name.len() > 6 {
                    if let Some(suffix) = name.strip_prefix("nvidia") {
                        if suffix.chars().all(|c| c.is_ascii_digit()) {
                            devices.push(entry.path().to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }

    // Also check for nvidia-uvm, nvidia-modeset, etc.
    let control_devices = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-modeset"];
    for device in &control_devices {
        if Path::new(device).exists() {
            devices.push(device.to_string());
        }
    }

    Ok(devices)
}

fn create_gpu_device(index: usize, device_path: &str) -> Result<GpuDevice> {
    let name = get_gpu_name(index)?;
    let pci_address = get_pci_address(index)?;
    let memory = get_gpu_memory(index)?;

    Ok(GpuDevice {
        id: index.to_string(),
        name,
        pci_address,
        driver_version: None,
        memory,
        device_path: device_path.to_string(),
    })
}

fn get_gpu_name(index: usize) -> Result<String> {
    let path = format!("/proc/driver/nvidia/gpus/nvidia{}/information", index);
    if let Ok(content) = fs::read_to_string(&path) {
        for line in content.lines() {
            if line.starts_with("Model:") {
                return Ok(line.replace("Model:", "").trim().to_string());
            }
        }
    }

    // Fallback: try nvidia-ml-py style path
    let alt_path = format!("/sys/class/drm/card{}/device/device", index);
    if let Ok(device_id) = fs::read_to_string(&alt_path) {
        return Ok(format!("NVIDIA GPU (Device ID: {})", device_id.trim()));
    }

    Ok(format!("NVIDIA GPU {}", index))
}

fn get_pci_address(index: usize) -> Result<String> {
    let path = format!("/proc/driver/nvidia/gpus/nvidia{}/information", index);
    if let Ok(content) = fs::read_to_string(&path) {
        for line in content.lines() {
            if line.starts_with("Bus Location:") {
                return Ok(line.replace("Bus Location:", "").trim().to_string());
            }
        }
    }

    // Fallback: try to get from sysfs
    let alt_path = format!("/sys/class/drm/card{}/device/uevent", index);
    if let Ok(content) = fs::read_to_string(&alt_path) {
        for line in content.lines() {
            if line.starts_with("PCI_SLOT_NAME=") {
                return Ok(line.replace("PCI_SLOT_NAME=", ""));
            }
        }
    }

    Ok(format!("unknown:{}", index))
}

fn get_gpu_memory(index: usize) -> Result<Option<u64>> {
    let path = format!("/proc/driver/nvidia/gpus/nvidia{}/information", index);
    if let Ok(content) = fs::read_to_string(&path) {
        let re = Regex::new(r"(\d+)\s*MB")?;
        for line in content.lines() {
            if line.contains("Memory:") {
                if let Some(caps) = re.captures(line) {
                    if let Ok(mb) = caps[1].parse::<u64>() {
                        return Ok(Some(mb * 1024 * 1024)); // Convert to bytes
                    }
                }
            }
        }
    }
    Ok(None)
}

pub async fn get_driver_info() -> Result<DriverInfo> {
    let (version, driver_type) = get_driver_version_and_type()?;
    let cuda_version = get_cuda_version().ok();
    let mut libraries = find_nvidia_libraries()?;

    // Add proprietary driver libraries if available
    if matches!(
        driver_type,
        DriverType::NvidiaProprietary | DriverType::NvidiaOpen
    ) && validate_proprietary_container_support()
    {
        let prop_libraries = get_proprietary_driver_libraries();
        libraries.extend(prop_libraries);
    }

    Ok(DriverInfo {
        version,
        driver_type,
        cuda_version,
        libraries,
    })
}

fn get_driver_version_and_type() -> Result<(String, DriverType)> {
    // First, try to determine driver type from kernel modules
    let driver_type = detect_driver_type();

    // Get version based on driver type with enhanced methods
    let version = match driver_type {
        DriverType::Nouveau => get_nouveau_version(),
        DriverType::NvidiaOpen => {
            // Try generic NVIDIA methods first, then fallback to Open-specific
            get_nvidia_version().or_else(|_| {
                debug!(
                    "Generic NVIDIA version detection failed for Open driver, trying alternatives"
                );
                validate_proprietary_driver() // This works for both Open and Proprietary
            })
        }
        DriverType::NvidiaProprietary => {
            // Try proprietary-specific methods first, then generic NVIDIA methods
            validate_proprietary_driver().or_else(|_| {
                debug!("Proprietary-specific detection failed, trying generic NVIDIA methods");
                get_nvidia_version()
            })
        }
    };

    match version {
        Ok(v) => Ok((v, driver_type)),
        Err(e) => {
            debug!("Failed to get driver version for {:?}: {}", driver_type, e);
            Err(e)
        }
    }
}

pub fn detect_driver_type() -> DriverType {
    // Check loaded kernel modules
    if let Ok(modules) = fs::read_to_string("/proc/modules") {
        if modules.contains("nouveau") {
            return DriverType::Nouveau;
        }
        if modules.contains("nvidia") {
            // Check if it's the open source version
            if is_nvidia_open_driver() {
                return DriverType::NvidiaOpen;
            } else {
                return DriverType::NvidiaProprietary;
            }
        }
    }

    // Fallback: check /sys/module
    if Path::new("/sys/module/nouveau").exists() {
        return DriverType::Nouveau;
    }
    if Path::new("/sys/module/nvidia").exists() {
        if is_nvidia_open_driver() {
            return DriverType::NvidiaOpen;
        } else {
            return DriverType::NvidiaProprietary;
        }
    }

    // Default assumption for NVIDIA systems
    DriverType::NvidiaProprietary
}

fn is_nvidia_open_driver() -> bool {
    // Enhanced NVIDIA Open GPU Kernel Modules detection
    // Specifically validate for version 580+ open driver support

    // Method 1: Check /proc/driver/nvidia/version for open kernel indicators
    if let Ok(version_content) = fs::read_to_string("/proc/driver/nvidia/version") {
        // Open driver versions 515+ mention "Open Kernel Modules" or similar
        if version_content.contains("Open Kernel")
            || version_content.contains("open-gpu-kernel-modules")
            || version_content.contains("GSP")
        {
            debug!(
                "Detected NVIDIA Open driver via version string: {}",
                version_content.lines().next().unwrap_or("")
            );

            // Extract version number to validate 580+
            if let Some(version_line) = version_content.lines().next() {
                if let Some(version_match) = Regex::new(r"(\d{3}\.\d+)")
                    .ok()
                    .and_then(|re| re.captures(version_line))
                {
                    if let Ok(version_num) = version_match[1].parse::<f32>() {
                        debug!("Detected NVIDIA driver version: {}", version_num);
                        // Open driver is fully supported from 515+, excellent from 580+
                        return version_num >= 515.0;
                    }
                }
            }
            return true;
        }
    }

    // Method 2: Check for GSP (GPU System Processor) firmware - primary indicator for open driver
    if let Ok(gsp_content) =
        fs::read_to_string("/sys/module/nvidia/parameters/NVreg_EnableGpuFirmware")
    {
        if gsp_content.trim() == "1" {
            debug!("Detected NVIDIA Open driver via GSP firmware enablement");
            return true;
        }
    }

    // Method 3: Check nvidia-drm modeset parameter (open driver defaults to Y)
    if let Ok(modeset_content) = fs::read_to_string("/sys/module/nvidia_drm/parameters/modeset") {
        if modeset_content.trim() == "Y" {
            // Additional validation - check for open driver specific paths
            if Path::new("/sys/module/nvidia").exists() {
                // Check for open driver specific parameters
                if Path::new("/sys/module/nvidia/parameters/NVreg_OpenRmEnableUnsupportedGpus")
                    .exists()
                {
                    debug!("Detected NVIDIA Open driver via open RM parameters");
                    return true;
                }
            }
        }
    }

    // Method 4: Check loaded modules for open driver indicators
    if let Ok(modules_content) = fs::read_to_string("/proc/modules") {
        // Open driver has specific module signatures
        let open_indicators = ["nvidia_drm", "nvidia_modeset", "nvidia_uvm"];
        let open_count = open_indicators
            .iter()
            .filter(|&indicator| modules_content.contains(indicator))
            .count();

        if open_count >= 2 && modules_content.contains("nvidia ") {
            // Likely open driver with multiple components loaded
            debug!("Detected NVIDIA Open driver via module analysis");
            return true;
        }
    }

    // Method 5: Check driver capabilities - open driver supports new features
    if Path::new("/dev/nvidia-caps").exists() {
        // nvidia-caps device is more common with open driver
        debug!("Detected NVIDIA Open driver via capabilities device");
        return true;
    }

    // Check module info for open source indicators
    if let Ok(output) = std::process::Command::new("modinfo").arg("nvidia").output() {
        if output.status.success() {
            let modinfo = String::from_utf8_lossy(&output.stdout);
            if modinfo.contains("open-gpu-kernel-modules")
                || modinfo.contains("NVIDIA Open GPU Kernel Module")
            {
                return true;
            }
        }
    }

    false
}

/// Enhanced NVIDIA Proprietary driver validation
/// Comprehensive detection for traditional NVIDIA drivers
fn validate_proprietary_driver() -> Result<String> {
    // Method 1: Check /proc/driver/nvidia/version (primary method for proprietary)
    if let Ok(version_content) = fs::read_to_string("/proc/driver/nvidia/version") {
        // Proprietary driver has specific version format
        if let Some(version_line) = version_content.lines().next() {
            debug!("Found NVIDIA version info: {}", version_line);

            // Extract version from proprietary driver format
            if let Some(version_match) =
                Regex::new(r"NVIDIA UNIX x86_64 Kernel Module\s+(\d+\.\d+)")
                    .ok()
                    .and_then(|re| re.captures(version_line))
            {
                let version = version_match[1].to_string();
                debug!("Detected NVIDIA Proprietary driver version: {}", version);
                return Ok(version);
            }

            // Alternative version extraction for different formats
            if let Some(version_match) = Regex::new(r"(\d{3}\.\d+)")
                .ok()
                .and_then(|re| re.captures(version_line))
            {
                let version = version_match[1].to_string();
                debug!("Detected NVIDIA driver version (alt format): {}", version);
                return Ok(version);
            }
        }
    }

    // Method 2: Try nvidia-smi for version detection
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            let version_str = String::from_utf8_lossy(&output.stdout);
            if !version_str.trim().is_empty() {
                let version = version_str.trim().to_string();
                debug!(
                    "Detected NVIDIA Proprietary driver via nvidia-smi: {}",
                    version
                );
                return Ok(version);
            }
        }
    }

    // Method 3: Check modinfo for proprietary driver information
    if let Ok(output) = std::process::Command::new("modinfo").arg("nvidia").output() {
        if output.status.success() {
            let modinfo_str = String::from_utf8_lossy(&output.stdout);
            if let Some(version_match) = Regex::new(r"version:\s*([^\s]+)")
                .ok()
                .and_then(|re| re.captures(&modinfo_str))
            {
                let version = version_match[1].to_string();
                debug!(
                    "Detected NVIDIA Proprietary driver via modinfo: {}",
                    version
                );
                return Ok(version);
            }
        }
    }

    // Method 4: Check nvidia-ml library version
    if let Ok(output) = std::process::Command::new("nvidia-ml-py3")
        .args([
            "-c",
            "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlSystemGetDriverVersion())",
        ])
        .output()
    {
        if output.status.success() {
            let version_str = String::from_utf8_lossy(&output.stdout);
            if !version_str.trim().is_empty() {
                let version = version_str.trim().to_string();
                debug!(
                    "Detected NVIDIA Proprietary driver via nvidia-ml: {}",
                    version
                );
                return Ok(version);
            }
        }
    }

    Err(anyhow::anyhow!(
        "Could not determine NVIDIA Proprietary driver version"
    ))
}

/// Check if the proprietary driver supports container operations
fn validate_proprietary_container_support() -> bool {
    // Check for essential proprietary driver container features
    let essential_features = [
        "/dev/nvidiactl",      // Primary control device
        "/dev/nvidia-uvm",     // Unified Virtual Memory
        "/dev/nvidia-modeset", // Mode setting (if available)
        "/proc/driver/nvidia", // Driver proc interface
    ];

    let mut supported_features = 0;
    for feature_path in &essential_features {
        if std::path::Path::new(feature_path).exists() {
            supported_features += 1;
            debug!("Proprietary driver feature available: {}", feature_path);
        }
    }

    // Need at least nvidiactl and nvidia-uvm for container support
    if supported_features >= 2 {
        debug!("Proprietary driver has sufficient container support features");
        return true;
    }

    // Check for CUDA runtime support
    if std::process::Command::new("nvidia-smi")
        .arg("--list-gpus")
        .output()
        .is_ok()
    {
        debug!("nvidia-smi available, proprietary driver has basic GPU management");
        return true;
    }

    // Check for libnvidia-ml (management library)
    let nvidia_ml_paths = [
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib64/libnvidia-ml.so.1",
        "/usr/lib/libnvidia-ml.so.1",
    ];

    for lib_path in &nvidia_ml_paths {
        if std::path::Path::new(lib_path).exists() {
            debug!("NVIDIA-ML library found: {}", lib_path);
            return true;
        }
    }

    false
}

/// Get proprietary driver specific information
fn get_proprietary_driver_libraries() -> Vec<String> {
    let mut libraries = Vec::new();

    // Essential proprietary driver libraries
    let essential_libs = [
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvcuvid.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-decode.so.1",
    ];

    // Alternative paths for different distributions
    let alt_lib_paths = ["/usr/lib64", "/usr/lib", "/lib/x86_64-linux-gnu", "/lib64"];

    for lib in &essential_libs {
        if std::path::Path::new(lib).exists() {
            libraries.push(lib.to_string());
        } else {
            // Check alternative paths
            let lib_name = std::path::Path::new(lib)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_else(|| {
                    warn!("Invalid library path: {}", lib);
                    "unknown"
                });
            for alt_path in &alt_lib_paths {
                let full_path = format!("{}/{}", alt_path, lib_name);
                if std::path::Path::new(&full_path).exists() {
                    libraries.push(full_path);
                    break;
                }
            }
        }
    }

    // Check for additional proprietary libraries
    let additional_libs = [
        "libnvidia-cfg.so.1",
        "libnvidia-compiler.so.1",
        "libnvidia-opencl.so.1",
        "libnvidia-ptxjitcompiler.so.1",
    ];

    for lib_name in &additional_libs {
        for alt_path in &alt_lib_paths {
            let full_path = format!("{}/{}", alt_path, lib_name);
            if std::path::Path::new(&full_path).exists() {
                libraries.push(full_path);
                break;
            }
        }
    }

    libraries.sort();
    libraries.dedup();
    libraries
}

fn get_nvidia_version() -> Result<String> {
    // Try reading from /proc/driver/nvidia/version
    if let Ok(content) = fs::read_to_string("/proc/driver/nvidia/version") {
        for line in content.lines() {
            if line.contains("Kernel Module") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(version) = parts.iter().find(|&s| {
                    s.contains('.') && s.chars().next().is_some_and(|c| c.is_ascii_digit())
                }) {
                    return Ok(version.to_string());
                }
            }
        }
    }

    // Fallback: try nvidia-smi
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=driver_version")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !version.is_empty() {
                return Ok(version);
            }
        }
    }

    Err(anyhow::anyhow!("Could not determine NVIDIA driver version"))
}

fn get_nouveau_version() -> Result<String> {
    // Try to get Nouveau version from kernel
    if let Ok(output) = std::process::Command::new("modinfo")
        .arg("nouveau")
        .output()
    {
        if output.status.success() {
            let modinfo = String::from_utf8_lossy(&output.stdout);
            for line in modinfo.lines() {
                if line.starts_with("version:") {
                    if let Some(version) = line.split_whitespace().nth(1) {
                        return Ok(version.to_string());
                    }
                }
            }
        }
    }

    // Fallback: get kernel version as Nouveau is built into kernel
    if let Ok(output) = std::process::Command::new("uname").arg("-r").output() {
        if output.status.success() {
            let kernel_version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            return Ok(format!("nouveau (kernel {})", kernel_version));
        }
    }

    Ok("nouveau (unknown version)".to_string())
}

fn get_cuda_version() -> Result<String> {
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=cuda_version")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !version.is_empty() {
                return Ok(version);
            }
        }
    }

    // Try reading CUDA version from driver info
    if let Ok(content) = fs::read_to_string("/proc/driver/nvidia/version") {
        if let Some(start) = content.find("CUDA Version") {
            let cuda_line = &content[start..];
            if let Some(version_start) = cuda_line.find(char::is_numeric) {
                let version_part = &cuda_line[version_start..];
                if let Some(version_end) = version_part.find(char::is_whitespace) {
                    return Ok(version_part[..version_end].to_string());
                }
            }
        }
    }

    Err(anyhow::anyhow!("Could not determine CUDA version"))
}

fn find_nvidia_libraries() -> Result<Vec<String>> {
    let mut libraries = Vec::new();
    let search_paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/local/lib",
        "/usr/local/lib64",
        "/lib64",
        "/lib/x86_64-linux-gnu",
    ];

    for search_path in &search_paths {
        if let Ok(entries) = fs::read_dir(search_path) {
            for entry in entries.flatten() {
                let file_name = entry.file_name();
                let name = file_name.to_string_lossy();

                if name.starts_with("libnvidia") || name.starts_with("libcuda") {
                    libraries.push(entry.path().to_string_lossy().to_string());
                }
            }
        }
    }

    libraries.sort();
    libraries.dedup();
    Ok(libraries)
}

pub fn find_nvidia_devices() -> Result<Vec<String>> {
    find_nvidia_devices_internal()
}

pub fn get_required_devices() -> Vec<String> {
    let mut devices = vec!["/dev/nvidiactl".to_string(), "/dev/nvidia-uvm".to_string()];

    // Add specific GPU devices
    for i in 0..16 {
        // Check up to 16 GPUs
        let device = format!("/dev/nvidia{}", i);
        if Path::new(&device).exists() {
            devices.push(device);
        }
    }

    devices
}

pub fn get_required_libraries() -> Result<Vec<String>> {
    find_nvidia_libraries()
}

pub fn is_nvidia_driver_available() -> bool {
    check_nvidia_driver_status().is_ok()
}

/// Enhanced NVIDIA driver status check with detailed error reporting
pub fn check_nvidia_driver_status() -> Result<(), GpuDriverError> {
    // Check for NVIDIA drivers (proprietary or open)
    let nvidia_paths = [
        "/proc/driver/nvidia/version",
        "/dev/nvidiactl",
        "/sys/module/nvidia",
    ];

    let nouveau_paths = ["/sys/module/nouveau", "/dev/dri"];

    // Check NVIDIA proprietary/open drivers first
    if nvidia_paths.iter().any(|path| Path::new(path).exists()) {
        // Driver exists, check if devices are accessible
        if !Path::new("/dev/nvidiactl").exists() {
            return Err(GpuDriverError::PermissionDenied);
        }
        return Ok(());
    }

    // Check Nouveau driver
    if nouveau_paths.iter().any(|path| Path::new(path).exists()) {
        warn!("Nouveau driver detected - performance may be limited");
        return Ok(());
    }

    Err(GpuDriverError::NoDriverFound)
}

/// Print detailed driver diagnostics to help users troubleshoot
pub fn print_driver_diagnostics() {
    println!("\n=== GPU Driver Diagnostics ===");

    // Check for common NVIDIA driver files
    let checks = [
        ("/proc/driver/nvidia/version", "NVIDIA driver version info"),
        ("/dev/nvidiactl", "NVIDIA control device"),
        ("/dev/nvidia-uvm", "NVIDIA unified memory"),
        ("/sys/module/nvidia", "NVIDIA kernel module"),
        ("/sys/module/nouveau", "Nouveau (open) driver"),
        ("/usr/bin/nvidia-smi", "nvidia-smi utility"),
    ];

    for (path, description) in &checks {
        let status = if Path::new(path).exists() {
            "✅ Found"
        } else {
            "❌ Missing"
        };
        println!("  {}: {}", description, status);
    }

    // Check if nvidia-smi works
    if Path::new("/usr/bin/nvidia-smi").exists() {
        if let Ok(output) = std::process::Command::new("nvidia-smi").output() {
            if output.status.success() {
                println!("  nvidia-smi: ✅ Working");
            } else {
                println!("  nvidia-smi: ❌ Failed to execute");
            }
        }
    }

    println!("\nTroubleshooting:");
    println!("  • Install NVIDIA drivers: sudo apt install nvidia-driver-xxx");
    println!("  • Load NVIDIA module: sudo modprobe nvidia");
    println!("  • Check driver status: sudo nvidia-smi");
    println!("  • For permissions: sudo usermod -a -G video $USER");
}

pub fn check_nvidia_requirements() -> Result<()> {
    if !is_nvidia_driver_available() {
        return Err(anyhow::anyhow!(
            "NVIDIA GPU driver not available. Please install drivers:\n\
             \n\
             🚀 RECOMMENDED - NVIDIA Open GPU Kernel Modules:\n\
             - GitHub: https://github.com/NVIDIA/open-gpu-kernel-modules\n\
             - Ubuntu/Debian: sudo apt install nvidia-open\n\
             - Arch: sudo pacman -S nvidia-open\n\
             \n\
             🔒 NVIDIA Proprietary Driver:\n\
             - Ubuntu/Debian: sudo apt install nvidia-driver-xxx\n\
             - RHEL/CentOS: sudo dnf install nvidia-driver\n\
             - Download: https://www.nvidia.com/drivers\n\
             \n\
             🆓 Nouveau Open Source Driver:\n\
             - Usually included in kernel (may have limited features)\n\
             - Enable: sudo modprobe nouveau"
        ));
    }

    let devices = find_nvidia_devices_internal()?;
    if devices.is_empty() {
        return Err(anyhow::anyhow!(
            "No NVIDIA devices found. Ensure:\n\
             - NVIDIA GPU is installed and detected\n\
             - GPU kernel modules are loaded (lsmod | grep -E 'nvidia|nouveau')\n\
             - User has permission to access /dev/nvidia* or /dev/dri/*"
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_device_creation() {
        let gpu = GpuDevice {
            id: "0".to_string(),
            name: "Test GPU".to_string(),
            pci_address: "0000:01:00.0".to_string(),
            driver_version: Some("580.82.09".to_string()),
            memory: Some(8 * 1024 * 1024 * 1024), // 8GB
            device_path: "/dev/nvidia0".to_string(),
        };

        assert_eq!(gpu.id, "0");
        assert_eq!(gpu.name, "Test GPU");
        assert_eq!(gpu.pci_address, "0000:01:00.0");
        assert_eq!(gpu.memory, Some(8 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_driver_info_creation() {
        let driver = DriverInfo {
            version: "580.82.09".to_string(),
            driver_type: DriverType::NvidiaOpen,
            cuda_version: Some("12.3".to_string()),
            libraries: vec![
                "/lib64/libcuda.so".to_string(),
                "/lib64/libnvidia-ml.so".to_string(),
            ],
        };

        assert_eq!(driver.version, "580.82.09");
        assert_eq!(driver.driver_type, DriverType::NvidiaOpen);
        assert_eq!(driver.cuda_version, Some("12.3".to_string()));
        assert_eq!(driver.libraries.len(), 2);
    }

    #[test]
    fn test_driver_types() {
        let open_driver = DriverType::NvidiaOpen;
        let proprietary_driver = DriverType::NvidiaProprietary;
        let nouveau_driver = DriverType::Nouveau;

        // Test PartialEq implementation
        assert_eq!(open_driver, DriverType::NvidiaOpen);
        assert_ne!(open_driver, proprietary_driver);
        assert_ne!(proprietary_driver, nouveau_driver);

        // Test Debug format (useful for logging)
        assert!(format!("{:?}", open_driver).contains("NvidiaOpen"));
        assert!(format!("{:?}", proprietary_driver).contains("NvidiaProprietary"));
        assert!(format!("{:?}", nouveau_driver).contains("Nouveau"));
    }

    #[test]
    fn test_detect_driver_type() {
        // This test verifies the function runs without panicking
        let driver_type = detect_driver_type();
        // Test that we get a valid driver type enum variant
        match driver_type {
            DriverType::NvidiaOpen | DriverType::NvidiaProprietary | DriverType::Nouveau => {
                // All variants are valid
            }
        }
    }

    #[test]
    fn test_is_nvidia_open_driver() {
        // This function checks various system paths and commands
        // We just verify it runs without panicking
        let _is_open = is_nvidia_open_driver();
        // Function completed without panicking
    }

    #[test]
    fn test_is_nvidia_driver_available() {
        // This test checks if the function works without panicking
        let _available = is_nvidia_driver_available();
        // Function completed without panicking
    }

    #[test]
    fn test_find_nvidia_devices() {
        let devices = find_nvidia_devices();
        // Should not panic and return a result
        assert!(devices.is_ok());
    }

    #[test]
    fn test_check_nvidia_requirements() {
        let result = check_nvidia_requirements();
        // Should return either Ok or Err without panicking
        match result {
            Ok(()) => {
                // NVIDIA requirements met
            }
            Err(e) => {
                // Error message should contain helpful information
                let error_msg = e.to_string();
                assert!(error_msg.contains("NVIDIA") || error_msg.contains("driver"));
            }
        }
    }

    #[test]
    fn test_get_required_devices() {
        let devices = get_required_devices();
        assert!(!devices.is_empty());
        assert!(devices.contains(&"/dev/nvidiactl".to_string()));
        assert!(devices.contains(&"/dev/nvidia-uvm".to_string()));
    }

    #[test]
    fn test_get_required_libraries() {
        let result = get_required_libraries();
        assert!(result.is_ok());
        // Libraries list could be empty on systems without NVIDIA
        let libraries = result.expect("Library detection should work in tests");
        for lib in &libraries {
            assert!(lib.contains("nvidia") || lib.contains("cuda"));
        }
    }

    #[tokio::test]
    async fn test_discover_gpus() {
        let result = discover_gpus().await;
        assert!(result.is_ok());
        let gpus = result.expect("GPU discovery should work in tests");
        // GPUs list could be empty on systems without NVIDIA GPUs
        for gpu in &gpus {
            assert!(!gpu.id.is_empty());
            assert!(!gpu.device_path.is_empty());
        }
    }

    // Tests for enhanced error handling features
    #[test]
    fn test_gpu_driver_error_types() {
        let no_driver = GpuDriverError::NoDriverFound;
        let permission_error = GpuDriverError::PermissionDenied;
        let version_error = GpuDriverError::IncompatibleVersion {
            version: "450.00".to_string(),
            min_cuda: "11.0".to_string(),
        };

        // Test error messages contain expected content
        assert!(no_driver.to_string().contains("driver not found"));
        assert!(permission_error.to_string().contains("permission"));
        assert!(version_error.to_string().contains("450.00"));
        assert!(version_error.to_string().contains("11.0"));
    }

    #[test]
    fn test_enhanced_driver_status_check() {
        let result = check_nvidia_driver_status();
        // Function should not panic regardless of result
        match result {
            Ok(_) => {
                // Driver available
                assert!(is_nvidia_driver_available());
            }
            Err(error) => {
                // Test specific error conditions
                match error {
                    GpuDriverError::NoDriverFound => {
                        assert!(!is_nvidia_driver_available());
                    }
                    GpuDriverError::PermissionDenied => {
                        // Permission issue - driver exists but not accessible
                    }
                    _ => {
                        // Other errors
                    }
                }
            }
        }
    }

    #[test]
    fn test_print_driver_diagnostics_no_panic() {
        // Test that diagnostics printing doesn't panic
        print_driver_diagnostics();
        // If we reach here, the function completed without panicking
    }

    #[tokio::test]
    async fn test_enhanced_info_function() {
        // Test the enhanced info function with better error handling
        let result = info().await;

        // Should either succeed or fail gracefully with helpful error
        match result {
            Ok(_) => {
                // Success case - driver available and working
            }
            Err(e) => {
                // Error case - should have helpful error message
                let error_msg = e.to_string();
                assert!(
                    error_msg.contains("driver")
                        || error_msg.contains("GPU")
                        || error_msg.contains("NVIDIA")
                );
            }
        }
    }

    #[test]
    fn test_driver_error_from_anyhow() {
        // Test that our custom errors can be converted to anyhow::Error
        let gpu_error = GpuDriverError::NoDriverFound;
        let anyhow_error: anyhow::Error = gpu_error.into();
        assert!(anyhow_error.to_string().contains("driver not found"));
    }
}
