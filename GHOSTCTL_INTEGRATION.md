# GhostCTL Integration Guide for nvbind

This document outlines the integration between nvbind GPU container runtime and GhostCTL, the all-purpose sysadmin CLI/TUI tool.

## 🎯 Integration Overview

GhostCTL + nvbind creates a unified GPU container management experience, leveraging GhostCTL's "🐳 DevOps & Container Management" framework to provide intuitive GPU runtime administration.

## 🏗️ Architecture Integration

### 1. GhostCTL Module Structure

```rust
// ghostctl/src/modules/gpu.rs
use nvbind::bolt::{NvbindGpuManager, BoltConfig};
use clap::{Parser, Subcommand};
use ratatui::prelude::*;

#[derive(Parser)]
#[command(name = "gpu")]
#[command(about = "GPU container runtime management via nvbind")]
pub struct GpuCommand {
    #[command(subcommand)]
    pub action: GpuAction,
}

#[derive(Subcommand)]
pub enum GpuAction {
    /// Show GPU status and information
    Status,
    /// Interactive GPU container launcher
    Launch,
    /// Manage GPU runtime profiles
    Profiles {
        #[command(subcommand)]
        action: ProfileAction,
    },
    /// Configure nvbind settings
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Run system diagnostics
    Doctor,
    /// Performance monitoring
    Monitor,
}

#[derive(Subcommand)]
pub enum ProfileAction {
    /// List available profiles
    List,
    /// Create new profile
    Create { name: String },
    /// Edit existing profile
    Edit { name: String },
    /// Delete profile
    Delete { name: String },
    /// Test profile performance
    Test { name: String },
}

pub struct GpuModule {
    nvbind_manager: NvbindGpuManager,
    profiles: HashMap<String, GpuProfile>,
}

impl GpuModule {
    pub fn new() -> Result<Self> {
        let nvbind_manager = NvbindGpuManager::with_defaults();
        let profiles = Self::load_profiles()?;

        Ok(Self {
            nvbind_manager,
            profiles,
        })
    }

    pub async fn handle_command(&mut self, cmd: GpuCommand) -> Result<()> {
        match cmd.action {
            GpuAction::Status => self.show_gpu_status().await,
            GpuAction::Launch => self.launch_interactive_tui().await,
            GpuAction::Profiles { action } => self.handle_profile_action(action).await,
            GpuAction::Config { action } => self.handle_config_action(action).await,
            GpuAction::Doctor => self.run_diagnostics().await,
            GpuAction::Monitor => self.start_monitoring_tui().await,
        }
    }
}
```

### 2. Interactive TUI Components

```rust
// GPU Container Launcher TUI
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Gauge, List, ListItem, Paragraph},
    Frame, Terminal,
};

pub struct GpuLauncherTui {
    nvbind_manager: NvbindGpuManager,
    selected_runtime: usize,
    selected_profile: usize,
    container_image: String,
    gpu_selection: String,
    runtimes: Vec<String>,
    profiles: Vec<String>,
    gpu_info: Vec<GpuDevice>,
    logs: Vec<String>,
}

impl GpuLauncherTui {
    pub fn new(nvbind_manager: NvbindGpuManager) -> Result<Self> {
        Ok(Self {
            nvbind_manager,
            selected_runtime: 0,
            selected_profile: 0,
            container_image: String::new(),
            gpu_selection: "all".to_string(),
            runtimes: vec!["bolt".to_string(), "docker".to_string(), "podman".to_string()],
            profiles: vec!["gaming".to_string(), "ai-ml".to_string(), "balanced".to_string()],
            gpu_info: Vec::new(),
            logs: Vec::new(),
        })
    }

    pub async fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Load GPU information
        self.gpu_info = self.nvbind_manager.get_gpu_info().await?;

        // Main event loop
        loop {
            terminal.draw(|f| self.draw(f))?;

            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Enter => {
                        self.launch_container().await?;
                    }
                    KeyCode::Tab => {
                        self.next_field();
                    }
                    KeyCode::Up => {
                        self.move_selection(-1);
                    }
                    KeyCode::Down => {
                        self.move_selection(1);
                    }
                    KeyCode::Char(c) => {
                        self.handle_char_input(c);
                    }
                    _ => {}
                }
            }
        }

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        Ok(())
    }

    fn draw(&mut self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(3),  // Title
                Constraint::Length(8),  // GPU Info
                Constraint::Length(6),  // Runtime Selection
                Constraint::Length(6),  // Profile Selection
                Constraint::Length(3),  // Container Image
                Constraint::Length(3),  // GPU Selection
                Constraint::Min(4),     // Logs
                Constraint::Length(3),  // Controls
            ].as_ref())
            .split(f.size());

        // Title
        let title = Paragraph::new("🚀 nvbind GPU Container Launcher")
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(title, chunks[0]);

        // GPU Information
        self.draw_gpu_info(f, chunks[1]);

        // Runtime Selection
        self.draw_runtime_selection(f, chunks[2]);

        // Profile Selection
        self.draw_profile_selection(f, chunks[3]);

        // Container Image Input
        self.draw_image_input(f, chunks[4]);

        // GPU Selection
        self.draw_gpu_selection(f, chunks[5]);

        // Logs
        self.draw_logs(f, chunks[6]);

        // Controls
        self.draw_controls(f, chunks[7]);
    }

    fn draw_gpu_info(&self, f: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self.gpu_info
            .iter()
            .map(|gpu| {
                let memory_info = if let Some(memory) = gpu.memory {
                    format!(" ({}MB)", memory)
                } else {
                    String::new()
                };

                ListItem::new(Line::from(vec![
                    Span::styled(format!("GPU {}: ", gpu.id), Style::default().fg(Color::Yellow)),
                    Span::raw(format!("{}{}", gpu.name, memory_info)),
                ]))
            })
            .collect();

        let gpu_list = List::new(items)
            .block(Block::default().title("🎮 Available GPUs").borders(Borders::ALL))
            .style(Style::default().fg(Color::White));

        f.render_widget(gpu_list, area);
    }

    fn draw_runtime_selection(&self, f: &mut Frame, area: Rect) {
        let items: Vec<ListItem> = self.runtimes
            .iter()
            .enumerate()
            .map(|(i, runtime)| {
                let style = if i == self.selected_runtime {
                    Style::default().fg(Color::Black).bg(Color::Cyan)
                } else {
                    Style::default().fg(Color::White)
                };

                ListItem::new(Line::from(Span::styled(runtime, style)))
            })
            .collect();

        let runtime_list = List::new(items)
            .block(Block::default().title("🐳 Container Runtime").borders(Borders::ALL));

        f.render_widget(runtime_list, area);
    }

    async fn launch_container(&mut self) -> Result<()> {
        let runtime = &self.runtimes[self.selected_runtime];
        let profile = &self.profiles[self.selected_profile];

        self.logs.push(format!("Launching container with runtime: {}, profile: {}", runtime, profile));

        // Launch container with nvbind
        match self.nvbind_manager.run_with_bolt_runtime(
            self.container_image.clone(),
            vec![],
            Some(self.gpu_selection.clone()),
        ).await {
            Ok(_) => {
                self.logs.push("✅ Container launched successfully".to_string());
            }
            Err(e) => {
                self.logs.push(format!("❌ Launch failed: {}", e));
            }
        }

        Ok(())
    }
}
```

### 3. Performance Monitoring TUI

```rust
pub struct GpuMonitorTui {
    nvbind_manager: NvbindGpuManager,
    gpu_metrics: Vec<GpuMetrics>,
    container_metrics: Vec<ContainerMetrics>,
    refresh_rate: Duration,
    last_update: Instant,
}

#[derive(Debug, Clone)]
pub struct GpuMetrics {
    pub gpu_id: String,
    pub name: String,
    pub memory_used: u64,
    pub memory_total: u64,
    pub utilization: f32,
    pub temperature: Option<f32>,
    pub power_usage: Option<f32>,
}

impl GpuMonitorTui {
    pub async fn run(&mut self) -> Result<()> {
        // Similar TUI setup as launcher
        loop {
            if self.last_update.elapsed() >= self.refresh_rate {
                self.update_metrics().await?;
                self.last_update = Instant::now();
            }

            terminal.draw(|f| self.draw_monitoring(f))?;

            // Handle events...
        }
    }

    fn draw_monitoring(&self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(f.size());

        // GPU utilization gauges
        self.draw_gpu_gauges(f, chunks[0]);

        // Container performance
        self.draw_container_metrics(f, chunks[1]);
    }

    fn draw_gpu_gauges(&self, f: &mut Frame, area: Rect) {
        let gpu_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(vec![Constraint::Percentage(100 / self.gpu_metrics.len() as u16); self.gpu_metrics.len()])
            .split(area);

        for (i, gpu) in self.gpu_metrics.iter().enumerate() {
            let gauge = Gauge::default()
                .block(Block::default().title(format!("GPU {} - {}", gpu.gpu_id, gpu.name)).borders(Borders::ALL))
                .gauge_style(Style::default().fg(Color::Green))
                .percent((gpu.utilization * 100.0) as u16)
                .label(format!("{:.1}% | {}/{}MB", gpu.utilization * 100.0, gpu.memory_used, gpu.memory_total));

            if i < gpu_chunks.len() {
                f.render_widget(gauge, gpu_chunks[i]);
            }
        }
    }
}
```

## 🔧 CLI Integration

### GhostCTL Command Structure

```bash
# Add to ghostctl main command structure
ghostctl gpu status                    # Show GPU information
ghostctl gpu launch                    # Interactive container launcher
ghostctl gpu profiles list             # List GPU profiles
ghostctl gpu profiles create gaming    # Create gaming profile
ghostctl gpu config show              # Show nvbind configuration
ghostctl gpu doctor                    # Run system diagnostics
ghostctl gpu monitor                   # Real-time monitoring TUI

# Advanced GPU management
ghostctl gpu containers list           # List GPU containers
ghostctl gpu containers kill <id>     # Stop GPU container
ghostctl gpu runtime switch bolt      # Switch default runtime
ghostctl gpu benchmark                 # Run GPU performance tests
```

### Configuration Integration

```toml
# ghostctl.toml - Add GPU section
[modules.gpu]
enabled = true
runtime = "nvbind"
default_profile = "balanced"
auto_detect = true

[modules.gpu.profiles.gaming]
name = "High Performance Gaming"
dlss = true
rt_cores = true
power_profile = "maximum"

[modules.gpu.profiles.ai-ml]
name = "AI/ML Workloads"
tensor_cores = true
memory_pool = "16GB"
cuda_cache = "4GB"

[modules.gpu.monitoring]
refresh_rate = "1s"
show_temperature = true
show_power = true
alert_threshold = 85
```

## 📦 Installation Strategy

### 1. Arch Linux (AUR Package)

```bash
# PKGBUILD for nvbind
pkgname=nvbind
pkgver=0.1.0
pkgrel=1
pkgdesc="Lightning-fast NVIDIA GPU container runtime"
arch=('x86_64')
url="https://github.com/ghostkellz/nvbind"
license=('MIT')
depends=('glibc')
makedepends=('rust' 'cargo')
source=("$pkgname-$pkgver.tar.gz::$url/archive/v$pkgver.tar.gz")

build() {
    cd "$pkgname-$pkgver"
    cargo build --release --locked
}

package() {
    cd "$pkgname-$pkgver"
    install -Dm755 target/release/nvbind "$pkgdir/usr/bin/nvbind"
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
```

### 2. Universal Installer Integration

```bash
# Install nvbind via ghostctl
ghostctl install nvbind              # Install nvbind
ghostctl install nvbind --user       # User installation
ghostctl update nvbind               # Update to latest version
ghostctl remove nvbind               # Uninstall nvbind

# Integration with GhostCTL's package management
ghostctl packages search gpu         # Find GPU-related packages
ghostctl packages install nvbind     # Install through GhostCTL
```

### 3. Cross-Platform Distribution

```
installers/
├── linux/
│   ├── install.sh                   # Universal Linux installer
│   ├── arch/
│   │   └── PKGBUILD                 # Arch Linux package
│   ├── debian/
│   │   ├── control                  # Debian package metadata
│   │   └── nvbind.deb              # Pre-built Debian package
│   ├── fedora/
│   │   └── nvbind.spec             # RPM spec file
│   └── flatpak/
│       └── com.ghostkellz.nvbind.yml # Flatpak manifest
├── scripts/
│   ├── build-packages.sh           # Build all package formats
│   └── test-install.sh             # Test installation on different distros
└── README.md                       # Installation guide
```

## 🚀 Integration Roadmap

### Phase 1: Core Integration (Week 1-2)
- [ ] Create `ghostctl gpu` module structure
- [ ] Implement basic GPU status and info commands
- [ ] Add nvbind as optional dependency in GhostCTL
- [ ] Create simple TUI for container launching

### Phase 2: Advanced Features (Week 3-4)
- [ ] Implement profile management system
- [ ] Add real-time GPU monitoring TUI
- [ ] Create container lifecycle management
- [ ] Add performance benchmarking tools

### Phase 3: Distribution (Week 5-6)
- [ ] Create AUR package for Arch Linux
- [ ] Build universal Linux installer
- [ ] Add Debian/Ubuntu package support
- [ ] Create Fedora/RHEL RPM packages

### Phase 4: Production Features (Month 2)
- [ ] Advanced GPU allocation algorithms
- [ ] Multi-user GPU sharing
- [ ] Container GPU quotas and limits
- [ ] Integration with GhostCTL's notification system

## 🎯 Strategic Benefits

**For GhostCTL Users:**
- **Unified Experience**: GPU management through familiar GhostCTL interface
- **Superior Performance**: nvbind's sub-microsecond GPU passthrough
- **Intuitive TUI**: Visual GPU container management
- **Zero Docker Dependency**: Complete replacement for Docker GPU workflows

**For nvbind:**
- **Broader Adoption**: Integration with popular sysadmin tool
- **Professional Validation**: Association with enterprise-grade CLI tool
- **User-Friendly Interface**: GUI/TUI wrapper around CLI commands
- **System Integration**: Deep OS-level GPU management

## 📋 Example Usage

### Quick Container Launch
```bash
# Simple command-line usage
ghostctl gpu launch --runtime bolt --profile gaming --image steam:latest

# Interactive TUI mode
ghostctl gpu launch
# Opens interactive launcher with dropdown menus for:
# - Runtime selection (bolt, docker, podman)
# - Profile selection (gaming, ai-ml, balanced)
# - GPU allocation (all, gpu0, gpu1, etc.)
# - Real-time GPU status display
```

### Performance Monitoring
```bash
# Real-time GPU monitoring
ghostctl gpu monitor
# Shows:
# - GPU utilization graphs
# - Memory usage per GPU
# - Temperature and power consumption
# - Active container performance
# - Historical performance trends
```

### Profile Management
```bash
# Create custom gaming profile
ghostctl gpu profiles create esports
# Opens profile editor with options for:
# - DLSS settings
# - Ray tracing preferences
# - Power management
# - Wine/Proton optimizations
```

This integration transforms GhostCTL into the **ultimate Linux sysadmin tool with premier GPU container management**, while positioning nvbind as the **enterprise-grade GPU runtime** for serious Linux administrators.

**Ready to revolutionize GPU administration! 🚀⚡🎮**