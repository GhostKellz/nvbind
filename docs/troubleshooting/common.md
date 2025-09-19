# Common Issues and Solutions

Quick solutions to the most frequently encountered nvbind issues.

## 🚨 Quick Diagnostics

```bash
# Run comprehensive system check
nvbind doctor

# Check GPU detection
nvbind info --detailed

# Test with simple container
nvbind run --runtime podman nvidia/cuda:latest nvidia-smi
```

## 🔧 Installation Issues

### nvbind command not found
```bash
# Check if nvbind is installed
which nvbind
ls -la /usr/local/bin/nvbind

# Fix PATH if needed
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Reinstall if binary missing
curl -sSL https://raw.githubusercontent.com/ghostkellz/nvbind/main/install.sh | sudo bash
```

### Permission denied
```bash
# Fix binary permissions
sudo chmod +x /usr/local/bin/nvbind

# For user installation
chmod +x ~/.local/bin/nvbind
export PATH="$HOME/.local/bin:$PATH"
```

### Cargo build fails
```bash
# Update Rust toolchain
rustup update

# Install required dependencies
sudo apt install build-essential pkg-config libssl-dev  # Ubuntu/Debian
sudo dnf install gcc openssl-devel                     # Fedora/RHEL

# Clean and rebuild
cargo clean
cargo build --release
```

## 🎮 GPU Detection Issues

### No GPUs detected
```bash
# Check NVIDIA driver status
nvidia-smi

# Verify kernel modules loaded
lsmod | grep nvidia

# Check device files exist
ls -la /dev/nvidia*

# Load NVIDIA modules if missing
sudo modprobe nvidia
sudo modprobe nvidia_uvm
```

### Driver version mismatch
```bash
# Check driver versions
cat /proc/driver/nvidia/version
nvidia-smi

# Update NVIDIA drivers
sudo apt update && sudo apt install nvidia-driver-535  # Ubuntu
sudo dnf install akmod-nvidia                          # Fedora

# Reboot after driver update
sudo reboot
```

### Permission denied on GPU devices
```bash
# Fix device permissions
sudo chmod 666 /dev/nvidia*
sudo chmod 666 /dev/nvidiactl
sudo chmod 666 /dev/nvidia-uvm

# Add user to video group
sudo usermod -aG video $USER

# Create persistent udev rules
sudo tee /etc/udev/rules.d/70-nvidia.rules > /dev/null <<EOF
KERNEL=="nvidia*", GROUP="video", MODE="0666"
KERNEL=="nvidiactl", GROUP="video", MODE="0666"
KERNEL=="nvidia-uvm", GROUP="video", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

## 🐳 Container Runtime Issues

### Container runtime not found
```bash
# Check available runtimes
which podman docker bolt

# Install missing runtime
sudo apt install podman                    # Ubuntu/Debian
sudo dnf install podman                    # Fedora/RHEL
curl -sSL https://install.bolt.sh | bash   # Bolt

# Test runtime
podman run hello-world
docker run hello-world
bolt --version
```

### Docker permission denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Apply group membership (logout/login or use newgrp)
newgrp docker

# Fix Docker socket permissions
sudo chmod 666 /var/run/docker.sock

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

### Podman rootless issues
```bash
# Configure rootless Podman
podman system migrate

# Set up user namespaces
echo "user.max_user_namespaces=28633" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Configure storage
mkdir -p ~/.config/containers
cat > ~/.config/containers/storage.conf <<EOF
[storage]
driver = "overlay"
runroot = "/run/user/1000/containers"
graphroot = "/home/$USER/.local/share/containers/storage"
EOF
```

## ⚙️ Configuration Issues

### Config file not found
```bash
# Create default config
nvbind config --output ~/.config/nvbind/config.toml

# Check config locations
ls -la ~/.config/nvbind/
ls -la /etc/nvbind/
ls -la ./nvbind.toml

# Generate and edit config
nvbind config --show > nvbind.toml
nano nvbind.toml
```

### Invalid configuration
```bash
# Validate config syntax
nvbind config --show

# Check TOML syntax
python3 -c "import toml; toml.load('nvbind.toml')" 2>/dev/null && echo "Valid TOML" || echo "Invalid TOML"

# Reset to defaults
rm ~/.config/nvbind/config.toml
nvbind config --output ~/.config/nvbind/config.toml
```

### Feature not available
```bash
# Check nvbind features
nvbind --version

# Rebuild with specific features
cargo build --release --features bolt

# Check feature availability
cargo run --bin nvbind -- run --help | grep -i bolt
```

## 🎮 Gaming-Specific Issues

### X11 display issues
```bash
# Allow X11 access
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY
export DISPLAY=:0

# Test X11 forwarding
nvbind run --runtime podman \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  ubuntu:latest \
  xeyes
```

### Audio not working
```bash
# Check audio system
pactl info          # PulseAudio
pipewire --version   # PipeWire

# Test audio in container
nvbind run --runtime podman \
  -v /run/user/$(id -u)/pulse:/run/pulse:ro \
  -e PULSE_SERVER=unix:/run/pulse/native \
  --device /dev/snd \
  ubuntu:latest \
  aplay /usr/share/sounds/alsa/Front_Left.wav
```

### Poor gaming performance
```bash
# Check GPU usage
nvidia-smi

# Verify gaming profile
nvbind run --runtime bolt --gpu all --profile gaming \
  --verbose nvidia/cuda:latest nvidia-smi

# Check power management
nvidia-smi -q -d POWER
```

### Controller not detected
```bash
# Add input devices
nvbind run --runtime podman --gpu all --profile gaming \
  --device /dev/input \
  --device /dev/uinput \
  -v /run/udev:/run/udev:ro \
  game:latest

# Check controller permissions
ls -la /dev/input/
sudo chmod 666 /dev/input/event*
```

## 🧠 AI/ML Issues

### CUDA out of memory
```bash
# Check GPU memory usage
nvidia-smi

# Clear GPU cache
nvbind run --runtime bolt --gpu all pytorch/pytorch:latest \
  python -c "import torch; torch.cuda.empty_cache()"

# Reduce memory usage
nvbind run --runtime bolt --gpu all --profile ai-ml \
  -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256" \
  pytorch/pytorch:latest
```

### Slow training performance
```bash
# Check Tensor Core usage
nvbind run --runtime bolt --gpu all --profile ai-ml \
  pytorch/pytorch:latest \
  python -c "import torch; print(f'TF32: {torch.backends.cuda.matmul.allow_tf32}')"

# Monitor GPU utilization
watch -n 1 nvidia-smi

# Check data loading
nvbind run --runtime bolt --gpu all --profile ai-ml \
  --shm-size=16g \
  pytorch/pytorch:latest
```

### Package import errors
```bash
# Check package installation
nvbind run --runtime bolt --gpu all pytorch/pytorch:latest \
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Install missing packages
nvbind run --runtime bolt --gpu all pytorch/pytorch:latest \
  pip install package-name
```

## 🔒 Security Issues

### SELinux blocking access
```bash
# Check SELinux status
sestatus

# Temporarily disable for testing
sudo setenforce 0

# Set SELinux booleans
sudo setsebool -P container_use_devices 1

# Disable SELinux for specific container
nvbind run --runtime podman --gpu all \
  --security-opt label=disable \
  nvidia/cuda:latest
```

### AppArmor issues
```bash
# Check AppArmor status
sudo aa-status

# Disable AppArmor for container
nvbind run --runtime podman --gpu all \
  --security-opt apparmor=unconfined \
  nvidia/cuda:latest
```

### Rootless permission issues
```bash
# Check user namespaces
cat /proc/sys/user/max_user_namespaces

# Set up user namespaces
echo "user.max_user_namespaces=28633" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Check subuid/subgid
grep $USER /etc/subuid /etc/subgid
```

## 🌐 Network Issues

### Container networking problems
```bash
# Test network connectivity
nvbind run --runtime podman ubuntu:latest ping -c 3 google.com

# Use host networking
nvbind run --runtime podman --network host ubuntu:latest

# Check DNS resolution
nvbind run --runtime podman ubuntu:latest nslookup google.com
```

### Port binding issues
```bash
# Check port availability
netstat -tlnp | grep :8080

# Use different port
nvbind run --runtime podman -p 8081:8080 nginx:latest

# Check firewall
sudo ufw status
sudo iptables -L
```

## 💾 Storage Issues

### Volume mount failures
```bash
# Check file permissions
ls -la $(pwd)

# Use SELinux context (on SELinux systems)
nvbind run --runtime podman -v $(pwd):/data:Z nginx:latest

# Create directory if missing
mkdir -p ~/nvbind-data
nvbind run --runtime podman -v ~/nvbind-data:/data nginx:latest
```

### Disk space issues
```bash
# Check disk usage
df -h

# Clean container images
podman system prune -f
docker system prune -f

# Check container storage
podman system df
docker system df
```

## 🔄 Upgrade Issues

### Version compatibility
```bash
# Check nvbind version
nvbind --version

# Update to latest
curl -sSL https://raw.githubusercontent.com/ghostkellz/nvbind/main/install.sh | sudo bash

# Check for breaking changes
nvbind config --show  # Should work with new version
```

### Configuration migration
```bash
# Backup old config
cp ~/.config/nvbind/config.toml ~/.config/nvbind/config.toml.bak

# Generate new config with defaults
nvbind config --output ~/.config/nvbind/config.new.toml

# Merge configurations manually
vimdiff ~/.config/nvbind/config.toml.bak ~/.config/nvbind/config.new.toml
```

## 🆘 Getting Help

### Collect debug information
```bash
# Generate debug report
nvbind doctor > nvbind-debug.txt

# Include system information
uname -a >> nvbind-debug.txt
nvidia-smi >> nvbind-debug.txt
lsb_release -a >> nvbind-debug.txt

# Check logs
journalctl -u docker >> nvbind-debug.txt    # Docker logs
podman logs --help >> nvbind-debug.txt      # Podman logs
```

### Report issues
1. **Check existing issues**: [GitHub Issues](https://github.com/ghostkellz/nvbind/issues)
2. **Create new issue** with:
   - nvbind version (`nvbind --version`)
   - OS and distribution (`lsb_release -a`)
   - GPU information (`nvidia-smi`)
   - Full error message
   - Steps to reproduce
   - Debug output (`nvbind doctor`)

### Community support
- **GitHub Discussions**: [Ask questions](https://github.com/ghostkellz/nvbind/discussions)
- **Documentation**: Check other docs sections for specific issues
- **Examples**: Look at working examples in `/docs/examples/`

---

**Still having issues? Don't hesitate to reach out - we're here to help! 🚀**