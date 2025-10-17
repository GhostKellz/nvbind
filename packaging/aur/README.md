# nvbind AUR Package

This directory contains the PKGBUILD for the nvbind Arch User Repository (AUR) package.

## For Users

### Installation from AUR

```bash
# Using yay
yay -S nvbind

# Using paru
paru -S nvbind

# Manual installation
git clone https://aur.archlinux.org/nvbind.git
cd nvbind
makepkg -si
```

### Post-Installation

After installation, verify your setup:

```bash
nvbind doctor
```

To use nvbind with Docker, configure the runtime in `/etc/docker/daemon.json`:

```json
{
  "runtimes": {
    "nvbind": {
      "path": "/usr/bin/nvbind",
      "runtimeArgs": ["runtime"]
    }
  }
}
```

Then restart Docker:

```bash
sudo systemctl restart docker
```

## For Maintainers

### Building Locally

To test the package locally before publishing:

```bash
cd packaging/aur
makepkg -si
```

### Publishing to AUR

1. Update version in PKGBUILD
2. Generate .SRCINFO:
   ```bash
   makepkg --printsrcinfo > .SRCINFO
   ```
3. Test build:
   ```bash
   makepkg -sf
   ```
4. Commit and push to AUR repository:
   ```bash
   git add PKGBUILD .SRCINFO
   git commit -m "Update to version X.Y.Z"
   git push
   ```

### Updating Checksums

When a new version is released:

1. Download the source tarball
2. Calculate SHA256:
   ```bash
   sha256sum nvbind-X.Y.Z.tar.gz
   ```
3. Update `sha256sums` in PKGBUILD
4. Regenerate .SRCINFO

## Dependencies

### Build Dependencies
- cargo
- rust (>= 1.70)
- git

### Runtime Dependencies
- glibc
- gcc-libs

### Optional Dependencies
- nvidia or nvidia-open (GPU driver)
- nvidia-utils (GPU management tools)
- docker (container runtime)
- podman (alternative container runtime)

## Configuration

Configuration file: `/etc/nvbind/config.toml`

Default configuration is created automatically on installation.

## Support

- Issues: https://github.com/ghostkellz/nvbind/issues
- AUR Package: https://aur.archlinux.org/packages/nvbind
- Documentation: https://github.com/ghostkellz/nvbind

## License

MIT License - see LICENSE file
