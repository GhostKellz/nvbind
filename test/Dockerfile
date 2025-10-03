FROM archlinux:latest

# Update system and install packages
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    base-devel \
    git \
    wget \
    curl \
    vim \
    nvidia-open \
    nvidia-utils \
    rust \
    cargo \
    docker \
    docker-compose \
    podman \
    python

# Create CDI directories
RUN mkdir -p /etc/cdi /var/run/cdi

WORKDIR /workspace

CMD ["/bin/bash"]