#!/bin/bash
set -e

echo "[nv-prometheus] Booting container..."
cd /runner

if [ ! -d ".runner" ]; then
  echo "[nv-prometheus] Configuring runner..."
  ./config.sh \
    --url https://github.com/GhostKellz/nvbind \
    --token HIDDEN \
    --name nv-prometheus \
    --labels self-hosted,nvidia,gpu,rust,prometheus \
    --unattended
else
  echo "[nv-prometheus] Already configured."
fi

echo "[nv-prometheus] Starting runner..."
exec ./run.sh
