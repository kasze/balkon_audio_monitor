#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  python3-numpy \
  python3-yaml \
  git \
  rsync \
  curl \
  sqlite3 \
  alsa-utils
