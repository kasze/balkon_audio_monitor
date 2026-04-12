#!/usr/bin/env bash
set -euo pipefail

PACKAGES=(
  python3
  python3-venv
  python3-pip
  python3-dev
  python3-numpy
  python3-yaml
  git
  rsync
  curl
  sqlite3
  alsa-utils
)

missing_packages=()
for package in "${PACKAGES[@]}"; do
  status="$(dpkg-query -W -f='${db:Status-Status}' "${package}" 2>/dev/null || true)"
  if [ "${status}" != "installed" ]; then
    missing_packages+=("${package}")
  fi
done

if [ "${#missing_packages[@]}" -eq 0 ]; then
  echo "System packages already installed."
  exit 0
fi

echo "Installing missing system packages: ${missing_packages[*]}"
sudo apt-get update
sudo apt-get install -y "${missing_packages[@]}"
