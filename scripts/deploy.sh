#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:?usage: ./scripts/deploy.sh user@host [/opt/audio-monitor]}"
REMOTE_DIR="${2:-/opt/audio-monitor}"
REMOTE_USER="${TARGET%@*}"

rsync -az --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude ".pytest_cache" \
  --exclude "__pycache__" \
  --exclude "data/" \
  "${PWD}/" "${TARGET}:${REMOTE_DIR}/"

ssh "${TARGET}" "cd ${REMOTE_DIR} && ./scripts/install_system_deps.sh && ./scripts/setup_venv.sh .venv && ./scripts/install_python_deps.sh .venv && if [ ! -f configs/config.yaml ]; then cp configs/config.yaml.example configs/config.yaml; fi"
sed "s/^User=.*/User=${REMOTE_USER}/" systemd/audio-monitor.service | ssh "${TARGET}" "sudo tee /etc/systemd/system/audio-monitor.service >/dev/null"
ssh "${TARGET}" "sudo systemctl daemon-reload && sudo systemctl enable audio-monitor && sudo systemctl restart audio-monitor"
