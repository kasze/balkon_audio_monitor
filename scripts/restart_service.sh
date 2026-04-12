#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
SERVICE="${2:-audio-monitor}"

if [[ -n "${TARGET}" ]]; then
  ssh "${TARGET}" "sudo systemctl restart ${SERVICE}"
else
  sudo systemctl restart "${SERVICE}"
fi

