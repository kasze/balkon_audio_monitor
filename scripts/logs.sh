#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
SERVICE="${2:-audio-monitor}"

if [[ -n "${TARGET}" ]]; then
  ssh "${TARGET}" "journalctl -u ${SERVICE} -n 150 -f"
else
  journalctl -u "${SERVICE}" -n 150 -f
fi

