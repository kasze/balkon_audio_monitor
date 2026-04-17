#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-${ROOT_DIR}/configs/deploy.env}"

if [ -f "${DEPLOY_ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${DEPLOY_ENV_FILE}"
fi

TARGET="${1:-${AUDIO_MONITOR_TARGET:-pi@raspberrypi.local}}"
REMOTE_DIR="${2:-${AUDIO_MONITOR_REMOTE_DIR:-/opt/audio-monitor}}"

cd "${ROOT_DIR}"

echo "[1/1] Bootstrapping remote dependencies"
AUDIO_MONITOR_SKIP_BOOTSTRAP=0 ./scripts/deploy.sh "${TARGET}" "${REMOTE_DIR}"
