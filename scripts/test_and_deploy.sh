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

echo "[1/2] Running tests"
./scripts/test.sh

echo "[2/3] Deploying code"
AUDIO_MONITOR_SKIP_BOOTSTRAP=1 ./scripts/deploy.sh "${TARGET}" "${REMOTE_DIR}"

echo "[3/3] Workspace summary"
changed_files="$(git status --short | awk '{print $2}')"
if [ -n "${changed_files}" ]; then
  changed_count="$(printf '%s\n' "${changed_files}" | sed '/^$/d' | wc -l | tr -d ' ')"
  echo "  - changed files: ${changed_count}"
  printf '%s\n' "${changed_files}" | sed '/^$/d' | while IFS= read -r file; do
    echo "  - ${file}"
  done
else
  echo "  - changed files: 0"
fi
