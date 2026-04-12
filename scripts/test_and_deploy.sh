#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:?usage: ./scripts/test_and_deploy.sh user@host [/opt/audio-monitor]}"
REMOTE_DIR="${2:-/opt/audio-monitor}"

echo "[1/2] Running tests"
./scripts/test.sh

echo "[2/2] Deploying"
./scripts/deploy.sh "${TARGET}" "${REMOTE_DIR}"
