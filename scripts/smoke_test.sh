#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PORT="$(CONFIG_PATH="${CONFIG}" "${PYTHON_BIN}" - <<'PY'
import os
from app.config import load_config
print(load_config(os.environ["CONFIG_PATH"]).web.port)
PY
)"

echo "[1/3] Initializing database"
"${PYTHON_BIN}" -m app.main --config "${CONFIG}" init-db

echo "[2/3] Probing audio input"
if ! "${PYTHON_BIN}" -m app.main --config "${CONFIG}" check-audio; then
  echo "Audio probe failed. If running on a laptop/offline, set SKIP_AUDIO=1 to bypass."
  if [[ "${SKIP_AUDIO:-0}" != "1" ]]; then
    exit 1
  fi
fi

echo "[3/3] Starting temporary web server"
SERVER_PID=""
if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m app.main --config "${CONFIG}" web > /tmp/audio-monitor-smoke.log 2>&1 &
  SERVER_PID=$!
  trap 'if [[ -n "${SERVER_PID}" ]]; then kill "${SERVER_PID}" >/dev/null 2>&1 || true; fi' EXIT
  sleep 2
fi

curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null
curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null

echo "Smoke test passed."
