#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

echo "[1/3] Stopping BirdNET service"
sudo systemctl disable --now birdnet-server 2>/dev/null || true

echo "[2/3] Clearing BirdNET API URL in configs/config.yaml"
python3 - <<'PY'
from pathlib import Path
import yaml

path = Path("configs/config.yaml")
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
data.setdefault("classifier", {})["birdnet_api_url"] = ""
path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

echo "[3/3] Restarting audio-monitor"
sudo systemctl restart audio-monitor
