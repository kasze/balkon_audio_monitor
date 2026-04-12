#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
python3 -m venv --system-site-packages "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip wheel
