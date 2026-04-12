#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
"${VENV_DIR}/bin/pip" install -r requirements.txt

