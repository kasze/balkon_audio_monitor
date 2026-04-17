#!/usr/bin/env bash
set -euo pipefail

BIRDNET_DIR="${1:-.birdnet/BirdNET-Analyzer}"
BIRDNET_VENV="${2:-.birdnet-venv}"
BIRDNET_REPO_URL="${BIRDNET_REPO_URL:-https://github.com/birdnet-team/BirdNET-Analyzer.git}"
BIRDNET_REPO_REF="${BIRDNET_REPO_REF:-main}"
BIRDNET_MIN_FREE_MB="${BIRDNET_MIN_FREE_MB:-2500}"
WORK_TMP_DIR="${BIRDNET_TMP_DIR:-$(pwd)/.birdnet-tmp}"

available_mb="$(df -Pm . | awk 'NR == 2 {print $4}')"
if [ "${available_mb}" -lt "${BIRDNET_MIN_FREE_MB}" ]; then
  echo "BirdNET install needs at least ${BIRDNET_MIN_FREE_MB} MB free in $(pwd), available: ${available_mb} MB." >&2
  exit 1
fi

mkdir -p "${WORK_TMP_DIR}"
export TMPDIR="${WORK_TMP_DIR}"
export TMP="${WORK_TMP_DIR}"
export TEMP="${WORK_TMP_DIR}"

if [ ! -d "${BIRDNET_DIR}/.git" ]; then
  mkdir -p "$(dirname "${BIRDNET_DIR}")"
  git clone --depth 1 --branch "${BIRDNET_REPO_REF}" "${BIRDNET_REPO_URL}" "${BIRDNET_DIR}"
else
  git -C "${BIRDNET_DIR}" fetch --depth 1 origin "${BIRDNET_REPO_REF}"
  git -C "${BIRDNET_DIR}" checkout -q FETCH_HEAD
fi

python3 -m venv "${BIRDNET_VENV}"
"${BIRDNET_VENV}/bin/python" -m pip install --upgrade pip
"${BIRDNET_VENV}/bin/python" -m pip cache purge || true
"${BIRDNET_VENV}/bin/python" -m pip install --no-cache-dir "${BIRDNET_DIR}[server]"
"${BIRDNET_VENV}/bin/python" -m pip install --no-cache-dir pyarrow
