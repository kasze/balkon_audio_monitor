#!/usr/bin/env bash
set -euo pipefail

OUT_FILE="${1:-sample_audio/manual_sample.wav}"
DURATION="${2:-10}"
DEVICE="${ARECORD_DEVICE:-default}"

mkdir -p "$(dirname "${OUT_FILE}")"
arecord -D "${DEVICE}" -f S16_LE -r 16000 -c 1 -d "${DURATION}" "${OUT_FILE}"
echo "Saved sample to ${OUT_FILE}"

