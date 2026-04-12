#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:?usage: ./scripts/deploy.sh user@host [/opt/audio-monitor]}"
REMOTE_DIR="${2:-/opt/audio-monitor}"
REMOTE_USER="${TARGET%@*}"
TMP_UNIT_FILE="$(mktemp)"
trap 'rm -f "${TMP_UNIT_FILE}"' EXIT

hash_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

hash_text() {
  shasum -a 256 | awk '{print $1}'
}

sed "s/^User=.*/User=${REMOTE_USER}/" systemd/audio-monitor.service > "${TMP_UNIT_FILE}"

SYSTEM_DEPS_HASH="$(hash_file scripts/install_system_deps.sh)"
PYTHON_DEPS_HASH="$(
  {
    hash_file requirements.txt
    hash_file pyproject.toml
    hash_file scripts/install_python_deps.sh
    hash_file scripts/setup_venv.sh
  } | hash_text
)"
SYSTEMD_UNIT_HASH="$(hash_file "${TMP_UNIT_FILE}")"

ssh -tt "${TARGET}" "sudo mkdir -p ${REMOTE_DIR} && sudo chown ${REMOTE_USER}:${REMOTE_USER} ${REMOTE_DIR}"

RSYNC_OUTPUT="$(
  rsync -azi --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude ".pytest_cache" \
  --exclude "__pycache__" \
  --exclude "data/" \
  "${PWD}/" "${TARGET}:${REMOTE_DIR}/"
)"

read -r -d '' REMOTE_BOOTSTRAP <<EOF || true
set -euo pipefail
cd "${REMOTE_DIR}"
mkdir -p .deploy

system_deps_changed=0
python_deps_changed=0
venv_created=0
unit_changed=0

if [ ! -f .deploy/system_deps.sha256 ] || [ "\$(cat .deploy/system_deps.sha256)" != "${SYSTEM_DEPS_HASH}" ]; then
  ./scripts/install_system_deps.sh
  printf '%s\n' "${SYSTEM_DEPS_HASH}" > .deploy/system_deps.sha256
  system_deps_changed=1
fi

if [ ! -d .venv ]; then
  ./scripts/setup_venv.sh .venv
  venv_created=1
fi

if [ "\${venv_created}" -eq 1 ] || [ ! -f .deploy/python_deps.sha256 ] || [ "\$(cat .deploy/python_deps.sha256)" != "${PYTHON_DEPS_HASH}" ]; then
  ./scripts/install_python_deps.sh .venv
  printf '%s\n' "${PYTHON_DEPS_HASH}" > .deploy/python_deps.sha256
  python_deps_changed=1
fi

if [ ! -f configs/config.yaml ]; then
  cp configs/config.yaml.example configs/config.yaml
fi

existing_unit_hash=""
if sudo test -f /etc/systemd/system/audio-monitor.service; then
  existing_unit_hash="\$(sudo sha256sum /etc/systemd/system/audio-monitor.service | awk '{print \$1}')"
fi
if [ "\${existing_unit_hash}" != "${SYSTEMD_UNIT_HASH}" ]; then
  unit_changed=1
fi

printf 'system_deps_changed=%s\n' "\${system_deps_changed}"
printf 'python_deps_changed=%s\n' "\${python_deps_changed}"
printf 'venv_created=%s\n' "\${venv_created}"
printf 'unit_changed=%s\n' "\${unit_changed}"
EOF

BOOTSTRAP_OUTPUT="$(ssh "${TARGET}" "${REMOTE_BOOTSTRAP}")"
UNIT_CHANGED="$(printf '%s\n' "${BOOTSTRAP_OUTPUT}" | awk -F= '/^unit_changed=/{print $2}')"

if [ "${UNIT_CHANGED}" = "1" ]; then
  ssh "${TARGET}" "sudo mkdir -p /etc/systemd/system"
  cat "${TMP_UNIT_FILE}" | ssh "${TARGET}" "sudo tee /etc/systemd/system/audio-monitor.service >/dev/null"
fi

if [ -n "${RSYNC_OUTPUT}" ] || [ "${UNIT_CHANGED}" = "1" ]; then
  ssh "${TARGET}" "sudo systemctl daemon-reload && sudo systemctl enable audio-monitor && sudo systemctl restart audio-monitor"
else
  echo "No file changes detected, skipping service restart."
fi
