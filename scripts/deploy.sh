#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_ENV_FILE="${DEPLOY_ENV_FILE:-${ROOT_DIR}/configs/deploy.env}"
SSH_PASSWORD=""
SSH_AUTH_MODE="key"

if [ -f "${DEPLOY_ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${DEPLOY_ENV_FILE}"
fi

TARGET="${1:-${AUDIO_MONITOR_TARGET:-pi@raspberrypi.local}}"
REMOTE_DIR="${2:-${AUDIO_MONITOR_REMOTE_DIR:-/opt/audio-monitor}}"
REMOTE_USER="${TARGET%@*}"
TMP_UNIT_FILE="$(mktemp)"
trap 'rm -f "${TMP_UNIT_FILE}"' EXIT

cd "${ROOT_DIR}"

log_step() {
  printf '\n[%s] %s\n' "$1" "$2"
}

log_info() {
  printf '  - %s\n' "$1"
}

hash_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

hash_text() {
  shasum -a 256 | awk '{print $1}'
}

load_ssh_password() {
  if [ -n "${AUDIO_MONITOR_SSH_PASSWORD:-}" ]; then
    SSH_PASSWORD="${AUDIO_MONITOR_SSH_PASSWORD}"
    return
  fi
  if [ -n "${AUDIO_MONITOR_SSH_PASSWORD_SECRET_CMD:-}" ]; then
    SSH_PASSWORD="$(eval "${AUDIO_MONITOR_SSH_PASSWORD_SECRET_CMD}")"
  fi
}

shell_join() {
  local joined=""
  local arg
  for arg in "$@"; do
    joined+="$(printf '%q' "${arg}") "
  done
  printf '%s\n' "${joined% }"
}

run_with_expect() {
  local command_string
  command_string="$(shell_join "$@")"
  EXPECT_COMMAND="${command_string}" EXPECT_PASSWORD="${SSH_PASSWORD}" expect <<'EOF'
log_user 1
set timeout -1
set command $env(EXPECT_COMMAND)
set password $env(EXPECT_PASSWORD)

spawn sh -lc $command
expect {
  -re "(?i)are you sure you want to continue connecting" {
    send -- "yes\r"
    exp_continue
  }
  -re "(?i)(?:password|passphrase).*:" {
    send -- "$password\r"
    exp_continue
  }
  eof {
    catch wait result
    exit [lindex $result 3]
  }
}
EOF
}

run_ssh() {
  if [ "${SSH_AUTH_MODE}" = "password_sshpass" ]; then
    sshpass -p "${SSH_PASSWORD}" ssh \
      -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no \
      "$@"
    return
  fi
  if [ "${SSH_AUTH_MODE}" = "password_expect" ]; then
    run_with_expect ssh \
      -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no \
      "$@"
    return
  fi
  ssh "$@"
}

run_rsync() {
  if [ "${SSH_AUTH_MODE}" = "password_sshpass" ]; then
    SSHPASS="${SSH_PASSWORD}" rsync \
      --rsh="sshpass -e ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no" \
      "$@"
    return
  fi
  if [ "${SSH_AUTH_MODE}" = "password_expect" ]; then
    run_with_expect rsync \
      --rsh="ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no" \
      "$@"
    return
  fi
  rsync "$@"
}

load_ssh_password

if [ -n "${SSH_PASSWORD}" ]; then
  if command -v sshpass >/dev/null 2>&1; then
    SSH_AUTH_MODE="password_sshpass"
  elif command -v expect >/dev/null 2>&1; then
    SSH_AUTH_MODE="password_expect"
  else
    echo "Password-based deploy requires sshpass or expect."
    exit 1
  fi
fi

sed "s/^User=.*/User=${REMOTE_USER}/" systemd/audio-monitor.service > "${TMP_UNIT_FILE}"

SYSTEM_DEPS_HASH="$(hash_file scripts/install_system_deps.sh)"
PYTHON_DEPS_HASH="$(
  {
    hash_file requirements.txt
    hash_file scripts/install_python_deps.sh
  } | hash_text
)"
SYSTEMD_UNIT_HASH="$(hash_file "${TMP_UNIT_FILE}")"

log_step 1 "Preparing remote directory on ${TARGET}"
log_info "remote dir: ${REMOTE_DIR}"
run_ssh -tt "${TARGET}" "sudo mkdir -p ${REMOTE_DIR} && sudo chown ${REMOTE_USER}:${REMOTE_USER} ${REMOTE_DIR}"

log_step 2 "Syncing repository"
RSYNC_OUTPUT="$(
  run_rsync -azi --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude ".pytest_cache" \
  --exclude "__pycache__" \
  --exclude ".deploy/" \
  --exclude "data/" \
  "${ROOT_DIR}/" "${TARGET}:${REMOTE_DIR}/"
)"
if [ -n "${RSYNC_OUTPUT}" ]; then
  log_info "files changed during rsync"
else
  log_info "rsync reported no file deltas"
fi

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

log_step 3 "Running remote bootstrap"
BOOTSTRAP_OUTPUT="$(run_ssh "${TARGET}" "${REMOTE_BOOTSTRAP}")"
printf '%s\n' "${BOOTSTRAP_OUTPUT}" | sed 's/^/  - /'
UNIT_CHANGED="$(printf '%s\n' "${BOOTSTRAP_OUTPUT}" | awk -F= '/^unit_changed=/{print $2}')"

if [ "${UNIT_CHANGED}" = "1" ]; then
  log_step 4 "Updating systemd unit"
  run_rsync -a "${TMP_UNIT_FILE}" "${TARGET}:${REMOTE_DIR}/.deploy/audio-monitor.service.tmp"
  run_ssh "${TARGET}" "sudo mkdir -p /etc/systemd/system"
  run_ssh "${TARGET}" "sudo install -m 0644 ${REMOTE_DIR}/.deploy/audio-monitor.service.tmp /etc/systemd/system/audio-monitor.service && rm -f ${REMOTE_DIR}/.deploy/audio-monitor.service.tmp"
fi

if [ -n "${RSYNC_OUTPUT}" ] || [ "${UNIT_CHANGED}" = "1" ]; then
  log_step 5 "Reloading and restarting service"
  run_ssh "${TARGET}" "sudo systemctl daemon-reload && sudo systemctl enable audio-monitor && sudo systemctl restart audio-monitor"
  log_info "service restarted"
else
  log_step 5 "Skipping restart"
  log_info "no file or unit changes detected"
fi

log_step done "Deploy finished"
log_info "target: ${TARGET}"
log_info "remote dir: ${REMOTE_DIR}"
