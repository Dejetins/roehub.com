#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
OPENBAO_ADDR="${OPENBAO_ADDR:-http://127.0.0.1:8200}"
ROEHUB_ENV_FILE="${ROEHUB_ENV_FILE:-/Users/daniildegtyarev/.config/roehub/roehub.env}"
PROVISION_SCRIPT="${ROEHUB_OPENBAO_PROVISION_SCRIPT:-/opt/roehub/bin/provision_openbao_transit_stage3a.sh}"
SMOKE_SCRIPT="${ROEHUB_OPENBAO_SMOKE_SCRIPT:-/opt/roehub/bin/smoke_openbao_transit_acl.sh}"
LOCK_DIR="${ROEHUB_OPENBAO_RECOVERY_LOCK_DIR:-/tmp/roehub-openbao-transit-recovery.lock}"
HEALTH_FILE="$(mktemp -t roehub-openbao-health.XXXXXX)"

export OPENBAO_ADDR ROEHUB_ENV_FILE

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "openbao_recovery=already_running"
  exit 0
fi
cleanup() {
  rm -f "$HEALTH_FILE"
  rmdir "$LOCK_DIR"
}
trap cleanup EXIT

wait_for_openbao() {
  local attempt=0
  local max_attempts="${ROEHUB_OPENBAO_RECOVERY_WAIT_ATTEMPTS:-60}"
  while (( attempt < max_attempts )); do
    : >"$HEALTH_FILE"
    if curl -sS --max-time 2 "$OPENBAO_ADDR/v1/sys/health" >"$HEALTH_FILE" 2>/dev/null; then
      return 0
    fi
    if [[ -s "$HEALTH_FILE" ]]; then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 1
  done
  echo "openbao_recovery=wait_failed"
  return 1
}

health_field() {
  local field="$1"
  python3 - "$field" "$HEALTH_FILE" <<'PY'
import json
import sys

field, path = sys.argv[1:]
try:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    raise SystemExit(1)
value = data.get(field)
if isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

wait_for_openbao
echo "openbao_initialized=$(health_field initialized)"
echo "openbao_sealed_before=$(health_field sealed)"

bash "$PROVISION_SCRIPT"

set -a
source "$ROEHUB_ENV_FILE"
set +a

bash "$SMOKE_SCRIPT"

curl -fsS "$OPENBAO_ADDR/v1/sys/health" >"$HEALTH_FILE"
sealed_after="$(health_field sealed)"
echo "openbao_sealed_after=${sealed_after}"
if [[ "$sealed_after" != "false" ]]; then
  echo "openbao_recovery=sealed_after_recovery" >&2
  exit 1
fi

echo "openbao_recovery=ok"
