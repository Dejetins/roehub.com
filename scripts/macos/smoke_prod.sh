#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
ROEHUB_ENV_FILE="/Users/daniildegtyarev/.config/roehub/roehub.env"

if [[ ! -s "$ROEHUB_ENV_FILE" ]]; then
  printf 'env file not found: %s\n' "$ROEHUB_ENV_FILE" >&2
  exit 1
fi

set -a
source "$ROEHUB_ENV_FILE"
set +a

brew services list
launchctl list | grep -E "roehub|clickhouse|blackbox|actions.runner|tailscale" || true

curl -I http://127.0.0.1:3000
curl -I http://127.0.0.1:9090
curl -I http://127.0.0.1:9100
curl -I http://127.0.0.1:9115
curl -I http://127.0.0.1:9116
curl -I http://127.0.0.1:9121
curl -I http://127.0.0.1:9187
curl -i http://127.0.0.1:8000/auth/current-user
curl -fsS http://127.0.0.1:9201/metrics >/tmp/roehub-metrics-9201.txt
curl -fsS http://127.0.0.1:9202/metrics >/tmp/roehub-metrics-9202.txt

/opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --query "SELECT 1"
redis-cli -h 127.0.0.1 -p 6379 PING
PGPASSWORD="${POSTGRES_PASSWORD}" psql \
  -h 127.0.0.1 \
  -p 5432 \
  -U "${POSTGRES_USER}" \
  -d "${POSTGRES_DB}" \
  -Atqc "SELECT to_regclass('public.identity_users'), to_regclass('public.identity_exchange_keys'), to_regclass('public.alembic_version')" \
  | grep -qx 'identity_users|identity_exchange_keys|alembic_version'
state="$(tailscale status --json | /usr/bin/awk -F '"' '/"BackendState"[[:space:]]*:/ {print $4; exit}')"
if [[ "$state" != "Running" ]]; then
  printf 'tailscale backend is not running: %s\n' "${state:-unknown}" >&2
  exit 1
fi
printf 'tailscale backend state: %s\n' "$state"
tailscale serve status
