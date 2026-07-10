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
curl -fsS http://127.0.0.1:9204/metrics >/tmp/roehub-metrics-9204.txt
rl_ready=0
for _attempt in {1..60}; do
  if curl -fsS http://127.0.0.1:9213/health/ready >/tmp/roehub-rl-ready-9213.json \
    && grep -q '"ready": true' /tmp/roehub-rl-ready-9213.json; then
    rl_ready=1
    break
  fi
  sleep 1
done
if [[ "$rl_ready" -ne 1 ]]; then
  printf 'RL monitor-only worker did not become ready within 60 seconds\n' >&2
  cat /tmp/roehub-rl-ready-9213.json >&2 2>/dev/null || true
  exit 1
fi
curl -fsS http://127.0.0.1:9213/metrics >/tmp/roehub-metrics-9213.txt
grep -q 'backtest_runner_tasks_claimed_total' /tmp/roehub-metrics-9204.txt
grep -q 'backtest_runner_last_success_unixtime' /tmp/roehub-metrics-9204.txt
grep -q '^rl_trading_inference_ready 1.0$' /tmp/roehub-metrics-9213.txt
grep -q '^rl_trading_inference_model_loaded 1.0$' /tmp/roehub-metrics-9213.txt
if grep -Eq '^rl_trading_inference_safety_breaches_total\{[^}]*\} [1-9]' /tmp/roehub-metrics-9213.txt; then
  printf 'RL monitor-only safety breach metric is non-zero\n' >&2
  exit 1
fi

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
