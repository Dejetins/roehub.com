#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

brew services list
launchctl list | grep -E "roehub|clickhouse|blackbox|actions.runner|tailscale" || true

curl -I http://127.0.0.1:3000
curl -I http://127.0.0.1:9090
curl -I http://127.0.0.1:9115
curl -i http://127.0.0.1:8000/auth/current-user
curl -fsS http://127.0.0.1:9201/metrics >/tmp/roehub-metrics-9201.txt
curl -fsS http://127.0.0.1:9202/metrics >/tmp/roehub-metrics-9202.txt

/opt/clickhouse/clickhouse client --host 127.0.0.1 --port 9000 --query "SELECT 1"
redis-cli -h 127.0.0.1 -p 6379 PING
tailscale serve status
