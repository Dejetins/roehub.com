#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:-prod}"
UID_VALUE="$(id -u)"
LAUNCH_AGENTS_DIR="/Users/daniildegtyarev/Library/LaunchAgents"

prod_services=(
  com.roehub.clickhouse.plist
  com.roehub.blackbox-exporter.plist
  com.roehub.postgres-exporter.plist
  com.roehub.redis-exporter.plist
  com.roehub.clickhouse-exporter.plist
  com.roehub.api.plist
  com.roehub.market-data-ws-worker.plist
  com.roehub.market-data-scheduler.plist
)

test_services=(
  com.roehub.test.postgres.plist
  com.roehub.test.redis.plist
  com.roehub.test.clickhouse.plist
  com.roehub.test.grafana.plist
  com.roehub.test.prometheus.plist
  com.roehub.test.blackbox-exporter.plist
  com.roehub.test.api.plist
  com.roehub.test.market-data-ws-worker.plist
  com.roehub.test.market-data-scheduler.plist
)

case "$PROFILE" in
  prod)
    services=("${prod_services[@]}")
    ;;
  test)
    services=("${test_services[@]}")
    ;;
  all)
    services=("${prod_services[@]}" "${test_services[@]}")
    ;;
  *)
    echo "usage: $0 [prod|test|all]" >&2
    exit 1
    ;;
esac

for service in "${services[@]}"; do
  launchctl bootout "gui/${UID_VALUE}" "${LAUNCH_AGENTS_DIR}/${service}" || true
  launchctl bootstrap "gui/${UID_VALUE}" "${LAUNCH_AGENTS_DIR}/${service}"
done

launchctl list | grep -E "roehub|clickhouse|blackbox|redis-exporter|postgres-exporter" || true
