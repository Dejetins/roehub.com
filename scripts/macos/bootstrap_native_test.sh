#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH_AGENTS_DIR="/Users/daniildegtyarev/Library/LaunchAgents"

mkdir -p /opt/roehub/test/postgresql /opt/roehub/test/redis
mkdir -p /opt/roehub/test/clickhouse/data /opt/roehub/test/clickhouse/tmp /opt/roehub/test/clickhouse/logs /opt/roehub/test/clickhouse/backups /opt/roehub/test/clickhouse/access
mkdir -p /opt/roehub/test/grafana /opt/roehub/test/prometheus /opt/roehub/test/blackbox
mkdir -p /opt/roehub/app/artifacts/backtest/v2
mkdir -p /Users/daniildegtyarev/.config/roehub /Users/daniildegtyarev/Library/Logs/roehub "$LAUNCH_AGENTS_DIR"

if [[ ! -f /Users/daniildegtyarev/.config/roehub/roehub.test.env ]]; then
  install -m 0600 /dev/null /Users/daniildegtyarev/.config/roehub/roehub.test.env
fi

install -m 0644 "$REPO_ROOT/infra/macos/prometheus/prometheus.test.yml" /opt/roehub/config/prometheus.test.yml
install -m 0644 "$REPO_ROOT/infra/macos/blackbox/blackbox.test.yml" /opt/roehub/config/blackbox.test.yml
install -m 0644 "$REPO_ROOT/infra/macos/clickhouse/config.test.xml" /opt/roehub/config/clickhouse.config.test.xml

for plist in \
  com.roehub.test.postgres.plist \
  com.roehub.test.redis.plist \
  com.roehub.test.clickhouse.plist \
  com.roehub.test.grafana.plist \
  com.roehub.test.prometheus.plist \
  com.roehub.test.blackbox-exporter.plist \
  com.roehub.test.api.plist \
  com.roehub.test.market-data-ws-worker.plist \
  com.roehub.test.market-data-scheduler.plist
  com.roehub.test.backtest-artifact-publisher.plist
do
  install -m 0644 "$REPO_ROOT/infra/macos/launchd/$plist" "$LAUNCH_AGENTS_DIR/$plist"
done

echo "test native templates installed"
