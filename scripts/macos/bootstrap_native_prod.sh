#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH_AGENTS_DIR="/Users/daniildegtyarev/Library/LaunchAgents"

mkdir -p /opt/roehub/app /opt/roehub/bin /opt/roehub/config /opt/roehub/state/backups /opt/roehub/clickhouse
mkdir -p /opt/roehub/clickhouse/data /opt/roehub/clickhouse/tmp /opt/roehub/clickhouse/logs /opt/roehub/clickhouse/backups /opt/roehub/clickhouse/access
mkdir -p /Users/daniildegtyarev/.config/roehub /Users/daniildegtyarev/.local/bin /Users/daniildegtyarev/Library/Logs/roehub "$LAUNCH_AGENTS_DIR"

install -m 0644 "$REPO_ROOT/infra/macos/prometheus/prometheus.prod.yml" /opt/roehub/config/prometheus.prod.yml
install -m 0644 "$REPO_ROOT/infra/macos/blackbox/blackbox.yml" /opt/roehub/config/blackbox.yml
install -m 0644 "$REPO_ROOT/infra/macos/clickhouse/config.xml" /opt/roehub/config/clickhouse.config.xml
install -m 0644 "$REPO_ROOT/infra/macos/clickhouse/users.d/roehub.xml" /opt/roehub/config/clickhouse.users.roehub.xml

for plist in \
  com.roehub.api.plist \
  com.roehub.market-data-ws-worker.plist \
  com.roehub.market-data-scheduler.plist \
  com.roehub.clickhouse.plist \
  com.roehub.blackbox-exporter.plist
do
  install -m 0644 "$REPO_ROOT/infra/macos/launchd/$plist" "$LAUNCH_AGENTS_DIR/$plist"
done

cat > /opt/homebrew/etc/prometheus.args <<'EOF'
--config.file=/opt/roehub/config/prometheus.prod.yml
--storage.tsdb.path=/opt/homebrew/var/prometheus
--storage.tsdb.retention.time=90d
EOF

echo "prod native templates installed"
