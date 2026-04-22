#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH_AGENTS_DIR="/Users/daniildegtyarev/Library/LaunchAgents"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

mkdir -p /opt/roehub/app /opt/roehub/bin /opt/roehub/config /opt/roehub/state/backups /opt/roehub/clickhouse
mkdir -p /opt/roehub/state/backtest_artifacts/v2
mkdir -p /opt/roehub/clickhouse/data /opt/roehub/clickhouse/tmp /opt/roehub/clickhouse/logs /opt/roehub/clickhouse/backups /opt/roehub/clickhouse/access
mkdir -p /Users/daniildegtyarev/.config/roehub /Users/daniildegtyarev/.local/bin /Users/daniildegtyarev/Library/Logs/roehub "$LAUNCH_AGENTS_DIR"
mkdir -p /opt/homebrew/etc/monit.d/scripts

install -m 0644 "$REPO_ROOT/infra/macos/prometheus/prometheus.prod.yml" /opt/roehub/config/prometheus.prod.yml
install -m 0644 "$REPO_ROOT/infra/macos/blackbox/blackbox.yml" /opt/roehub/config/blackbox.yml
install -m 0644 "$REPO_ROOT/infra/macos/clickhouse/config.xml" /opt/roehub/config/clickhouse.config.xml
install -m 0644 "$REPO_ROOT/infra/macos/clickhouse/users.d/roehub.xml" /opt/roehub/config/clickhouse.users.roehub.xml
install -m 0755 "$REPO_ROOT/infra/scripts/monit/launchctl_service_control.sh" /opt/homebrew/etc/monit.d/scripts/launchctl_service_control.sh
install -m 0600 "$REPO_ROOT/infra/scripts/monit/roehub-market-data.monitrc" /opt/homebrew/etc/monit.d/roehub-market-data.monitrc
install -m 0600 "$REPO_ROOT/infra/scripts/monit/roehub-backtest-artifact-publisher.monitrc" /opt/homebrew/etc/monit.d/roehub-backtest-artifact-publisher.monitrc

for plist in \
  com.roehub.api.plist \
  com.roehub.market-data-ws-worker.plist \
  com.roehub.market-data-scheduler.plist \
  com.roehub.backtest-artifact-publisher.plist \
  com.roehub.clickhouse.plist \
  com.roehub.blackbox-exporter.plist \
  com.roehub.postgres-exporter.plist \
  com.roehub.redis-exporter.plist \
  com.roehub.clickhouse-exporter.plist \
  com.roehub.tailscale-runtime.plist
do
  install -m 0644 "$REPO_ROOT/infra/macos/launchd/$plist" "$LAUNCH_AGENTS_DIR/$plist"
done

worker_plists=()
while IFS= read -r worker_plist; do
  worker_plists+=("$worker_plist")
done < <(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/macos/render_backtest_job_runner_launchd.py" \
    --profile prod \
    --repo-root "$REPO_ROOT" \
    --launch-agents-dir "$LAUNCH_AGENTS_DIR" \
    --clean
)

cat > /opt/homebrew/etc/prometheus.args <<'EOF'
--config.file=/opt/roehub/config/prometheus.prod.yml
--storage.tsdb.path=/opt/homebrew/var/prometheus
--storage.tsdb.retention.time=90d
EOF

cat > /opt/homebrew/etc/node_exporter.args <<'EOF'
--web.listen-address=127.0.0.1:9100
EOF

echo "prod backtest-job-runner fleet materialized: ${#worker_plists[@]} instance(s)"
if ((${#worker_plists[@]} > 0)); then
  printf 'prod backtest-job-runner service: %s\n' "${worker_plists[@]}"
fi

echo "prod native templates installed"
