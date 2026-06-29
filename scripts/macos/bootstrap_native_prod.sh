#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH_AGENTS_DIR="/Users/daniildegtyarev/Library/LaunchAgents"

mkdir -p /opt/roehub/app /opt/roehub/bin /opt/roehub/config /opt/roehub/config/prometheus.rules /opt/roehub/config/openbao /opt/roehub/config/openbao/policies /opt/roehub/state/backups /opt/roehub/state/openbao/data /opt/roehub/clickhouse
mkdir -p /opt/roehub/state/backtest_artifacts/v2
mkdir -p /opt/roehub/clickhouse/data /opt/roehub/clickhouse/tmp /opt/roehub/clickhouse/logs /opt/roehub/clickhouse/backups /opt/roehub/clickhouse/access
mkdir -p /Users/daniildegtyarev/.config/roehub /Users/daniildegtyarev/.local/bin /Users/daniildegtyarev/Library/Logs/roehub "$LAUNCH_AGENTS_DIR"
mkdir -p /opt/homebrew/etc/monit.d/scripts

artifact_publisher_label="com.roehub.backtest-artifact-publisher"
launchctl disable "gui/$(id -u)/${artifact_publisher_label}" >/dev/null 2>&1 || true
launchctl bootout "gui/$(id -u)/${artifact_publisher_label}" >/dev/null 2>&1 || true
rm -f \
  "$LAUNCH_AGENTS_DIR/${artifact_publisher_label}.plist" \
  /opt/homebrew/etc/monit.d/roehub-backtest-artifact-publisher.monitrc

install -m 0644 "$REPO_ROOT/infra/macos/prometheus/prometheus.prod.yml" /opt/roehub/config/prometheus.prod.yml
install -m 0644 "$REPO_ROOT/infra/macos/prometheus/rules/live-execution-stage17.rules.yml" /opt/roehub/config/prometheus.rules/live-execution-stage17.rules.yml
install -m 0644 "$REPO_ROOT/infra/macos/prometheus/rules/strategy-producer.rules.yml" /opt/roehub/config/prometheus.rules/strategy-producer.rules.yml
install -m 0644 "$REPO_ROOT/infra/macos/prometheus/rules/market-data-funding.rules.yml" /opt/roehub/config/prometheus.rules/market-data-funding.rules.yml
install -m 0644 "$REPO_ROOT/infra/macos/prometheus/rules/notifications-admin.rules.yml" /opt/roehub/config/prometheus.rules/notifications-admin.rules.yml
install -m 0644 "$REPO_ROOT/infra/macos/openbao/openbao.prod.hcl" /opt/roehub/config/openbao/openbao.prod.hcl
install -m 0644 "$REPO_ROOT/infra/macos/openbao/policies/roehub-exchange-control-transit.hcl" /opt/roehub/config/openbao/policies/roehub-exchange-control-transit.hcl
install -m 0644 "$REPO_ROOT/infra/macos/openbao/policies/roehub-api-transit-deny-decrypt.hcl" /opt/roehub/config/openbao/policies/roehub-api-transit-deny-decrypt.hcl
install -m 0644 "$REPO_ROOT/infra/macos/blackbox/blackbox.yml" /opt/roehub/config/blackbox.yml
install -m 0644 "$REPO_ROOT/infra/macos/clickhouse/config.xml" /opt/roehub/config/clickhouse.config.xml
install -m 0644 "$REPO_ROOT/infra/macos/clickhouse/users.d/roehub.xml" /opt/roehub/config/clickhouse.users.roehub.xml
install -m 0755 "$REPO_ROOT/scripts/macos/provision_openbao_transit_stage3a.sh" /opt/roehub/bin/provision_openbao_transit_stage3a.sh
install -m 0755 "$REPO_ROOT/scripts/macos/smoke_openbao_transit_acl.sh" /opt/roehub/bin/smoke_openbao_transit_acl.sh
install -m 0755 "$REPO_ROOT/scripts/macos/recover_openbao_transit.sh" /opt/roehub/bin/recover_openbao_transit.sh
install -m 0755 "$REPO_ROOT/infra/scripts/monit/launchctl_service_control.sh" /opt/homebrew/etc/monit.d/scripts/launchctl_service_control.sh
install -m 0600 "$REPO_ROOT/infra/scripts/monit/roehub-market-data.monitrc" /opt/homebrew/etc/monit.d/roehub-market-data.monitrc
install -m 0600 "$REPO_ROOT/infra/scripts/monit/roehub-backtest-job-runner.monitrc" /opt/homebrew/etc/monit.d/roehub-backtest-job-runner.monitrc
install -m 0600 "$REPO_ROOT/infra/scripts/monit/roehub-keycloak.monitrc" /opt/homebrew/etc/monit.d/roehub-keycloak.monitrc
install -m 0600 "$REPO_ROOT/infra/scripts/monit/roehub-exchange-control.monitrc" /opt/homebrew/etc/monit.d/roehub-exchange-control.monitrc
install -m 0600 "$REPO_ROOT/infra/scripts/monit/roehub-exchange-execution.monitrc" /opt/homebrew/etc/monit.d/roehub-exchange-execution.monitrc
install -m 0600 "$REPO_ROOT/infra/scripts/monit/roehub-strategy-live-runner.monitrc" /opt/homebrew/etc/monit.d/roehub-strategy-live-runner.monitrc
install -m 0600 "$REPO_ROOT/infra/scripts/monit/roehub-openbao.monitrc" /opt/homebrew/etc/monit.d/roehub-openbao.monitrc

for plist in \
  com.roehub.api.plist \
  com.roehub.openbao.plist \
  com.roehub.openbao-recover.plist \
  com.roehub.exchange-control.plist \
  com.roehub.exchange-execution.plist \
  com.roehub.strategy-live-runner.plist \
  com.roehub.backtest-job-runner.plist \
  com.roehub.market-data-ws-worker.plist \
  com.roehub.market-data-scheduler.plist \
  com.roehub.clickhouse.plist \
  com.roehub.blackbox-exporter.plist \
  com.roehub.postgres-exporter.plist \
  com.roehub.redis-exporter.plist \
  com.roehub.clickhouse-exporter.plist \
  com.roehub.tailscale-runtime.plist
do
  install -m 0644 "$REPO_ROOT/infra/macos/launchd/$plist" "$LAUNCH_AGENTS_DIR/$plist"
done

cat > /opt/homebrew/etc/prometheus.args <<'EOF'
--config.file=/opt/roehub/config/prometheus.prod.yml
--storage.tsdb.path=/opt/homebrew/var/prometheus
--storage.tsdb.retention.time=90d
EOF

cat > /opt/homebrew/etc/node_exporter.args <<'EOF'
--web.listen-address=127.0.0.1:9100
EOF

echo "prod native templates installed"
