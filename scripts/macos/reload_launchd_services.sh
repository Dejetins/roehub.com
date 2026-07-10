#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:-prod}"
UID_VALUE="$(id -u)"
PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH_AGENTS_DIR="/Users/daniildegtyarev/Library/LaunchAgents"

prod_services=(
  com.roehub.clickhouse.plist
  com.roehub.blackbox-exporter.plist
  com.roehub.tailscale-runtime.plist
  com.roehub.postgres-exporter.plist
  com.roehub.redis-exporter.plist
  com.roehub.clickhouse-exporter.plist
  com.roehub.openbao.plist
  com.roehub.openbao-recover.plist
  com.roehub.exchange-control.plist
  com.roehub.exchange-execution.plist
  com.roehub.strategy-live-runner.plist
  com.roehub.rl-trading-inference.plist
  com.roehub.notification-dispatcher.plist
  com.roehub.api.plist
  com.roehub.backtest-job-runner.plist
  com.roehub.market-data-ws-worker.plist
  com.roehub.market-data-scheduler.plist
)

prod_pre_openbao_services=(
  com.roehub.clickhouse.plist
  com.roehub.blackbox-exporter.plist
  com.roehub.tailscale-runtime.plist
  com.roehub.postgres-exporter.plist
  com.roehub.redis-exporter.plist
  com.roehub.clickhouse-exporter.plist
)

prod_post_openbao_services=(
  com.roehub.openbao-recover.plist
  com.roehub.exchange-control.plist
  com.roehub.exchange-execution.plist
  com.roehub.strategy-live-runner.plist
  com.roehub.rl-trading-inference.plist
  com.roehub.notification-dispatcher.plist
  com.roehub.api.plist
  com.roehub.backtest-job-runner.plist
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
  com.roehub.test.exchange-execution.plist
  com.roehub.test.backtest-job-runner.plist
  com.roehub.test.market-data-ws-worker.plist
  com.roehub.test.market-data-scheduler.plist
  com.roehub.test.backtest-artifact-publisher.plist
)

collect_worker_services() {
  local profile="$1"
  local prefix=""
  case "$profile" in
    prod)
      prefix="com.roehub.backtest-job-runner."
      ;;
    test)
      prefix="com.roehub.test.backtest-job-runner."
      ;;
    *)
      echo "unsupported profile for worker collection: $profile" >&2
      return 1
      ;;
  esac
  find "$LAUNCH_AGENTS_DIR" -maxdepth 1 -type f -name "${prefix}*.plist" -print \
    | sed "s#${LAUNCH_AGENTS_DIR}/##" \
    | LC_ALL=C sort
}

service_label_from_plist() {
  local service="$1"
  printf '%s\n' "${service%.plist}"
}

service_is_loaded() {
  local label="$1"
  launchctl print "gui/${UID_VALUE}/${label}" >/dev/null 2>&1
}

wait_until_unloaded() {
  local label="$1"
  local attempts="${2:-10}"
  local attempt=0
  while service_is_loaded "$label"; do
    attempt=$((attempt + 1))
    if (( attempt >= attempts )); then
      echo "service ${label} is still loaded after bootout" >&2
      return 1
    fi
    sleep 1
  done
}

bootout_service() {
  local service="$1"
  local plist_path="${LAUNCH_AGENTS_DIR}/${service}"
  local label=""
  label="$(service_label_from_plist "$service")"
  if ! service_is_loaded "$label"; then
    echo "bootout skip ${label}: service not loaded"
    return 0
  fi
  echo "bootout ${label}"
  if launchctl bootout "gui/${UID_VALUE}" "${plist_path}"; then
    wait_until_unloaded "$label"
    return 0
  fi
  echo "bootout path failed for ${label}; retrying by service target" >&2
  launchctl bootout "gui/${UID_VALUE}/${label}"
  wait_until_unloaded "$label"
}

bootstrap_service() {
  local service="$1"
  local plist_path="${LAUNCH_AGENTS_DIR}/${service}"
  local label=""
  label="$(service_label_from_plist "$service")"
  plutil -lint "${plist_path}" >/dev/null
  echo "enable ${label}"
  launchctl enable "gui/${UID_VALUE}/${label}" || true
  if service_is_loaded "$label"; then
    echo "bootstrap preflight ${label}: service still loaded, forcing service-target bootout"
    launchctl bootout "gui/${UID_VALUE}/${label}" || true
    wait_until_unloaded "$label"
  fi
  echo "bootstrap ${label}"
  if launchctl bootstrap "gui/${UID_VALUE}" "${plist_path}"; then
    return 0
  fi
  if service_is_loaded "$label"; then
    echo "bootstrap ${label}: launchctl reported failure but service is loaded"
    return 0
  fi
  return 1
}

recover_openbao_transit() {
  local recovery_script="/opt/roehub/bin/recover_openbao_transit.sh"
  if [[ ! -x "$recovery_script" ]]; then
    echo "missing OpenBao recovery script: ${recovery_script}" >&2
    return 1
  fi
  echo "recover OpenBao transit"
  "$recovery_script"
}

reload_profile() {
  local profile="$1"
  local -a static_services=()
  local -a existing_worker_services=()
  local existing_worker_count=0
  local service=""

  case "$profile" in
    prod)
      static_services=("${prod_services[@]}")
      ;;
    test)
      static_services=("${test_services[@]}")
      ;;
    *)
      echo "unsupported profile: $profile" >&2
      return 1
      ;;
  esac

  while IFS= read -r service; do
    existing_worker_services+=("$service")
  done < <(collect_worker_services "$profile")
  existing_worker_count="${#existing_worker_services[@]}"
  echo "reloading ${profile} static services: ${#static_services[@]}"
  echo "removing legacy ${profile} backtest-job-runner services: ${existing_worker_count}"
  for service in "${static_services[@]}"; do
    bootout_service "${service}" || true
  done
  if (( existing_worker_count > 0 )); then
    for service in "${existing_worker_services[@]}"; do
      bootout_service "${service}" || true
    done
  fi

  if [[ "$profile" == "prod" ]]; then
    for service in "${prod_pre_openbao_services[@]}"; do
      bootstrap_service "${service}"
    done
    bootstrap_service com.roehub.openbao.plist
    recover_openbao_transit
    for service in "${prod_post_openbao_services[@]}"; do
      bootstrap_service "${service}"
    done
  else
    for service in "${static_services[@]}"; do
      bootstrap_service "${service}"
    done
  fi
}

case "$PROFILE" in
  prod|test)
    reload_profile "$PROFILE"
    ;;
  all)
    reload_profile prod
    reload_profile test
    ;;
  *)
    echo "usage: $0 [prod|test|all]" >&2
    exit 1
    ;;
esac

launchctl list | grep -E "roehub|clickhouse|blackbox|redis-exporter|postgres-exporter|tailscale" || true
