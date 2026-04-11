#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:-prod}"
UID_VALUE="$(id -u)"
PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH_AGENTS_DIR="/Users/daniildegtyarev/Library/LaunchAgents"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

prod_services=(
  com.roehub.clickhouse.plist
  com.roehub.blackbox-exporter.plist
  com.roehub.tailscale-runtime.plist
  com.roehub.postgres-exporter.plist
  com.roehub.redis-exporter.plist
  com.roehub.clickhouse-exporter.plist
  com.roehub.api.plist
  com.roehub.market-data-ws-worker.plist
  com.roehub.market-data-scheduler.plist
  com.roehub.backtest-artifact-publisher.plist
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

render_worker_services() {
  local profile="$1"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/macos/render_backtest_job_runner_launchd.py" \
    --profile "$profile" \
    --repo-root "$REPO_ROOT" \
    --launch-agents-dir "$LAUNCH_AGENTS_DIR" \
    --clean
}

service_label_from_plist() {
  local service="$1"
  printf '%s\n' "${service%.plist}"
}

service_is_loaded() {
  local label="$1"
  launchctl print "gui/${UID_VALUE}/${label}" >/dev/null 2>&1
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
    return 0
  fi
  echo "bootout path failed for ${label}; retrying by service target" >&2
  launchctl bootout "gui/${UID_VALUE}/${label}"
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
  fi
  echo "bootstrap ${label}"
  launchctl bootstrap "gui/${UID_VALUE}" "${plist_path}"
}

reload_profile() {
  local profile="$1"
  local -a static_services=()
  local -a existing_worker_services=()
  local -a worker_services=()
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
  echo "reloading ${profile} static services: ${#static_services[@]}"
  echo "reloading ${profile} backtest-job-runner fleet (existing instances): ${#existing_worker_services[@]}"
  for service in "${static_services[@]}" "${existing_worker_services[@]}"; do
    bootout_service "${service}" || true
  done

  while IFS= read -r service; do
    worker_services+=("$service")
  done < <(render_worker_services "$profile")
  echo "reloading ${profile} backtest-job-runner fleet (desired instances): ${#worker_services[@]}"
  for service in "${static_services[@]}" "${worker_services[@]}"; do
    bootstrap_service "${service}"
  done
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

launchctl list | grep -E "roehub|backtest-job-runner|clickhouse|blackbox|redis-exporter|postgres-exporter|tailscale" || true
