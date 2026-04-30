#!/usr/bin/env bash
set -Eeuo pipefail

action="${1:-}"
label="${2:-}"
plist_path="${3:-}"

if [[ -z "$action" || -z "$label" || -z "$plist_path" ]]; then
  echo "usage: $0 <start|stop|restart|status> <label> <plist_path>" >&2
  exit 64
fi

uid_value="$(id -u)"
service_target="gui/${uid_value}/${label}"

is_loaded() {
  launchctl print "$service_target" >/dev/null 2>&1
}

start_service() {
  launchctl enable "$service_target" || true
  if ! is_loaded; then
    launchctl bootstrap "gui/${uid_value}" "$plist_path" || is_loaded
  fi
  launchctl kickstart "$service_target"
}

stop_service() {
  launchctl disable "$service_target" || true
  if ! is_loaded; then
    return 0
  fi
  launchctl bootout "$service_target" || launchctl bootout "gui/${uid_value}" "$plist_path"
}

case "$action" in
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  restart)
    stop_service
    start_service
    ;;
  status)
    launchctl print "$service_target" >/dev/null
    ;;
  *)
    echo "unsupported action: $action" >&2
    exit 64
    ;;
esac
