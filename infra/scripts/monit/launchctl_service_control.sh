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
  if is_loaded; then
    launchctl kickstart -k "$service_target"
  else
    launchctl bootstrap "gui/${uid_value}" "$plist_path"
  fi
}

stop_service() {
  if ! is_loaded; then
    exit 0
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
    if is_loaded; then
      launchctl kickstart -k "$service_target"
    else
      start_service
    fi
    ;;
  status)
    launchctl print "$service_target" >/dev/null
    ;;
  *)
    echo "unsupported action: $action" >&2
    exit 64
    ;;
esac
