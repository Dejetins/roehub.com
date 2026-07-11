#!/usr/bin/env bash
set -Eeuo pipefail

log_dir="${ROEHUB_LOG_DIR:-/Users/daniildegtyarev/Library/Logs/roehub}"
max_bytes="${ROEHUB_LOG_ROTATE_MAX_BYTES:-$((10 * 1024 * 1024))}"
keep="${ROEHUB_LOG_ROTATE_KEEP:-7}"
lock_dir="${log_dir}/.notification-log-rotation.lock"

mkdir -p "$log_dir"
if ! mkdir "$lock_dir" 2>/dev/null; then
  exit 0
fi
trap 'rmdir "$lock_dir"' EXIT

for name in \
  notification-dispatcher.out.log \
  notification-dispatcher.err.log \
  telegram-egress-tunnel.out.log \
  telegram-egress-tunnel.err.log
do
  path="${log_dir}/${name}"
  [[ -f "$path" ]] || continue
  size="$(stat -f '%z' "$path")"
  (( size >= max_bytes )) || continue

  rm -f "${path}.${keep}.gz"
  for ((index = keep - 1; index >= 1; index--)); do
    [[ -f "${path}.${index}.gz" ]] || continue
    mv "${path}.${index}.gz" "${path}.$((index + 1)).gz"
  done
  cp "$path" "${path}.1"
  gzip -f "${path}.1"
  : > "$path"
  chmod 0640 "$path"
done
