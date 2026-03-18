#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_tailscale_bin() {
  local candidate
  for candidate in \
    "${TAILSCALE_BIN:-}" \
    /usr/local/bin/tailscale \
    /opt/homebrew/bin/tailscale
  do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  candidate="$(command -v tailscale 2>/dev/null || true)"
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  printf 'error: tailscale binary not found\n' >&2
  return 1
}

get_backend_state() {
  "$TAILSCALE_BIN" status --json 2>/dev/null | /usr/bin/awk -F '"' '/"BackendState"[[:space:]]*:/ {print $4; exit}' || true
}

has_required_serve_config() {
  local status
  local required

  status="$($TAILSCALE_BIN serve status 2>/dev/null || true)"
  if [[ -z "$status" ]]; then
    return 1
  fi

  for required in \
    "tcp://127.0.0.1:5432" \
    "tcp://127.0.0.1:8123" \
    "tcp://127.0.0.1:9000" \
    "tcp://127.0.0.1:15433" \
    "tcp://127.0.0.1:18124" \
    "tcp://127.0.0.1:19001" \
    "http://127.0.0.1:8000" \
    "http://127.0.0.1:3000" \
    "http://127.0.0.1:18000" \
    "http://127.0.0.1:13000"
  do
    if [[ "$status" != *"$required"* ]]; then
      return 1
    fi
  done

  return 0
}

TAILSCALE_BIN="$(resolve_tailscale_bin)"
state="$(get_backend_state)"

if [[ "$state" != "Running" ]]; then
  "$TAILSCALE_BIN" up >/dev/null 2>&1 || true
  sleep 2
  state="$(get_backend_state)"
fi

if [[ "$state" != "Running" ]]; then
  printf 'error: tailscale backend state is %s\n' "${state:-unknown}" >&2
  exit 1
fi

if ! has_required_serve_config; then
  "$SCRIPT_DIR/configure_tailscale_serve.sh"
fi
