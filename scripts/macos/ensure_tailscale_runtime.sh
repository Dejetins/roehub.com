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
  "$TAILSCALE_BIN" status --json 2>/dev/null | python -c 'import json,sys; print(json.load(sys.stdin).get("BackendState", ""))' 2>/dev/null || true
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

TAILSCALE_SKIP_RESET=1 "$SCRIPT_DIR/configure_tailscale_serve.sh"
