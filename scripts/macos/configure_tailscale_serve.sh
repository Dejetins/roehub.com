#!/usr/bin/env bash
set -Eeuo pipefail

TAILSCALE_BIN="${TAILSCALE_BIN:-/usr/local/bin/tailscale}"

"$TAILSCALE_BIN" serve reset || true

# Production endpoints.
"$TAILSCALE_BIN" serve --bg http://127.0.0.1:8000
"$TAILSCALE_BIN" serve --bg --https=3443 3000
"$TAILSCALE_BIN" serve --bg --tcp=15432 tcp://127.0.0.1:5432
"$TAILSCALE_BIN" serve --bg --tcp=18123 tcp://127.0.0.1:8123
"$TAILSCALE_BIN" serve --bg --tcp=19000 tcp://127.0.0.1:9000

# Test endpoints.
"$TAILSCALE_BIN" serve --bg --https=8443 18000
"$TAILSCALE_BIN" serve --bg --https=3444 13000
"$TAILSCALE_BIN" serve --bg --tcp=25432 tcp://127.0.0.1:15433
"$TAILSCALE_BIN" serve --bg --tcp=28123 tcp://127.0.0.1:18124
"$TAILSCALE_BIN" serve --bg --tcp=29000 tcp://127.0.0.1:19001

"$TAILSCALE_BIN" serve status
