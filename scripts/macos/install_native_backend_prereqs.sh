#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

CLICKHOUSE_VERSION="v26.2.4.23-stable"
CLICKHOUSE_URL="https://github.com/ClickHouse/ClickHouse/releases/download/${CLICKHOUSE_VERSION}/clickhouse-macos-aarch64"
CLICKHOUSE_SHA256="fa6589cd762fb4d77f839c84e78a87706a30a414506da3ae9ebcc8720fbed7a1"

BLACKBOX_VERSION="0.28.0"
BLACKBOX_ARCHIVE="blackbox_exporter-${BLACKBOX_VERSION}.darwin-arm64.tar.gz"
BLACKBOX_URL="https://github.com/prometheus/blackbox_exporter/releases/download/v${BLACKBOX_VERSION}/${BLACKBOX_ARCHIVE}"
BLACKBOX_SHA256="ec6c70ccca92e209dd22be76a4fa244f4bd31afdae3ddb2bb082144100ec52bb"

ensure_writable_dir() {
  local dir="$1"
  mkdir -p "$dir" 2>/dev/null || true
  if [[ -d "$dir" && -w "$dir" ]]; then
    return
  fi

  cat >&2 <<EOF
error: directory is not writable: $dir
run once and retry:
  sudo install -d -m 755 -o $(id -un) -g staff "$dir"
EOF
  exit 1
}

brew update
brew install uv postgresql@16 redis grafana prometheus

ensure_writable_dir /opt/clickhouse
ensure_writable_dir /opt/roehub/bin
ensure_writable_dir /opt/roehub/state/downloads
ensure_writable_dir /opt/homebrew/var/lib/grafana
ensure_writable_dir /opt/homebrew/var/log/grafana

curl -fsSL "$CLICKHOUSE_URL" -o /opt/clickhouse/clickhouse
echo "${CLICKHOUSE_SHA256}  /opt/clickhouse/clickhouse" | shasum -a 256 -c -
chmod +x /opt/clickhouse/clickhouse

curl -fsSL "$BLACKBOX_URL" -o "/opt/roehub/state/downloads/${BLACKBOX_ARCHIVE}"
echo "${BLACKBOX_SHA256}  /opt/roehub/state/downloads/${BLACKBOX_ARCHIVE}" | shasum -a 256 -c -

rm -rf "/opt/roehub/state/downloads/blackbox_exporter-${BLACKBOX_VERSION}"
mkdir -p "/opt/roehub/state/downloads/blackbox_exporter-${BLACKBOX_VERSION}"
tar -xzf "/opt/roehub/state/downloads/${BLACKBOX_ARCHIVE}" -C "/opt/roehub/state/downloads/blackbox_exporter-${BLACKBOX_VERSION}"
install -m 0755 "/opt/roehub/state/downloads/blackbox_exporter-${BLACKBOX_VERSION}/blackbox_exporter-${BLACKBOX_VERSION}.darwin-arm64/blackbox_exporter" /opt/roehub/bin/blackbox_exporter

/opt/clickhouse/clickhouse local --query "SELECT version()"
/opt/roehub/bin/blackbox_exporter --version
