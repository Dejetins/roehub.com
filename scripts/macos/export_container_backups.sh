#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
ROEHUB_ENV_FILE="${ROEHUB_ENV_FILE:-/Users/daniildegtyarev/.config/roehub/roehub.env}"
BACKUP_ROOT="${BACKUP_ROOT:-/opt/roehub/state/backups}"

if [[ ! -s "$ROEHUB_ENV_FILE" ]]; then
  echo "env file not found: $ROEHUB_ENV_FILE" >&2
  exit 1
fi

mkdir -p "$BACKUP_ROOT/postgres" "$BACKUP_ROOT/clickhouse" "$BACKUP_ROOT/grafana" "$BACKUP_ROOT/redis"

set -a
source "$ROEHUB_ENV_FILE"
set +a

docker exec roehub-postgres-1 psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "select now();"
docker exec roehub-clickhouse-1 clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "SELECT count() FROM market_data.canonical_candles_1m"
docker exec roehub-clickhouse-1 clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "SELECT max(ts_open) FROM market_data.canonical_candles_1m"

docker exec -t roehub-postgres-1 pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc > "$BACKUP_ROOT/postgres/roehub_prod.dump"
shasum -a 256 "$BACKUP_ROOT/postgres/roehub_prod.dump" > "$BACKUP_ROOT/postgres/roehub_prod.dump.sha256"

docker exec roehub-clickhouse-1 mkdir -p /var/lib/clickhouse/backup
docker exec roehub-clickhouse-1 clickhouse-client --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" --query "BACKUP DATABASE market_data TO File('/var/lib/clickhouse/backup/market_data_prod.zip')"
docker cp roehub-clickhouse-1:/var/lib/clickhouse/backup/market_data_prod.zip "$BACKUP_ROOT/clickhouse/market_data_prod.zip"
shasum -a 256 "$BACKUP_ROOT/clickhouse/market_data_prod.zip" > "$BACKUP_ROOT/clickhouse/market_data_prod.zip.sha256"

docker run --rm -v grafana_data:/from -v "$BACKUP_ROOT/grafana:/to" alpine sh -c "cp -a /from/. /to/"
tar -C "$BACKUP_ROOT" -czf "$BACKUP_ROOT/grafana_prod.tar.gz" grafana
shasum -a 256 "$BACKUP_ROOT/grafana_prod.tar.gz" > "$BACKUP_ROOT/grafana_prod.tar.gz.sha256"

docker exec redis redis-cli SAVE || true
docker run --rm -v roehub_redis_data:/from -v "$BACKUP_ROOT/redis:/to" alpine sh -c "cp -a /from/. /to/" || true

cp "$ROEHUB_ENV_FILE" "$BACKUP_ROOT/roehub.env.backup"
cp /opt/roehub/docker-compose.backend.yml "$BACKUP_ROOT/docker-compose.backend.yml.backup"

ls -lah "$BACKUP_ROOT/postgres" "$BACKUP_ROOT/clickhouse" "$BACKUP_ROOT"
