#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

CH_BIN="${CH_BIN:-/opt/clickhouse/clickhouse}"
CH_RECEIVE_TIMEOUT="${CH_RECEIVE_TIMEOUT:-3600}"
CH_MAX_MEMORY_USAGE="${CH_MAX_MEMORY_USAGE:-8000000000}"
CH_MAX_BYTES_BEFORE_EXTERNAL_SORT="${CH_MAX_BYTES_BEFORE_EXTERNAL_SORT:-1000000000}"

ZERO_UUID="00000000-0000-0000-0000-000000000000"
MODE="all"
INCLUDE_TODAY="0"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/macos/clickhouse_partition_dedup.sh [mode] [--include-today]

Modes:
  all        Run raw + canonical dedup (default)
  raw        Run only raw tables dedup (binance + bybit)
  canonical  Run only canonical_candles_1m partition FINAL optimize

Options:
  --include-today  Include current day partitions in processing

Environment overrides:
  CH_BIN
  CH_RECEIVE_TIMEOUT
  CH_MAX_MEMORY_USAGE
  CH_MAX_BYTES_BEFORE_EXTERNAL_SORT
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    all|raw|canonical)
      MODE="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-today)
      INCLUDE_TODAY="1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ch_query() {
  "${CH_BIN}" client --receive_timeout "${CH_RECEIVE_TIMEOUT}" --query "$1"
}

ch_multiquery() {
  "${CH_BIN}" client --receive_timeout "${CH_RECEIVE_TIMEOUT}" --multiquery --query "$1"
}

ensure_raw_stage_tables() {
  ch_query "CREATE TABLE IF NOT EXISTS market_data.raw_binance_klines_1m_dedup_stage AS market_data.raw_binance_klines_1m"
  ch_query "CREATE TABLE IF NOT EXISTS market_data.raw_bybit_klines_1m_dedup_stage AS market_data.raw_bybit_klines_1m"
}

dedup_raw_binance_day() {
  local day="$1"
  local partition_id="${day//-/}"
  ch_multiquery "
TRUNCATE TABLE market_data.raw_binance_klines_1m_dedup_stage;
INSERT INTO market_data.raw_binance_klines_1m_dedup_stage
SELECT *
FROM market_data.raw_binance_klines_1m
WHERE toDate(open_time) = toDate('${day}')
ORDER BY
  market_id,
  symbol,
  open_time,
  ingested_at DESC,
  coalesce(ingest_id, toUUID('${ZERO_UUID}')) DESC
LIMIT 1 BY market_id, symbol, open_time
SETTINGS
  max_memory_usage = ${CH_MAX_MEMORY_USAGE},
  max_bytes_before_external_sort = ${CH_MAX_BYTES_BEFORE_EXTERNAL_SORT};
ALTER TABLE market_data.raw_binance_klines_1m
REPLACE PARTITION ${partition_id}
FROM market_data.raw_binance_klines_1m_dedup_stage;
"
}

dedup_raw_bybit_day() {
  local day="$1"
  local partition_id="${day//-/}"
  ch_multiquery "
TRUNCATE TABLE market_data.raw_bybit_klines_1m_dedup_stage;
INSERT INTO market_data.raw_bybit_klines_1m_dedup_stage
SELECT *
FROM market_data.raw_bybit_klines_1m
WHERE toDate(start_time_utc) = toDate('${day}')
ORDER BY
  market_id,
  symbol,
  start_time_utc,
  ingested_at DESC,
  coalesce(ingest_id, toUUID('${ZERO_UUID}')) DESC
LIMIT 1 BY market_id, symbol, start_time_utc
SETTINGS
  max_memory_usage = ${CH_MAX_MEMORY_USAGE},
  max_bytes_before_external_sort = ${CH_MAX_BYTES_BEFORE_EXTERNAL_SORT};
ALTER TABLE market_data.raw_bybit_klines_1m
REPLACE PARTITION ${partition_id}
FROM market_data.raw_bybit_klines_1m_dedup_stage;
"
}

run_raw_dedup() {
  local tmp_binance_days
  local tmp_bybit_days
  local binance_filter
  local bybit_filter
  local binance_total
  local bybit_total
  local day
  local idx

  tmp_binance_days="$(mktemp /tmp/roehub_binance_days.XXXXXX)"
  tmp_bybit_days="$(mktemp /tmp/roehub_bybit_days.XXXXXX)"
  trap 'rm -f "${tmp_binance_days}" "${tmp_bybit_days}"' RETURN

  ensure_raw_stage_tables

  binance_filter="toDate(open_time) < today()"
  bybit_filter="toDate(start_time_utc) < today()"
  if [[ "${INCLUDE_TODAY}" = "1" ]]; then
    binance_filter="1"
    bybit_filter="1"
  fi

  log "Collecting duplicate days for raw_binance_klines_1m"
  ch_query "
SELECT formatDateTime(day, '%F')
FROM
(
  SELECT
    toDate(open_time) AS day,
    count() - uniqExact(tuple(market_id, symbol, open_time)) AS dup_rows
  FROM market_data.raw_binance_klines_1m
  WHERE ${binance_filter}
  GROUP BY day
  HAVING dup_rows > 0
  ORDER BY day
)
FORMAT TSV
" > "${tmp_binance_days}"

  binance_total="$(wc -l < "${tmp_binance_days}" | tr -d ' ')"
  log "raw_binance duplicate days: ${binance_total}"

  idx=0
  while IFS= read -r day; do
    [[ -z "${day}" ]] && continue
    idx=$((idx + 1))
    log "[binance ${idx}/${binance_total}] day=${day} start"
    dedup_raw_binance_day "${day}"
    log "[binance ${idx}/${binance_total}] day=${day} done"
  done < "${tmp_binance_days}"

  log "Collecting duplicate days for raw_bybit_klines_1m"
  ch_query "
SELECT formatDateTime(day, '%F')
FROM
(
  SELECT
    toDate(start_time_utc) AS day,
    count() - uniqExact(tuple(market_id, symbol, start_time_utc)) AS dup_rows
  FROM market_data.raw_bybit_klines_1m
  WHERE ${bybit_filter}
  GROUP BY day
  HAVING dup_rows > 0
  ORDER BY day
)
FORMAT TSV
" > "${tmp_bybit_days}"

  bybit_total="$(wc -l < "${tmp_bybit_days}" | tr -d ' ')"
  log "raw_bybit duplicate days: ${bybit_total}"

  idx=0
  while IFS= read -r day; do
    [[ -z "${day}" ]] && continue
    idx=$((idx + 1))
    log "[bybit ${idx}/${bybit_total}] day=${day} start"
    dedup_raw_bybit_day "${day}"
    log "[bybit ${idx}/${bybit_total}] day=${day} done"
  done < "${tmp_bybit_days}"

  log "Final duplicate counters for raw tables"
  ch_query "
SELECT
  'raw_binance' AS table_name,
  sum(rows_per_day) AS rows,
  sum(uniq_per_day) AS uniq_rows,
  sum(rows_per_day - uniq_per_day) AS duplicates
FROM
(
  SELECT
    toDate(open_time) AS day,
    count() AS rows_per_day,
    uniqExact(tuple(market_id, symbol, open_time)) AS uniq_per_day
  FROM market_data.raw_binance_klines_1m
  WHERE ${binance_filter}
  GROUP BY day
)
UNION ALL
SELECT
  'raw_bybit' AS table_name,
  sum(rows_per_day) AS rows,
  sum(uniq_per_day) AS uniq_rows,
  sum(rows_per_day - uniq_per_day) AS duplicates
FROM
(
  SELECT
    toDate(start_time_utc) AS day,
    count() AS rows_per_day,
    uniqExact(tuple(market_id, symbol, start_time_utc)) AS uniq_per_day
  FROM market_data.raw_bybit_klines_1m
  WHERE ${bybit_filter}
  GROUP BY day
)
"
}

run_canonical_dedup() {
  local tmp_partitions
  local partition_filter
  local total
  local idx
  local partition_id

  tmp_partitions="$(mktemp /tmp/roehub_canonical_parts.XXXXXX)"
  trap 'rm -f "${tmp_partitions}"' RETURN

  partition_filter="partition < formatDateTime(today(), '%Y%m%d')"
  if [[ "${INCLUDE_TODAY}" = "1" ]]; then
    partition_filter="1"
  fi

  log "Collecting partitions for canonical_candles_1m"
  ch_query "
SELECT partition
FROM system.parts
WHERE database = 'market_data'
  AND table = 'canonical_candles_1m'
  AND active
  AND ${partition_filter}
GROUP BY partition
ORDER BY partition
FORMAT TSV
" > "${tmp_partitions}"

  total="$(wc -l < "${tmp_partitions}" | tr -d ' ')"
  log "canonical partitions to optimize: ${total}"

  idx=0
  while IFS= read -r partition_id; do
    [[ -z "${partition_id}" ]] && continue
    idx=$((idx + 1))
    log "[canonical ${idx}/${total}] partition=${partition_id} start"
    ch_query "OPTIMIZE TABLE market_data.canonical_candles_1m PARTITION ID '${partition_id}' FINAL"
    log "[canonical ${idx}/${total}] partition=${partition_id} done"
  done < "${tmp_partitions}"
}

log "clickhouse_partition_dedup start (mode=${MODE}, include_today=${INCLUDE_TODAY})"

case "${MODE}" in
  raw)
    run_raw_dedup
    ;;
  canonical)
    run_canonical_dedup
    ;;
  all)
    run_raw_dedup
    run_canonical_dedup
    ;;
  *)
    printf 'Unknown mode: %s\n' "${MODE}" >&2
    usage
    exit 1
    ;;
esac

log "clickhouse_partition_dedup done"
