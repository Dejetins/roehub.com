#!/usr/bin/env bash
set -Eeuo pipefail

# Sequentially deduplicates ReplacingMergeTree partitions in canonical_candles_1m.
# It detects partitions with duplicate (market_id, symbol, minute) keys and runs:
#   OPTIMIZE TABLE ... PARTITION <id> FINAL
#
# Optional filter:
#   INSTRUMENT_KEY='binance:spot:BTCUSDT' to scan (and thus optimize) only partitions
#   where this instrument has duplicates.
#
# Defaults are aligned with prod runbook paths but can be overridden via env vars.
#
# Examples:
#   DRY_RUN=1 bash scripts/ops/optimize_canonical_partitions.sh
#   INSTRUMENT_KEY='binance:spot:BTCUSDT' DRY_RUN=1 bash scripts/ops/optimize_canonical_partitions.sh
#   INSTRUMENT_KEY='binance:spot:BTCUSDT' MAX_PARTITIONS=5 SLEEP_SECONDS=3 bash scripts/ops/optimize_canonical_partitions.sh

ENV_FILE="${ENV_FILE:-/Users/daniildegtyarev/.config/roehub/roehub.env}"

CLICKHOUSE_BIN="${CLICKHOUSE_BIN:-/opt/clickhouse/clickhouse}"
CH_HOST="${CH_HOST:-127.0.0.1}"
CH_PORT="${CH_PORT:-9000}"
CH_USER="${CH_USER:-}"
CH_PASSWORD="${CH_PASSWORD:-}"

DB="${DB:-market_data}"
TABLE="${TABLE:-canonical_candles_1m}"

MIN_DUP_ROWS="${MIN_DUP_ROWS:-1}"
MAX_PARTITIONS="${MAX_PARTITIONS:-0}"   # 0 = all partitions with duplicates
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"     # pause between OPTIMIZE commands
DRY_RUN="${DRY_RUN:-0}"                 # 1 = print plan only, do not optimize

# Optional instrument filter
INSTRUMENT_KEY="${INSTRUMENT_KEY:-}"
INSTRUMENT_KEY_COL="${INSTRUMENT_KEY_COL:-instrument_key}"

if [[ -s "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

CH_USER="${CH_USER:-${CLICKHOUSE_USER:-default}}"
CH_PASSWORD="${CH_PASSWORD:-${CLICKHOUSE_PASSWORD:-}}"

ch_query() {
  local query="$1"
  if [[ -n "${CH_PASSWORD}" ]]; then
    "${CLICKHOUSE_BIN}" client --host "${CH_HOST}" --port "${CH_PORT}" --user "${CH_USER}" --password "${CH_PASSWORD}" --query "${query}"
    return
  fi
  "${CLICKHOUSE_BIN}" client --host "${CH_HOST}" --port "${CH_PORT}" --user "${CH_USER}" --query "${query}"
}

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

sql_quote() {
  # ClickHouse SQL string literal with basic single-quote escaping.
  local s="$1"
  s="${s//\'/\\\'}"
  printf "'%s'" "${s}"
}

where_filter=""
if [[ -n "${INSTRUMENT_KEY}" ]]; then
  where_filter="WHERE ${INSTRUMENT_KEY_COL} = $(sql_quote "${INSTRUMENT_KEY}")"
fi

if [[ -n "${INSTRUMENT_KEY}" ]]; then
  echo "[$(timestamp)] scanning partitions with duplicates in ${DB}.${TABLE} (filter: ${INSTRUMENT_KEY_COL}=${INSTRUMENT_KEY}) ..."
else
  echo "[$(timestamp)] scanning partitions with duplicates in ${DB}.${TABLE} ..."
fi

detect_query="
SELECT
  toYYYYMMDD(ts_open) AS partition_id,
  count() AS rows_total,
  uniqExact((market_id, symbol, toStartOfMinute(ts_open))) AS uniq_rows,
  rows_total - uniq_rows AS dup_rows
FROM ${DB}.${TABLE}
${where_filter}
GROUP BY partition_id
HAVING dup_rows >= ${MIN_DUP_ROWS}
ORDER BY partition_id
FORMAT TSVRaw
"

partition_rows_raw="$(ch_query "${detect_query}")" || {
  echo "[$(timestamp)] failed to scan duplicate partitions." >&2
  exit 1
}

partition_rows=()
while IFS= read -r row; do
  [[ -z "${row}" ]] && continue
  partition_rows+=("${row}")
done <<<"${partition_rows_raw}"

if ((${#partition_rows[@]} == 0)); then
  echo "[$(timestamp)] no partitions with dup_rows >= ${MIN_DUP_ROWS}; nothing to do."
  exit 0
fi

echo "[$(timestamp)] found ${#partition_rows[@]} partition(s) with duplicates."

optimized=0
planned=0
touched_partitions=()

for row in "${partition_rows[@]}"; do
  IFS=$'\t' read -r partition_id rows_total uniq_rows dup_rows <<<"${row}"

  if ((MAX_PARTITIONS > 0 && planned >= MAX_PARTITIONS)); then
    break
  fi
  planned=$((planned + 1))

  echo "[$(timestamp)] [${planned}] partition=${partition_id} rows=${rows_total} uniq=${uniq_rows} dup=${dup_rows}"
  touched_partitions+=("${partition_id}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  ch_query "OPTIMIZE TABLE ${DB}.${TABLE} PARTITION ${partition_id} FINAL SETTINGS optimize_throw_if_noop=0;"
  optimized=$((optimized + 1))

  if [[ "${SLEEP_SECONDS}" != "0" ]]; then
    sleep "${SLEEP_SECONDS}"
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[$(timestamp)] DRY_RUN=1, optimization was not executed."
  exit 0
fi

if ((${optimized} == 0)); then
  echo "[$(timestamp)] no partitions optimized (check MAX_PARTITIONS/MIN_DUP_ROWS)."
  exit 0
fi

in_list=""
for p in "${touched_partitions[@]}"; do
  if [[ -n "${in_list}" ]]; then
    in_list+=","
  fi
  in_list+="${p}"
done

post_where="WHERE toYYYYMMDD(ts_open) IN (${in_list})"
if [[ -n "${INSTRUMENT_KEY}" ]]; then
  post_where+=" AND ${INSTRUMENT_KEY_COL} = $(sql_quote "${INSTRUMENT_KEY}")"
fi

echo "[$(timestamp)] optimized ${optimized} partition(s). Post-check:"
ch_query "
SELECT
  toYYYYMMDD(ts_open) AS partition_id,
  count() - uniqExact((market_id, symbol, toStartOfMinute(ts_open))) AS dup_rows
FROM ${DB}.${TABLE}
${post_where}
GROUP BY partition_id
ORDER BY partition_id
FORMAT PrettyCompact
"

echo "[$(timestamp)] done."
