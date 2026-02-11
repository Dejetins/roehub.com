#!/usr/bin/env bash
set -Eeuo pipefail

# Sequentially deduplicates ReplacingMergeTree partitions in canonical_candles_1m.
# It detects partitions with duplicate (market_id, symbol, minute) keys and runs:
#   OPTIMIZE TABLE ... PARTITION <id> FINAL
#
# Defaults are aligned with prod runbook paths but can be overridden via env vars.
#
# Example:
#   DRY_RUN=1 bash scripts/ops/optimize_canonical_partitions.sh
#   MAX_PARTITIONS=5 SLEEP_SECONDS=3 bash scripts/ops/optimize_canonical_partitions.sh

COMPOSE_FILE="${COMPOSE_FILE:-/opt/roehub/docker-compose.yml}"
ENV_FILE="${ENV_FILE:-/etc/roehub/roehub.env}"
SERVICE="${SERVICE:-clickhouse}"

DB="${DB:-market_data}"
TABLE="${TABLE:-canonical_candles_1m}"

MIN_DUP_ROWS="${MIN_DUP_ROWS:-1}"
MAX_PARTITIONS="${MAX_PARTITIONS:-0}"   # 0 = all partitions with duplicates
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"     # pause between OPTIMIZE commands
DRY_RUN="${DRY_RUN:-0}"                 # 1 = print plan only, do not optimize

compose_cmd=(
  docker compose
  -f "${COMPOSE_FILE}"
  --env-file "${ENV_FILE}"
)

ch_query() {
  local query="$1"
  "${compose_cmd[@]}" exec -T "${SERVICE}" clickhouse-client -q "${query}"
}

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

echo "[$(timestamp)] scanning partitions with duplicates in ${DB}.${TABLE} ..."

detect_query="
SELECT
  toYYYYMMDD(ts_open) AS partition_id,
  count() AS rows_total,
  uniqExact((market_id, symbol, toStartOfMinute(ts_open))) AS uniq_rows,
  rows_total - uniq_rows AS dup_rows
FROM ${DB}.${TABLE}
GROUP BY partition_id
HAVING dup_rows >= ${MIN_DUP_ROWS}
ORDER BY partition_id
FORMAT TSVRaw
"

mapfile -t partition_rows < <(ch_query "${detect_query}")

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

echo "[$(timestamp)] optimized ${optimized} partition(s). Post-check:"
ch_query "
SELECT
  toYYYYMMDD(ts_open) AS partition_id,
  count() - uniqExact((market_id, symbol, toStartOfMinute(ts_open))) AS dup_rows
FROM ${DB}.${TABLE}
WHERE toYYYYMMDD(ts_open) IN (${in_list})
GROUP BY partition_id
ORDER BY partition_id
FORMAT PrettyCompact
"

echo "[$(timestamp)] done."
