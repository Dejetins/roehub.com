#!/usr/bin/env bash
set -Eeuo pipefail

# Deduplicate canonical_candles_1m by optimizing only partitions that contain duplicates
# for a specific instrument_key. Duplicate key definition:
#   (market_id, symbol, toStartOfMinute(ts_open))
#
# It detects dup_rows per partition with memory caps + external GROUP BY (spill to disk),
# then runs:
#   OPTIMIZE TABLE ... PARTITION <YYYYMMDD> FINAL
#
# Examples:
#   INSTRUMENT_KEY='binance:spot:BTCUSDT' DRY_RUN=1 \
#     bash scripts/ops/optimize_canonical_by_instrument.sh
#
#   INSTRUMENT_KEY='binance:spot:BTCUSDT' \
#     SCAN_MAX_THREADS=1 SCAN_MAX_MEMORY_BYTES=$((600*1024*1024)) \
#     bash scripts/ops/optimize_canonical_by_instrument.sh
#
#   INSTRUMENT_KEY='binance:spot:BTCUSDT' PARTITION_FROM=20260201 PARTITION_TO=20260210 DRY_RUN=1 \
#     bash scripts/ops/optimize_canonical_by_instrument.sh
#
#   INSTRUMENT_KEY='binance:spot:BTCUSDT' MAX_PARTITIONS=3 SLEEP_SECONDS=5 \
#     bash scripts/ops/optimize_canonical_by_instrument.sh

COMPOSE_FILE="${COMPOSE_FILE:-/opt/roehub/docker-compose.yml}"
ENV_FILE="${ENV_FILE:-/etc/roehub/roehub.env}"
SERVICE="${SERVICE:-clickhouse}"

DB="${DB:-market_data}"
TABLE="${TABLE:-canonical_candles_1m}"

INSTRUMENT_KEY="${INSTRUMENT_KEY:-}"     # REQUIRED

MIN_DUP_ROWS="${MIN_DUP_ROWS:-1}"
MAX_PARTITIONS="${MAX_PARTITIONS:-0}"    # 0 = no limit
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
DRY_RUN="${DRY_RUN:-0}"

# Optional inclusive bounds for partition_id (YYYYMMDD) based on toYYYYMMDD(ts_open)
PARTITION_FROM="${PARTITION_FROM:-}"     # e.g. 20260201
PARTITION_TO="${PARTITION_TO:-}"         # e.g. 20260210

# Scan (duplicate detection) resource limits (ClickHouse SETTINGS)
SCAN_MAX_THREADS="${SCAN_MAX_THREADS:-1}"
SCAN_MAX_MEMORY_BYTES="${SCAN_MAX_MEMORY_BYTES:-800000000}"          # ~800MB
SCAN_EXTERNAL_GB_BYTES="${SCAN_EXTERNAL_GB_BYTES:-200000000}"         # spill after ~200MB
SCAN_MAX_TMP_DISK_BYTES="${SCAN_MAX_TMP_DISK_BYTES:-8000000000}"      # temp on disk cap (8GB)
SCAN_MAX_EXECUTION_TIME="${SCAN_MAX_EXECUTION_TIME:-0}"              # 0=unlimited, else seconds

# OPTIMIZE query limits (limits the optimize query; merges can still load the server)
OPTIMIZE_MAX_THREADS="${OPTIMIZE_MAX_THREADS:-1}"
OPTIMIZE_MAX_MEMORY_BYTES="${OPTIMIZE_MAX_MEMORY_BYTES:-800000000}"
OPTIMIZE_MAX_EXECUTION_TIME="${OPTIMIZE_MAX_EXECUTION_TIME:-0}"      # 0=unlimited, else seconds

# If scan fails for a partition, continue instead of exiting
CONTINUE_ON_SCAN_ERROR="${CONTINUE_ON_SCAN_ERROR:-0}"

compose_cmd=(
  docker compose
  -f "${COMPOSE_FILE}"
  --env-file "${ENV_FILE}"
)

ch_query() {
  local query="$1"
  "${compose_cmd[@]}" exec -T "${SERVICE}" clickhouse-client -q "${query}"
}

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
is_yyyymmdd() { [[ "${1:-}" =~ ^[0-9]{8}$ ]]; }

settings_kv() {
  local max_time="$1"
  if [[ "$max_time" != "0" ]]; then
    echo ", max_execution_time = ${max_time}"
  else
    echo ""
  fi
}

sql_escape_string() {
  # Escape single quotes for ClickHouse SQL literals.
  printf "%s" "$1" | sed "s/'/''/g"
}

# ---- validation ----
if [[ -z "$INSTRUMENT_KEY" ]]; then
  echo "INSTRUMENT_KEY is required, e.g.:" >&2
  echo "  INSTRUMENT_KEY='binance:spot:BTCUSDT' DRY_RUN=1 bash $0" >&2
  exit 2
fi

if [[ -n "$PARTITION_FROM" ]] && ! is_yyyymmdd "$PARTITION_FROM"; then
  echo "PARTITION_FROM must be YYYYMMDD (got: $PARTITION_FROM)" >&2
  exit 2
fi
if [[ -n "$PARTITION_TO" ]] && ! is_yyyymmdd "$PARTITION_TO"; then
  echo "PARTITION_TO must be YYYYMMDD (got: $PARTITION_TO)" >&2
  exit 2
fi
if [[ -n "$PARTITION_FROM" && -n "$PARTITION_TO" ]] && [[ "$PARTITION_FROM" > "$PARTITION_TO" ]]; then
  echo "PARTITION_FROM must be <= PARTITION_TO" >&2
  exit 2
fi

instrument_key_sql="$(sql_escape_string "$INSTRUMENT_KEY")"

echo "[$(timestamp)] config:"
echo "  table=${DB}.${TABLE}"
echo "  instrument_key='${INSTRUMENT_KEY}'"
echo "  min_dup_rows=${MIN_DUP_ROWS} max_partitions=${MAX_PARTITIONS} sleep_seconds=${SLEEP_SECONDS} dry_run=${DRY_RUN}"
if [[ -n "$PARTITION_FROM" || -n "$PARTITION_TO" ]]; then
  echo "  partition_range=[${PARTITION_FROM:-min}..${PARTITION_TO:-max}]"
fi
echo "  scan_limits: threads=${SCAN_MAX_THREADS} mem=${SCAN_MAX_MEMORY_BYTES} external_gb=${SCAN_EXTERNAL_GB_BYTES} tmp_disk=${SCAN_MAX_TMP_DISK_BYTES} max_time=${SCAN_MAX_EXECUTION_TIME}"
echo "  optimize_limits: threads=${OPTIMIZE_MAX_THREADS} mem=${OPTIMIZE_MAX_MEMORY_BYTES} max_time=${OPTIMIZE_MAX_EXECUTION_TIME}"
echo

# Partition list from metadata (cheap)
parts_where_extra=""
if [[ -n "$PARTITION_FROM" ]]; then
  parts_where_extra+=" AND partition >= '${PARTITION_FROM}'"
fi
if [[ -n "$PARTITION_TO" ]]; then
  parts_where_extra+=" AND partition <= '${PARTITION_TO}'"
fi

echo "[$(timestamp)] listing partitions from system.parts ..."
parts_query="
SELECT
  partition AS partition_id,
  sum(rows) AS rows_total
FROM system.parts
WHERE database = '${DB}'
  AND table = '${TABLE}'
  AND active
  ${parts_where_extra}
GROUP BY partition_id
ORDER BY partition_id
FORMAT TSVRaw
"
mapfile -t parts_rows < <(ch_query "${parts_query}")

if ((${#parts_rows[@]} == 0)); then
  echo "[$(timestamp)] no active partitions found in range; nothing to do."
  exit 0
fi

scan_time_setting="$(settings_kv "$SCAN_MAX_EXECUTION_TIME")"
opt_time_setting="$(settings_kv "$OPTIMIZE_MAX_EXECUTION_TIME")"

planned=0
optimized=0
touched_partitions=()

echo "[$(timestamp)] scanning for duplicates (per-partition, memory-capped) ..."

for row in "${parts_rows[@]}"; do
  IFS=$'\t' read -r partition_id rows_total <<<"${row}"

  if ! is_yyyymmdd "$partition_id"; then
    echo "[$(timestamp)] skip non-YYYYMMDD partition_id='${partition_id}'"
    continue
  fi

  if (( MAX_PARTITIONS > 0 && planned >= MAX_PARTITIONS )); then
    break
  fi

  # dup_rows for this instrument in this partition:
  # sum(cnt-1) over groups where cnt>1
  dup_query="
SELECT
  ifNull(sum(cnt - 1), 0) AS dup_rows
FROM
(
  SELECT count() AS cnt
  FROM ${DB}.${TABLE}
  WHERE toYYYYMMDD(ts_open) = ${partition_id}
    AND instrument_key = '${instrument_key_sql}'
  GROUP BY market_id, symbol, toStartOfMinute(ts_open)
  HAVING cnt > 1
)
SETTINGS
  max_threads = ${SCAN_MAX_THREADS},
  max_memory_usage = ${SCAN_MAX_MEMORY_BYTES},
  max_bytes_before_external_group_by = ${SCAN_EXTERNAL_GB_BYTES},
  max_temporary_data_on_disk = ${SCAN_MAX_TMP_DISK_BYTES}
  ${scan_time_setting}
FORMAT TSVRaw
"

  dup_rows=""
  if dup_rows="$(ch_query "${dup_query}")"; then
    :
  else
    if [[ "$CONTINUE_ON_SCAN_ERROR" == "1" ]]; then
      echo "[$(timestamp)] scan failed for partition=${partition_id}; skipping (CONTINUE_ON_SCAN_ERROR=1)" >&2
      continue
    fi
    echo "[$(timestamp)] scan failed for partition=${partition_id}; aborting" >&2
    exit 1
  fi

  dup_rows="$(echo -n "$dup_rows" | tr -d $'\r\n')"
  [[ -z "$dup_rows" ]] && dup_rows="0"

  if (( dup_rows < MIN_DUP_ROWS )); then
    continue
  fi

  planned=$((planned + 1))
  echo "[$(timestamp)] [${planned}] partition=${partition_id} instrument_dup_rows=${dup_rows}"
  touched_partitions+=("${partition_id}")

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  echo "[$(timestamp)] optimizing partition=${partition_id} (FINAL) ..."
  ch_query "
OPTIMIZE TABLE ${DB}.${TABLE} PARTITION ${partition_id} FINAL
SETTINGS
  optimize_throw_if_noop = 0,
  max_threads = ${OPTIMIZE_MAX_THREADS},
  max_memory_usage = ${OPTIMIZE_MAX_MEMORY_BYTES}
  ${opt_time_setting}
;
"
  optimized=$((optimized + 1))

  if [[ "${SLEEP_SECONDS}" != "0" ]]; then
    sleep "${SLEEP_SECONDS}"
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[$(timestamp)] DRY_RUN=1, optimization was not executed."
  if ((${#touched_partitions[@]} == 0)); then
    echo "[$(timestamp)] no partitions matched dup_rows >= ${MIN_DUP_ROWS} for instrument_key='${INSTRUMENT_KEY}'."
  else
    echo "[$(timestamp)] planned partitions (${#touched_partitions[@]}): ${touched_partitions[*]}"
  fi
  exit 0
fi

if (( optimized == 0 )); then
  echo "[$(timestamp)] no partitions optimized (check MIN_DUP_ROWS/MAX_PARTITIONS/partition range)."
  exit 0
fi

echo "[$(timestamp)] optimized ${optimized} partition(s). Post-check (memory-capped):"
for p in "${touched_partitions[@]}"; do
  check_query="
SELECT
  ifNull(sum(cnt - 1), 0) AS dup_rows
FROM
(
  SELECT count() AS cnt
  FROM ${DB}.${TABLE}
  WHERE toYYYYMMDD(ts_open) = ${p}
    AND instrument_key = '${instrument_key_sql}'
  GROUP BY market_id, symbol, toStartOfMinute(ts_open)
  HAVING cnt > 1
)
SETTINGS
  max_threads = ${SCAN_MAX_THREADS},
  max_memory_usage = ${SCAN_MAX_MEMORY_BYTES},
  max_bytes_before_external_group_by = ${SCAN_EXTERNAL_GB_BYTES},
  max_temporary_data_on_disk = ${SCAN_MAX_TMP_DISK_BYTES}
  ${scan_time_setting}
FORMAT TSVRaw
"
  d="$(ch_query "${check_query}" | tr -d $'\r\n')"
  echo "  partition=${p} instrument_dup_rows=${d:-0}"
done

echo "[$(timestamp)] done."
