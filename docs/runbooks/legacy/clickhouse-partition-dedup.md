# ClickHouse Partition Dedup (Mac Studio)

Runbook для безопасного удаления дублей в таблицах:

- `market_data.raw_binance_klines_1m`
- `market_data.raw_bybit_klines_1m`
- `market_data.canonical_candles_1m`

Скрипт выполняет dedup по дням/partition, чтобы не упираться в память на full-history мутациях.

## Script

Путь:

- `scripts/macos/clickhouse_partition_dedup.sh`

Режимы:

- `raw` — дедуп только `raw_*` таблиц через stage table + `REPLACE PARTITION`
- `canonical` — дедуп `canonical_candles_1m` через `OPTIMIZE ... PARTITION ... FINAL`
- `all` — `raw` и затем `canonical`

По умолчанию текущий день исключается (чтобы не конфликтовать с активной записью).

## Запуск

Подключиться к Mac Studio:

```bash
ssh macstudio
```

Запустить всё сразу (рекомендуется):

```bash
bash scripts/macos/clickhouse_partition_dedup.sh all
```

Отдельно raw:

```bash
bash scripts/macos/clickhouse_partition_dedup.sh raw
```

Отдельно canonical:

```bash
bash scripts/macos/clickhouse_partition_dedup.sh canonical
```

Включить текущий день (только при остановленном ingestion):

```bash
bash scripts/macos/clickhouse_partition_dedup.sh all --include-today
```

Фоновый запуск с логом:

```bash
log_file="/tmp/roehub_clickhouse_partition_dedup_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts/macos/clickhouse_partition_dedup.sh all >"${log_file}" 2>&1 < /dev/null &
echo "$!"
echo "${log_file}"
```

## Мониторинг выполнения

Лог скрипта:

```bash
tail -f /tmp/roehub_clickhouse_partition_dedup_*.log
```

Проверка активных merge/mutation:

```bash
/opt/clickhouse/clickhouse client --query "
SELECT database, table, elapsed, progress, is_mutation, result_part_name
FROM system.merges
WHERE database = 'market_data'
ORDER BY elapsed DESC
LIMIT 20
"
```

Проверка мутаций:

```bash
/opt/clickhouse/clickhouse client --query "
SELECT table, mutation_id, is_done, parts_to_do, latest_fail_reason
FROM system.mutations
WHERE database = 'market_data'
ORDER BY create_time DESC
LIMIT 20
"
```

Память ClickHouse:

```bash
/opt/clickhouse/clickhouse client --query "
SELECT metric, value, formatReadableSize(value)
FROM system.metrics
WHERE metric IN ('MemoryTracking', 'BackgroundMergesAndMutationsPoolTask', 'MergesInFlight')
ORDER BY metric
"
```

## Post-check

Сводка дублей по raw (без текущего дня):

```bash
/opt/clickhouse/clickhouse client --query "
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
  WHERE toDate(open_time) < today()
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
  WHERE toDate(start_time_utc) < today()
  GROUP BY day
)
"
```

## Что такое multiquery

`--multiquery` нужен, когда в одном запуске передаётся несколько SQL через `;`.

Пример:

```bash
/opt/clickhouse/clickhouse client --multiquery --query "
SELECT 1;
SELECT 2;
"
```

Если выполняется один SQL-оператор за раз, `--multiquery` не нужен.
