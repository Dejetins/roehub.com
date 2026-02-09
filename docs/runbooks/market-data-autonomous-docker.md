# Market Data Autonomous Docker Runbook

Runbook для автономного запуска `market-data-ws-worker` и `market-data-scheduler` в Docker.

## 1. Режимы запуска

### Режим по умолчанию (prod-friendly)

`infra/docker/docker-compose.market_data.yml` по умолчанию **не поднимает отдельный ClickHouse** и подключается к уже существующему CH (`CH_HOST=clickhouse`) через внешнюю сеть Docker:

- external network: `${ROEHUB_SHARED_NETWORK:-roehub_default}`
- без конфликта с `infra/docker/docker-compose.yml`

### Standalone режим (опционально)

Поднимает отдельный `market-data-clickhouse` только при профиле `standalone` на альтернативных портах:

- HTTP: `127.0.0.1:${MARKET_DATA_CH_HTTP_PORT:-18123}`
- Native: `127.0.0.1:${MARKET_DATA_CH_NATIVE_PORT:-19000}`

## 2. Локальный запуск (из репозитория)

```bash
cd /path/to/roehub.com
docker network inspect roehub_default >/dev/null 2>&1 || docker network create roehub_default
docker compose -f infra/docker/docker-compose.market_data.yml config >/dev/null
docker compose -f infra/docker/docker-compose.market_data.yml up -d --build
docker compose -f infra/docker/docker-compose.market_data.yml ps
```

Логи:

```bash
docker compose -f infra/docker/docker-compose.market_data.yml logs -f --tail=200 market-data-scheduler
docker compose -f infra/docker/docker-compose.market_data.yml logs -f --tail=200 market-data-ws-worker
```

Остановка:

```bash
docker compose -f infra/docker/docker-compose.market_data.yml down
```

Standalone запуск:

```bash
docker compose -f infra/docker/docker-compose.market_data.yml --profile standalone up -d --build
```

## 3. Prod запуск в фоне (на хосте)

```bash
docker compose -f /opt/roehub/docker-compose.market_data.yml --env-file /etc/roehub/roehub.env up -d --build --remove-orphans
docker compose -f /opt/roehub/docker-compose.market_data.yml --env-file /etc/roehub/roehub.env ps
docker compose -f /opt/roehub/docker-compose.market_data.yml --env-file /etc/roehub/roehub.env logs -f --tail=200 market-data-scheduler
docker compose -f /opt/roehub/docker-compose.market_data.yml --env-file /etc/roehub/roehub.env logs -f --tail=200 market-data-ws-worker
docker compose -f /opt/roehub/docker-compose.market_data.yml --env-file /etc/roehub/roehub.env down
```

## 4. Проверка метрик

```bash
curl -fsS http://localhost:9201/metrics | head
curl -fsS http://localhost:9202/metrics | head
```

Ключевые scheduler-метрики для startup backfill:

```bash
curl -fsS http://localhost:9202/metrics | rg "scheduler_tasks_(planned|enqueued)_total|scheduler_startup_scan_instruments_total"
```

## 5. Проверка данных (SQL, операторские проверки)

Последняя canonical свеча на инструмент:

```sql
SELECT instrument_key, max(ts_open) AS last_ts
FROM market_data.canonical_candles_1m
GROUP BY instrument_key
ORDER BY last_ts DESC
LIMIT 50;
```

Lag по инструментам:

```sql
SELECT
  instrument_key,
  dateDiff('second', max(ts_open), now()) AS lag_s
FROM market_data.canonical_candles_1m
GROUP BY instrument_key
ORDER BY lag_s DESC
LIMIT 50;
```

Контроль дубликатов минут в canonical:

```sql
SELECT
  instrument_key,
  count() - uniqExact(toStartOfMinute(ts_open)) AS dup_minutes
FROM market_data.canonical_candles_1m
WHERE ts_open >= now() - INTERVAL 1 DAY
GROUP BY instrument_key
ORDER BY dup_minutes DESC
LIMIT 50;
```

## 6. First 10 Minutes Checklist

1. В логах scheduler есть `startup_scan: instruments_scanned=...` и причины задач `historical_backfill` при необходимости.
2. `scheduler_tasks_planned_total{reason="historical_backfill"}` и `scheduler_tasks_enqueued_total{reason="historical_backfill"}` растут на старте, если canonical начинается позже earliest boundary.
3. В логах worker нет серии ошибок вставки или session-конфликтов.
4. `/metrics` для портов `9201` и `9202` отвечают.
5. В canonical уменьшается lag, а исторические диапазоны постепенно заполняются.

## 7. Траблшутинг

`Attempt to execute concurrent queries within the same session`:
- убедиться, что используется версия с `ThreadLocalClickHouseConnectGateway`.

`DataError ... Unable to create Python array`:
- проверить, что в payload не попадают `None` в non-nullable numeric колонки raw.

Виден только хвост, истории нет:
- проверить `market_data.markets[*].rest.earliest_available_ts_utc` в конфиге;
- проверить метрики `scheduler_tasks_planned_total{reason="historical_backfill"}` и `scheduler_tasks_enqueued_total{reason="historical_backfill"}`;
- проверить логи startup scan (`planned_task[...]`).
