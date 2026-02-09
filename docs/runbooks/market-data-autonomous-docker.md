# Market Data Autonomous Docker Runbook

Этот runbook описывает автономный запуск EPIC 3 (ClickHouse + WS worker + scheduler)
через `docker-compose.market_data.yml`.

## 1. Подготовка

```bash
cd /path/to/roehub.com
```

Проверить, что порты свободны:
- `8123` (ClickHouse HTTP)
- `9000` (ClickHouse native)
- `9201` (worker /metrics)
- `9202` (scheduler /metrics)

## 2. Запуск compose

```bash
docker compose -f infra/docker/docker-compose.market_data.yml up -d --build
```

Логи:

```bash
docker compose -f infra/docker/docker-compose.market_data.yml logs -f clickhouse
docker compose -f infra/docker/docker-compose.market_data.yml logs -f market-data-ws-worker
docker compose -f infra/docker/docker-compose.market_data.yml logs -f market-data-scheduler
```

Остановка:

```bash
docker compose -f infra/docker/docker-compose.market_data.yml down
```

Полная очистка volume ClickHouse:

```bash
docker compose -f infra/docker/docker-compose.market_data.yml down -v
```

## 3. Инициализация схемы ClickHouse

Если таблицы ещё не созданы, выполните DDL из `docs/architecture/market_data/market_data_ddl.sql`
в вашем CH-клиенте/IDE перед запуском worker/scheduler в production.

Пример проверки существования таблиц:

```sql
SHOW TABLES FROM market_data;
```

## 4. Проверка health и метрик

Worker:

```bash
curl -fsS http://localhost:9201/metrics | head
```

Scheduler:

```bash
curl -fsS http://localhost:9202/metrics | head
```

## 5. Проверка данных (SQL)

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

Дубликаты минут в canonical (контроль качества):

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

1. В логах worker нет непрерывных ошибок `insert`/`session`.
2. В логах scheduler есть успешные `sync_whitelist`, `enrich`, `startup_scan`.
3. `/metrics` endpoints отвечают на `9201` и `9202`.
4. В `raw_*_klines_1m` и `canonical_candles_1m` растут строки.
5. `lag_s` по ключевым инструментам стабилизируется и не накапливается.

## 7. Траблшутинг

`Attempt to execute concurrent queries within the same session`:
- убедитесь, что используется текущая версия с `ThreadLocalClickHouseConnectGateway`.

`DataError ... Unable to create Python array` при вставке:
- проверьте, что в payload нет `None` для non-nullable numeric колонок raw-таблиц.

История не догружается, виден только свежий хвост:
- проверьте, что в config задан `market_data.markets[*].rest.earliest_available_ts_utc`;
- убедитесь, что startup scan scheduler отработал и enqueue’ит `historical_backfill`.
