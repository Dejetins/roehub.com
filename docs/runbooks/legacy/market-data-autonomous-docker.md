# Market Data Docker Runbook

Статус: legacy/local-only path.

Production path для market-data на `Mac Studio` теперь native:

- `docs/runbooks/mac-studio-native-backend-operations.md`

Этот документ оставлен только для локального автономного Docker запуска в dev/testing сценариях.

## Local standalone compose

Используется файл:

- `infra/docker/docker-compose.market_data.yml`

Запуск:

```bash
docker compose -f infra/docker/docker-compose.market_data.yml up -d --build
docker compose -f infra/docker/docker-compose.market_data.yml ps
```

Остановка:

```bash
docker compose -f infra/docker/docker-compose.market_data.yml down
```

## Local checks

```bash
curl -fsS http://127.0.0.1:9201/metrics | head
curl -fsS http://127.0.0.1:9202/metrics | head
```

Опциональные SQL проверки:

```sql
SELECT instrument_key, max(ts_open) AS last_ts
FROM market_data.canonical_candles_1m
GROUP BY instrument_key
ORDER BY last_ts DESC
LIMIT 50;
```

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

## Важно

- Не используйте этот ранбук для production operations.
- Для production market-data restart/smoke используйте native scripts из `scripts/macos/`.
