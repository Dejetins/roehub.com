# Market Data Docker Runbook

Runbook для `market-data-ws-worker` и `market-data-scheduler`.

## 1. Рекомендуемая прод-модель

Прод использует **единый стек** `infra/docker/docker-compose.yml`:

- `market-data-ws-worker`
- `market-data-scheduler`
- `clickhouse` (общий сервис)
- `prometheus` скрапит market-data по DNS внутри compose-сети:
  - `market-data-ws-worker:9201`
  - `market-data-scheduler:9202`

`host.docker.internal` для market-data метрик в Linux больше не нужен.

## 2. Деплой/перезапуск в фоне (prod host)

```bash
export COMPOSE_PROJECT_NAME=roehub
export MARKET_DATA_BUILD_CONTEXT=/opt/roehub/market-data-src
export MARKET_DATA_DOCKERFILE=infra/docker/Dockerfile.market_data

docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env up -d --build --remove-orphans
docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env ps
```

Логи:

```bash
docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env logs -f --tail=200 market-data-ws-worker
docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env logs -f --tail=200 market-data-scheduler
```

Остановка только market-data сервисов:

```bash
docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env stop market-data-ws-worker market-data-scheduler
```

## 3. Проверка метрик и scrape

Проверка scrape из контейнера Prometheus (DNS внутри сети):

```bash
docker exec -it prometheus wget -T 2 -qO- http://market-data-ws-worker:9201/metrics | head
docker exec -it prometheus wget -T 2 -qO- http://market-data-scheduler:9202/metrics | head
```

Проверка startup historical backfill:

```bash
docker exec -it prometheus wget -T 2 -qO- http://market-data-scheduler:9202/metrics | rg 'scheduler_tasks_(planned|enqueued)_total.*historical_backfill'
docker exec -it prometheus wget -T 2 -qO- http://market-data-scheduler:9202/metrics | rg 'scheduler_job_(runs|errors)_total.*startup_scan|rest_insurance_catchup'
```

## 4. SQL-проверки (выполняет оператор)

Последний canonical close:

```sql
SELECT instrument_key, max(ts_open) AS last_ts
FROM market_data.canonical_candles_1m
GROUP BY instrument_key
ORDER BY last_ts DESC
LIMIT 50;
```

Lag:

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

## 5. Standalone compose (dev/local only)

`infra/docker/docker-compose.market_data.yml` оставлен для локального автономного запуска.
Он поднимает отдельный `market-data-clickhouse` на портах:

- `127.0.0.1:18123 -> 8123`
- `127.0.0.1:19000 -> 9000`

Запуск:

```bash
docker compose -f infra/docker/docker-compose.market_data.yml up -d --build
docker compose -f infra/docker/docker-compose.market_data.yml ps
```

В проде этот файл не является основным путем деплоя.

В standalone режиме метрики доступны с хоста:

```bash
curl -fsS http://localhost:9201/metrics | head
curl -fsS http://localhost:9202/metrics | head
```

## 6. First 10 Minutes Checklist

1. `docker compose ... ps` показывает `market-data-ws-worker` и `market-data-scheduler` в `running`.
2. `docker exec prometheus wget ... market-data-ws-worker:9201/metrics` и `...market-data-scheduler:9202/metrics` возвращают ответ.
3. В логах scheduler есть `startup_scan: instruments_scanned=...` и задачи `historical_backfill` при необходимости.
4. `scheduler_tasks_planned_total{reason="historical_backfill"}` и `scheduler_tasks_enqueued_total{reason="historical_backfill"}` растут, если истории не хватает.
5. В логах worker нет серийных `insert_errors_total`/`rest_fill_errors_total`.


## 7. Troubleshooting: Bybit WS Stalled

Симптом:
- Bybit инструменты догоняются через REST после рестарта, но live-обновления не приходят.

Проверка live WS по рынкам (последние 10 минут):

```sql
SELECT
  market_id,
  count() AS ws_rows_10m,
  max(ts_open) AS last_ws_ts
FROM market_data.canonical_candles_1m
WHERE source = 'ws'
  AND ts_open >= now() - INTERVAL 10 MINUTE
  AND market_id IN (1,2,3)
GROUP BY market_id
ORDER BY market_id;
```

Если `market_id=3` отсутствует, проверьте логи на subscribe ACK ошибки:

```bash
docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env logs --tail=2000 market-data-ws-worker | rg "bybit ws stream failed|ret_msg|subscribe|args size"
```

Типичный корень проблемы:
- Bybit V5 вернул `{"success":false,"ret_msg":"args size >10","op":"subscribe"}`.

## 8. Restart WS + Scheduler

```bash
docker compose -f /opt/roehub/docker-compose.yml --env-file /etc/roehub/roehub.env restart market-data-scheduler market-data-ws-worker
```
