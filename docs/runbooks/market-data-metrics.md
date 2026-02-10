# Market Data Metrics

Документ фиксирует основные Prometheus-метрики для:
- `market-data-ws-worker` (`:9201/metrics`)
- `market-data-scheduler` (`:9202/metrics`)

Подробный русскоязычный справочник по каждой метрике:
- `docs/runbooks/market-data-metrics-reference-ru.md`

## Scrape модель в Docker

В production scrape идет **внутри docker-сети** по DNS-именам сервисов:

- `market-data-ws-worker:9201`
- `market-data-scheduler:9202`

Prometheus job-ы настроены в `infra/monitoring/monitoring/prometheus/prometheus.yml`.
Для Linux не используется `host.docker.internal` и не используется scrape через host loopback.

## Worker Metrics (`market-data-ws-worker`)

WebSocket runtime:
- `ws_connected`
- `ws_reconnects_total`
- `ws_messages_total`
- `ws_errors_total`
- `ignored_non_closed_total`

Raw inserts:
- `insert_rows_total`
- `insert_batches_total`
- `insert_duration_seconds`
- `insert_errors_total`

SLO latency:
- `ws_closed_to_insert_start_seconds`
- `ws_closed_to_insert_done_seconds`

Gap / ordering:
- `ws_out_of_order_total`
- `ws_duplicates_total`

REST fill queue:
- `rest_fill_tasks_total`
- `rest_fill_active`
- `rest_fill_errors_total`
- `rest_fill_duration_seconds`

Redis live feed publish:
- `redis_publish_total`
- `redis_publish_errors_total`
- `redis_publish_duplicates_total`
- `redis_publish_duration_seconds`

## Scheduler Metrics (`market-data-scheduler`)

- `scheduler_job_runs_total{job="..."}`
- `scheduler_job_errors_total{job="..."}`
- `scheduler_job_duration_seconds{job="..."}`
- `scheduler_tasks_planned_total{reason="..."}`
- `scheduler_tasks_enqueued_total{reason="..."}`
- `scheduler_startup_scan_instruments_total`

Ожидаемые job labels:
- `sync_whitelist`
- `enrich`
- `startup_scan`
- `rest_insurance_catchup`

## Quick Checks

Standalone/локальный endpoint (если опубликованы порты на хост):

```bash
curl -fsS http://localhost:9201/metrics | rg "ws_|insert_|rest_fill_"
```

Standalone/локальный scheduler endpoint:

```bash
curl -fsS http://localhost:9202/metrics | rg "scheduler_(job_|tasks_|startup_scan_)"
```

Ошибки job’ов:

```bash
curl -fsS http://localhost:9202/metrics | rg "scheduler_job_errors_total"
```

SLO latency buckets:

```bash
curl -fsS http://localhost:9201/metrics | rg "ws_closed_to_insert_(start|done)_seconds"
```

Redis live feed metrics:

```bash
curl -fsS http://localhost:9201/metrics | rg "redis_publish_(total|errors_total|duplicates_total|duration_seconds)"
```

Проверка scrape из контейнера Prometheus (production recommended):

```bash
docker exec -it prometheus wget -T 2 -qO- http://market-data-ws-worker:9201/metrics | head
docker exec -it prometheus wget -T 2 -qO- http://market-data-scheduler:9202/metrics | head
```

PromQL quick checks (Redis live feed):

```promql
increase(redis_publish_errors_total[15m])
```

```promql
histogram_quantile(
  0.95,
  sum(rate(redis_publish_duration_seconds_bucket[5m])) by (le)
)
```

## Интерпретация

- Рост `ws_reconnects_total` при стабильном `ws_messages_total` может указывать на сетевые проблемы.
- Рост `insert_errors_total` или `rest_fill_errors_total` требует проверки CH и REST лимитов.
- Рост `redis_publish_errors_total` при стабильном `insert_rows_total` означает проблему канала live feed, но не остановку ingestion.
- Рост `redis_publish_duplicates_total` обычно указывает на повтор/опоздание WS candle по тому же `ts_open`.
- Для SLO ориентируйтесь на p95 из `ws_closed_to_insert_done_seconds`.
- Рост `scheduler_job_errors_total{job="startup_scan"}` блокирует ранний historical backfill.
- При старте, если canonical начинается позже earliest boundary, должны расти:
  - `scheduler_tasks_planned_total{reason="historical_backfill"}`
  - `scheduler_tasks_enqueued_total{reason="historical_backfill"}`.
