# Market Data Metrics

Документ фиксирует основные Prometheus-метрики для:

- `market-data-ws-worker` (`127.0.0.1:9201/metrics`)
- `market-data-scheduler` (`127.0.0.1:9202/metrics`)

Подробный справочник по каждой метрике:

- `docs/runbooks/market-data-metrics-reference-ru.md`

## Scrape модель (native)

В текущем production scrape идет на loopback цели `Mac Studio`:

- job `market-data-ws-worker` -> `http://127.0.0.1:9201/metrics`
- job `market-data-scheduler` -> `http://127.0.0.1:9202/metrics`

Source of truth:

- `infra/macos/prometheus/prometheus.prod.yml`

Параллельно в production мониторятся infra exporter jobs (`node-exporter`, `postgres-exporter`, `redis-exporter`, `clickhouse-exporter`) для диагностики причин деградации pipeline.

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

Ожидаемые `job` labels:

- `sync_whitelist`
- `enrich`
- `startup_scan`
- `rest_insurance_catchup`

## Quick checks

```bash
curl -fsS http://127.0.0.1:9201/metrics | rg "ws_|insert_|rest_fill_"
curl -fsS http://127.0.0.1:9202/metrics | rg "scheduler_(job_|tasks_|startup_scan_)"
curl -fsS http://127.0.0.1:9202/metrics | rg "scheduler_job_errors_total"
curl -fsS http://127.0.0.1:9201/metrics | rg "ws_closed_to_insert_(start|done)_seconds"
curl -fsS http://127.0.0.1:9201/metrics | rg "redis_publish_(total|errors_total|duplicates_total|duration_seconds)"
```

Проверка target health в Prometheus:

```bash
curl -fsS http://127.0.0.1:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="market-data-ws-worker" or .labels.job=="market-data-scheduler") | {job: .labels.job, health: .health, scrapeUrl: .scrapeUrl}'
```

## PromQL quick checks

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

- рост `ws_reconnects_total` при стабильном `ws_messages_total` часто указывает на проблемы с внешним WS;
- рост `insert_errors_total` или `rest_fill_errors_total` требует проверки ClickHouse и REST лимитов;
- рост `redis_publish_errors_total` при стабильном `insert_rows_total` означает проблему канала live feed, но не остановку ingestion;
- для SLO ориентируйтесь на p95 из `ws_closed_to_insert_done_seconds`;
- рост `scheduler_job_errors_total{job="startup_scan"}` блокирует ранний historical backfill.
