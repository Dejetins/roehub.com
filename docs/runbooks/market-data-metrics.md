# Market Data Metrics

Документ фиксирует основные Prometheus-метрики для:
- `market-data-ws-worker` (`:9201/metrics`)
- `market-data-scheduler` (`:9202/metrics`)

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

Worker endpoint:

```bash
curl -fsS http://localhost:9201/metrics | rg "ws_|insert_|rest_fill_"
```

Scheduler endpoint:

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

## Интерпретация

- Рост `ws_reconnects_total` при стабильном `ws_messages_total` может указывать на сетевые проблемы.
- Рост `insert_errors_total` или `rest_fill_errors_total` требует проверки CH и REST лимитов.
- Для SLO ориентируйтесь на p95 из `ws_closed_to_insert_done_seconds`.
- Рост `scheduler_job_errors_total{job="startup_scan"}` блокирует ранний historical backfill.
- При старте, если canonical начинается позже earliest boundary, должны расти:
  - `scheduler_tasks_planned_total{reason="historical_backfill"}`
  - `scheduler_tasks_enqueued_total{reason="historical_backfill"}`.
