# Mac Studio Monitoring Plan

Статус: актуальный production monitoring для native backend runtime на `Mac Studio`.

Документ фиксирует target production monitoring baseline без Docker/Colima runtime:

- `Mac Studio` держит backend/data/monitoring;
- `VPS` остается публичным edge;
- monitoring собирается локально с `127.0.0.1` целей;
- source of truth для scrape-конфигурации: `infra/macos/prometheus/prometheus.prod.yml`.

## Что реально мониторится сейчас

## Prometheus jobs

Текущий `prometheus.prod.yml` содержит jobs:

- `prometheus` (`127.0.0.1:9090`)
- `node-exporter` (`127.0.0.1:9100`)
- `postgres-exporter` (`127.0.0.1:9187`)
- `redis-exporter` (`127.0.0.1:9121`)
- `clickhouse-exporter` (`127.0.0.1:9116`)
- `blackbox-http` (через `127.0.0.1:9115/probe`)
- `blackbox-tcp` (через `127.0.0.1:9115/probe`)
- `market-data-ws-worker` (`127.0.0.1:9201/metrics`)
- `market-data-scheduler` (`127.0.0.1:9202/metrics`)

HTTP probes:

- `http://127.0.0.1:3000/api/health` (Grafana)
- `http://127.0.0.1:8123/ping` (ClickHouse HTTP)
- `http://127.0.0.1:8000/openapi.json` (API process availability)

TCP probes:

- `127.0.0.1:5432` (Postgres)
- `127.0.0.1:9000` (ClickHouse native)

## Service ownership

- `prometheus`, `grafana`, `postgresql@16`, `redis` — `brew services`
- `node_exporter` — `brew services`
- `blackbox-exporter`, `postgres-exporter`, `redis-exporter`, `clickhouse-exporter`, `clickhouse`, `api`, `market-data-*` — user `launchd` services

## Install and bootstrap commands

```bash
bash scripts/macos/install_native_backend_prereqs.sh
bash scripts/macos/bootstrap_native_prod.sh
brew services start node_exporter
bash scripts/macos/reload_launchd_services.sh prod
```

Файлы, которые ставят production monitoring baseline:

- `infra/macos/prometheus/prometheus.prod.yml`
- `infra/macos/launchd/com.roehub.blackbox-exporter.plist`
- `infra/macos/launchd/com.roehub.postgres-exporter.plist`
- `infra/macos/launchd/com.roehub.redis-exporter.plist`
- `infra/macos/launchd/com.roehub.clickhouse-exporter.plist`
- `scripts/macos/install_native_backend_prereqs.sh`
- `scripts/macos/bootstrap_native_prod.sh`
- `scripts/macos/reload_launchd_services.sh`

## Metric coverage

В baseline гарантированно покрыты:

- availability/liveness ключевых endpoints через `probe_success`
- host базовые метрики (`node_*`)
- PostgreSQL метрики (`pg_*`)
- Redis метрики (`redis_*`)
- ClickHouse exporter метрики (`clickhouse_*`)
- market data pipeline metrics (`ws_*`, `insert_*`, `rest_fill_*`, `scheduler_*`, `redis_publish_*`)
- API auth path health (`http_requests_total`, `http_request_duration_seconds`)
- Prometheus self metrics

## Ключевые метрики по сервисам

- monitoring stack: `up{job=...}`, `probe_success`, `probe_http_status_code`, `prometheus_tsdb_head_series`
- host: `node_cpu_seconds_total`, `node_load1`, `node_memory_free_bytes`, `node_filesystem_avail_bytes`
- PostgreSQL: `pg_up`, `pg_exporter_last_scrape_error`, `pg_stat_database_xact_commit`, `pg_stat_database_numbackends`
- Redis: `redis_up`, `redis_exporter_last_scrape_error`, `redis_commands_processed_total`, `redis_memory_used_bytes`, `redis_mem_fragmentation_ratio`
- ClickHouse exporter: `clickhouse_exporter_scrape_success`, `clickhouse_uptime_seconds`, `clickhouse_system_event_total{event="InsertedRows"}`
- market-data worker: `ws_connected`, `ws_messages_total`, `ws_errors_total`, `insert_errors_total`, `ws_closed_to_insert_done_seconds`
- market-data scheduler: `scheduler_job_errors_total`, `scheduler_job_duration_seconds`, `scheduler_tasks_enqueued_total`, `scheduler_rest_catchup_gap_rows_written_total`
- auth/login API (через `http://127.0.0.1:8000/metrics`): `http_requests_total{path="/auth/telegram/login",status_code=~"5.."}`, `http_request_duration_seconds_count{path="/auth/telegram/login"}`

## Вне scope

Следующие docker-era элементы не используются и не считаются частью target state:

- `cadvisor`
- scrape по compose DNS именам

## Runtime checks

## 1) Проверка jobs/targets

```bash
curl -fsS http://127.0.0.1:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health, scrapeUrl: .scrapeUrl}'
```

Ожидаемо: все текущие jobs в `health: "up"`.

## 2) Проверка ключевых probe/availability метрик

```bash
curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up'
curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=probe_success'
curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up{job=~"node-exporter|postgres-exporter|redis-exporter|clickhouse-exporter"}'
curl -fsS http://127.0.0.1:8000/metrics | rg 'http_requests_total\{method="POST",path="/auth/telegram/login",status_code="500"\}'
```

## 3) Проверка exporter endpoint'ов

```bash
curl -fsS http://127.0.0.1:9100/metrics | rg '^node_'
curl -fsS http://127.0.0.1:9187/metrics | rg '^pg_'
curl -fsS http://127.0.0.1:9121/metrics | rg '^redis_'
curl -fsS http://127.0.0.1:9116/metrics | rg '^clickhouse_'
```

## 4) Проверка market-data метрик

```bash
curl -fsS http://127.0.0.1:9201/metrics | rg 'ws_|insert_|rest_fill_|redis_publish_'
curl -fsS http://127.0.0.1:9202/metrics | rg 'scheduler_(job_|tasks_|startup_scan_|rest_catchup_)'
```

## 5) Проверка сервисов хоста

```bash
brew services list
launchctl list | grep -E 'com.roehub.(blackbox-exporter|postgres-exporter|redis-exporter|clickhouse-exporter|clickhouse|api|market-data)'
curl -I http://127.0.0.1:3000
curl -I http://127.0.0.1:9090
curl -I http://127.0.0.1:9100
curl -I http://127.0.0.1:9115
curl -I http://127.0.0.1:9116
curl -I http://127.0.0.1:9121
curl -I http://127.0.0.1:9187
curl -i http://127.0.0.1:8000/auth/current-user
```

## Minimum done state

Monitoring считается в рабочем состоянии, когда одновременно выполняется все ниже:

- Prometheus target list показывает текущие jobs в `up`
- `probe_success` не сигнализирует массовых падений probes
- `node-exporter`, `postgres-exporter`, `redis-exporter`, `clickhouse-exporter` отдают метрики
- `market-data-ws-worker` и `market-data-scheduler` метрики доступны
- `Grafana` отвечает (`302` на `/` или `200` на `/api/health`)
- API отвечает (`401` на `/auth/current-user` без cookie)

## Связанные документы

- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/runbooks/market-data-metrics.md`
- `docs/runbooks/market-data-metrics-reference-ru.md`
- `docs/runbooks/prod-dashboard-metrics-reference-ru.md`
- `infra/macos/prometheus/prometheus.prod.yml`
- `infra/macos/blackbox/blackbox.yml`
