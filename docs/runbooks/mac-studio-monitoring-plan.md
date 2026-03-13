# Mac Studio Monitoring Plan

Реализованный план мониторинга для текущей production topology:

- `Mac Studio` = приватный backend/data/compute host
- `VPS` = публичный edge и TLS termination
- `Prometheus`, `Grafana`, `Blackbox` уже живут на `Mac Studio`
- backend runtime по-прежнему принадлежит пользователю `daniildegtyarev`

Документ фиксирует уже внедренную repo-managed схему мониторинга: что именно собирается, какими
компонентами, какие файлы отвечают за runtime, и как проверить rollout без ручных Grafana click-ops.

## Что собирается

### Host metrics

Host-уровень закрыт через `node_exporter` на macOS host service.

- target для scrape из `Prometheus` контейнера: `host.lima.internal:9100`
- установка и user-level autostart: `infra/monitoring/host-macos/install-node-exporter.sh`
- expected model: `brew services start node_exporter` под `daniildegtyarev`

Собираются:

- CPU / load / uptime
- RAM / filesystem / disk IO
- network traffic / errors

Сознательно не собираются на этом этапе:

- hardware temperature
- fan speed
- power / energy metrics

Для них потребуется отдельный macOS-specific exporter или textfile collector с кастомным script.

### Container metrics

Container/runtime-уровень закрыт через `cadvisor` в backend compose stack.

Файл:

- `infra/docker/docker-compose.backend.yml`

Собираются:

- CPU / RAM / filesystem / network per container
- restart indicators через `changes(container_start_time_seconds[...])`
- `container_last_seen` для lifecycle visibility

### Service metrics

В compose добавлены сервисные exporters:

- `postgres_exporter`
- `redis_exporter`
- `clickhouse_exporter`

`postgres_exporter` и `redis_exporter` используют стандартные upstream images.

`clickhouse_exporter` реализован как repo-managed Python service в `ROEHUB_APP_IMAGE`:

- module: `apps.monitoring.clickhouse_exporter`
- scrape endpoint: `clickhouse_exporter:9116/metrics`
- reason: избежать зависимости от stale/неочевидного third-party image на ARM runtime

Он публикует:

- `clickhouse_exporter_scrape_success`
- `clickhouse_exporter_scrape_duration_seconds`
- `clickhouse_uptime_seconds`
- `clickhouse_system_metric_value{metric=...}`
- `clickhouse_system_event_total{event=...}`

### API health and API metrics

API теперь публикует:

- `GET /health` -> `200 {"status": "ok"}`
- `GET /metrics` -> Prometheus exposition

HTTP instrumentation добавлена в `apps/api/monitoring.py`.

Собираются:

- `http_requests_total`
- `http_request_duration_seconds`
- `http_requests_in_progress`

`/health` и `/metrics` исключены из request counters/histograms, чтобы monitoring traffic не загрязнял
основную API телеметрию.

## Repo-managed monitoring assets

### Compose

Файл:

- `infra/docker/docker-compose.backend.yml`

Добавлены сервисы:

- `cadvisor`
- `postgres_exporter`
- `redis_exporter`
- `clickhouse_exporter`

И provisioning mounts:

- Grafana dashboards + datasources
- Prometheus alert rules

Persistent volumes `prom_data` и `grafana_data` не меняются и не удаляются.

### Prometheus

Файлы:

- `infra/monitoring/monitoring/prometheus/prometheus.yml`
- `infra/monitoring/monitoring/prometheus/rules/mac-studio-monitoring.rules.yml`

Scrape jobs:

- `api`
- `blackbox`
- `blackbox_http`
- `blackbox_tcp`
- `cadvisor`
- `clickhouse_exporter`
- `market-data-scheduler`
- `market-data-ws-worker`
- `node_exporter`
- `postgres_exporter`
- `prometheus`
- `redis_exporter`

Blackbox probes:

- `http://api:8000/health`
- `http://clickhouse:8123/ping`
- `http://grafana:3000/api/health`
- `http://prometheus:9090/-/healthy`
- `postgres:5432`
- `redis:6379`
- `clickhouse:9000`

Alert rules покрывают:

- exporter down / service down
- API `/health` down
- host high CPU / host low disk free
- container high CPU / high memory / recent restart
- market-data worker and scheduler error growth

### Grafana provisioning

Файлы:

- `infra/monitoring/monitoring/grafana/provisioning/datasources/roehub-prometheus.yml`
- `infra/monitoring/monitoring/grafana/provisioning/dashboards/roehub-dashboards.yml`
- `infra/monitoring/monitoring/grafana/dashboards/roehub/01-platform-overview.json`
- `infra/monitoring/monitoring/grafana/dashboards/roehub/02-mac-studio-host.json`
- `infra/monitoring/monitoring/grafana/dashboards/roehub/03-containers.json`
- `infra/monitoring/monitoring/grafana/dashboards/roehub/04-datastores.json`
- `infra/monitoring/monitoring/grafana/dashboards/roehub/05-api-market-data.json`

Provisioned dashboards:

1. `Platform Overview`
2. `Mac Studio Host`
3. `Containers`
4. `Datastores`
5. `API and Market Data`

Все dashboards используют datasource UID `roehub-prometheus` и загружаются из репозитория.

## Rollout sequence

### 1. Host node_exporter

На `Mac Studio`:

```bash
cd /opt/roehub
bash infra/monitoring/host-macos/install-node-exporter.sh
brew services list | grep node_exporter
curl -fsS http://127.0.0.1:9100/metrics | head
```

### 2. Monitoring services in backend compose

```bash
export ROEHUB_ENV_FILE=/Users/daniildegtyarev/.config/roehub/roehub.env
cd /opt/roehub

docker compose -f /opt/roehub/docker-compose.backend.yml --env-file "$ROEHUB_ENV_FILE" up -d \
  prometheus grafana blackbox cadvisor postgres_exporter redis_exporter clickhouse_exporter api \
  market-data-ws-worker market-data-scheduler
```

### 3. Prometheus target validation

```bash
curl -fsS http://127.0.0.1:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health, scrapeUrl: .scrapeUrl}'
```

Ожидаемые healthy targets:

- `api`
- `blackbox`
- `cadvisor`
- `clickhouse_exporter`
- `market-data-scheduler`
- `market-data-ws-worker`
- `node_exporter`
- `postgres_exporter`
- `prometheus`
- `redis_exporter`

### 4. Probe and metric validation

```bash
curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up'
curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=probe_success'
curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=clickhouse_exporter_scrape_success'
curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=http_requests_total'
```

### 5. Grafana validation

```bash
curl -fsS http://127.0.0.1:3000/api/health
```

Дальше в UI Grafana проверить папку `Roehub Monitoring` и наличие пяти provisioned dashboards.

## Minimum done state

Реализация считается в рабочем состоянии, когда одновременно выполняется всё ниже:

- `Prometheus` успешно scrapes `host.lima.internal:9100`
- `Prometheus` успешно scrapes `cadvisor`
- `Prometheus` успешно scrapes `postgres_exporter`, `redis_exporter`, `clickhouse_exporter`
- `Blackbox` probes успешны для `API /health`, `Prometheus`, `Grafana`, `Postgres`, `Redis`, `ClickHouse`
- `Grafana` автоматически поднимает dashboards из репозитория
- `API /health` используется для monitoring и больше не опирается на `401` из auth endpoints

## Follow-up, который сознательно отложен

- host hardware temperature / fan / power metrics
- Grafana unified alerting rules внутри Grafana; сейчас источник истины только Prometheus alert rules
- deploy metadata panels (git SHA, image tag) как отдельные exported metrics

## Связанные документы

- `docs/runbooks/mac-studio-backend-operations.md`
- `docs/runbooks/market-data-metrics-reference-ru.md`
- `docs/runbooks/prod-migration-linux-to-mac-studio.md`
- `infra/docker/docker-compose.backend.yml`
- `infra/monitoring/monitoring/prometheus/prometheus.yml`
