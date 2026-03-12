# Mac Studio Monitoring Plan

План внедрения системного и контейнерного мониторинга для текущей production topology:

- `Mac Studio` = backend/data/compute host
- `VPS` = public edge
- `Prometheus`, `Grafana`, `Blackbox` уже живут на `Mac Studio`

Документ фиксирует, что именно нужно собирать, какими компонентами, какие изменения внести в runtime,
и какие Grafana dashboards построить поверх этих метрик.

## Цели

Нужно видеть в `Prometheus` и `Grafana`:

- состояние всех backend сервисов и контейнеров;
- uptime сервисов и контейнеров;
- ошибки и деградации по API, market-data, storage, monitoring;
- потребление ресурсов контейнерами: CPU, RAM, filesystem, network;
- потребление ресурсов `Mac Studio` как хоста: CPU, RAM, disk, filesystem, network;
- health-checks внешне и внутренне доступных endpoints.

## Что уже есть

Сейчас уже работают:

- `Prometheus`
- `Grafana`
- `Blackbox exporter`
- metrics от:
  - `market-data-ws-worker`
  - `market-data-scheduler`

Сейчас уже доступны tailnet-only endpoints:

- `Grafana`: `https://macstudio-daniil.tail0ebbbc.ts.net:3443/`
- `Postgres`: `macstudio-daniil.tail0ebbbc.ts.net:15432`
- `ClickHouse HTTP`: `macstudio-daniil.tail0ebbbc.ts.net:18123`
- `ClickHouse native`: `macstudio-daniil.tail0ebbbc.ts.net:19000`

## Что обязательно добавить

### 1. Метрики хоста Mac Studio

Нужен отдельный host exporter на самом `Mac Studio`, не в Docker.

Рекомендуемый вариант:

- `node_exporter` на macOS как host service

Что он даст:

- CPU usage / load / context switches
- RAM / swap
- filesystem usage
- disk IO counters
- network traffic / packets / errors
- uptime
- process/file descriptor level базовые системные метрики

Что он не даст хорошо на macOS без доп. костылей:

- hardware temperature / fan speed / power metrics

Если позже это понадобится, можно добавить отдельно:

- textfile collector + кастомный exporter script
- или отдельный lightweight sensor exporter

### 2. Метрики контейнеров и Docker runtime

Рекомендуемый вариант:

- `cAdvisor` как контейнер в backend compose stack

Что он даст:

- per-container CPU
- per-container memory
- per-container filesystem usage
- per-container network RX/TX
- container start time / last seen
- container-level lifecycle observability

Это обязательно, если вы хотите видеть, какой контейнер ест CPU/RAM/IO.

### 3. Метрики баз данных и инфраструктурных сервисов

Рекомендуемые exporters:

- `postgres_exporter`
- `redis_exporter`
- `clickhouse_exporter`

Что это даст:

- `Postgres`: connections, transactions, locks, table/index stats, bgwriter, database size
- `Redis`: memory, clients, ops/sec, evictions, keyspace
- `ClickHouse`: query/load/merge metrics, parts, background tasks, storage state, replication-free health signals

### 4. Health checks сервисов

Текущий `Blackbox exporter` нужно расширить.

Нужно проверять:

- `Grafana` HTTP health
- `Prometheus` HTTP health
- `ClickHouse` HTTP `/ping`
- `Postgres` TCP connect
- `ClickHouse` native TCP connect
- `Redis` TCP connect
- `API` explicit `/health` endpoint

Важно:

- для `API` желательно добавить нормальный `/health` endpoint, который отдает `200`
- не использовать `auth/current-user` как health-check, потому что `401` — это auth contract, а не health contract

## Что рекомендуется добавить сверх минимума

### 1. Метрики самого API

Если их еще нет, добавить Prometheus instrumentation в `api`:

- request count
- request duration histogram
- error count by status code
- in-flight requests

Минимальный набор:

- `http_requests_total`
- `http_request_duration_seconds`
- `http_requests_in_progress`

### 2. Alert rules

Нужны хотя бы базовые rules в Prometheus:

- container down
- high container CPU
- high container memory
- high host CPU
- low host disk free
- API health down
- Postgres TCP down
- ClickHouse HTTP/TCP down
- Redis TCP down
- scheduler job errors grow
- worker insert/publish errors grow

### 3. Uptime dashboard

Отдельный overview dashboard по состоянию платформы:

- host up/down
- exporter up/down
- container running/down
- last deploy time / image tag (опционально позже)

## Целевой состав monitoring stack

На `Mac Studio` в backend stack:

- `prometheus`
- `grafana`
- `blackbox`
- `cadvisor`
- `postgres_exporter`
- `redis_exporter`
- `clickhouse_exporter`

На host `Mac Studio` вне Docker:

- `node_exporter`

В приложении:

- `api` metrics endpoint
- existing market-data metrics endpoints

## План внедрения по шагам

### Phase A — Host metrics

Сделать:

- поставить `node_exporter` на `Mac Studio`
- поднять его как user-level startup service под `daniildegtyarev`
- открыть scrape из `Prometheus` контейнера на host target

Нужно решить target address для scrape из Colima VM к macOS host:

- рекомендованный вариант: `host.lima.internal:9100`

### Phase B — Container metrics

Сделать:

- добавить `cadvisor` в `infra/docker/docker-compose.backend.yml`
- добавить scrape job в `prometheus.yml`

Получим:

- container CPU/RAM/FS/network
- статус и lifecycle контейнеров

### Phase C — Service exporters

Сделать:

- добавить `postgres_exporter`
- добавить `redis_exporter`
- добавить `clickhouse_exporter`
- добавить scrape jobs в `prometheus.yml`

### Phase D — Health checks

Сделать:

- расширить `blackbox` targets
- добавить `API /health`
- завести HTTP/TCP probes для всех ключевых сервисов

### Phase E — Dashboards

Сделать dashboards:

1. `Platform Overview`
   - host up
   - exporters up
   - backend services up
   - public edge/API health

2. `Mac Studio Host`
   - CPU
   - RAM
   - disk usage
   - disk IO
   - network traffic
   - uptime

3. `Containers`
   - per-container CPU
   - per-container RAM
   - per-container fs usage
   - per-container network
   - restart indicators / last seen

4. `Datastores`
   - Postgres
   - ClickHouse
   - Redis

5. `Market Data Pipeline`
   - использовать existing metrics из
     `docs/runbooks/market-data-metrics-reference-ru.md`

6. `API`
   - request rate
   - latency
   - 5xx/4xx
   - uptime/health

## Что нужно изменить в репозитории

Минимальный набор изменений:

- `infra/docker/docker-compose.backend.yml`
  - добавить `cadvisor`
  - добавить `postgres_exporter`
  - добавить `redis_exporter`
  - добавить `clickhouse_exporter`

- `infra/monitoring/monitoring/prometheus/prometheus.yml`
  - добавить scrape jobs:
    - `node-exporter`
    - `cadvisor`
    - `postgres-exporter`
    - `redis-exporter`
    - `clickhouse-exporter`
    - расширенные `blackbox` jobs

- `apps/api`
  - добавить `/health`
  - при необходимости добавить Prometheus metrics middleware / endpoint

- `docs/runbooks/mac-studio-backend-operations.md`
  - добавить команды проверки exporters и dashboards

Опционально позже:

- provisioning dashboards в `Grafana`
- alert rules в репо

## Минимальный Definition of Done

Считаем внедрение завершенным, когда:

- `Prometheus` видит host exporter
- `Prometheus` видит `cadvisor`
- `Prometheus` видит exporters для `Postgres`, `Redis`, `ClickHouse`
- `Blackbox` проверяет `Grafana`, `Prometheus`, `API`, `Postgres`, `ClickHouse`, `Redis`
- в `Grafana` есть минимум 4 dashboards:
  - host
  - containers
  - datastores
  - market-data/api overview
- по дашбордам видно:
  - статус всех сервисов и контейнеров
  - uptime
  - CPU / RAM / SSD / network хоста
  - CPU / RAM / FS / network контейнеров
  - service errors и деградации

## Рекомендуемый порядок следующей реализации

1. `node_exporter` на `Mac Studio`
2. `cadvisor` в backend compose
3. `postgres_exporter` + `redis_exporter` + `clickhouse_exporter`
4. обновление `prometheus.yml`
5. `API /health`
6. первые 4 Grafana dashboards
7. alert rules

## Связанные документы

- `docs/runbooks/mac-studio-backend-operations.md`
- `docs/runbooks/market-data-metrics-reference-ru.md`
- `infra/monitoring/monitoring/prometheus/prometheus.yml`
- `infra/docker/docker-compose.backend.yml`
