# Native Service Control, Monitoring, and Admin Target (v1)

Документ фиксирует целевую production-картину управления, контроля и мониторинга сервисов Roehub на `Mac Studio` в native-модели (без обязательного Docker runtime).

## Статус и границы

- Статус: target state (планируемое целевое устройство контура).
- Хост backend: `Mac Studio`.
- Публичный edge остается на VPS (`roehub.com`), backend на `Mac Studio` остается private.
- Собственный billing-модуль не разрабатывается.
- `Backtest artifact publisher` управляется через `Monit` поверх `launchd`.
- Backtest-контур (`com.roehub.backtest-job-runner.*`, `com.roehub.backtest-artifact-publisher` и последующие backtest services) входит в обязательный Prometheus monitoring baseline.

## Архитектурные решения

- `launchd` отвечает за запуск/перезапуск сервисов как process supervisor.
- `Monit` отвечает за service-level контроль состояния, рестарты по health-правилам и операционные алерты.
- `Prometheus + exporters + blackbox` отвечают за сбор метрик и probes.
- Для backtest-контура Prometheus является обязательным системным источником правды по runtime/freshness/failure метрикам и alerting rules.
- `Grafana` отвечает за визуализацию и дашборды.
- `Alertmanager` отвечает за маршрутизацию алертов.
- Доменная админка и операционные действия выполняются через FastAPI admin surface (`SQLAdmin` + явные action endpoints).
- Identity и RBAC для админских действий централизуются в `Keycloak`.
- Billing/subscriptions подключаются как внешний OSS-компонент (основной и альтернативный варианты ниже), без написания своего billing ядра.

## Общая матрица

| Сервис | Управление | Мониторинг | Админка (операции) |
| --- | --- | --- | --- |
| `com.roehub.api` | `launchd` + `Monit` process/port checks | `/metrics`, blackbox HTTP probe, API SLI | `SQLAdmin` для доменных сущностей, ops endpoints для control actions |
| `com.roehub.market-data-ws-worker` | `launchd` + `Monit` | `:9201` (`ws_*`, `insert_*`), restart/error alerts | restart/pause/resume через admin actions |
| `com.roehub.market-data-scheduler` | `launchd` + `Monit` | `:9202` (`scheduler_job_runs_total`, `scheduler_job_errors_total`) | trigger/reschedule job actions |
| `com.roehub.backtest-job-runner.*` | `launchd` fleet + `Monit` per instance | `:9204+N` `/metrics`, queue lag/state metrics, lease health, runner failure/duration metrics, Prometheus alert rules | cancel/retry/requeue/top через admin API |
| `com.roehub.backtest-artifact-publisher` | `launchd` + `Monit` (обязательно) | Prometheus-compatible publish/freshness/failure metrics (`/metrics` или exporter bridge для batch-режима), freshness/lag alerts | run-now/rebuild/switch-slot действия |
| `strategy-live-runner` (target) | `launchd` + `Monit` + control-plane lease model | `:9203`, heartbeat/command lag/failure metrics | start/stop/pause/resume, kill-switch, incident actions |
| PostgreSQL | `brew services`/`launchd` + `Monit` | `postgres-exporter` + TCP probe | `pgAdmin 4` (role/db/session ops) |
| Redis | `brew services`/`launchd` + `Monit` | `redis-exporter` + memory/latency alerts | `redis-commander` (streams/keys ops) |
| ClickHouse | `launchd` + `Monit` | `clickhouse-exporter`, HTTP `/ping`, native TCP probe | `CH-UI`/DBeaver CE + runbook operations |
| Prometheus | `brew services` + `Monit` | self-monitoring (`up`, scrape health, TSDB) | rule/scrape management through repo config |
| Grafana | `brew services` + `Monit` | `/api/health`, dashboard errors | dashboards, on-call views, incident drilldown |
| Alertmanager | `brew services` + `Monit` | `/api/v2/status`, notification delivery | silences, routing policies |
| `Keycloak` | `launchd` + `Monit` | JVM/process + readiness probe + auth SLI | users, groups, roles, realms, clients |
| Billing engine (`Kill Bill` или `Lago`) | `launchd` + `Monit` | billing API health, queue/webhook health, invoice/subscription lag | тарифы, подписки, инвойсы, payment state |

## Users and subscriptions (без своего billing модуля)

### Базовое решение

- Identity/RBAC: `Keycloak`.
- Billing/subscriptions: внешний OSS billing engine.
- Entitlements/feature flags для API и UI: хранение в Roehub БД + проверка в backend policy layer.

### Варианты billing engine

### Вариант A (предпочтительный): `Keycloak + Kill Bill`

- Плюсы: зрелый OSS billing stack, Apache-2.0, сильный subscription lifecycle.
- Минусы: более тяжелая интеграция и операционный контур.
- Когда выбирать: если нужен сложный lifecycle тарифов и долгий горизонт масштабирования.

### Вариант B: `Keycloak + Lago (OSS core)`

- Плюсы: быстрый старт usage/metering сценариев, современный API-first подход.
- Минусы: часть enterprise-функций вне OSS core.
- Когда выбирать: если нужен быстрый запуск metering/subscription сценариев с простым UX.

### Обязательные интеграционные контракты

- Источник identity: `sub`/user-id из Keycloak токена.
- Source of truth subscriptions: billing engine.
- Roehub backend хранит только локальный projection:
  - active plan,
  - limits,
  - feature entitlements,
  - access period.
- Синхронизация подписок:
  - webhook-first,
  - periodic reconciliation job.

## Target picture по каждому сервису

### Backtest monitoring contour

- Все backtest services рассматриваются как единый production monitoring contour, а не только как process checks через `Monit`.
- Каждый backtest service обязан публиковать Prometheus-compatible service metrics; для batch/ephemeral сценариев допустим exporter/textfile bridge, если прямой `/metrics` endpoint не подходит.
- В `Prometheus` фиксируются отдельные scrape jobs, recording rules и alert rules для backtest-контура.
- В `Grafana` фиксируется отдельный dashboard set по backtest execution/publishing pipeline.
- Минимальный обязательный набор сигналов по backtest-контуру:
  - throughput/state transitions,
  - failure/error counters,
  - duration histograms,
  - freshness/lag metrics,
  - last-success timestamp для batch-процессов.

### 1) API (`com.roehub.api`)

- Запуск: `launchd`.
- Контроль: `Monit` проверяет процесс и HTTP `/health`; при деградации делает restart.
- Мониторинг:
  - Prometheus scrape `/metrics`,
  - blackbox probe публичного `/api/health`,
  - latency/error SLI по ключевым endpoints.
- Админ-операции:
  - `SQLAdmin` views,
  - отдельные action endpoints для restart-safe business actions (cancel/requeue/repair).

### 2) Market data WS worker (`com.roehub.market-data-ws-worker`)

- Запуск: `launchd`.
- Контроль: `Monit` process + port check.
- Мониторинг:
  - `ws_reconnections_total`,
  - `ws_errors_total`,
  - `insert_batches_total`,
  - ingest lag alerts.
- Админ-операции:
  - soft restart,
  - forced reconnect,
  - pause/resume ingestion.

### 3) Market data scheduler (`com.roehub.market-data-scheduler`)

- Запуск: `launchd`.
- Контроль: `Monit` process + periodic health script.
- Мониторинг:
  - `scheduler_job_runs_total`,
  - `scheduler_job_errors_total`,
  - duration histograms.
- Админ-операции:
  - run-now для конкретной задачи,
  - pause/resume отдельных задач,
  - error ack с reason.

### 4) Backtest job runner fleet (`com.roehub.backtest-job-runner.*`)

- Запуск: materialized `launchd` instances из `worker_processes`.
- Контроль: `Monit` на каждый instance (process + metrics endpoint).
- Мониторинг:
  - обязательный Prometheus scrape каждого instance,
  - claimed/finished/failed counters,
  - active claimed jobs,
  - duration histogram,
  - queue lag по БД состояниям,
  - lease/heartbeat health,
  - alert rules на stalled runner, failure burst и queue lag saturation.
- Админ-операции:
  - cancel/retry/requeue job,
  - quarantine noisy user job,
  - временный scale fleet вверх/вниз через config+reload.

### 5) Backtest artifact publisher (`com.roehub.backtest-artifact-publisher`)

- Запуск: `launchd` schedule/service.
- Контроль: `Monit` обязателен:
  - process control,
  - health script на freshness манифестов,
  - restart/alert policy.
- Мониторинг:
  - Prometheus-compatible metrics обязательны, даже если publisher работает как batch/scheduled service,
  - `backtest_artifact_publish_runs_total`,
  - `backtest_artifact_publish_failures_total`,
  - `backtest_artifact_publish_duration_seconds`,
  - `backtest_artifact_last_success_unixtime`,
  - `backtest_artifact_freshness_lag_seconds`,
  - alert rules на publish failures, stale artifacts и отсутствие successful publish в допустимом окне.
- Админ-операции:
  - run-now,
  - full rebuild trigger,
  - slot switch с валидацией.

### 6) Strategy live runner (target)

- Запуск: `launchd`.
- Контроль: не только процессный.
- Обязательный control-plane:
  - desired vs actual state,
  - lease/heartbeat,
  - command queue (`start/stop/pause/resume`),
  - global and per-user kill-switch.
- Мониторинг:
  - runner heartbeat age,
  - command execution latency,
  - failed runs,
  - stream lag.
- Админ-операции:
  - stop all runs,
  - stop user runs,
  - replay/recovery,
  - incident mode.

### 7) PostgreSQL

- Управление: `brew services`/`launchd` + `Monit`.
- Мониторинг: `postgres-exporter`, connection saturation, replication/IO (если появится replica).
- Админ-операции: `pgAdmin 4`, role grants, session control, backup verification.

### 8) Redis

- Управление: `brew services`/`launchd` + `Monit`.
- Мониторинг: `redis-exporter`, memory fragmentation, evictions, stream lag.
- Админ-операции: `redis-commander`, stream inspection, consumer group diagnostics.

### 9) ClickHouse

- Управление: `launchd` + `Monit`.
- Мониторинг: `clickhouse-exporter`, `/ping`, insert/query latency, disk usage.
- Админ-операции: `CH-UI`/DBeaver CE, partition maintenance runbooks, dedup jobs.

### 10) Monitoring stack (`Prometheus`, `Grafana`, `Alertmanager`)

- Управление: `brew services` + `Monit`.
- Мониторинг:
  - self-health checks для самого monitoring stack,
  - alert delivery success,
  - scrape failure budget,
  - выделенные scrape/rule groups для backtest execution и artifact publishing.
- Админ-операции:
  - alert routing/silences,
  - dashboard ownership и release discipline через git,
  - versioned Prometheus/Grafana config для backtest monitoring baseline.

### 11) Identity (`Keycloak`)

- Управление: `launchd` + `Monit`.
- Мониторинг:
  - readiness/auth flow success rate,
  - token issue/error rate,
  - login latency.
- Админ-операции:
  - users/groups/roles,
  - client scopes,
  - admin realm policies.

### 12) Billing engine (`Kill Bill` или `Lago`)

- Управление: `launchd` + `Monit`.
- Мониторинг:
  - webhook ingestion health,
  - invoice/subscription state transition errors,
  - reconciliation lag.
- Админ-операции:
  - plan catalog,
  - subscriptions,
  - invoices,
  - payment attempts and dunning actions.

## Security and governance baseline

- Все admin surfaces за auth (Keycloak/OIDC), без публичного anonymous admin доступа.
- `/metrics` закрываются от внешнего интернета, кроме строго необходимого controlled path.
- Все control actions пишутся в audit log (кто, когда, что сделал, correlation id).
- Production changes только через versioned config и runbook-процедуры.

## Поэтапное внедрение

1. Зафиксировать `Monit` для всех текущих `launchd` сервисов, включая `backtest-artifact-publisher`.
2. Добавить Prometheus instrumentation и scrape-конфигурацию для backtest services, включая `backtest-artifact-publisher`.
3. Собрать отдельные Grafana dashboards и Prometheus/Alertmanager rules для backtest execution и artifact publishing.
4. Закрыть внешний доступ к внутренним metrics endpoints.
5. Включить `Alertmanager` routing для критичных алертов.
6. Поднять `Keycloak` и интегрировать API auth/RBAC.
7. Выбрать billing engine (`Kill Bill` как baseline, `Lago` как ускоренный вариант) и подключить subscription sync.
8. Внедрить `strategy-control-plane` и после этого запускать live runner в production режиме.

## Связанные документы

- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/runbooks/mac-studio-monitoring-plan.md`
- `docs/runbooks/backtest-job-runner.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`
- `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md`
- `docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md`
