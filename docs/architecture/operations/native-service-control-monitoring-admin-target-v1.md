# Native Service Control, Monitoring, and Admin Target (v1)

Документ фиксирует целевую production-картину управления, контроля и мониторинга сервисов Roehub на `Mac Studio` в native-модели (без обязательного Docker runtime).

## Статус и границы

- Статус: target state (планируемое целевое устройство контура).
- Хост backend: `Mac Studio`.
- Публичный edge остается на VPS (`roehub.com`), backend на `Mac Studio` остается private.
- Собственный billing-модуль не разрабатывается.
- `Backtest artifact publisher` управляется через `Monit` поверх `launchd`.

## Архитектурные решения

- `launchd` отвечает за запуск/перезапуск сервисов как process supervisor.
- `Monit` отвечает за service-level контроль состояния, рестарты по health-правилам и операционные алерты.
- `Prometheus + exporters + blackbox` отвечают за сбор метрик и probes.
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
| `com.roehub.backtest-job-runner.*` | `launchd` fleet + `Monit` per instance | `:9204+N`, queue lag/state metrics, lease health | cancel/retry/requeue/top через admin API |
| `com.roehub.backtest-artifact-publisher` | `launchd` + `Monit` (обязательно) | process + freshness/lag check program + publish metrics | run-now/rebuild/switch-slot действия |
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
  - claimed/finished/failed counters,
  - active claimed jobs,
  - duration histogram,
  - queue lag по БД состояниям.
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
  - publish duration,
  - last successful publish timestamp,
  - artifact freshness lag.
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
  - scrape failure budget.
- Админ-операции:
  - alert routing/silences,
  - dashboard ownership и release discipline через git.

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
2. Закрыть внешний доступ к внутренним metrics endpoints.
3. Включить `Alertmanager` routing для критичных алертов.
4. Поднять `Keycloak` и интегрировать API auth/RBAC.
5. Выбрать billing engine (`Kill Bill` как baseline, `Lago` как ускоренный вариант) и подключить subscription sync.
6. Внедрить `strategy-control-plane` и после этого запускать live runner в production режиме.

## Связанные документы

- `docs/runbooks/mac-studio-native-backend-operations.md`
- `docs/runbooks/mac-studio-monitoring-plan.md`
- `docs/runbooks/backtest-job-runner.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`
- `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md`
- `docs/architecture/strategy/strategy-realtime-output-redis-streams-v1.md`
