# Stage 00 — заморозка текущего состояния и контрактов

## Результат

- Дата: `2026-07-13`.
- Stage: `00`.
- Режим: `goal_driven`.
- Статус: `accepted`.
- Граница доказательства: `N/A` — этап не меняет runtime-поведение и не
  утверждает готовность целевой реализации.
- Audited Git revision: `e62667c4bd11bd6d6ea4d5bfd5a07ffcbb9c1eb0`.
- Project-map structural digest:
  `061cb4423267fdafd052aa03447fe0e7b5b6b7f4fc8b0113fbf55b55ddc545a4`
  (`inventory_file_count=2485`).
- Бизнес-результат: текущая native-first система, её пользовательская модель,
  хранилища, внешние вызовы и опасные границы зафиксированы как implementation
  baseline для greenfield self-hosted продукта. Ни один из 33 текущих
  компонентов не потерян. Обнаруженные в production ownership graph
  cross-owner и orphan references сохраняются как evidence о необходимых
  fresh-schema constraints, но текущие строки не импортируются и не
  исправляются. Greenfield-поправка прошла повторные локальные и независимые
  проверки; Stage `01` разрешён.

## Источники и метод

Факты получены из текущего checkout, детерминированной карты проекта,
entrypoints, портов, адаптеров, миграций, конфигурации, deployment-артефактов и
ранее выполненной безопасной read-only сверки production PostgreSQL на
`macstudio`. Запрос к БД возвращал только имена колонок и агрегированные counts;
UUID, имена, адреса, DSN, токены и другие чувствительные значения не
извлекались. После решения `A07` новые production-запросы не требуются:
current data не является migration input.

Обозначения:

- `fact` — наблюдается в актуальном коде, схеме, конфигурации или read-only
  агрегате;
- `inference` — вывод из перечисленных фактов;
- `decision` — принятое целевое решение плана, ещё не реализованное;
- `unknown` — не доказано в Stage `00` и явно передано будущему этапу.

Ключевые источники:

- `docs/architecture/project-map/project-map.json`;
- `README.md`;
- `infra/docker/docker-compose.yml`;
- `infra/macos/launchd/` и `infra/scripts/monit/`;
- `apps/`, `src/trading/contexts/`, `src/trading/{fastpath,integration,platform,shared_kernel}/`;
- `migrations/postgres/`, `migrations/clickhouse/`, `alembic/versions/`;
- `docs/architecture/operations/native-service-control-monitoring-admin-target-v1.md`.

## Проверка трёх источников исполнения

| Поле | Наблюдение | Итог |
|---|---|---|
| `plan_doc` | `docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md` существует и ссылается на тот же pack/ledger | `passed` |
| `prompt_pack_dir` | `.codex/agents/generated/roehub-self-hosted-oss-platform-v1/` содержит уникальный Stage `00` prompt | `passed` |
| `stage_ledger` | Initial gate: `current_stage: 00`, `execution_mode: goal_driven`; после первого audit ledger был `blocked`, затем пользовательское решение `A07` возобновило Stage `00`; после greenfield validation журнал содержит `00=accepted`, `current_stage: 01` | `passed` |
| Предшественник | У Stage `00` нет stage-предшественника; cold-head review исходного плана записан | `passed` |
| Authority | Только docs writes; commit, push, deploy и production mutation не разрешены | `passed` |

## Текущий runtime и поставка

| Поверхность | Наблюдаемое состояние | Вид утверждения | Evidence |
|---|---|---|---|
| Production topology | Backend описан как native runtime на Mac Studio, edge/TLS остаётся на VPS | `fact` | `README.md`; `infra/macos/launchd/` |
| Process supervision | Основные API/workers/exchange/OpenBao процессы имеют `launchd` units; часть сервисов покрыта Monit-конфигурациями | `fact` | `infra/macos/launchd/`; `infra/scripts/monit/` |
| Containers | Общий Compose покрывает datastores, API/Web и часть market-data/observability, но не все 33 компонента; образы observability не везде закреплены digest | `fact` | `infra/docker/docker-compose.yml`; `infra/docker/docker-compose.backend.yml` |
| User configuration | Runtime собирается из env, YAML и service-specific defaults; единого `roehub.yaml` пока нет | `fact` | `configs/`; `apps/**/wiring/`; `infra/macos/launchd/` |
| Target delivery | Одна подписанная release unit и `roehubctl up` | `decision` | `docs/architecture/platform/roehub-self-hosted-oss-platform-v1.md` |
| Runtime proof | Изменённого runtime-кода в Stage `00` нет; production readiness не заявляется | `fact` | Stage authority/proof boundary |

## Инвентаризация 33 текущих компонентов

`project-map.json` содержит ровно 33 компонента. Список в plan doc содержит те
же 33 ID по одному разу; генератор карты проходит `--check`. В колонке
«Зависимости» указаны текущие import/runtime зависимости из карты, а не
будущая целевая схема.

| Компонент | Текущий owner / entrypoint | Зависимости | State / deployment / trust boundary | Target stages | Evidence |
|---|---|---|---|---|---|
| `app:api` | API delivery; `apps/api/main/{app,main}.py` | CLI, backtest/artifacts, identity, indicators, live execution, market data, notifications, RL, strategy, platform/shared kernel | PostgreSQL/ClickHouse/Redis; native `launchd` и Compose; browser/API auth boundary | `05`–`13`,`16`,`19` | `apps/api/`; project map |
| `app:cli` | Operator CLI; `apps/cli/main/main.py` | API, artifacts, indicators, market data, platform/shared kernel | Host filesystem/env и ClickHouse; operator-host trust | `03`,`09`,`17`,`18` | `apps/cli/`; project map |
| `app:exchange_control` | Internal HTTP service; `apps/exchange_control/main/{app,main}.py` | `context:exchange_control` | PostgreSQL/OpenBao/exchanges; native `launchd`; secret-bearing trusted boundary | `08`,`10`,`16`,`17` | `apps/exchange_control/`; launchd unit |
| `app:exchange_execution` | Execution service; `apps/exchange_execution/main/{app,main}.py` | exchange control, live execution, strategy, shared kernel | PostgreSQL/Redis/exchanges; native `launchd`; money-moving boundary | `10`,`16`,`17` | `apps/exchange_execution/`; launchd unit |
| `app:migrations` | Schema bootstrap; `apps/migrations/main.py` | SQL/Alembic, not imported contexts | PostgreSQL/ClickHouse schema authority; host/container one-shot | `04`–`12`,`14` | `apps/migrations/`; `migrations/`; `alembic/` |
| `app:monitoring` | ClickHouse exporter; no `main.py` in map | direct ClickHouse HTTP | Metrics-only state; native `launchd`; observability boundary | `17`,`20` | `apps/monitoring/clickhouse_exporter.py`; launchd unit |
| `app:scheduler` | Market-data scheduler and artifact publisher entrypoints | API/CLI, artifacts, indicators, market data, platform/shared kernel | ClickHouse/filesystem/PostgreSQL; native `launchd`; scheduled side effects | `09`,`11`,`14`,`15`,`17` | `apps/scheduler/`; launchd units |
| `app:web` | SSR/same-origin gateway; `apps/web/main/{app,main}.py` | API through HTTP, not Python import | Browser session/cookie and upstream API; Compose/native edge path | `06`,`07`,`13`,`19` | `apps/web/`; `apps/web/main/api_client.py` |
| `context:backtest` | Historical compute and job orchestration | artifacts, indicators, platform/shared kernel | PostgreSQL jobs + filesystem arrays + compute memory; CPU/hot-path boundary | `09`,`15` | `src/trading/contexts/backtest/` |
| `context:backtest_artifacts` | Artifact precompute/publish/read | backtest, indicators, market data, platform/shared kernel | Filesystem slot/manifests + PostgreSQL metadata; artifact integrity boundary | `09`,`14` | `src/trading/contexts/backtest_artifacts/` |
| `context:exchange_control` | Connections, validation, secret cipher | shared kernel | PostgreSQL + OpenBao transit + exchange HTTP; credentials boundary | `08`,`10`,`16` | `src/trading/contexts/exchange_control/` |
| `context:identity` | Users, sessions, settings, exchange credentials | shared kernel | PostgreSQL; Keycloak subject + Roehub session; auth/recovery/secret boundary | `05`–`08` | `src/trading/contexts/identity/`; migrations `0001`,`0005`,`0006`,`0008` |
| `context:indicators` | Registry and compute ports/kernels | market data, platform/shared kernel | YAML registry and process memory; verified compute/hot-path boundary | `09`,`13`,`15` | `src/trading/contexts/indicators/` |
| `context:live_execution` | Intent/order/risk/reconciliation model | exchange control, strategy, shared kernel | PostgreSQL ledger + Redis streams + exchange adapter; money-moving/unknown-state boundary | `10`,`16` | `src/trading/contexts/live_execution/` |
| `context:market_data` | Exchange ingest/normalization/storage | shared kernel | ClickHouse + PostgreSQL refs + Redis + external REST/WS; network/data freshness boundary | `09`,`12`,`17` | `src/trading/contexts/market_data/` |
| `context:ml` | Package skeleton only | none | No implemented contract or deployment | `15`,`17`,`24` | `src/trading/contexts/ml/__init__.py` |
| `context:notifications` | Event routing, outbox, providers, reports | shared kernel | PostgreSQL + Telegram HTTP; provider unknown-state/redaction boundary | `11`,`12` | `src/trading/contexts/notifications/` |
| `context:optimize` | Package skeleton only | none | No independent implementation/deployment | `09`,`15` | `src/trading/contexts/optimize/__init__.py` |
| `context:risk` | Package skeleton only | none | Current execution risk lives mainly in `live_execution`; target boundary not yet extracted | `10`,`16` | `src/trading/contexts/risk/__init__.py`; live-execution risk files |
| `context:rl_trading` | Training/inference/model registry and ACL | live execution, strategy, shared kernel | PostgreSQL + filesystem artifacts + Redis runtime; model/inference/execution boundary | `09`,`14`–`16` | `src/trading/contexts/rl_trading/` |
| `context:strategy` | Strategy spec/run/signals/live lifecycle | live execution, market data, notifications, platform/shared kernel | PostgreSQL + Redis; producer-to-execution and notification boundaries | `10`,`15`,`16` | `src/trading/contexts/strategy/` |
| `core:fastpath` | Package skeleton | none | No shared fastpath implementation; hot paths remain inside contexts | `09`,`15`,`24` | `src/trading/fastpath/__init__.py` |
| `core:integration` | Package skeleton | none | No public versioned wire/SDK contract | `12`–`15` | `src/trading/integration/__init__.py` |
| `core:platform` | Config/error/time primitives | market data, shared kernel | Process-local config/defaults; no product-wide config schema | `03`,`04`,`12`,`17` | `src/trading/platform/` |
| `core:shared_kernel` | User/market/instrument/time primitives | none | Includes `UserId` and `PaidLevel`; no organization primitive | `05`,`09`,`10` | `src/trading/shared_kernel/` |
| `worker:backtest_job_runner` | Queue/job worker entrypoint | backtest/artifacts, platform/shared kernel | PostgreSQL jobs + filesystem artifacts; native `launchd`; resource/process boundary | `09`,`15`,`17` | `apps/worker/backtest_job_runner/`; launchd unit |
| `worker:market_data_ws` | WS ingest entrypoint | CLI, market data, platform/shared kernel | ClickHouse/Redis/exchanges; native/Compose; long-lived network boundary | `09`,`17`,`20` | `apps/worker/market_data_ws/`; launchd/Compose |
| `worker:notification_dispatcher` | Delivery worker entrypoint | notifications | PostgreSQL + Telegram HTTP; native `launchd`; provider side-effect boundary | `11`,`17`,`20` | `apps/worker/notification_dispatcher/`; launchd unit |
| `worker:notification_report_scheduler` | Package wiring without mapped `main.py` | notifications | PostgreSQL report/delivery scheduling; deployment entrypoint is not independently frozen | `11`,`17`,`20` | `apps/worker/notification_report_scheduler/` |
| `worker:rl_trading_inference` | Inference/monitor/testnet entrypoint | live execution, RL, strategy | Redis/PostgreSQL/model files; native `launchd`; execution-capable trust boundary | `15`–`17` | `apps/worker/rl_trading_inference/`; launchd unit |
| `worker:rl_trading_trainer` | Training entrypoint | no mapped cross-component import | Filesystem model artifacts + data sources; resource-intensive isolated-job candidate | `14`,`15`,`17` | `apps/worker/rl_trading_trainer/` |
| `worker:strategy_live_runner` | Live runner entrypoint | CLI, live execution, market data, notifications, strategy, platform | PostgreSQL/Redis/Telegram; native `launchd`; intent producer boundary | `10`,`15`–`17` | `apps/worker/strategy_live_runner/`; launchd unit |
| `worker:telegram_bot_worker` | Telegram command worker without mapped `main.py` | notifications | Telegram/provider + PostgreSQL; external inbound/provider trust boundary | `11`,`17`,`20` | `apps/worker/telegram_bot_worker/` |

Новые `context:extensions`, `context:operations`, `app:control_agent` и
`app:roehubctl` являются `decision`, отсутствуют в текущем component inventory и
создаются только соответствующими будущими stages.

## Владение данными и greenfield baseline

### PostgreSQL

Ранее выполненная read-only production schema сверка подтверждает:

- 205 строк `identity_users` и 205 различных `user_id`;
- 203 текущих Keycloak identity links и 2 Telegram-linked legacy users;
- 51 колонку владения `user_id`/`owner_user_id` в прикладных таблицах;
- отсутствие `organization_id`, `tenant_id` и `org_id` в коде, SQL migrations и
  текущей production schema;
- нулевые `NULL` owners во всех обязательных ownership columns;
- допустимые nullable owners только у системных/admin notification rows:
  3 events и 2 routes;
- полный ownership-graph audit ниже: 49 declared-FK edges не имеют orphan или
  cross-owner rows, но среди 51 semantic edges без FK обнаружены 218 orphan
  edge references и 10 cross-owner mismatches.

Вывод (`fact` + `decision`): текущая схема выражает user ownership и показывает,
какие классы связей новая схема должна запретить. Решение `A07` исключает
mapping текущих users/rows в personal organizations, поэтому причины конкретных
production anomalies не являются Stage `00` gate. В clean database системные
notification rows создаются сразу в installation/system scope, а все
organization-owned ссылки получают same-org и referential-integrity invariants.

### Воспроизводимый ownership-graph audit

- Время audit snapshot: `2026-07-12T22:23:14Z`.
- Scope: 46 production tables с 51 колонкой `user_id`/`owner_user_id`.
- Phase A: все single-column PostgreSQL foreign keys между двумя
  ownership-bearing tables; проверено 49 edges, `orphan_rows=0`,
  `owner_mismatch_rows=0`.
- Phase B: все внутренние semantic reference candidates без FK. Правила
  сопоставления: `exchange_connection_id → exchange_connections.connection_id`,
  `intent_id → execution_intents.intent_id`,
  `strategy_signal_id|source_signal_id → strategy_signals.signal_id`,
  `strategy_id → strategy_strategies.strategy_id`,
  `strategy_run_id → strategy_runs.run_id`,
  `live_profile_id → strategy_live_profiles.profile_id`,
  `source_job_id → backtest_jobs.job_id`,
  `reservation_id → strategy_capital_reservations.reservation_id`,
  `source_event_id → execution_source_events.source_event_id`,
  `paper_order_id → paper_orders.paper_order_id`,
  `paper_fill_id → paper_fills.paper_fill_id`,
  `accounting_id → strategy_paper_accounting.accounting_id`,
  `scenario_matrix_row_id → strategy_variant_scenario_matrix_rows.scenario_matrix_row_id`
  и `source_account_snapshot_id → exchange_account_snapshots.account_snapshot_id`.
  `market_id`, `instrument_id`, provider/Redis/chat/external IDs и polymorphic
  audit `target_id` исключены как не ссылающиеся на user-owned parent table.

Sanitized query template, выполненный только через `SELECT`:

```sql
WITH owner_tables AS (
  SELECT table_name,
         coalesce(
           min(column_name) FILTER (WHERE column_name = 'owner_user_id'),
           min(column_name) FILTER (WHERE column_name = 'user_id')
         ) AS owner_column
  FROM information_schema.columns
  WHERE table_schema = 'public'
    AND column_name IN ('owner_user_id', 'user_id')
  GROUP BY table_name
), edge_sources(child_table, child_fk, parent_table, parent_pk) AS (
  -- Phase A is generated from pg_constraint; Phase B uses the mapping rules above.
  VALUES ('child_table', 'reference_id', 'parent_table', 'primary_id')
), edges AS (
  SELECT edge_sources.*,
         child_owner.owner_column AS child_owner,
         parent_owner.owner_column AS parent_owner
  FROM edge_sources
  JOIN owner_tables AS child_owner
    ON child_owner.table_name = edge_sources.child_table
  JOIN owner_tables AS parent_owner
    ON parent_owner.table_name = edge_sources.parent_table
)
SELECT format(
  'SELECT %L AS edge,
     count(*) FILTER (WHERE c.%I IS NOT NULL AND p.%I IS NULL) AS orphan_rows,
     count(*) FILTER (
       WHERE c.%I IS NOT NULL AND p.%I IS NOT NULL
         AND c.%I IS DISTINCT FROM p.%I
     ) AS owner_mismatch_rows
   FROM %I AS c LEFT JOIN %I AS p ON c.%I = p.%I;',
  child_table || '.' || child_fk || '->' || parent_table || '.' || parent_pk,
  child_fk, parent_pk,
  child_fk, parent_pk, child_owner, parent_owner,
  child_table, parent_table, child_fk, parent_pk
)
FROM edges
ORDER BY child_table, child_fk, parent_table
\gexec
```

Все non-zero semantic results (counts — агрегаты, IDs не извлекались):

| Edge | Orphan references | Cross-owner mismatches |
|---|---:|---:|
| `exchange_account_config_guard_results.exchange_connection_id → exchange_connections.connection_id` | 2 | 0 |
| `exchange_account_snapshots.exchange_connection_id → exchange_connections.connection_id` | 9 | 0 |
| `exchange_balance_snapshots.exchange_connection_id → exchange_connections.connection_id` | 9 | 0 |
| `exchange_instrument_filter_snapshots.exchange_connection_id → exchange_connections.connection_id` | 9 | 0 |
| `execution_intents.exchange_connection_id → exchange_connections.connection_id` | 85 | 0 |
| `execution_intents.strategy_signal_id → strategy_signals.signal_id` | 20 | 0 |
| `execution_notification_outbox.strategy_signal_id → strategy_signals.signal_id` | 1 | 0 |
| `execution_orders.exchange_connection_id → exchange_connections.connection_id` | 1 | 0 |
| `execution_source_events.strategy_signal_id → strategy_signals.signal_id` | 22 | 0 |
| `paper_orders.source_signal_id → strategy_signals.signal_id` | 4 | 0 |
| `strategy_backtest_variant_provenance.source_job_id → backtest_jobs.job_id` | 0 | 1 |
| `strategy_capital_reservations.exchange_connection_id → exchange_connections.connection_id` | 28 | 0 |
| `strategy_capital_reservations.strategy_run_id → strategy_runs.run_id` | 15 | 0 |
| `strategy_live_profiles.exchange_connection_id → exchange_connections.connection_id` | 11 | 6 |
| `strategy_position_ownership.exchange_connection_id → exchange_connections.connection_id` | 2 | 3 |

Остальные semantic edges имеют нулевые orphan/mismatch counts. Non-zero rows
могут быть историческими, synthetic/test или следствием архивирования, но это
не доказано. Определение причины либо изменение production данных выходит за
docs-only authority и greenfield scope Stage `00`. Эти строки не переносятся;
наблюдение передано Stages `05`,`09`,`10` как обязательный класс отрицательных
fresh-schema tests.

### Прочие владельцы состояния

| State | Текущий owner | Greenfield-требование | Evidence |
|---|---|---|---|
| Market/analytics time series | ClickHouse; текущие таблицы в основном installation-global | Добавление org scope не должно дублировать общедоступные market facts без решения data ownership | `migrations/clickhouse/`; market-data adapters |
| Transport/cache/leases | Redis Streams/cache | Redis не может стать единственным source of truth; org должен войти в stream payload/consumer semantics | live-execution/strategy/market-data Redis adapters |
| Exchange secrets | OpenBao transit и legacy encrypted PostgreSQL records/env references | Нужен новый versioned secret reference и per-org policy; текущие значения не копируются | exchange-control adapters; identity exchange-key migrations |
| Backtest artifacts | Host filesystem slots/manifests + PostgreSQL metadata | Текущие paths/hash pointers не являются tenant-aware CAS | backtest/backtest-artifacts adapters |
| RL/ML artifacts | Host filesystem manifests/models + PostgreSQL registry | Нужны organization scope, digest identity и resource policy | RL domain/migrations |
| Metrics/logs | Prometheus-compatible endpoints и host logs | Labels/redaction/runbook ownership меняются при container/org model | monitoring code; infra monitoring/launchd |

Постоянное filesystem state является доказанным текущим фактом и implementation
input для поиска path coupling, но не migration input и не допустимым target
container state.

## Текущие service calls и trust boundaries

| Вызов | Current auth / secrets | Timeout / retry | Idempotency / unknown state | Metrics / alerts | Runbook gap и owner stage | Evidence |
|---|---|---|---|---|---|---|
| Browser/Web → API | Roehub HttpOnly session cookie; Web proxy не переносит provider token в browser | Current-user client 5 s, generic proxy 30 s; автоматический retry не задан | Единого idempotency contract для mutating Web calls нет; HTTP failure отображается как unavailable/error | Generic API HTTP metrics; отдельного auth-flow alert contract нет | `web-ui-gateway-same-origin.md`; org/RBAC/recent-auth/CSRF и versioned errors — `05`–`07` | `apps/web/main/{app,api_client}.py`; identity current-user port |
| API → Keycloak OIDC | Authorization code flow, state cookie, host-local client credential; Keycloak-specific endpoints | Token/introspection HTTP timeout 5 s; transport failure завершает login; blind retry отсутствует | State одноразовый; uncertain token POST не создаёт Roehub session | Keycloak readiness/Monit есть; per-step discovery/JWKS/token metrics/alerts отсутствуют | `keycloak-local-setup-and-ops.md`; generic OIDC, discovery/JWKS, `local/hybrid`, recovery — `06`–`07` | `auth_oidc.py`; `apps/api/wiring/modules/identity.py` |
| API → exchange-control | Static bearer credential из host-local env + `X-Roehub-Internal-Service: apps/api` + request ID | Default 2 s; client не выполняет автоматический retry; transport/status errors sanitized | Operation IDs существуют в отдельных commands, но uniform replay/unknown contract отсутствует; timeout = failure, не success | Exchange-control readiness/metrics/Monit существуют; service-auth failure alert/runbook linkage не унифицирован | `exchange-secret-management.md`; short-lived service identity, org capability, typed operations — `08`,`10`,`18` | `apps/api/exchange_control_client.py:13-21,485-521` |
| exchange-control → OpenBao | `X-Vault-Token` со scoped transit token; plaintext существует только внутри trusted cipher adapter | Fixed 3 s; retry отсутствует | Encrypt/decrypt/HMAC request не имеет operation id; timeout/transport error теряет результат и поднимает sanitized error | OpenBao metrics/Monit/runbook существуют; per-operation unknown counter/alert отсутствует | `exchange-secret-management.md`; per-org refs, rotation/recovery и explicit unknown handling — `08` | `openbao_transit.py`; `secret_cipher.py` |
| market-data → exchanges | Public market REST/WS; private credentials в этом ingest path не требуются | Configured timeout; REST retries `429/418/5xx` и transport errors с bounded exponential backoff+jitter | Reads are naturally retryable; checkpoint/dedupe живут в ingestion/storage, но provider result identity неодинакова | Detailed WS/insert/repair metrics, alerts and runbooks exist | `market-data-live-tail-repair.md`, metrics reference; plugin source instance/egress/org-vs-system ownership — `09`,`12` | market-data clients/workers |
| strategy/RL → live execution | In-process ports/ACL, `owner_user_id`; secrets не передаются | Redis socket/connect timeouts; retry stream/DLQ предусмотрены | Source/intent idempotency hashes + PostgreSQL ledger; `unknown`/reconciliation states существуют | Execution source/intent/risk/dispatch/outbox metrics and critical runbook actions exist | `strategy-live-worker.md`, `rl-trading-operations.md`; org/account namespace migration — `10`,`15`,`16` | strategy/RL ACLs; live-execution ports/adapters |
| exchange execution → exchange | Credential resolved через exchange-control/OpenBao boundary; adapter gets scoped account command | Provider-specific HTTP timeouts; blind submit retry запрещён | Exchange client order ID, order/event/fill/reconciliation ledger; unknown result требует reconcile | Rich execution readiness/DLQ/reconciliation metrics and incident runbook exist | `exchange-execution.md`; org/account/risk/mainnet enforcement and universal unknown policy — `16` | native HTTP adapter; live-execution domain |
| dispatcher → Telegram | Provider credential из host-local env; recipient address redacted in request hash | Default 2 s; `429` → bounded retry with `Retry-After`; timeout/5xx → `unknown`, not blind retry | Delivery UUID/outbox/attempts; `unknown` requires explicit handling, max attempts bound retryable results | sent/retry/unknown/dead-letter metrics and notification runbooks exist | notification admin/egress runbooks; per-org provider instance/secret ref/replay — `11` | notification provider/dispatcher |
| Workers → PostgreSQL/ClickHouse/Redis | Service env credentials; no uniform short-lived service identity | Driver/context-specific timeouts; Redis retry/claim semantics vary | PostgreSQL is durable truth for jobs/intents; Redis is transport/cache, but not every worker documents unknown/replay uniformly | Per-worker metrics exist, cross-service alert/action schema is fragmented | Current domain runbooks; service identity, org envelope and `ops.roehub.io/v1` — `02`,`17`,`20` | worker wiring; persistence/messaging adapters |
| API/CLI → Docker Engine | Current call absent; no Docker socket in API | `N/A` | `N/A` | `N/A` | Future typed allowlist, operation journal and degraded recovery only through `control-agent` — `18` | repo-wide search; target plan |
| Plugin gateway → plugin | Current call/secret/capability identity absent | `N/A` | `N/A` | `N/A` | Entire auth/timeout/retry/idempotency/metrics/runbook boundary is owned by `12`–`15`; execution before then forbidden | skeleton `core:integration`; target plan |

## Поиск специальных контрактов

| Тема | Наблюдение | Вывод |
|---|---|---|
| `paid_level` / subscription | `paid_level` присутствует в identity principal, API/Web DTO, backtest admission, RL entitlements и persisted schema; найдено 25 runtime/schema files | Удаление является `breaking-change`; Stage `05`,`09`,`15` должны делать consumer-by-consumer replacement, не простой drop |
| Keycloak-only | Current identity wiring требует Keycloak endpoints/client and persists `keycloak_subject` | `local` и generic OIDC — новый `breaking-change`; current identities/`user_id` не импортируются, новые links создаются в clean database |
| Native/Monit | Launchd units существуют для API, workers, exchange, OpenBao и exporters; Monit configs покрывают часть контура | Это implementation reference, не target architecture и не migration source; Stage `25` запускает отдельную greenfield production installation |
| Docker | Compose покрывает часть topology и содержит env-oriented configuration | Stage `17` обязан построить полный generated topology и pinned manifest |
| Telemetry / phone-home | Отдельного product analytics/update-check outbound path не найдено; слово telemetry относится к локальным compute metrics | Это bounded static evidence, не сетевой runtime proof; Stage `22` доказывает no-phone-home на bundle |
| Plugins | Plugin package/instance/API/SDK отсутствуют; совпадения `plugin` относятся к другим техническим терминам | Target contract начинается с Stage `12`, без утверждения compatibility с несуществующим public API |
| Artifacts | Backtest/RL используют filesystem manifests/hashes и PostgreSQL pointers; общего `ArtifactStore/v1` нет | Stage `14` создаёт новый breaking storage/identity contract и signed demo bundle; current paths не импортируются |

## Request hash, cache key и persistence identity

Stage `00` устраняет исходный `unknown`: добавление organization scope затрагивает
реальные persisted/dedupe identities и является `breaking-change`.

| Context | Текущая identity | Organization impact | Classification | Evidence |
|---|---|---|---|---|
| Backtest jobs | Canonical request hash + отдельно owner/user-scoped idempotency lookup | Новые content hashes остаются byte-semantic, а dedupe key сразу включает versioned org namespace | `breaking-change` | backtest preflight/use case/repository |
| Strategy provenance | `user_id`, source job/variant, spec hash, launch request hash и idempotency hash | Org membership и org-owned strategy меняют persistence uniqueness; spec content hash может остаться | `breaking-change` | create-from-variant use case; migrations `0016`,`0034` |
| Live execution | Hash сырого idempotency key, lookup в паре с `owner_user_id`; exchange `client_order_id` | Новые intents сразу используют org/account namespace; current in-flight intents не импортируются | `breaking-change` | execution ingress/repository/order domain |
| Notifications | Event/report dedupe keys включают owner/route/time; deliveries имеют stable UUID | Per-org route/provider instance меняет dedupe namespace и replay ownership | `breaking-change` | source router/report scheduler/repository |
| Market-data global cache | Instrument-based in-process keys; shared market facts не содержат user/org | Для installation-global facts org нельзя добавлять автоматически | `compatible-change` | market-data REST/cache adapters |
| Market-data private source instance | Контракт пока отсутствует | Org/plugin instance требует versioned key namespace | `breaking-change` | target plugin/data-source contract |
| Artifact manifests | SHA-256 manifests/config hashes + filesystem slot pointers | Content digest может сохраниться; tenancy, location, lease/quota/pointer identity меняются | `breaking-change` | backtest-artifacts contracts/config/publisher |
| RL/ML models | Manifest/artifact hashes и registry identities | Org ownership и bundle/runtime digest добавляются вокруг content hash | `breaking-change` | RL registry/training/evaluation domain |

Greenfield rule: новые records сразу используют versioned
org/account/instance identity fields; dual-read, alias и backfill текущих rows
не создаются. Content hashes остаются общими только когда их байтовая семантика
действительно не меняется.

## Матрица влияния на контракты

| Surface | Old contract | New contract | Consumers/evidence | Classification | Migration/deprecation | Rollout/rollback | Verification | Unknowns |
|---|---|---|---|---|---|---|---|---|
| `public_api` | User + `paid_level`, Keycloak login, user-owned resources | Organization/RBAC, local + optional OIDC, admin/plugin/ops API | API/Web routes and DTO | `breaking-change` | Versioned v1 DTO; no compatibility projection for current installation | Coordinated greenfield Web/API rollout; rollback within new release | API/browser/two-org tests | Exact API version names fixed by later stages |
| `port_contract` | User-scoped ports; no extension/operations ports | Org-aware ports + plugin/control contracts | identity/strategy/live/notification ports | `breaking-change` | Update adapters in the same staged codebase | Context-by-context implementation before bundle acceptance | Contract tests | None critical for Stage `00` |
| `dto_schema` | `user_id`, `paid_level`, legacy account/settings DTO | org/role/capability/provider instance | API DTO and Web client | `breaking-change` | Coordinated v1 DTO replacement | Feature-gated implementation until clean release | API/Web schema tests | Exact v1 schema names |
| `persisted_schema` | 51 user ownership columns; no org | Organizations, membership, org-owned resources, plugin/job/artifact state | Migrations + current aggregate evidence | `breaking-change` | Greenfield schema with FK/same-org invariants; no backfill/dual-read | Empty-store bootstrap; rollback by restoring new-format backup | Disposable DB + two-org negative tests | Exact constraint shapes by context |
| `config_schema` | env + service YAML + hand-written Compose | `roehub.yaml` + generated internal files | wiring/infra | `breaking-change` | Greenfield schema and version; no current-config converter | Generate clean release config; rollback within new release | Deterministic generation | External datastore profile details |
| `request_cache_identity` | User/route/job namespaces and content hashes | Org/account/instance namespaces around byte-semantic content hashes | table above | `breaking-change` | Versioned keys from first v1 write; no alias/backfill | No current rows imported | Replay/dedupe/cache tests | Per-context version numbers |
| `service_call_semantics` | Keycloak-specific and env service credentials | generic OIDC/service identity/org capability | service-call table | `breaking-change` | Replace in target composition | Disposable provider/runtime canary and degradation | Fakes + real safe boundaries | Exact mTLS/token mechanism |
| `external_side_effects` | User-scoped exchange/Telegram identities | Org/account/provider-instance identities | execution/notifications | `breaking-change` | Preserve external reconciliation keys | Fail closed; reconcile before retry | Emulator/testnet/provider fake | Mainnet remains approval-gated |
| `logs_metrics_audit_redaction` | Mixed labels and current user audit | org/operation/plugin/runbook labels with strict redaction | monitoring/audit code | `breaking-change` | New labels/event versions | Greenfield dashboards/alerts | Metrics/audit schema checks | Label cardinality budgets |
| `alerts_runbooks` | Native/Monit runbooks | `ops.roehub.io/v1` and container actions | runbooks/infra | `breaking-change` operationally | Generate new docs; current runbooks remain historical/runtime-specific | Failure drills before greenfield launch | Failure drills | Exact alert routing owners |
| `benchmark_rollout_gate` | Context-specific benchmarks and native gates | platform matrix/release bundle gates | stage plan | `compatible-change` | Preserve existing context baselines | Stage-gated rollout | Stage `24` matrix | macOS ML result |
| `verified_hot_path` | Existing indicators/backtest/RL paths | Org/runtime wrappers must preserve compute semantics | perf docs/tests | `unknown` | No performance claim in Stage `00` | Measure at `09`,`15`,`24` | Comparable benchmarks | macOS CPU/MPS performance |
| `browser_visible` | Keycloak login and current user settings | local + optional OIDC + org/admin/plugin UI | Web routes/templates | `breaking-change` | New greenfield login/settings UX | Browser-gated release | Auth/admin/a11y/responsive QA | Final visual composition |

## Findings и решения Stage 00

### Решённые Blocker / High

1. Первый проход зафиксировал `Blocker`: full ownership graph содержит 10
   cross-owner mismatches и 218 orphan edge references, поэтому безопасный org
   backfill доказать нельзя. Решение владельца `A07` удалило backfill/current
   data migration из scope v1. Production rows остаются неизменными, а класс
   дефекта передан Stages `05`,`09`,`10` как fresh-schema constraints/tests.
2. Independent review первоначально нашёл `High`: service-call matrix не
   фиксировала все auth/secrets/timeout/retry/idempotency/unknown/metrics/alerts/
   runbook dimensions и скрывала static bearer boundary API → exchange-control.
   Исправлено расширенной матрицей выше.

Исправленная plan gap: `request hash / cache key / persistence identity` больше
не `unknown`; переход user-scoped identities к organization scope —
`breaking-change`.

### Medium

1. `worker:notification_report_scheduler` и `worker:telegram_bot_worker` не
   имеют самостоятельного mapped `main.py`. Stage `11`/`17` должны либо создать
   явные entrypoints, либо документировать намеренное встраивание и не обещать
   отдельный service.
2. `context:ml`, `context:optimize`, `context:risk`, `core:fastpath` и
   `core:integration` являются skeleton packages. Их component IDs полезны как
   зарезервированные boundaries, но не доказывают реализованную capability.
3. Несколько current Compose images используют floating tags. Это известный
   release-security gap Stage `03`/`17`/`22`, а не допустимое target состояние.
4. Runtime wiring содержит development-only secret defaults. Значения не
   воспроизводятся в evidence; greenfield production fail-fast и новые secret
   refs должны быть проверены в `06`–`08`.

### Unknown, не блокирующие Stage 00

- точные CPU/ML performance results на macOS — владелец Stage `24`;
- final plugin wire protocol и SDK compatibility window — владелец Stage `12`;
- конкретный alert label cardinality budget — владелец Stage `20`;
- exact release image digests/SBOM — владелец Stage `22`.

Эти unknown не скрывают текущий компонент, money-moving path, secret boundary
или greenfield bootstrap owner; каждый имеет явный будущий gate.

## Validation evidence

До независимой проверки выполнено:

- `uv run python -m tools.docs.generate_project_map --check` — `passed`, 5
  generated artifacts current;
- machine comparison project-map component IDs vs plan component table —
  `passed`, 33/33, duplicates absent;
- repo search for organization, `paid_level`, Keycloak, native/Monit, Docker,
  telemetry, plugin and artifact contracts — completed;
- read-only production PostgreSQL ownership/schema aggregates — completed,
  no sensitive values emitted;
- full ownership graph — 49 FK edges clean; 51 semantic edges checked; 15
  non-zero edge results, including 3 cross-owner edges; первый проход был
  `blocked`, после `A07` это non-gating design evidence для fresh-schema
  constraints;
- `uv run python -m tools.docs.generate_project_map` and `--check` — `passed`;
- `uv run python -m tools.docs.generate_docs_index` and `--check` — `passed`;
- `git diff --check` for owned paths — `passed` after the final ledger update;
- independent read-only architecture review первого прохода — completed,
  verdict `Block`, two High findings; service-call High fixed, ownership High
  привёл к исходной остановке до решения владельца `A07`.
- greenfield prompt-pack structural check — `passed`: 26 valid YAML
  frontmatters, unique stages `00`–`25`, consistent three execution links,
  acyclic dependencies и ledger prompt-path parity;
- machine comparison current project-map IDs vs plan component table —
  `passed`, 33/33, duplicates absent;
- единственная independent read-only greenfield review — `Release after fixes`;
  три findings исправлены;
- local follow-up после greenfield fixes — project-map generation/check,
  docs-index generation/check, focused tests, whitespace scan и
  `git diff --check` прошли.

## Файловый и authority отчёт

- Создано Stage `00`:
  `docs/architecture/platform/roehub-self-hosted-oss-platform-v1-stage-reports/00-freeze-current-state-and-contracts.md`.
- Modified owned paths: plan doc, stage ledger, затронутые prompts и generated
  docs-index entry.
- Runtime/code/schema/config files: не изменялись.
- Удалённые prompt paths: `07-oidc-hybrid-and-keycloak-migration.md`,
  `23-legacy-migration-rehearsal.md`,
  `25-production-cutover-and-native-closure.md`; они заменены greenfield-
  эквивалентами с теми же Stage IDs.
- Production mutation: none.
- Git publish/deploy: не выполнялись и не разрешены.
- Foreign baseline preserved: `.codex/PLANS.md`, supersession docs и
  существующие изменения project-map/docs index не принадлежат Stage `00`.
- Mixed-file limitation: `docs/architecture/README.md` уже имел foreign
  generated changes; Stage `00` может добавить только generated index entry и
  не будет публиковать файл без отдельного scoped review.

## Историческая независимая проверка первого прохода Stage 00

- Режим: `independent subagent`, read-only, reviewer chain отсутствует.
- Первоначальный verdict: `Block`.
- Findings: incomplete ownership proof (`High`) и incomplete service-call
  contract freeze (`High`).
- Исправлено: выполнен полный aggregate-only ownership graph audit; service-call
  matrix расширена по всем обязательным dimensions; classification cells
  нормализованы к одному допустимому значению.
- Production ownership mismatches/orphans не изменялись без data authority;
  они стали причиной исторического `Block` до нового пользовательского решения.
- Итог первого прохода: `Block`.

## Независимая проверка greenfield-поправки

- Режим: ровно один `independent subagent`, read-only; reviewer chain
  отсутствует, файлы рецензентом не изменялись.
- Первоначальный verdict: `Release after fixes`.
- Найдено: противоречивый старый blocked handoff в этом отчёте (`Blocker`),
  migration clauses `A01`/`A03`, не помеченные как superseded (`High`), и
  двусмысленная миграция notification env values (`Medium`).
- Исправлено: старый `Block` оформлен как история первого прохода; `A01`/`A03`
  явно заменены `A07` в миграционной части; plan/Stage `11` требуют fresh
  provider configuration без импорта current env/tokens.
- Подтверждено рецензентом: 26 frontmatters, этапы `00`–`25`, три execution
  links, ацикличные зависимости, 33/33 компонентов, greenfield Stages
  `07`/`23`/`25`, secrets/browser/production/mainnet/proof stop gates.
- Local follow-up после исправлений: `completed`; structural pack check,
  project-map/docs generators и checks, 33/33 comparison, `4 passed`,
  whitespace scan и `git diff --check` прошли.
- Итог: `Release after fixes`; Stage `00` принят после исправлений.
- Остаточные риски: будущие runtime/platform proofs и точный service-identity
  mechanism; legacy-data migration больше не является риском или scope v1.

## Handoff

- Stage `00=accepted`, `current_stage: 01`; Stage `01` является единственным
  разрешённым следующим prompt, Stage `02` остаётся ожидающим
  последовательного ledger gate.
- Current production data не исследуется, не исправляется и не переносится.
  15 non-zero ownership edges сохранены только как требование fresh-schema
  constraints/tests для Stages `05`,`09`,`10`.
