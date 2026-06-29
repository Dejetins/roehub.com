# Stage 01: Contract Foundation And Audit Schema

Статус: `accepted`.

Дата: `2026-06-29`.

## Pre-Start

User required before start: nothing.

Ledger gate: `market-data-live-tail-repair-v1-stage-ledger.md` указывал `current_stage=01`, Stage `01` был `pending`, предыдущий обязательный stage отсутствует. Stage `02` до acceptance/delivery закрыт.

Runtime blocker source: `12-4-sustained-6h-soak.md` фиксирует, что selected run `d87917a1-1d72-49a8-b5c5-e40290bd3096` остановился после короткого Redis candle gap и ClickHouse HTTP failure. Stage `01` поэтому ограничен contract/audit foundation и не меняет `_repair_gap`.

## Что Реализовано

| Область | Итог |
|---|---|
| Strategy port | Добавлен `ClosedCandleTailProvider` в `src/trading/contexts/strategy/application/ports/closed_candle_tail_provider.py`. |
| Market Data DTO | Добавлены `ClosedCandleTailResult`, `ClosedCandleTailRow`, `CandleRepairSourceAttempt`, `ClosedCandleTailRepairPolicy`, `MarketDataCandleRepairAuditEvent`. |
| Repair source/status contract | Зафиксированы bounded source values `redis_hot_cache`, `clickhouse`, `rest` и status values `attempted`, `succeeded`, `miss`, `failed`, `circuit_open`, `rate_limited`. |
| Audit persistence port | Добавлен `CandleRepairAuditRepository` для append/read audit events. |
| Postgres adapter | Добавлен Market Data-owned `MarketDataPostgresGateway` и `PostgresCandleRepairAuditRepository`; Strategy persistence internals не импортируются. |
| Migration | Добавлена additive Alembic migration `20260629_0038_market_data_candle_repair_events_v1.py`. |
| Tests | Добавлены focused tests для fake provider contract, repository insert/read без ClickHouse и SQL migration shape/redaction. |

## Business Impact

Stage `01` does not repair production gaps by itself. Its business value is reducing implementation risk for the following repair stages: the system now has one explicit contract for "can the strategy get a continuous closed 1m tail?" and one durable audit schema for "why did repair succeed or fail?". This makes the later runtime fix reviewable, auditable, and safer for the paper/testnet strategy-producer recovery path before any live-runner behavior is changed.

## Service Calls / Ops Coverage

| Surface | Stage `01` decision |
|---|---|
| Runtime service calls | `N/A`; no caller/callee wiring is changed in this stage. |
| Auth/secrets/provider payloads | `N/A`; no exchange, ClickHouse, Redis, or external REST call is made. DTO and migration constraints only allow bounded source/status names and redacted error codes/summaries. |
| Timeout/retry/fallback behavior | `N/A`; policy fields are dormant primitives, and executable timeout/retry behavior belongs to Stages `03`-`04`. |
| Idempotency/unknown-state behavior | `N/A`; no stream ACK/checkpoint behavior changes. |
| Alerts/runbooks | `N/A`; metrics, alerts, and operator runbook are Stage `05`. |
| Rollback | Additive migration can be rolled back before production writes by dropping `market_data_candle_repair_events`; after production writes, rollback should disable later runtime writers and retain the audit table for investigation. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/strategy/application/ports/closed_candle_tail_provider.py` | none | none | Strategy-side port for later Market Data repair provider injection. | `compatible-change`; new optional port, no existing runner wiring changed. |
| `src/trading/contexts/market_data/application/dto/live_tail_repair.py` | none | none | Repair result/audit DTOs and config primitive. | `compatible-change`; new DTO surface only. |
| `src/trading/contexts/market_data/application/ports/stores/candle_repair_audit_repository.py` | none | none | Audit repository application port. | `compatible-change`; new port only. |
| `src/trading/contexts/market_data/adapters/outbound/persistence/postgres/gateway.py` | none | none | Market Data-owned Postgres gateway. | `compatible-change`; new adapter boundary. |
| `src/trading/contexts/market_data/adapters/outbound/persistence/postgres/candle_repair_audit_repository.py` | none | none | Postgres implementation of repair audit repository. | `compatible-change`; new adapter. |
| `src/trading/contexts/market_data/adapters/outbound/persistence/postgres/__init__.py` | none | none | Local exports for new adapter package. | `compatible-change`; import surface only. |
| `alembic/versions/20260629_0038_market_data_candle_repair_events_v1.py` | none | none | Additive audit table and indexes. | `compatible-change`; additive persisted schema. |
| `tests/unit/contexts/strategy/application/test_closed_candle_tail_provider_contract.py` | none | none | Fake provider tests for continuous/missing results. | `none`; tests only. |
| `tests/unit/contexts/market_data/adapters/test_postgres_candle_repair_audit_repository.py` | none | none | Repository insert/read test without ClickHouse dependency. | `none`; tests only. |
| `tests/unit/apps/migrations/test_market_data_candle_repair_events_sql.py` | none | none | Migration SQL shape/redaction check. | `none`; tests only. |
| none | `src/trading/contexts/market_data/application/dto/__init__.py` | none | Export new DTOs. | `compatible-change`; additive import surface. |
| none | `src/trading/contexts/market_data/application/ports/stores/__init__.py` | none | Export audit repository port. | `compatible-change`; additive import surface. |
| none | `src/trading/contexts/strategy/application/ports/__init__.py` | none | Export new Strategy port. | `compatible-change`; additive import surface. |
| none | `src/trading/contexts/strategy/application/__init__.py` | none | Export new Strategy port from application package. | `compatible-change`; additive import surface. |
| `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/01-contract-foundation-audit-schema.md` | none | none | Stage report and validation evidence. | `none`; documentation/evidence only. |
| none | `docs/architecture/README.md` | none | Docs index refreshed after adding Stage `01` report. | `none`; generated documentation index only. |
| none | `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md` | none | To be updated after delivery decision. | `none`; staged workflow state only. |

Files outside prompt expected paths: `tests/unit/apps/migrations/test_market_data_candle_repair_events_sql.py` is outside the prompt's context test directories, but it follows the existing migration-test convention under `tests/unit/apps/migrations`.

## Validation Evidence

| Gate | Result | Evidence |
|---|---:|---|
| Focused new tests | passed | `uv run pytest -q tests/unit/contexts/strategy/application/test_closed_candle_tail_provider_contract.py tests/unit/contexts/market_data/adapters/test_postgres_candle_repair_audit_repository.py tests/unit/apps/migrations/test_market_data_candle_repair_events_sql.py` -> `5 passed in 0.30s`. |
| Prompt `ruff` gate | passed | `uv run ruff check src/trading/contexts/market_data src/trading/contexts/strategy tests/unit/contexts/market_data tests/unit/contexts/strategy` -> `All checks passed!`. |
| Prompt `pyright` gate | passed | `uv run pyright src/trading/contexts/market_data src/trading/contexts/strategy tests/unit/contexts/market_data tests/unit/contexts/strategy` -> `0 errors, 0 warnings, 0 informations`. |
| Prompt context pytest gate | passed | `uv run pytest -q tests/unit/contexts/market_data tests/unit/contexts/strategy` -> `229 passed in 1.98s`. |
| Migration SQL test | passed | `uv run pytest -q tests/unit/apps/migrations/test_market_data_candle_repair_events_sql.py` -> `2 passed in 0.02s`. |
| Alembic head check | passed | `uv run alembic heads` -> `20260629_0038 (head)`. |
| Docs index | passed | `uv run python -m tools.docs.generate_docs_index --check` -> `OK: /Users/daniildegtyarev/Projects/roehub.com/docs/architecture/README.md is up-to-date.` |
| Publish-route repo-wide ruff | passed | `uv run ruff check .` -> `All checks passed!`. |
| Publish-route repo-wide pyright | passed | `uv run pyright` -> `0 errors, 0 warnings, 0 informations`. |
| Publish-route repo-wide pytest | passed | `uv run pytest -q -ra` -> `1448 passed, 3 warnings in 68.15s`. |

### Validation Boundary Note

Stage `01` is not a runtime behavior stage. It creates dormant contracts, DTOs, an additive migration, and an audit repository; it deliberately does not wire `StrategyLiveRunner`, Redis hot cache, ClickHouse fallback, REST tail repair, ACK policy, metrics, or production runtime behavior. Therefore `post_main_production_runtime_proof`, Redis integration proof, REST/provider proof, and browser/runtime proof are not applicable in this stage and are explicitly deferred to Stages `02`-`06`.

The nearest Stage `01` real boundary available without runtime mutation is:

| Boundary | Evidence |
|---|---|
| Application port/result boundary | Fake `ClosedCandleTailProvider` tests prove deterministic continuous and missing `ClosedCandleTailResult` representation. |
| Audit repository SQL boundary | `PostgresCandleRepairAuditRepository` insert/read test exercises the repository through a `MarketDataPostgresGateway` boundary and proves no ClickHouse read is used. |
| Migration chain boundary | Migration SQL shape/redaction test plus `uv run alembic heads` prove the new revision is the single Alembic head. |

A live Postgres mutation was not run because Stage `01` does not require production/runtime proof, no local DSN is part of the prompt contract, and secrets/DSNs must not be requested or recorded in chat. Stage `06` owns changed-code production runtime proof after main delivery.

## Audit Repository Proof

`PostgresCandleRepairAuditRepository` was exercised through a deterministic `MarketDataPostgresGateway` fake that stores `INSERT INTO market_data_candle_repair_events` parameters and serves `SELECT` reads by `event_id` and `correlation_id`. The proof inserted and read back one event with:

| Field | Value |
|---|---|
| `event_id` | `00000000-0000-0000-0000-000000003801` |
| `correlation_id` | `stage01-audit-proof` |
| `instrument_key` | `binance:spot:BTCUSDT` |
| `status` | `failed` |
| `sources_attempted` | `redis_hot_cache:miss`, `clickhouse:failed/http_connection_reset`, `rest:miss` |
| `restored_ts_opens` | `2026-06-29T12:00:00.000Z` |
| `missing_ts_opens` | `2026-06-29T12:01:00.000Z` |

The repository query log assertion confirms it does not read `canonical_candles` and does not call ClickHouse.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API routes or payloads changed. |
| Port contract | `compatible-change` | Adds `ClosedCandleTailProvider` and `CandleRepairAuditRepository`; existing ports unchanged. |
| DTO schema | `compatible-change` | Adds new DTOs only; existing DTOs unchanged. |
| Persisted schema | `compatible-change` | Adds `market_data_candle_repair_events`; no existing tables changed. |
| Config schema | `none` | `ClosedCandleTailRepairPolicy` is a code primitive only; no config loader/default changed. |
| Request hash / cache key / persistence identity | `none` | No existing identity/hash semantics changed. |
| Service-call auth/timeout/retry/error semantics | `none` | No runtime provider chain wired in Stage `01`. |
| External side effects / unknown-state semantics | `none` | No Strategy runner ACK/checkpoint behavior changed. |
| Logs/metrics/alerts/runbooks | `none` | Metrics/alerts are Stage `05`; no runtime logging changed. |
| Browser-visible behavior | `none` | Backend contract/persistence only. |
| Performance risk | `none` | No verified hot path behavior changed; new code is dormant and not wired into runtime. |

## Performance / Benchmark Applicability

Comparable benchmark evidence is `N/A` for Stage `01`: this stage makes no speed, latency, throughput, allocation, CPU, or memory improvement claim and does not alter a verified hot path. The statement above is a scope classification, not a performance result. Comparable runtime measurement becomes applicable only after later stages wire Redis hot cache/provider chain/runner behavior into an executable path.

## Delivery Status

Пользователь подтвердил, что scoped direct-main publish должен включать pre-existing untracked durable artifacts, необходимые для связной публикации Stage `01`:

- `.codex/agents/generated/market-data-live-tail-repair-v1/`;
- `docs/architecture/market_data/market-data-live-tail-repair-v1.md`;
- `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md`.

`docs/architecture/README.md` индексирует новый market-data plan/report/ledger, поэтому публикация только Stage `01` code/tests/report оставила бы broken documentation references в `origin/main`. Approved publish scope включает Stage `01` code/tests/report plus prompt-pack/plan/ledger docs и исключает foreign change `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md`.

Accepted delivery evidence для этого stage — reviewed scoped staging, direct-main commit/push, local publish gates и GitHub Actions/CI check после push. Точный commit hash фиксируется в финальном отчете исполнителя, потому что сам hash нельзя заранее записать в commit, который его создает.

## Next Stage Handoff

Stage `02` может начинаться только после Stage `01` delivery evidence в `origin/main` и зеленого publish route. Следующий executor может опираться на эти контракты:

- `ClosedCandleTailProvider.get_closed_1m_tail(...) -> ClosedCandleTailResult`;
- `ClosedCandleTailResult.continuous`, `restored_ts_opens`, `missing_ts_opens`, and `sources_attempted`;
- `CandleRepairAuditRepository.record/get_by_id/list_for_correlation`;
- Postgres audit table `market_data_candle_repair_events`.

## Residual Risks

The audit repository proof is repository-boundary unit evidence with a deterministic gateway, not a live Postgres insert against a running database. Runtime wiring, Redis hot cache, REST fallback, ACK policy, metrics, production deploy, and Stage `12.4` rerun remain later-stage work.
