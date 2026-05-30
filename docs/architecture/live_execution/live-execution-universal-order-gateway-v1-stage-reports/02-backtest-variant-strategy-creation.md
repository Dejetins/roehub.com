# Stage 02: Backtest Variant Strategy Creation

Stage 02 adds the canonical path from a launchable persisted backtest variant to an immutable `StrategySpecV1` strategy.

Date: 2026-05-30.

Status: implementation locally validated; runtime acceptance pending post-deploy boundary evidence.

## Scope

Included:

- owner-scoped backtest job and public variant lookup;
- immutable strategy creation from canonical persisted variant parameters;
- atomic strategy/provenance persistence;
- idempotent replay and duplicate source-variant handling;
- stable forbidden/not-found/not-launchable/idempotency error codes;
- `/backtests` CTA and modal wording that states no live or mainnet run is started;
- additive metric `strategy_variant_launch_total`.

Out of scope:

- no mainnet order submission;
- no testnet order submission;
- no strategy run/profile creation;
- no Redis execution stream;
- no exchange SDK/API call;
- no raw credential access.

## Prerequisite

| Requirement | Evidence | Verdict |
|---|---|---|
| Stage `01` accepted before Stage `02`. | Ledger row for `01` is `accepted`; `main` and `origin/main` were at `1c4a9e8e docs: accept live execution stage 01 baseline`; GitHub workflows for that commit were green and `scripts/macos/smoke_prod.sh` passed on `macstudio`. | Pass. |
| Work on `main`, no stage branch or PR. | `git branch --show-current` returned `main`. | Pass. |
| Runtime acceptance is not tests-only. | This report is not accepted until post-deploy API/SQL/browser proof is appended. | Pending. |

## Files Changed

Code:

- `apps/api/common/errors.py`
- `apps/api/monitoring.py`
- `apps/api/routes/backtests.py`
- `apps/api/wiring/modules/backtest.py`
- `src/trading/contexts/backtest/application/dto/backtest_jobs.py`
- `src/trading/contexts/strategy/domain/entities/strategy_backtest_variant_provenance.py`
- `src/trading/contexts/strategy/application/ports/backtest_variant_launch_reader.py`
- `src/trading/contexts/strategy/application/ports/repositories/strategy_backtest_variant_provenance_repository.py`
- `src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/in_memory/strategy_backtest_variant_provenance_repository.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_backtest_variant_provenance_repository.py`
- strategy package `__init__.py` export files touched for the new ports/use-case/adapters/entity.

Schema:

- `alembic/versions/20260530_0016_strategy_backtest_variant_provenance_v1.py`

UI:

- `apps/web/templates/pages/backtests.html`
- `apps/web/dist/js/pages/backtests.js`
- `apps/web/locales/en.json`
- `apps/web/locales/ru.json`

Tests:

- `tests/unit/contexts/strategy/application/test_strategy_use_cases.py`
- `tests/unit/apps/api/test_backtests_routes.py`

Docs:

- `docs/architecture/backtest/backtest-service-artifact-runtime-v1.md`
- `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/02-backtest-variant-strategy-creation.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md`

## Implementation

The application use-case is `CreateStrategyFromBacktestVariantUseCase`.

The use-case:

- requires a non-empty `Idempotency-Key` and stores only its SHA-256 hash;
- asks `BacktestVariantLaunchReader` for an owner-scoped launch snapshot;
- rejects non-`succeeded` jobs and malformed launch payloads as `strategy_variant_launch.not_launchable`;
- builds `StrategySpecV1` from canonical indicator params and persisted job coordinates;
- computes deterministic `strategy_spec_hash` and `launch_request_hash`;
- returns existing strategy/provenance for idempotent replay;
- returns an explicit duplicate result for same owner/job/public variant/spec;
- creates strategy plus provenance atomically through the repository port;
- appends `strategy_created_from_backtest_variant` for new strategies.

The Postgres adapter uses one transaction statement to insert `strategy_strategies` and `strategy_backtest_variant_provenance`. Unique indexes enforce:

- `(user_id, idempotency_key_hash)`;
- `(user_id, source_job_id, source_variant_key, strategy_spec_hash)`.

The API endpoint is:

```http
POST /backtests/jobs/{job_id}/variants/{variant_key}/strategies
```

The browser-facing same-origin endpoint is:

```http
POST /api/backtests/jobs/{job_id}/variants/{variant_key}/strategies
```

## Local Evidence

| Surface | Evidence | Result |
|---|---|---|
| Use-case create + provenance | `uv run pytest -q tests/unit/contexts/strategy/application/test_strategy_use_cases.py` | Passed in focused gate. |
| API route idempotency requirement and provenance response | `uv run pytest -q tests/unit/apps/api/test_backtests_routes.py` | Passed in focused gate. |
| App route wiring compatibility | `uv run pytest -q tests/unit/apps/api/test_app_strategy_router_toggle.py tests/unit/apps/api/test_backtests_routes.py` | `74 passed`. |
| UI template route load | `uv run pytest -q tests/unit/apps/web/test_app_routes.py` | Passed in focused gate. |
| Broad unit slice | `uv run pytest -q tests/unit/contexts/backtest tests/unit/contexts/strategy tests/unit/apps` | `659 passed, 3 warnings`. |
| Type checking | `uv run pyright src/trading/contexts/backtest src/trading/contexts/strategy apps/api tests` | `0 errors, 0 warnings, 0 informations`. |
| Lint | `uv run ruff check src/trading/contexts/backtest src/trading/contexts/strategy apps/api tests` | Passed. |

The three warnings are existing `httpx` per-request cookie deprecation warnings in web route tests.

## Runtime Evidence

Pending after publish/deploy:

- real HTTP `GET /backtests/jobs/{job_id}/variants/{variant_key}`;
- real HTTP `POST /backtests/jobs/{job_id}/variants/{variant_key}/strategies`;
- idempotent replay with same key;
- forbidden owner proof;
- not-launchable proof;
- SQL proof for strategy/provenance rows and no duplicate strategy on replay;
- metrics proof for `strategy_variant_launch_total`;
- browser proof for `/backtests` variant detail to create-strategy modal/action.

Until those are recorded, Stage `02` remains pending runtime acceptance.

## Error Behavior

| Case | Code | Expected behavior |
|---|---|---|
| Missing idempotency key | `strategy_variant_launch.idempotency_key_required` | `422`; no strategy/provenance row. |
| Source job belongs to another user | `strategy_variant_launch.forbidden` | `403`; no strategy/provenance row. |
| Source job or public variant key is absent | `strategy_variant_launch.not_found` | `404`; no strategy/provenance row. |
| Source job is not launchable | `strategy_variant_launch.not_launchable` | `409`; no strategy/provenance row. |
| Idempotency key reused for different launch request | `strategy_variant_launch.idempotency_key_conflict` | `409`; no second strategy. |
| Create service not configured | `strategy_variant_launch.unavailable` | `503`; route fails closed. |

## Runtime Config

No new environment variables, YAML files, feature flags or kill switches were added.

Fail-closed default:

- if `STRATEGY_PG_DSN` is absent, the create-from-variant use-case is not wired and the route returns `strategy_variant_launch.unavailable`;
- if the source job lacks launch metadata, the use-case returns `strategy_variant_launch.not_launchable`.

## Monitoring

`strategy_variant_launch_total` was added with bounded labels:

- `result`: `created`, `duplicate`, `rejected`;
- `reason`: stable reason code for duplicate/rejected paths only.

No user id, job id, variant key, strategy id, credential, cookie or raw request payload is used as a metric label.

## Logging And Redaction

No secrets, cookies, Authorization headers, API keys, private keys, passphrases, ciphertext, signed exchange payloads, raw exchange provider responses or raw idempotency keys were intentionally printed or committed.

The persisted idempotency value is a SHA-256 hash only. The public API response omits the idempotency key hash; SQL evidence will verify it without printing raw keys.

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `compatible-change` | Adds `POST /backtests/jobs/{job_id}/variants/{variant_key}/strategies` and stable `strategy_variant_launch.*` errors. |
| Persistence | `compatible-change` | Adds `strategy_backtest_variant_provenance`; existing strategy/backtest tables are unchanged except for writes through existing strategy table. |
| Redis | `none` | No stream, consumer group, dispatch, or Redis config change. |
| Config | `none` | No new env/YAML/default. |
| Runtime / ops | `compatible-change` | Adds bounded Prometheus counter. No service/process change. |
| UI / browser | `compatible-change` | Adds create-strategy CTA/modal to existing `/backtests` variant detail surface. |
| Docs | `compatible-change` | Adds Stage 02 report and updates affected backtest/strategy API docs and ledger. |

## Rollback

Safe rollback path before any later live/paper stages depend on this table:

- revert the code/UI/docs commit;
- run Alembic downgrade from `20260530_0016` if provenance rows are not needed;
- otherwise disable the route by removing `STRATEGY_PG_DSN` from the affected runtime only as an emergency fail-closed workaround, leaving immutable strategy/provenance rows as audit records.

No exchange state, Redis execution stream, order ledger, paper order or live order rollback is required for this stage.

## Next-Stage Handoff

Stage `03` can rely on these facts only after runtime acceptance is appended:

- a strategy created from a backtest variant is immutable and inert;
- provenance links owner, source job, public variant key/hash, strategy id, strategy spec hash and idempotency hash;
- duplicate replay does not create a second strategy;
- no profile/run/execution state exists until a later explicit stage creates it.
