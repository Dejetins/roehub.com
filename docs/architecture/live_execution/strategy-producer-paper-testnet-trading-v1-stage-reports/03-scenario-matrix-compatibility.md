# Stage 03 — Scenario matrix and compatibility

Дата: 2026-06-17

## Pre-Start

User required before start: nothing

Stage `02` проверен в `strategy-producer-paper-testnet-trading-v1-stage-ledger.md` до implementation edits: статус `accepted`, `Next stage allowed = yes`, активных blockers нет.

## Pre-Edit Scope

Ожидаемые broad paths были сужены до конкретных файлов до implementation edits:

| Planned file | Planned action | Reason |
|---|---|---|
| `src/trading/contexts/strategy/application/use_cases/scenario_matrix.py` | create | Построить per-variant scenario matrix из текущих contracts: modes, entry sizing, risk mode, direction, compatibility/readiness. |
| `src/trading/contexts/strategy/application/ports/repositories/scenario_matrix_repository.py` | create | Добавить порт durable записи matrix report без привязки application layer к Postgres. |
| `src/trading/contexts/strategy/adapters/outbound/persistence/in_memory/scenario_matrix_repository.py` | create | Unit-test adapter и dev-safe repository. |
| `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/scenario_matrix_repository.py` | create | Durable Postgres upsert для matrix rows and reason codes. |
| `alembic/versions/20260617_0032_strategy_variant_scenario_matrix_v1.py` | create | Additive SQL table for scenario matrix evidence. |
| `apps/api/routes/backtests.py` | modify | Additive API endpoint for current top/available variant scenario matrix. |
| `apps/api/wiring/modules/backtest.py` | modify | Wire scenario matrix service and optional Postgres persistence. |
| `apps/api/routes/strategies.py` | modify | Reuse shared Stage 03 constants for Stage 02 launch validation. |
| `src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py` | modify | Preserve canonical MA-cross signal fields needed by compatibility checker. |
| strategy package `__init__.py` export files | modify | Export new use case/port/adapters through existing package boundaries. |
| `tests/unit/contexts/strategy/application/test_strategy_use_cases.py` | modify | Compatibility/readiness and scenario matrix unit coverage. |
| `tests/unit/apps/api/test_backtests_routes.py` | modify | Real FastAPI route call against public top variant key. |
| `tests/unit/apps/migrations/test_strategy_variant_scenario_matrix_sql.py` | create | SQL migration contract test. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/03-scenario-matrix-compatibility.md` | create | Stage report, evidence, manifest. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify | Stage status/evidence/handoff. |
| `docs/architecture/README.md` | modify/check | Docs index after new report. Existing unrelated `docs/architecture/ml/` index changes were already present before this stage's docs edit. |

`apps/api/routes/ui_backtests.py` был expected entrypoint for UI variant availability inspection, but implementation did not need to change it: the new API endpoint uses the existing public `variant_key` exposed by `/backtests/jobs/{job_id}/top`.

## Matrix Contract

Scenario combinations are discovered from current contracts, not guessed:

| Dimension | Source |
|---|---|
| `mode` | Stage scope constants: `paper`, `testnet`. |
| `entry_sizing` | `LiveStrategyProfileSizingMethod` literal values: `fixed_quote`, `fixed_equity_pct`. |
| `risk_mode` | Stage 02 launch contract: `single_position_cap`. |
| `direction` | `canonical_variant_params.execution.direction_mode`: `long_only` -> `long`; `long_short_reversal` -> `long`, `short`; unknown defaults fail narrow to `long`. |
| `market_type` / `symbol` | Actual `BacktestVariantLaunchSnapshot` from the selected variant. |
| compatibility/readiness | `StrategyCompatibilityReadinessService.check_backtest_variant(...)`. |

Durable row identity is `scenario_key = sha256(schema, source_job_id, source_variant_key, variant_hash, strategy_spec_hash, mode, market_type, symbol, entry_sizing, risk_mode, direction, backtest_risk_mode, backtest_direction_mode)`.

## Matrix Rows Covered

For a live-compatible `BTCUSDT` variant with `backtest_direction_mode=long_short_reversal`, the service produces `8` rows per source market type: `2 modes x 2 entry sizing x 1 launch risk x 2 directions`.

| Source market | Mode | Direction | Entry sizing rows | Scenario state | Scenario reason | Order capability | Capability reason |
|---|---|---:|---:|---|---|---|---|
| `spot` | `paper` | `long` | 2 | `launchable` | `paper_no_exchange_submit` | `paper_only` | `paper_no_exchange_submit` |
| `spot` | `paper` | `short` | 2 | `launchable` | `paper_no_exchange_submit` | `paper_only` | `spot_short_not_real_order_capable` |
| `spot` | `testnet` | `long` | 2 | `blocked` | `exchange_connection_required` | `real_order_capable` | `testnet_order_path_supported_when_exchange_ready` |
| `spot` | `testnet` | `short` | 2 | `blocked` | `spot_short_not_supported` | `unsupported` | `spot_short_not_supported` |
| `futures` | `paper` | `long` | 2 | `launchable` | `paper_no_exchange_submit` | `paper_only` | `paper_no_exchange_submit` |
| `futures` | `paper` | `short` | 2 | `launchable` | `paper_no_exchange_submit` | `paper_only` | `paper_no_exchange_submit` |
| `futures` | `testnet` | `long` | 2 | `blocked` | `exchange_connection_required` | `real_order_capable` | `testnet_order_path_supported_when_exchange_ready` |
| `futures` | `testnet` | `short` | 2 | `blocked` | `exchange_connection_required` | `real_order_capable` | `futures_short_requires_isolated_1x_guard` |

Current fake API top variant in `tests/unit/apps/api/test_backtests_routes.py` is intentionally not live-compatible (`ma.dema` evaluator shape), so the real FastAPI route test proves the public top variant key still returns `8` rows and blocks ordinary rows with `unsupported_live_evaluator`; `testnet` spot-short keeps the higher-priority explicit `spot_short_not_supported` reason.

## API / SQL Evidence

| Surface | Evidence |
|---|---|
| API endpoint | `GET /backtests/jobs/{job_id}/variants/{variant_key}/scenario-matrix` returns `BacktestVariantScenarioMatrixResponse` with row-level state, reason codes, order capability, compatibility check id, and market-data readiness id. |
| API top variant call | `test_get_backtest_variant_scenario_matrix_uses_public_top_variant_key` creates a backtest job, completes it in the fake repository, fetches `/top`, then calls `/scenario-matrix` using the public `variant_key`; response has `8` rows. |
| Compatibility/readiness call | `StrategyVariantScenarioMatrixService.build_for_backtest_variant` calls `StrategyCompatibilityReadinessService.check_backtest_variant` before row generation; tests cover launchable MA-cross snapshots and unsupported top-variant evaluator snapshots. |
| SQL durable artifact | Alembic migration `20260617_0032` creates `strategy_variant_scenario_matrix_rows` with JSONB reason-code arrays, row hash checks, mode/sizing/risk/direction constraints, BTCUSDT scope, and unique upsert key. |
| Persistence adapter | `PostgresStrategyVariantScenarioMatrixRepository.record` upserts every row when `STRATEGY_PG_DSN` is configured; without DSN the API still exposes the matrix and does not fake persistence. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/strategy/application/use_cases/scenario_matrix.py` | none | none | New use case for per-backtest-variant matrix rows, stable reason codes, order capability, compatibility/readiness join, and API JSON serialization. | compatible-change |
| `src/trading/contexts/strategy/application/ports/repositories/scenario_matrix_repository.py` | none | none | New repository port for durable matrix recording. | compatible-change |
| `src/trading/contexts/strategy/adapters/outbound/persistence/in_memory/scenario_matrix_repository.py` | none | none | In-memory matrix repository for focused tests. | none |
| `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/scenario_matrix_repository.py` | none | none | Postgres upsert adapter for durable matrix rows. | compatible-change |
| `alembic/versions/20260617_0032_strategy_variant_scenario_matrix_v1.py` | none | none | Additive matrix table and indexes. | compatible-change |
| `tests/unit/apps/migrations/test_strategy_variant_scenario_matrix_sql.py` | none | none | SQL text contract for the additive migration. | none |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/03-scenario-matrix-compatibility.md` | none | none | Stage evidence/report. | none |
| none | `apps/api/routes/backtests.py` | none | Additive scenario matrix route and response DTOs for selected job variant. | compatible-change |
| none | `apps/api/wiring/modules/backtest.py` | none | Wires matrix service and optional Postgres repository behind existing backtest router composition. | compatible-change |
| none | `apps/api/routes/strategies.py` | none | Reuses Stage 03 constants for existing Stage 02 launch allowlists; no broadening of launch behavior. | none |
| none | `src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py` | none | Preserves canonical `signal_template`, `fast`, and `slow` so MA-cross variants can pass existing live compatibility without loosening the checker. | compatible-change |
| none | strategy package export `__init__.py` files | none | Exposes new use case/port/adapters through existing public package boundaries. | compatible-change |
| none | `tests/unit/contexts/strategy/application/test_strategy_use_cases.py` | none | Adds spot/futures matrix coverage and representative reason-code assertions. | none |
| none | `tests/unit/apps/api/test_backtests_routes.py` | none | Adds FastAPI route test against public top variant key. | none |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Stage status/evidence handoff. | none |
| none | `docs/architecture/README.md` | none | Docs index update for this new stage report; note the existing `docs/architecture/ml/` index changes predated this stage and are not Stage 03 implementation work. | none |

No files were deleted.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API / DTO | compatible-change | Additive `GET /backtests/jobs/{job_id}/variants/{variant_key}/scenario-matrix`; existing endpoints unchanged. |
| DTO schema | compatible-change | New response models only; no existing response fields removed or retyped. |
| Persisted schema | compatible-change | Additive Alembic table and indexes; no existing table rewrite. |
| Port contract | compatible-change | New optional scenario matrix repository port; existing ports unchanged. |
| Config schema | none | No new required env; `STRATEGY_PG_DSN` reuses existing strategy Postgres configuration. |
| Request hash / cache / persistence identity | compatible-change | New `scenario_key` identity only; existing backtest request hash, variant hash, and strategy launch hash remain unchanged. |
| Service-call auth / timeout / retry | none | No new external service call; uses existing in-process readiness service. |
| External side effects / idempotency | none | No exchange submit, no mainnet, no account mutation. Postgres row write is idempotent by unique key. |
| Logs / metrics / audit / redaction | compatible-change | Adds non-sensitive row/reason-code evidence; no secrets, tokens, cookies, API keys, signed payloads, or raw provider payloads. |
| Alert / runbook | none | Stage 05+ owns exchange/account operational guards. |
| Benchmark / hot path | none | Matrix generation is API/readiness path only, not backtest compute or producer hot path. |
| Browser-visible behavior | none | No UI surface changed in this stage. |

## Validation

| Command | Result |
|---|---|
| `python -m compileall -q apps/api/routes/backtests.py apps/api/routes/strategies.py apps/api/wiring/modules/backtest.py src/trading/contexts/strategy/application/use_cases/scenario_matrix.py src/trading/contexts/strategy/application/use_cases/create_strategy_from_backtest_variant.py src/trading/contexts/strategy/application/ports/repositories/scenario_matrix_repository.py src/trading/contexts/strategy/adapters/outbound/persistence/in_memory/scenario_matrix_repository.py src/trading/contexts/strategy/adapters/outbound/persistence/postgres/scenario_matrix_repository.py` | passed |
| `uv run pytest -q tests/unit/contexts/strategy/application/test_strategy_use_cases.py` | `13 passed` |
| `uv run pytest -q tests/unit/apps/api/test_backtests_routes.py::test_get_backtest_variant_scenario_matrix_uses_public_top_variant_key tests/unit/apps/migrations/test_strategy_variant_scenario_matrix_sql.py` | `2 passed` |
| `uv run pytest -q tests/unit/contexts/strategy/application/test_strategy_use_cases.py::test_scenario_matrix_derives_spot_rows_and_blocks_testnet_spot_short tests/unit/contexts/strategy/application/test_strategy_use_cases.py::test_scenario_matrix_marks_futures_short_real_order_capable_but_not_bound` | `2 passed` |
| `uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/backtest tests/unit/apps` | `2 failed, 751 passed, 3 warnings in 39.91s`; local environment has `IDENTITY_FAIL_FAST=true` without `KEYCLOAK_BASE_URL`, causing `tests/unit/apps/api/test_app_strategy_router_toggle.py::{test_create_app_includes_strategy_router_when_enabled,test_create_app_skips_strategy_router_when_disabled}` to fail during app import. |
| `IDENTITY_FAIL_FAST=false uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/backtest tests/unit/apps` | `753 passed, 3 warnings in 41.06s` |
| `uv run ruff check src/trading/contexts/strategy src/trading/contexts/backtest apps tests` | passed after import ordering auto-fix in touched files. |
| `uv run pyright src/trading/contexts/strategy src/trading/contexts/backtest apps tests` | `0 errors, 0 warnings, 0 informations` |
| `python -m tools.docs.generate_docs_index --check` | passed after docs index regeneration. |

## Delivery Status

| Item | Status |
|---|---|
| Local implementation | complete |
| Local API/SQL/unit/docs gates | complete |
| Runtime/API boundary | complete: backend deploy applied migrations and smoke passed. |
| GitHub publish prerequisite | complete: `gh version 2.85.0`; `gh auth status` authenticated as `Dejetins` with `repo` and `workflow` scopes. |
| Main delivery / origin main | complete: direct `main` commit `ab01c4fd38f09a3d0a1acb304adc766a7ab5cc75` (`Add strategy scenario matrix`) pushed to `origin/main`; no temporary branch or PR was used, so branch cleanup is `N/A`. |
| CI | complete: [CI run `27652037267`](https://github.com/Dejetins/roehub.com/actions/runs/27652037267) succeeded for `ab01c4fd38f09a3d0a1acb304adc766a7ab5cc75`. |
| Deploy Backend | complete: [Deploy Backend run `27652195850`](https://github.com/Dejetins/roehub.com/actions/runs/27652195850) succeeded; it synced backend source, built runtime, ran DB bootstrap/migrations, reloaded production launchd surface, and passed backend smoke. |
| Publish App Image | complete: [Publish App Image run `27652195870`](https://github.com/Dejetins/roehub.com/actions/runs/27652195870) succeeded. |
| Deploy Web | complete: [Deploy Web run `27652195856`](https://github.com/Dejetins/roehub.com/actions/runs/27652195856) succeeded. |
| Mac Studio checkout sync | complete: `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded from `8918ff72fb9112ad45c44f124a642c31d3a8d9d3` to `ab01c4fd38f09a3d0a1acb304adc766a7ab5cc75`; `git status -sb` reported `## main...origin/main`. |
| Mac Studio deploy smoke | complete: `bash scripts/macos/smoke_prod.sh` exited `0`; service list included `com.roehub.api`, `com.roehub.backtest-job-runner`, `com.roehub.exchange-control`, and `com.roehub.exchange-execution`; unauthenticated `/auth/current-user` returned expected `401 missing_session_id`; Redis returned `PONG`; Tailscale backend state was `Running`. |
| Runtime SQL | complete: production Postgres reports `alembic_version=20260617_0032`, table `strategy_variant_scenario_matrix_rows`, and constraints `strategy_variant_scenario_matrix_capability_chk`, `strategy_variant_scenario_matrix_mode_chk`, `strategy_variant_scenario_matrix_symbol_chk`, `strategy_variant_scenario_matrix_unique_row`. |

Stage `03` is accepted. `mainnet` remains out of scope and no exchange submit path was added.

## Blockers / Residual Risk

| Risk | Status | Handoff |
|---|---|---|
| Local broad pytest command with default env | Environmental/pre-existing local config issue: `IDENTITY_FAIL_FAST=true` without `KEYCLOAK_BASE_URL`; env-adjusted suite and GitHub CI passed. | Not an acceptance blocker for Stage `03`; keep local shell env aligned before using the exact prompt pytest command as a local green signal. |
| Current fake API top variant | Matrix exposes it as blocked by `unsupported_live_evaluator`; this is expected until the top variant uses a supported live evaluator shape. | MA-cross variants with `signal_template`, `fast`, and `slow` are covered by live-compatible unit snapshots. |
| Testnet rows | Still blocked by `exchange_connection_required`; futures short is marked real-order-capable but not launchable. | Stage `05` owns exchange binding and isolated `1x` futures guard proof. |
| Spot short | Explicitly unsupported for testnet real orders. | Later stages must not fake spot-short real orders unless a separate margin/borrow product is implemented and proven. |
| `docs/architecture/ml/` worktree changes | Pre-existing unrelated untracked docs/index changes were present before Stage 03 docs edits. | Do not stage or revert them as Stage 03 work. |

## Next Handoff

- Stage `04` may start from `main` commit `ab01c4fd38f09a3d0a1acb304adc766a7ab5cc75` or later.
- Stage `05` should consume `order_capability=real_order_capable` futures short rows but must still prove exchange connection, account/config guard, isolated margin, leverage `1x`, precision, min notional, and balance projection before any real testnet submit.
- Stage `07` can use `paper_only` rows as full paper coverage targets, including spot-short as paper-only/not-real-order-capable.
