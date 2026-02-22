---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-bkt-epic-07-api-post-backtests-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement BKT-EPIC-07 end-to-end: FastAPI endpoint POST /backtests (saved strategy + ad-hoc grid) with typed request/response DTOs, owner-only auth via identity current_user, deterministic top-K output, reproducibility hashes, and unified deterministic 422 errors"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/backtest/backtest-api-post-backtests-v1.md
  - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
  - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
  - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
  - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
  - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
  - docs/architecture/roadmap/base_milestone_plan.md
  - configs/prod/backtest.yaml
  - .codex/AGENTS.md
style_references:
  - apps/api/routes/strategies.py
  - apps/api/routes/indicators.py
  - apps/api/common/errors.py
  - apps/api/wiring/modules/strategy.py
  - apps/api/wiring/modules/indicators.py
  - src/trading/contexts/backtest/application/use_cases/run_backtest.py
hard_requirements:
  keep_fail_fast_on_startup: true
  do_not_break_public_imports: true
  deterministic_ordering_everywhere: true
  stable_exports_via_init: true
  add_or_update_unit_tests_as_needed: true
  docstrings_must_link_to_docs_and_related_files: true
  maintain_existing_variant_key_v1_semantics: true
task_toggles:
  read_codex_agents_first: true
  implement_api_route_post_backtests: true
  implement_api_dto_for_backtests: true
  implement_backtest_wiring_module: true
  implement_strategy_acl_adapter_for_backtest: true
  implement_grid_defaults_provider_adapter: true
  include_reproducibility_hashes_in_response: true
  enforce_owner_only_saved_mode: true
  allow_top_trades_n_override: true
  do_not_add_async_jobs: true
  update_docs_index_if_any_md_changed: true
target_envs:
  - dev
  - test
  - prod
required_literals:
  - "/backtests"
  - "POST /backtests"
  - "Total Return [%]"
  - "validation_error"
  - "not_found"
  - "forbidden"
  - "conflict"
  - spec_hash
  - grid_request_hash
  - engine_params_hash
non_goals:
  - "Do not implement new backtest compute logic (grid/engine/reporting) beyond API wiring."
  - "Do not add any other endpoints besides POST /backtests."
  - "Do not implement async jobs/progress (Milestone 5)."
final_report_format:
  language: ru
  sections:
    - "Результат"
    - "Изменённые файлы"
    - "Ключевые решения"
    - "Как проверить"
    - "Риски / что дальше"
quality_gates:
  - cmd: "uv run ruff check ."
    expect: "exit code 0"
  - cmd: "uv run pyright"
    expect: "exit code 0"
  - cmd: "uv run pytest -q"
    expect: "exit code 0"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect:
      - "exit code 0"
      - "run only if any .md file was added/changed"
expected_touched_paths:
  - apps/api/routes/backtests.py
  - apps/api/routes/__init__.py
  - apps/api/dto/**
  - apps/api/wiring/modules/backtest.py
  - apps/api/wiring/modules/__init__.py
  - apps/api/main/app.py
  - src/trading/contexts/backtest/adapters/**
  - tests/unit/apps/api/**
  - tests/unit/contexts/backtest/**
safety_notes:
  - "Backtest use-case must stay the source of truth; API route must remain thin."
  - "All 422 payloads must be deterministic and consistent with apps/api/common/errors.py contract."
---

# Task

Fully implement the API described in `docs/architecture/backtest/backtest-api-post-backtests-v1.md`.

"Done" means:

1) A new FastAPI route `POST /backtests` exists at `apps/api/routes/backtests.py` and is included into the app in `apps/api/main/app.py`.
2) The route supports two modes using one envelope (no explicit `mode` field):
   - saved: `strategy_id` provided, `template` missing
   - ad-hoc: `template` provided, `strategy_id` missing
3) Auth:
   - Endpoint is protected by identity dependency (authenticated user required).
   - Saved-mode ownership is enforced by backtest use-case (not in route SQL).
4) Typed API request models exist under `apps/api/dto/backtests.py` (or similar) and map deterministically into backtest application DTOs.
5) Response includes:
   - top-K variants only (K default from config, override from request)
   - deterministic sorting (return desc, variant_key asc)
   - reporting payload already produced by backtest (report rows/table/trades policy)
   - explicit variant payload (indicator selections + signals + risk + execution + direction/sizing) sufficient for UI to save StrategySpec.
   - reproducibility hashes: `spec_hash` or `grid_request_hash`, plus `engine_params_hash`.
6) Errors:
   - unified deterministic 422 payload via RoehubError and existing handlers.
   - route converts known mapping/validation errors into RoehubError(validation_error) or raises HTTPException(422) with deterministic detail.
7) Wiring:
   - A new API wiring module exists: `apps/api/wiring/modules/backtest.py`.
   - It wires CandleFeed, IndicatorCompute, BacktestStrategyReader ACL adapter, BacktestGridDefaultsProvider adapter, BacktestRuntimeConfig (configs/<env>/backtest.yaml).
   - Fail-fast behavior applies at startup (similar to indicators/strategy modules).
8) Unit tests exist for:
   - request validation determinism (extra fields forbidden, mode exclusivity)
   - unauthorized behavior
   - saved-mode forbidden/not_found mapping
   - response includes hashes and payload fields
   - deterministic ordering of variants in response for ties
9) Quality gates pass.

Your final report MUST be written in Russian.

## Context / Current State

- Backtest use-case exists and can run end-to-end when wired with CandleFeed, IndicatorCompute, StrategyReader ACL, and defaults providers.
- Strategy/Identity/Indicators API modules exist and demonstrate patterns for:
  - typed pydantic DTO mapping
  - wiring modules
  - deterministic error handling
- There is currently no backtests API route.

## Requirements (Must)

- Read `.codex/AGENTS.md` first.
- Create `apps/api/routes/backtests.py` with `build_backtests_router(...)`.
- Add typed request/response models in `apps/api/dto/backtests.py` and export them via `apps/api/dto/__init__.py`.
- Route must forbid extra fields (pydantic Config extra="forbid" style).
- Implement deterministic mapping:
  - time_range, instrument_id, timeframe
  - grid specs (`explicit` / `range`)
  - execution fields and percent semantics
  - signal grids mapping
  - risk grid mapping with enable flags
  - top_k/preselect/warmup_bars/top_trades_n overrides
- Saved mode overrides MUST support signal grids (2B decision).
- Implement reproducibility hashes:
  - `spec_hash` when saved mode
  - `grid_request_hash` when ad-hoc
  - `engine_params_hash` always
  - Hashes must be sha256 of canonical JSON with sorted keys and stable separators.
- Response must include per-variant explicit payload required for saving strategy.
- Add `apps/api/wiring/modules/backtest.py` and include it in `apps/api/wiring/modules/__init__.py` and `apps/api/wiring/__init__.py`.
- Update `apps/api/main/app.py` to include backtests router and wire runtime dependencies.
- Ensure unified deterministic errors:
  - prefer raising RoehubError from use-case; rely on `apps/api/common/errors.py` handler.
  - when route needs to throw 422 directly, ensure payload is deterministic.
- Add unit tests:
  - should not require DB/clickhouse; use fakes/in-memory adapters.

## Requirements (Should)

- Keep API layer thin: mapping + call use-case + mapping response.
- Avoid importing concrete adapters into application/domain.
- Follow existing strategies/indicators route style.

## Requirements (Nice-to-have)

- Add a minimal contract test that the error payload for missing required fields is sorted/deterministic.

# Required reading (do first)

1) `.codex/AGENTS.md`
2) `docs/architecture/backtest/backtest-api-post-backtests-v1.md`
3) `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md`
4) `docs/architecture/api/api-errors-and-422-payload-v1.md`
5) `apps/api/common/errors.py`
6) `apps/api/routes/strategies.py`
7) `apps/api/routes/indicators.py`
8) `apps/api/wiring/modules/strategy.py`
9) `apps/api/wiring/modules/indicators.py`
10) `src/trading/contexts/backtest/application/use_cases/run_backtest.py`

# Work plan (agent should follow)

1) Implement typed pydantic DTOs for request/response with extra="forbid" and discriminators (where needed).
2) Implement converters from API DTOs to backtest application DTOs.
3) Implement response converters from backtest DTOs to API response payload.
4) Implement reproducibility hash helpers (canonical JSON -> sha256).
5) Implement wiring module for backtest and include it into app factory.
6) Implement `POST /backtests` route with identity dependency.
7) Add unit tests for route behavior and deterministic payload ordering.
8) Run quality gates.

# Acceptance criteria (Definition of Done)

- `POST /backtests` works in both saved and ad-hoc modes with typed DTOs.
- Errors are deterministic and match RoehubError contract.
- Response includes top-K only, reports, explicit variant payload, and hashes.
- Unit tests exist and pass.
- Quality gates pass.

# Implementation constraints

## Determinism & ordering

- Always sort keys when building canonical JSON for hashes.
- Ensure response variants are sorted deterministically even when scoring ties.

## API / contracts

- Do not introduce any additional endpoints.
- Do not change existing backtest/strategy/identity contracts unless required; if changed, update docs and tests.

## Documentation

- Add doc links in docstrings for new routers/DTOs/wiring modules.
- If any `.md` is changed/added, run docs index generator.

## Tests

- Unit tests only, deterministic.
- Use FastAPI TestClient (or starlette test client) with in-memory fakes.

# Files to indicate (expected touched areas)

- `apps/api/routes/backtests.py`
- `apps/api/routes/__init__.py`
- `apps/api/dto/backtests.py`
- `apps/api/dto/__init__.py`
- `apps/api/wiring/modules/backtest.py`
- `apps/api/wiring/modules/__init__.py`
- `apps/api/wiring/__init__.py`
- `apps/api/main/app.py`
- `src/trading/contexts/backtest/adapters/**`
- `tests/unit/apps/api/**`

# Non-goals

- No async jobs.
- No additional endpoints.

# Quality gates (must run and pass)

- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -q`
- `python -m tools.docs.generate_docs_index --check`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Результат**

2) **Изменённые файлы**

3) **Ключевые решения**

4) **Как проверить**

5) **Риски / что дальше**
