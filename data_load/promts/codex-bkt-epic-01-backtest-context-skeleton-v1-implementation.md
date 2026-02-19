---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-bkt-epic-01-backtest-context-skeleton-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement BKT-EPIC-01 end-to-end: introduce backtest bounded context v1 (domain + application + ports + use-case skeleton + runtime config + deterministic errors), following the architecture spec"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
  - docs/architecture/roadmap/milestone-4-epics-v1.md
  - docs/architecture/roadmap/base_milestone_plan.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
  - docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md
  - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
  - docs/_templates/architecture-doc-template.md
  - .codex/AGENTS.md
style_references:
  - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
  - src/trading/contexts/market_data/application/ports/stores/enabled_instrument_reader.py
  - src/trading/contexts/market_data/application/ports/stores/raw_kline_writer.py
  - src/trading/contexts/strategy/application/use_cases/get_my_strategy.py
  - src/trading/contexts/strategy/application/use_cases/errors.py
  - src/trading/platform/errors/roehub_error.py
  - apps/api/common/errors.py
  - src/trading/contexts/indicators/application/ports/feeds/candle_feed.py
  - src/trading/contexts/indicators/application/dto/compute_request.py
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
  introduce_backtest_context_skeleton: true
  use_indicators_candle_feed_port: true
  use_indicators_indicator_compute_port: true
  saved_strategy_ownership_checked_in_backtest_use_case: true
  runtime_config_backtest_yaml: true
  request_dto_full_contract_now: true
  add_variant_index_and_variant_key_v1: true
  add_backtest_error_mapping_to_roehub_error: true
  update_docs_index_if_needed: true
target_envs:
  - dev
  - test
  - prod
required_literals:
  - ROEHUB_ENV
  - ROEHUB_BACKTEST_CONFIG
  - warmup_bars_default
  - top_k_default
  - preselect_default
  - "200"
  - "300"
  - "20000"
non_goals:
  - "Do not implement BKT-EPIC-04/05/06 logic (staged grid builder, execution engine, reporting metrics)."
  - "Do not add API routes (/backtests) in this epic. API is BKT-EPIC-07."
  - "Do not modify indicators variant_key v1 semantics or hash canonicalization."
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
      - "If it fails, run: python -m tools.docs.generate_docs_index"
expected_touched_paths:
  - src/trading/contexts/backtest/**
  - configs/dev/backtest.yaml
  - configs/test/backtest.yaml
  - configs/prod/backtest.yaml
  - tests/unit/contexts/backtest/**
  - docs/architecture/README.md
safety_notes:
  - "Read .codex/AGENTS.md first. If it conflicts with this prompt, this prompt has priority."
  - "Do not commit or add any __pycache__/ or *.pyc files."
---

# Task

Implement the architecture spec `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md` fully.

This is **BKT-EPIC-01** only: create the backtest bounded context skeleton (domain + application + ports + DTOs + error mapping + runtime config loader + unit tests). Do NOT implement the staged grid builder, the execution engine, reporting metrics, or any API routes.

Your final report MUST be written in Russian.

## Context / Current State

- `src/trading/contexts/backtest/__init__.py` exists but is currently empty.
- There is no backtest domain/application skeleton yet.
- Existing building blocks:
  - `CanonicalCandleReader` (market_data port) reads sparse canonical 1m candles.
  - `CandleFeed` (indicators port) materializes a dense 1m timeline with NaN holes using `CanonicalCandleReader` via `MarketDataCandleFeed`.
  - `IndicatorCompute` (indicators port) computes indicator tensors and has established guards.
  - `StrategyRepository` and Strategy use-cases enforce ownership/visibility explicitly in use-case layer.
  - `RoehubError` + deterministic 422 payload is the canonical API error contract.

## Requirements (Must)

- Follow `.codex/AGENTS.md` DDD rules and error model.
- Create `src/trading/contexts/backtest/` with `domain/`, `application/`, `adapters/` in the established style.
- Implement a single application use-case `RunBacktestUseCase` with DTO request/response skeleton.
  - One request DTO supports both modes:
    - saved mode: `strategy_id` (and current_user)
    - ad-hoc mode: `template` (instrument_id + timeframe + indicator grid template)
- Use indicators ports:
  - candles: `trading.contexts.indicators.application.ports.feeds.CandleFeed`
  - compute: `trading.contexts.indicators.application.ports.compute.IndicatorCompute`
- For saved strategy mode, enforce ownership/visibility in the backtest use-case.
  - Introduce a backtest application port (ACL boundary) to load a strategy snapshot by id without owner filtering.
  - Perform owner/deleted checks in the backtest use-case.
- Add runtime config v1:
  - `configs/<env>/backtest.yaml` for `dev`, `test`, `prod`
  - loader/validator under `src/trading/contexts/backtest/adapters/outbound/config/`
  - config resolution: `ROEHUB_BACKTEST_CONFIG` override > `configs/<ROEHUB_ENV>/backtest.yaml` fallback (`ROEHUB_ENV` default `dev`)
  - defaults and keys:
    - `warmup_bars_default = 200`
    - `top_k_default = 300`
    - `preselect_default = 20000`
- Errors:
  - Add backtest use-case error mapping to `RoehubError` with deterministic payloads (like Strategy).
  - Use `docs/architecture/api/api-errors-and-422-payload-v1.md` contract.
- Stable exports via `__init__.py`:
  - Ensure `src/trading/contexts/backtest/__init__.py` exports the public surface you introduce.
  - Ensure subpackages use stable `__init__.py` re-exports where appropriate.
- Docstrings:
  - Protocols/classes MUST contain docstrings linking to the primary doc and related files.
- Tests:
  - Add unit tests under `tests/unit/contexts/backtest/**` to cover:
    - runtime config loader behavior and precedence
    - request DTO invariants (mutual exclusivity of saved vs template)
    - error mapping determinism (at least one or two representative cases)
- Documentation index:
  - Ensure `python -m tools.docs.generate_docs_index --check` passes.
  - If it fails, run `python -m tools.docs.generate_docs_index` and commit the updated index.

## Requirements (Should)

- Keep the backtest domain minimal but not “data bags”: enforce invariants in value objects.
- Reuse existing deterministic canonicalization patterns:
  - Use `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True)` when constructing canonical payloads.
- For variant identity, implement both:
  - `variant_index` (stable ordering position)
  - `variant_key` v1 (sha256 of canonical JSON).
  - Do NOT change indicators `build_variant_key_v1` behavior; compose on top of it.

## Requirements (Nice-to-have)

- Provide a small fake Strategy reader and fake CandleFeed/IndicatorCompute for unit tests (avoid heavy mocks).

# Required reading (do first)

1) `.codex/AGENTS.md`
2) `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md`
3) `docs/architecture/api/api-errors-and-422-payload-v1.md`
4) `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`
5) `apps/api/wiring/modules/indicators.py`
6) `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md`

# Work plan (agent should follow)

1) Read required docs and inspect existing style references (ports, errors, config loaders).
2) Create backtest package structure (`domain/`, `application/`, `adapters/`) and stable exports.
3) Implement runtime config (`backtest.yaml`) + loader/validator + tests.
4) Implement domain errors and minimal value objects + DTOs + tests.
5) Implement `RunBacktestUseCase` skeleton and application ports (CandleFeed, IndicatorCompute, Strategy reader).
6) Implement backtest error mapping to `RoehubError` (pattern from strategy) + tests.
7) Run quality gates (ruff/pyright/pytest). Fix all failures.
8) Run docs index check; update generated index if needed.

# Acceptance criteria (Definition of Done)

- `src/trading/contexts/backtest/**` exists with domain/application/adapters skeleton and stable `__init__.py` exports.
- `configs/dev/backtest.yaml`, `configs/test/backtest.yaml`, `configs/prod/backtest.yaml` exist and load successfully.
- Unit tests for backtest pass and cover at least config + DTO invariants + one error mapping.
- `uv run ruff check .`, `uv run pyright`, `uv run pytest -q` all pass.
- `python -m tools.docs.generate_docs_index --check` passes (or index updated and then check passes).

# Implementation constraints

## Determinism & ordering

- Any lists returned in error payloads MUST be deterministically sorted.
- Canonical JSON used for any hashes MUST use stable ordering and ASCII serialization (as used elsewhere in the repo).

## API / contracts (if relevant)

- Do NOT add `/backtests` API routes. Only implement application use-case + DTO contracts.
- Error codes MUST use existing canonical Roehub codes (`validation_error`, `not_found`, `forbidden`, `conflict`, `unexpected_error`).

## Documentation

- Add doc links in all protocols/classes added in `backtest`.
- Only add new docs if strictly needed; if you add any `.md`, run `python -m tools.docs.generate_docs_index`.

## Tests

- Unit tests only. Do not add integration tests (no ClickHouse/Postgres/Redis) for this epic.

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/**`
- `configs/*/backtest.yaml`
- `tests/unit/contexts/backtest/**`
- `docs/architecture/README.md`

# Non-goals

- Do not implement staged grid builder logic, CPU parallelism, execution engine, or reporting.
- Do not implement API wiring/routes.

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
