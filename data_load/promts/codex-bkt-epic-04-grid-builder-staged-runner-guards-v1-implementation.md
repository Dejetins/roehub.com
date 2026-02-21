---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-bkt-epic-04-grid-builder-staged-runner-guards-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement BKT-EPIC-04 end-to-end: backtest grid builder v1 (variants generation) + staged pipeline (Stage A preselect -> Stage B expand SL/TP -> top-K) + sync guards (600k variants, 5 GiB) + deterministic ordering + unit tests"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
  - docs/architecture/roadmap/milestone-4-epics-v1.md
  - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
  - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
  - docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
  - configs/prod/indicators.yaml
  - configs/prod/backtest.yaml
  - .codex/AGENTS.md
style_references:
  - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
  - src/trading/contexts/indicators/application/services/grid_builder.py
  - src/trading/contexts/indicators/application/dto/variant_key.py
  - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
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
  implement_backtest_grid_builder_v1: true
  implement_staged_pipeline_stage_a_stage_b: true
  implement_sync_guards_variants_and_memory: true
  implement_top_k_output_policy: true
  keep_indicators_variant_key_compute_only: true
  extend_backtest_variant_key_v1_with_signals_field: true
  sl_tp_values_are_percent_numbers: true
  sl_tp_disable_is_separate_flag: true
  do_not_implement_trade_execution_engine: true
  do_not_implement_reporting_metrics_engine: true
  do_not_add_api_routes: true
  update_docs_index_if_any_md_changed: true
target_envs:
  - dev
  - test
  - prod
required_literals:
  - MAX_VARIANTS_PER_COMPUTE_DEFAULT
  - MAX_COMPUTE_BYTES_TOTAL_DEFAULT
  - "Total Return [%]"
  - top_k_default
  - preselect_default
  - stage_a
  - stage_b
  - signals
non_goals:
  - "Do not implement backtest execution loop, position sizing, fees/slippage, SL/TP fill logic, or trade simulation (BKT-EPIC-05)."
  - "Do not implement reporting/metrics table beyond the minimal ranking hook needed for deterministic sorting (BKT-EPIC-06)."
  - "Do not implement API routes (/backtests) or transport DTOs in apps/ (BKT-EPIC-07)."
  - "Do not change indicators.build_variant_key_v1 semantics; it must remain compute-only."
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
  - src/trading/contexts/backtest/application/services/**
  - src/trading/contexts/backtest/application/dto/**
  - src/trading/contexts/backtest/application/ports/**
  - src/trading/contexts/backtest/application/use_cases/run_backtest.py
  - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
  - tests/unit/contexts/backtest/**
  - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
  - docs/architecture/README.md
safety_notes:
  - "Read .codex/AGENTS.md first. If it conflicts with this prompt, this prompt has priority."
  - "Keep all ordering deterministic; never depend on dict insertion order."
  - "Prefer ports (Protocols) and dependency injection; do not import concrete adapters into application/domain."
---

# Task

Fully implement the architecture defined in `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`.

"Done" means:

1) Backtest has a deterministic Grid Builder v1 that can materialize variants from:
   - indicator compute grids (`GridSpec`: inputs/params) coming from request and/or defaults,
   - signal params grids (`defaults.<indicator_id>.signals.v1.params.*`) coming from request and/or defaults,
   - and risk axes SL/TP coming from request (Stage B only).
2) Sync guards are enforced for staged pipeline:
   - Stage A variants guard (<= 600_000)
   - Stage B variants guard (<= 600_000)
   - Memory guard based on indicators estimator policy (<= 5 GiB by default)
   - On exceed, a deterministic and machine-readable 422 (`RoehubError(code="validation_error")`) is raised.
3) The staged pipeline logic exists and is deterministic:
   - Stage A shortlist is stably sorted by `Total Return [%]` desc, tie-break by base variant key.
   - Stage B top-K is stably sorted by `Total Return [%]` desc, tie-break by full variant key.
   - Only top-K is returned (K from config default or request override).
4) Variant identity is correct and reproducible:
   - `indicators.build_variant_key_v1(...)` stays compute-only.
   - `backtest.build_backtest_variant_key_v1(...)` is extended to include a `signals` field (deterministically normalized).
5) Public exports are stable:
   - new DTOs/ports/services are exported via `__init__.py` and `__all__`.
6) Unit tests exist and cover determinism, guards, staged sorting, and variant key semantics.
7) `uv run ruff check .`, `uv run pyright`, `uv run pytest -q` pass.

Your final report MUST be written in Russian.

## Context / Current State

- Backtest context skeleton exists (BKT-EPIC-01) with `RunBacktestUseCase` and variant key builder.
- Candle timeline builder exists (BKT-EPIC-02) and provides warmup-aware minute-normalized load semantics.
- Signals-from-indicators v1 exists (BKT-EPIC-03) and defines where signal params ranges must live in `configs/<env>/indicators.yaml`.
- Indicators context already has deterministic grid materialization + estimator + guards (600k, 5 GiB), but backtest needs staged semantics and must not treat full compute*risk product as a single sync run.
- There is no API route for backtests yet; do not add it.

## Requirements (Must)

- Read `.codex/AGENTS.md` first.
- Implement a backtest Grid Builder v1 under `src/trading/contexts/backtest/application/services/`.
- Implement staged pipeline orchestration for Stage A / Stage B under backtest application layer.
  - IMPORTANT: Do not implement trade execution; instead introduce ports (Protocols) that provide `Total Return [%]` for a given variant.
  - Use fakes in unit tests to provide deterministic return values.
- Enforce guards exactly as per `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`:
  - Stage A variants guard: base grid size (compute x signals; no SL/TP)
  - Stage B variants guard: `shortlist_len * sl_variants * tp_variants` (with enable flags)
  - Memory guard: estimate based on (T bars incl warmup) + sum of per-indicator compute tensor sizes + reserve.
- Extend `src/trading/contexts/backtest/domain/value_objects/variant_identity.py`:
  - Add `signals` field into `build_backtest_variant_key_v1` payload.
  - Normalize `signals` mapping deterministically:
    - indicator ids sorted (lowercase)
    - inside each indicator: param names sorted (lowercase)
  - Keep `indicators.build_variant_key_v1` unchanged (compute-only).
- SL/TP percent semantics:
  - store values as numbers where `3.0 == 3%`.
  - allow disabling SL and/or TP via separate flags (not via special numeric values).
- Errors:
  - When guards are exceeded, raise a deterministic `RoehubError(code="validation_error")` with machine-readable `details`.
  - Include `stage` in details (`stage_a` or `stage_b`) and guard name + actual/limit values.
- Update stable exports via `__init__.py` in each affected package.
- Add/update unit tests as needed.
- Any new classes/protocols MUST have docstrings with links to:
  - `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
  - and relevant related files.
- If you change any `.md`, run `python -m tools.docs.generate_docs_index` and ensure `--check` passes.

## Requirements (Should)

- Keep boundary design clean:
  - Domain: variant identity/key canonicalization.
  - Application: staged orchestration and guards.
  - Ports: `typing.Protocol` for any external dependency (runner/scorer, config loaders).
- Reuse existing deterministic policies from indicators where appropriate (range materialization semantics).
- Provide deterministic variant_index computation via mixed-radix encoding (do not materialize the full grid just to compute indexes).

## Requirements (Nice-to-have)

- Add small helpers for debugging determinism (e.g., build deterministic variant coordinate key strings used in tests).

# Required reading (do first)

1) `.codex/AGENTS.md`
2) `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
3) `docs/architecture/backtest/backtest-signals-from-indicators-v1.md`
4) `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`
5) `docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md`
6) `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
7) `src/trading/contexts/indicators/application/dto/variant_key.py`

# Work plan (agent should follow)

1) Inspect current backtest DTOs/ports/use-case and identify where staged runner and grid builder should live.
2) Implement variant key update (signals field) first, including unit tests proving determinism and that indicators variant_key remains unchanged.
3) Implement backtest grid materialization:
   - compute variants (via `IndicatorCompute.estimate(...)` or via indicators grid builder policy)
   - signal params variants (materialize range/explicit with deterministic rules)
   - risk axes variants with enable flags
4) Implement guard enforcement services and deterministic errors.
5) Implement staged pipeline orchestration:
   - Stage A variant enumeration and shortlist selection using injected scorer port.
   - Stage B expansion along SL/TP and top-K selection using injected scorer port.
6) Update `RunBacktestUseCase` wiring to use the new staged pipeline (still without trade execution).
   - In tests, provide fake scorer that returns deterministic `Total Return [%]` values.
7) Update exports (`__init__.py`) and any impacted docs.
8) Add unit tests covering:
   - determinism of variants enumeration
   - stage guards
   - stable sorting with tie-break by variant_key
   - top-K and preselect behavior
9) Run quality gates and fix all failures.
10) If any `.md` changed, regenerate docs index and ensure it is up-to-date.

# Acceptance criteria (Definition of Done)

- Backtest grid builder v1 exists and can enumerate Stage A base variants and Stage B expanded variants deterministically.
- Guards are enforced per stage with clear deterministic 422 errors.
- Staged selection logic returns only top-K results and is deterministic for ties.
- `build_backtest_variant_key_v1` includes `signals` and is order-insensitive (sorted) while `indicators.build_variant_key_v1` remains compute-only.
- Unit tests cover the above and pass.
- Quality gates pass:
  - ruff
  - pyright
  - pytest
  - docs index check (only if `.md` changed)

# Implementation constraints

## Determinism & ordering

- Never rely on dict insertion order; always sort keys when building canonical payloads.
- Variant enumeration order must be explicitly defined and tested.
- Sorting for Stage A and Stage B MUST use stable tie-break by key (`base_variant_key` / `variant_key`).

## API / contracts

- Do not add `apps/api/routes/backtests.py` or any new API endpoints.
- Any DTO/port changes are contract changes: keep them backward compatible when possible (add optional fields with defaults).

## Documentation

- Update related docs if you changed the variant_key contract or introduced new ports/DTOs.
- If any `.md` changed/added, run `python -m tools.docs.generate_docs_index`.

## Tests

- Unit tests only; no network, no real time, no randomness.
- Prefer deterministic fakes over heavy mocks.

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/application/services/**`
- `src/trading/contexts/backtest/application/ports/**`
- `src/trading/contexts/backtest/application/dto/**`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/domain/value_objects/variant_identity.py`
- `tests/unit/contexts/backtest/**`

# Non-goals

- Do not implement trade execution, fills, SL/TP execution semantics, or full metrics/reporting.

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
