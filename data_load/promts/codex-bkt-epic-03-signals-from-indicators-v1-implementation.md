---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-bkt-epic-03-signals-from-indicators-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement BKT-EPIC-03 end-to-end: indicator signal rules v1 (LONG/SHORT/NEUTRAL) for all prod indicators + AND aggregation + deterministic behavior + unit tests"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
  - docs/architecture/roadmap/milestone-4-epics-v1.md
  - docs/architecture/roadmap/base_milestone_plan.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
  - docs/architecture/indicators/indicators-overview.md
  - docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
  - configs/prod/indicators.yaml
  - .codex/AGENTS.md
style_references:
  - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
  - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
  - src/trading/contexts/backtest/application/use_cases/run_backtest.py
  - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
  - src/trading/contexts/indicators/application/dto/variant_key.py
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
  implement_signal_rule_engine_v1: true
  implement_rules_for_all_prod_indicator_ids: true
  implement_and_aggregation: true
  nan_and_warmup_is_neutral: true
  use_primary_output_only_from_indicator_compute: true
  allow_candle_series_in_rules: true
  implement_delta_rule_with_per_direction_lag: true
  do_not_build_rich_dsl: true
  add_unit_tests_for_each_rule_family: true
  update_docs_index_if_any_md_changed: true
target_envs:
  - prod
required_literals:
  - LONG
  - SHORT
  - NEUTRAL
  - conflicting_signals
non_goals:
  - "Do not implement staged runner, position management, fees/slippage, SL/TP, or trade execution."
  - "Do not implement API routes (/backtests)."
  - "Do not modify indicators compute engine outputs or variant_key v1 semantics."
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
  - src/trading/contexts/backtest/domain/**
  - src/trading/contexts/backtest/application/use_cases/run_backtest.py
  - tests/unit/contexts/backtest/**
  - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
  - docs/architecture/README.md
safety_notes:
  - "Read .codex/AGENTS.md first. If it conflicts with this prompt, this prompt has priority."
  - "Keep the solution deterministic and testable; avoid hidden global state."
---

# Task

Implement `docs/architecture/backtest/backtest-signals-from-indicators-v1.md` fully.

Done means:

1) Backtest has a signal rule engine v1 that can compute per-indicator signals `LONG|SHORT|NEUTRAL` for any indicator_id present in `configs/prod/indicators.yaml`.
2) Backtest can aggregate multiple indicator signals using AND policy into `final_long/final_short`, with deterministic conflict handling.
3) NaN/warmup semantics are enforced: any missing input (NaN) or insufficient history yields `NEUTRAL`.
4) Rule families from the doc are implemented:
   - compare_price_to_output (uses selected `source` where applicable, else close)
   - threshold_band (parameterized)
   - sign
   - delta_sign with per-direction lags (`long_delta_periods` and `short_delta_periods` in UI-config form as negative integers)
   - compare_volume_to_output
   - candle_body_direction
   - pivot_events (requires wrapper dependency handling for `structure.pivot_low` and `structure.pivot_high`)
5) Unit tests exist and cover:
   - each rule family on small deterministic arrays
   - AND aggregation + conflict resolution
   - delta semantics (negative periods meaning lookback)
   - determinism (stable ordering does not change output)

Your final report MUST be written in Russian.

## Context / Current State

- Backtest bounded context exists (BKT-EPIC-01) and has timeline builder (BKT-EPIC-02).
- There is no signals subsystem yet under backtest.
- Indicators compute returns only one primary output per indicator_id (float32 tensor).
- `configs/prod/indicators.yaml` defines parameter ranges and steps for indicators; signal parameters must be grid-configurable as well.
- We must NOT implement a full signal DSL; use rule families.

## Requirements (Must)

- Read `.codex/AGENTS.md` first.
- Implement a backtest signal subsystem under `src/trading/contexts/backtest/application/services/`.
- Add domain types:
  - an enum or Literal for `SignalV1` with `LONG|SHORT|NEUTRAL`.
  - DTOs for per-indicator signals and aggregated signals.
- Implement per-indicator rule evaluation based on the catalog in the doc, covering all indicator_ids in `configs/prod/indicators.yaml`.
- Implement AND aggregation:
  - `final_long = all == LONG`
  - `final_short = all == SHORT`
  - if both true, return NEUTRAL + increment/report `conflicting_signals` counter in output metadata.
- Enforce NaN/warmup semantics -> NEUTRAL.
- Delta rule semantics:
  - UI/config supplies negative ints like -1..-10; use `N = abs(value)`.
  - `delta(x,N) = x[t] - x[t-N]`.
  - insufficient history -> NEUTRAL.
- compare_price_to_output semantics:
  - if indicator has `inputs.source`, compare selected source series on that bar against output.
  - else compare `close` against output.
- pivot_events:
  - When a strategy includes `structure.pivots`, the signal evaluator must ensure it can access both pivot_low and pivot_high events.
  - Use wrapper ids `structure.pivot_low` and `structure.pivot_high` as compute dependencies.
- Update stable exports via `__init__.py`.
- Add unit tests in `tests/unit/contexts/backtest/**`.
- Keep deterministic ordering everywhere.
- Docstrings in classes/protocols MUST link to the primary doc and related files.

## Requirements (Should)

- Provide a clean separation:
  - rule evaluation should be pure and independent from compute engine where possible (accept numpy arrays).
  - orchestration that fetches primary outputs from `IndicatorCompute` can be a thin wrapper.
- Provide a clear mapping table (dict) from indicator_id -> rule family + required inputs.

## Requirements (Nice-to-have)

- Add a small “rule registry” debug helper that can list supported indicator_ids and their rule family.

# Required reading (do first)

1) `.codex/AGENTS.md`
2) `docs/architecture/backtest/backtest-signals-from-indicators-v1.md`
3) `configs/prod/indicators.yaml`
4) `docs/architecture/api/api-errors-and-422-payload-v1.md`
5) `src/trading/contexts/indicators/application/ports/compute/indicator_compute.py`
6) `src/trading/contexts/backtest/application/use_cases/run_backtest.py`

# Work plan (agent should follow)

1) Inspect indicator ids in `configs/prod/indicators.yaml` and the backtest request/template DTOs to see how indicators are represented.
2) Implement the signal types + rule evaluation functions (pure functions first).
3) Implement rule registry mapping for all indicator_ids.
4) Implement AND aggregation and conflict handling.
5) Implement orchestration glue (if needed) to map indicator compute outputs + candle series into the pure evaluator.
6) Add comprehensive unit tests for rule families + aggregation.
7) Run `ruff`, `pyright`, `pytest` and fix all issues.
8) If any `.md` changed, regenerate docs index.

# Acceptance criteria (Definition of Done)

- There is a signal engine v1 implementation in backtest and it supports every `indicator_id` in `configs/prod/indicators.yaml`.
- All rule families documented are implemented and unit-tested.
- AND aggregation is deterministic and conflict handling matches the doc.
- `uv run ruff check .`, `uv run pyright`, `uv run pytest -q` pass.

# Implementation constraints

## Determinism & ordering

- Any iteration over indicator ids must sort deterministically.
- Avoid reliance on dict insertion order.

## API / contracts (if relevant)

- Do not add API routes.
- Do not change indicator compute outputs.

## Documentation

- Do not change docs unless required to clarify implementation details.
- If you change/add any `.md`, run `python -m tools.docs.generate_docs_index`.

## Tests

- Unit tests only; no integration tests.

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/application/services/**`
- `src/trading/contexts/backtest/domain/**`
- `tests/unit/contexts/backtest/**`

# Non-goals

- No staged runner, no execution engine, no metrics/reporting, no API.

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
