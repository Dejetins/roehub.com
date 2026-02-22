---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-bkt-epic-08-tests-determinism-golden-perf-smoke-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement BKT-EPIC-08 end-to-end: deterministic golden tests for backtest execution+reporting (string table_md fixtures), additional determinism unit tests, and backtest perf-smoke tests (small sync grid within guards)"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/backtest/backtest-tests-determinism-golden-perf-smoke-v1.md
  - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
  - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
  - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
  - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
  - docs/architecture/roadmap/milestone-4-epics-v1.md
  - .codex/AGENTS.md
style_references:
  - tests/unit/contexts/backtest/application/services/test_execution_engine_v1.py
  - tests/unit/contexts/backtest/application/services/test_reporting_service_v1.py
  - tests/unit/contexts/backtest/application/services/test_staged_runner_v1.py
  - tests/perf_smoke/contexts/indicators/test_indicators_ma.py
  - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
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
  add_golden_tests_for_table_md: true
  golden_scenarios_count: 2
  golden_use_engine_plus_reporting_only: true
  add_backtest_perf_smoke: true
  add_complexity_invariant_unit_tests: true
  do_not_add_heavy_integration_tests: true
  update_docs_index_if_any_md_changed: true
target_envs:
  - test
required_literals:
  - "|Metric|Value|"
  - "Total Return [%]"
  - "Num. Trades"
  - stage_a
  - stage_b
non_goals:
  - "Do not introduce strict time-based SLAs in tests (avoid flakiness)."
  - "Do not add long integration tests requiring real ClickHouse/Postgres."
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
  - tests/unit/contexts/backtest/**
  - tests/unit/contexts/backtest/golden/**
  - tests/perf_smoke/contexts/backtest/**
  - src/trading/contexts/backtest/application/services/**
safety_notes:
  - "Golden tests must not depend on Numba/JIT; feed precomputed final_signal arrays."
  - "Golden fixtures must be stored as ASCII .md files for readable diffs."
  - "Perf-smoke should validate completion + invariants, not strict latency."
---

# Task

Fully implement the test suite described in `docs/architecture/backtest/backtest-tests-determinism-golden-perf-smoke-v1.md`.

"Done" means:

1) Golden tests exist under `tests/unit/contexts/backtest/` and compare `report.table_md` 1:1 against repository golden fixtures stored as ASCII `.md` files under `tests/unit/contexts/backtest/golden/`.
2) Exactly two golden scenarios are implemented:
   - no-trades: final_signal is all NEUTRAL
   - multi-trade: deterministic final_signal with multiple edge entries/exits producing >= 2 trades and non-trivial drawdown/ratios.
3) Golden tests run via `BacktestExecutionEngineV1` + `BacktestReportingServiceV1` only:
   - they must NOT call indicator compute or depend on Numba.
4) Add additional unit tests for determinism/complexity invariants:
   - verify Stage B risk expansion does not increase indicator compute calls (signal cache reuse) using fakes/instrumentation.
   - verify deterministic ordering is independent from dict ordering.
5) Add a backtest perf-smoke suite under `tests/perf_smoke/contexts/backtest/`:
   - constructs a small sync grid (within guards)
   - runs staged runner end-to-end using in-memory fakes (no external services)
   - asserts completion, output size, and basic invariants
   - optional: a very generous "catastrophic" time upper bound (e.g. < 60s) is allowed.
6) All tests are stable across platforms and do not rely on locale/timezone.
7) Quality gates pass.

Your final report MUST be written in Russian.

## Context / Current State

- Unit tests for backtest already exist and cover grid builder, signals v1, execution engine v1, staged runner, reporting v1, and API.
- Perf-smoke suite exists for indicators under `tests/perf_smoke/contexts/indicators/` and serves as the reference pattern (no strict SLA, just viability and shape).
- Reporting produces deterministic `report.table_md` via `BacktestReportingServiceV1` and formatter.

## Requirements (Must)

- Read `.codex/AGENTS.md` first.
- Add golden fixture files:
  - `tests/unit/contexts/backtest/golden/no-trades.md`
  - `tests/unit/contexts/backtest/golden/multi-trade.md`
  - fixtures MUST be ASCII and include the full markdown table generated by formatter.
- Add golden tests that:
  - build deterministic candles (CandleArrays)
  - supply deterministic final_signal arrays
  - run engine -> outcome
  - run reporting -> report
  - compare `report.table_md` to golden file 1:1
- Add at least one test that asserts key numeric rows (e.g., Total Return, Num. Trades) match expected strings.
- Add complexity-invariant unit tests:
  - instrument scorer/cache and assert Stage B does not recompute base signals per risk variant.
- Add backtest perf-smoke tests under `tests/perf_smoke/contexts/backtest/`.
- Keep tests deterministic:
  - fixed datetimes in UTC
  - fixed numpy arrays
  - no randomness
- Add doc links in docstrings for any new helper classes.

## Requirements (Should)

- Prefer small fixtures and keep runtime low.
- Keep golden fixtures readable and easy to diff.

## Requirements (Nice-to-have)

- Provide a helper function to regenerate golden fixtures locally (optional, not required).

# Required reading (do first)

1) `.codex/AGENTS.md`
2) `docs/architecture/backtest/backtest-tests-determinism-golden-perf-smoke-v1.md`
3) `src/trading/contexts/backtest/application/services/execution_engine_v1.py`
4) `src/trading/contexts/backtest/application/services/reporting_service_v1.py`
5) `src/trading/contexts/backtest/application/services/staged_runner_v1.py`
6) `tests/perf_smoke/contexts/indicators/test_indicators_ma.py`

# Work plan (agent should follow)

1) Implement deterministic candle + signal fixtures for golden scenarios.
2) Run engine+reporting to generate expected table_md strings once, store them into `.md` golden files.
3) Add golden tests reading those files and asserting exact match.
4) Add complexity-invariant unit tests (e.g., compute call counters, cache hit ratios).
5) Add backtest perf-smoke tests modeled after indicators perf-smoke.
6) Run quality gates and fix all failures.

# Acceptance criteria (Definition of Done)

- `uv run pytest -q` passes consistently across repeated runs.
- Golden tests fail on any change in table formatting or metric formulas.
- Perf-smoke tests complete and validate basic invariants without strict latency SLA.
- ruff, pyright, pytest pass.

# Implementation constraints

## Determinism & ordering

- Do not use random or time-dependent inputs.
- Always use UTC timezone-aware datetimes.
- Ensure string comparisons are exact and newline-normalized (\n).

## API / contracts

- Do not change public API contracts; this epic adds tests only.

## Documentation

- Do not add new docs unless necessary; if you do, regenerate docs index.

## Tests

- Keep fixtures small and runtime low.
- Do not require external services.

# Files to indicate (expected touched areas)

- `tests/unit/contexts/backtest/golden/**`
- `tests/unit/contexts/backtest/**`
- `tests/perf_smoke/contexts/backtest/**`

# Non-goals

- No strict time SLAs.
- No heavy integration tests.

# Quality gates (must run and pass)

- `uv run ruff check .`
- `uv run pyright`
- `uv run pytest -q`

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Результат**

2) **Изменённые файлы**

3) **Ключевые решения**

4) **Как проверить**

5) **Риски / что дальше**
