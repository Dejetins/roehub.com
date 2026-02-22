---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-bkt-epic-06-reporting-metrics-table-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement BKT-EPIC-06 end-to-end: deterministic reporting v1 for backtests (equity curve + trade log + metrics table |Metric|Value|) including benchmark return and ratios (Sharpe/Sortino/Calmar) with stable formatting, for all top-K variants"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
  - docs/architecture/roadmap/base_milestone_plan.md
  - docs/architecture/roadmap/milestone-4-epics-v1.md
  - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
  - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
  - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
  - configs/prod/backtest.yaml
  - .codex/AGENTS.md
style_references:
  - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
  - src/trading/contexts/backtest/application/services/execution_engine_v1.py
  - src/trading/contexts/backtest/domain/entities/execution_v1.py
  - src/trading/contexts/backtest/application/services/staged_runner_v1.py
  - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
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
  implement_reporting_service_v1: true
  implement_equity_curve_builder_v1: true
  implement_metrics_table_v1: true
  implement_benchmark_return_v1: true
  implement_ratios_resample_1d_v1: true
  implement_deterministic_formatting_v1: true
  return_trades_only_for_top_n_variants: true
  update_runtime_config_reporting_defaults: true
  do_not_add_api_routes: true
  update_docs_index_if_any_md_changed: true
target_envs:
  - dev
  - test
  - prod
required_literals:
  - "|Metric|Value|"
  - "Total Return [%]"
  - "Benchmark Return [%]"
  - "Sharpe Ratio"
  - "Sortino Ratio"
  - "Calmar Ratio"
  - "Max. Drawdown [%]"
  - "Avg. Drawdown Duration"
  - top_trades_n_default
non_goals:
  - "Do not add apps/api routes or transport DTOs (BKT-EPIC-07)."
  - "Do not change execution semantics (engine) or grid/guards semantics from EPIC-04/05."
  - "Do not implement asynchronous jobs/history (Milestone 5)."
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
  - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
  - configs/*/backtest.yaml
  - tests/unit/contexts/backtest/**
  - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
  - docs/architecture/README.md
safety_notes:
  - "Read .codex/AGENTS.md first. If it conflicts with this prompt, this prompt has priority."
  - "Keep formatting deterministic; never use pandas Series.__repr__ or locale-dependent formatting."
  - "Avoid heavy dependencies in hot paths; reporting runs only for top-K variants."
---

# Task

Fully implement the architecture defined in `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md`.

"Done" means:

1) For each Stage B top-K variant, backtest returns a deterministic reporting payload containing:
   - metrics table rows in the exact fixed order,
   - optionally a markdown table string `|Metric|Value|...` derived from those rows,
   - and trades only for the best N variants (configurable default).
2) Metrics are computed deterministically on `target_slice` only:
   - equity curve on bar closes,
   - drawdown + drawdown durations,
   - coverage and gross exposure,
   - trade stats,
   - Expectancy and SQN (backtesting.py-compatible semantics),
   - benchmark buy&hold return (no fee/slippage),
   - Sharpe/Sortino/Calmar ratios (equity resample to 1d, risk_free=0, annualization=365).
3) Value formatting is deterministic and stable (no platform/locale drift).
4) Runtime config is extended with reporting defaults (fail-fast): `backtest.reporting.top_trades_n_default`.
5) Unit tests exist and cover:
   - metric ordering,
   - deterministic formatting,
   - benchmark and ratio calculations on small synthetic equity curves,
   - edge cases (no trades, too-short period for ratios).
6) Quality gates pass.

Your final report MUST be written in Russian.

## Context / Current State

- Execution engine v1 exists and produces deterministic trades and final equity.
- Staged runner returns only top-K variants and currently returns only `total_return_pct` preview.
- Reporting layer does not exist yet.
- `configs/<env>/backtest.yaml` already includes execution defaults; reporting defaults must be added.

## Requirements (Must)

- Read `.codex/AGENTS.md` first.
- Implement reporting services under `src/trading/contexts/backtest/application/services/`:
  - equity curve builder v1
  - metrics calculator v1
  - table formatter v1
- Extend response DTOs to include the reporting payload for each returned variant (top-K):
  - keep backwards compatibility when possible (add optional fields).
- Ensure the metrics table output uses exactly the fixed metric list and order from the doc.
- Benchmark:
  - buy&hold long, no fee/slippage,
  - entry at `close[first_target_bar]`, exit at `close[last_target_bar]`.
- Ratios:
  - resample equity to 1d (UTC) using last
  - daily returns via pct_change
  - risk_free=0, annualization=365
- Expectancy and SQN:
  - use the backtesting.py-compatible formulas described in the doc.
- Formatting:
  - deterministic precision and trailing-zero trimming
  - `N/A` for undefined
- Trades:
  - always compute trades for each top-K variant to compute metrics
  - include full trade list in response only for top N variants (`top_trades_n_default`)
- Runtime config:
  - add `backtest.reporting.top_trades_n_default` to `configs/dev|test|prod/backtest.yaml`
  - update loader/validator fail-fast in `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- Add/update unit tests.
- Update stable exports via `__init__.py` and `__all__`.
- Docstrings for new public types MUST link to the EPIC-06 doc and relevant files.
- Do not add API endpoints.
- If any `.md` is changed/added, run `python -m tools.docs.generate_docs_index`.

## Requirements (Should)

- Keep reporting isolated from execution logic:
  - engine remains source of truth for trades
  - reporting consumes trades + candles + target_slice + execution params
- Prefer numpy for numeric loops; pandas MAY be used only for 1d resample if it stays deterministic.

## Requirements (Nice-to-have)

- Add a tiny helper to output a canonical markdown table string used in golden tests.

# Required reading (do first)

1) `.codex/AGENTS.md`
2) `docs/architecture/backtest/backtest-reporting-metrics-table-v1.md`
3) `docs/architecture/roadmap/base_milestone_plan.md`
4) `src/trading/contexts/backtest/application/services/execution_engine_v1.py`
5) `src/trading/contexts/backtest/domain/entities/execution_v1.py`
6) `src/trading/contexts/backtest/application/services/staged_runner_v1.py`
7) `src/trading/contexts/backtest/application/dto/run_backtest.py`
8) `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`

# Work plan (agent should follow)

1) Inspect current response DTOs and staged runner output; design minimal backwards-compatible report fields.
2) Implement reporting domain/application types:
   - `BacktestMetricRowV1` (name, value string)
   - `BacktestReportV1` (rows + optional table_md + optional trades)
3) Implement equity curve computation on `target_slice` from trades + candles.
4) Implement metrics calculator for the fixed list.
5) Implement deterministic formatter and markdown table builder.
6) Integrate reporting into staged runner so that:
   - Stage B runs produce top-K variants
   - For each returned variant, build a report
   - Include trades only for top N
7) Extend runtime config with reporting defaults.
8) Write unit tests:
   - deterministic formatting
   - metric order
   - benchmark return correctness
   - ratios on a small deterministic series
   - drawdown durations detection
   - no-trades edge cases
9) Run quality gates.

# Acceptance criteria (Definition of Done)

- For a deterministic synthetic backtest run, report table rows match expected fixed order and stable strings.
- Benchmark and ratios match the documented method.
- Trades are included only for top N variants in response.
- Config loader validates reporting defaults fail-fast.
- All tests pass and quality gates pass.

# Implementation constraints

## Determinism & ordering

- Metric rows must be emitted in the exact fixed order.
- Formatting must be deterministic and platform-independent.
- If using pandas resample, ensure timezone/UTC handling is explicit and deterministic.

## API / contracts

- Do not add API endpoints.
- Add optional fields to DTOs rather than breaking existing fields.

## Documentation

- If you modify any docs, regenerate docs index.

## Tests

- Unit tests only; deterministic.
- Prefer numpy-based assertions.

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/application/services/**`
- `src/trading/contexts/backtest/application/dto/**`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- `configs/*/backtest.yaml`
- `tests/unit/contexts/backtest/**`

# Non-goals

- No API routes.
- No async jobs/history.

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
