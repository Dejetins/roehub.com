---
# =========================
# Prompt Front-Matter (YAML)
# =========================
prompt_name: codex-bkt-epic-05-execution-engine-close-fill-v1-implementation
repo: Dejetins/roehub.com
branch: main
scope: "Implement BKT-EPIC-05 end-to-end: deterministic close-fill backtest execution engine v1 (single position) with edge-based entry gating, direction modes, sizing (4 modes incl profit lock), close-based SL/TP, fee+slippage semantics, CPU-parallel scoring port, and unit tests"
language:
  implementation: python
  agent_report: ru
primary_docs:
  - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
  - docs/architecture/roadmap/milestone-4-epics-v1.md
  - docs/architecture/roadmap/base_milestone_plan.md
  - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
  - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
  - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
  - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
  - docs/architecture/api/api-errors-and-422-payload-v1.md
  - configs/prod/backtest.yaml
  - .codex/AGENTS.md
style_references:
  - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
  - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
  - src/trading/contexts/backtest/application/services/grid_builder_v1.py
  - src/trading/contexts/backtest/application/services/staged_runner_v1.py
  - src/trading/contexts/backtest/application/services/signals_from_indicators_v1.py
  - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
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
  implement_close_fill_engine_v1: true
  implement_edge_based_entry_gating: true
  implement_single_position_constraint: true
  implement_direction_modes_v1: true
  implement_sizing_modes_v1: true
  implement_profit_lock_policy_v1: true
  implement_close_based_sl_tp_v1: true
  implement_fee_slippage_semantics_v1: true
  implement_forced_close_last_bar: true
  implement_cpu_parallel_variant_scoring: true
  integrate_with_existing_staged_runner_port: true
  update_backtest_runtime_config_schema_for_execution_defaults: true
  do_not_add_api_routes: true
  do_not_implement_reporting_metrics_table: true
  update_docs_index_if_any_md_changed: true
target_envs:
  - dev
  - test
  - prod
required_literals:
  - "Total Return [%]"
  - long-only
  - short-only
  - long-short
  - all_in
  - fixed_quote
  - strategy_compound
  - strategy_compound_profit_lock
  - safe_profit_percent
  - sl_enabled
  - sl_pct
  - tp_enabled
  - tp_pct
  - fee_pct
  - slippage_pct
non_goals:
  - "Do not add apps/api routes or transport DTOs (BKT-EPIC-07)."
  - "Do not implement intrabar fills, leverage, funding, liquidation, borrow modelling (explicit non-goals)."
  - "Do not implement reporting metrics table/ratios/benchmark (BKT-EPIC-06), except the minimal deterministic Total Return [%] needed for staged ranking."
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
  - src/trading/contexts/backtest/domain/**
  - src/trading/contexts/backtest/application/services/**
  - src/trading/contexts/backtest/application/ports/**
  - src/trading/contexts/backtest/application/dto/**
  - src/trading/contexts/backtest/application/use_cases/run_backtest.py
  - src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py
  - configs/*/backtest.yaml
  - tests/unit/contexts/backtest/**
  - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
  - docs/architecture/README.md
safety_notes:
  - "Read .codex/AGENTS.md first. If it conflicts with this prompt, this prompt has priority."
  - "Keep ordering deterministic; never rely on dict insertion order or concurrency scheduling."
  - "Stay within DDD boundaries: domain/application must not import concrete adapters."
---

# Task

Fully implement the architecture defined in `docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md`.

"Done" means:

1) A deterministic close-fill execution engine v1 exists in backtest context that can run one variant over a candle timeline with:
   - single-position constraint,
   - direction modes (`long-only`, `short-only`, `long-short`),
   - sizing modes (4 modes) including profit lock policy,
   - close-based SL/TP,
   - fee+slippage semantics as specified.
2) Stage semantics are supported:
   - Stage A: run WITHOUT SL/TP (risk disabled) for shortlist.
   - Stage B: run WITH SL/TP (close-based) for final ranking.
3) Entry gating uses signal EDGE semantics:
   - a new entry is allowed only when the aggregated signal changes to LONG/SHORT;
   - persisted signals must not re-open trades after SL/TP or other exits.
4) Forced close happens on the last bar of `target_slice` (with slippage+fee).
5) A concrete scorer implementing `BacktestStagedVariantScorer` exists and uses the engine to output deterministic `Total Return [%]`.
6) The scorer supports CPU-parallel evaluation across variants (threads or processes), while the final ordering stays deterministic.
7) Backtest runtime config is extended (fail-fast) to provide defaults for:
   - init cash
   - fee/slippage defaults (spot vs futures)
   - fixed_quote default
   - safe_profit_percent default
8) Unit tests cover deterministic behavior (trade log / equity end value), fee/slippage correctness, direction modes, sizing modes, edge gating, SL/TP rules, and forced close.
9) `uv run ruff check .`, `uv run pyright`, and `uv run pytest -q` pass.

Your final report MUST be written in Russian.

## Context / Current State

- Backtest context exists with:
  - candle timeline builder (BKT-EPIC-02)
  - signals-from-indicators v1 (BKT-EPIC-03)
  - deterministic grid builder + staged runner + guards (BKT-EPIC-04)
- Current staged runner depends on a port `BacktestStagedVariantScorer` and currently uses a constant fallback scorer.
- There is no execution engine yet.
- There is no reporting module yet; BKT-EPIC-06 will later compute full metrics. For EPIC-05 you only need deterministic `Total Return [%]` used for staged ranking.

## Requirements (Must)

- Read `.codex/AGENTS.md` first.
- Implement `ExecutionEngineV1` (name may vary) under `src/trading/contexts/backtest/application/services/`.
- Implement domain entities/value objects for position/trade/accounting under `src/trading/contexts/backtest/domain/`:
  - must enforce invariants and be test-covered.
- Implement a concrete scorer adapter under backtest application layer that implements `BacktestStagedVariantScorer`:
  - must produce a mapping containing `Total Return [%]`.
  - must accept stage literal and apply risk policy:
    - stage_a => SL/TP disabled regardless of risk params
    - stage_b => SL/TP enabled according to risk params
- Integrate the scorer into composition of `RunBacktestUseCase` WITHOUT adding any API endpoints.
  - Keep backwards compatibility: the use-case constructor must still accept an injected scorer.
- Edge-based entry gating MUST be implemented as per the doc:
  - entry only on signal change to LONG/SHORT; first bar assumes prev=NEUTRAL.
- Exit/entry ordering MUST be deterministic on each bar:
  - risk exit (SL/TP) -> signal exit/entry -> forced close (if last bar)
- Direction modes MUST implement exit-only semantics for forbidden signals:
  - long-only: SHORT closes long but cannot open short
  - short-only: LONG closes short but cannot open long
- SL/TP semantics MUST be close-based with SL priority on tie.
- Fee/slippage semantics MUST follow the doc exactly:
  - buy fill uses +slippage, sell fill uses -slippage
  - fee on entry and exit
  - reversal in one bar counts as two operations (fees+slippage apply to both)
- Short accounting MUST be synthetic/margin-style and symmetric (no "cash explosion" from short sale).
- Sizing MUST support all 4 modes:
  - all_in
  - fixed_quote (capped by available)
  - strategy_compound
  - strategy_compound_profit_lock (safe_profit_percent)
- Runtime config MUST be extended in `configs/<env>/backtest.yaml` and loader must be updated fail-fast:
  - add `backtest.execution.*` defaults as per the doc proposal.
- Add/update unit tests as needed.
- All new classes/protocols MUST include docstrings with links to:
  - `docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md`
  - and relevant related files.
- Update `__init__.py` exports (`__all__`) for any new public modules.
- If any `.md` changed/added, run `python -m tools.docs.generate_docs_index` and ensure `--check` passes.

## Requirements (Should)

- Keep boundaries clean:
  - domain: position/trade/account state behaviors
  - application: execution orchestration and scorer
  - ports: still the existing `BacktestStagedVariantScorer` (do not introduce a new scorer port unless necessary)
- Prefer deterministic fakes in tests.
- Provide a deterministic "trade id" scheme (e.g., incrementing integer or bar-index-based) so results are reproducible.

## Requirements (Nice-to-have)

- Provide a tiny perf smoke test for engine scoring on a small grid (optional), ensuring it stays deterministic.

# Required reading (do first)

1) `.codex/AGENTS.md`
2) `docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md`
3) `docs/architecture/roadmap/milestone-4-epics-v1.md`
4) `docs/architecture/roadmap/base_milestone_plan.md`
5) `docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md`
6) `docs/architecture/backtest/backtest-signals-from-indicators-v1.md`
7) `docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md`
8) `src/trading/contexts/backtest/application/services/staged_runner_v1.py`
9) `src/trading/contexts/backtest/application/ports/staged_runner.py`
10) `src/trading/contexts/backtest/application/dto/run_backtest.py`
11) `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`

# Work plan (agent should follow)

1) Review current backtest staged runner/scorer port and identify required data inputs (candles, target slice, signals).
2) Design domain model for:
   - PositionV1 (direction, qty, entry fill price, entry bar index)
   - TradeV1 (open/close fills, fees, pnl)
   - AccountStateV1 (available/safe/equity accounting)
   - ExecutionParamsV1 (init cash, fee_pct, slippage_pct, sizing params)
   - RiskParamsV1 (sl/tp enabled + pct)
3) Implement deterministic engine loop operating on `target_slice`:
   - compute aggregated signal series (reusing existing signals service)
   - apply edge gating
   - apply risk exit, then signal-driven exit/entry, then forced close
4) Implement sizing modes and profit lock policy.
5) Implement scorer that returns `Total Return [%]` and can optionally return additional internal debug values for tests.
6) Add CPU-parallel scoring support for Stage A/Stage B without breaking determinism:
   - parallelize computation per variant
   - collect results and sort deterministically by keys
7) Extend `configs/*/backtest.yaml` and update loader/validator accordingly.
8) Wire the scorer into `RunBacktestUseCase` default (replace constant scorer) while still allowing injection.
9) Write unit tests:
   - edge gating prevents re-entry on persisted signals after SL/TP
   - direction modes
   - fee/slippage math
   - SL/TP tie => SL
   - forced close
   - profit lock accounting
   - deterministic results across repeated runs
10) Run quality gates and fix issues.

# Acceptance criteria (Definition of Done)

- Execution engine v1 exists and matches all semantics in the doc.
- Stage A vs Stage B risk behavior is implemented.
- Edge-based entry gating is implemented and covered by tests.
- Config defaults exist in YAML and loader validates fail-fast.
- Unit tests pass and cover key invariants.
- Quality gates pass:
  - `uv run ruff check .`
  - `uv run pyright`
  - `uv run pytest -q`

# Implementation constraints

## Determinism & ordering

- Do not rely on dict insertion order; sort keys when building any canonical payload.
- Trade IDs, ordering, and sorting must be deterministic.
- Parallel execution must not change results; always sort outputs deterministically.

## API / contracts

- Do not add API endpoints.
- Keep existing port names and semantics stable unless a change is explicitly required and documented.

## Documentation

- Keep doc links in docstrings for new public classes/protocols.
- If you change any `.md`, regenerate docs index.

## Tests

- Tests must be deterministic: no time, no randomness, no network.
- Prefer fakes over mocks.

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/domain/**`
- `src/trading/contexts/backtest/application/services/**`
- `src/trading/contexts/backtest/application/use_cases/run_backtest.py`
- `src/trading/contexts/backtest/adapters/outbound/config/backtest_runtime_config.py`
- `configs/*/backtest.yaml`
- `tests/unit/contexts/backtest/**`

# Non-goals

- No intrabar fills, leverage, funding, liquidation.
- No API routes.
- No reporting table/ratios/benchmark beyond deterministic `Total Return [%]`.

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
