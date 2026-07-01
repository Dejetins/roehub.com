---
prompt_name: backtest_service_iteration_4_2_exact_scoring_self_check
repo: roehub.com
branch: current
scope: "Iteration 4.2: implement no-risk exact scoring kernels, full metrics, and self-check without heap/proxy-fill optimization."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and verification rules"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "Iteration 4.2 contract"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark evidence manifest"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "4.1 service boundary"
      inspect_symbols:
        - BacktestNoRiskExactScoringService
    - path: src/trading/contexts/backtest/application/services/v2/combo_planning.py
      why: "backend registry and combo contexts"
      inspect_symbols:
        - BacktestComboPlanningService
        - BacktestExactBackend
    - path: tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
      why: "canonical algorithm source"
      inspect_symbols:
        - evaluate_no_risk_exact_chunk
        - run_fast_vs_reference_self_check_two
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      why: "canonical target values"
      inspect_symbols:
        - runs
        - timers
  conditional_bundles:
    dto_and_tests:
      read_when: "updating exact result DTOs or test fixtures"
      paths:
        - src/trading/contexts/backtest/application/dto/no_risk_exact.py
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
    numba_context:
      read_when: "implementing or debugging Numba kernels"
      paths:
        - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
        - src/trading/contexts/backtest/application/dto/prepare_pools.py
    accepted_prior_evidence:
      read_when: "building Mac Studio benchmark record"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-04-27_iteration_3_combo_planning_contexts/benchmark_results.json
        - docs/architecture/backtest/benchmark_iterations/2026-04-27_iteration_3_combo_planning_contexts/benchmark_summary.md
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_summary.md
      read_when: "human-readable target details are needed"

style_references:
  - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
  - src/trading/contexts/backtest/application/services/v2/combo_planning.py

hard_requirements:
  exact_scoring_only: true
  no_heap_optimization_scope: true
  no_proxy_fill_scope: true
  macstudio_acceptance_required: true
  max_implementation_attempts: 2

task_toggles:
  implement_exact_kernels: true
  implement_self_check: true
  implement_full_no_risk_metrics: true
  implement_heap_update: false
  implement_top_result_proxy_fill: false

skill_routing:
  - skill: numba
    use_when: "writing or debugging JIT kernels"
    timing: during implementation
    reason: "Numba kernel typing and warmup behavior"
  - skill: backend-performance-evidence
    use_when: "measuring exact_scoring/self_check or claiming acceptance"
    timing: during verification
    reason: "stage comparability and Mac Studio evidence"
  - skill: backend-quality-gates
    use_when: "running local checks"
    timing: during verification
    reason: "backend lint/type/test gates"
  - skill: contract-impact-analysis
    use_when: "changing DTO or telemetry fields"
    timing: before implementation
    reason: "protect internal and benchmark contracts"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "event_segments_2_no_risk"
  - "event_segments_n_no_risk"
  - "streaming_2_no_risk"
  - "exact_scoring"
  - "self_check"
  - "benchmark_top_k = 5"

non_goals:
  - "Do not optimize heap_update in this prompt."
  - "Do not implement top_result_proxy_fill in this prompt."
  - "Do not add public variant identity or persistence."
  - "Do not complete all public sizing modes or profit_lock parity; Iteration 8 owns full sizing/profit_lock completion unless behavior already exists."
  - "Do not mark accepted without Mac Studio evidence."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "Mac Studio benchmark"
    - "Contract impact"
    - "Проверки"
    - "Ограничения / следующий шаг"

quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/backtest/application/services/v2 src/trading/contexts/backtest/application/dto tests/unit/contexts/backtest/application/services/v2"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"
    expect: "passes"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/services/v2/no_risk_exact.py"
  - "src/trading/contexts/backtest/application/dto/no_risk_exact.py"
  - "tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/services/v2/__init__.py"
  - "src/trading/contexts/backtest/application/dto/__init__.py"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_2_exact_scoring_self_check/"

safety_notes:
  - "Exact scoring has canonical Mac Studio target values; local timing is not acceptance evidence."
  - "If corrected benchmark runner accounting is unavailable, do not claim acceptance; report runner blocker and route to Iteration 4.6."
  - "The executor has only 2 implementation attempts; after the second failed corrective cycle, stop and report the blocker."
---

# Task

Implement Iteration 4.2: no-risk exact scoring kernels, full no-risk metrics, and fast-vs-reference self-check. Keep `heap_update`, `top_result_proxy_fill`, result hash normalization, persistence, and API identity out of scope.

Done means:

- `event_segments_2_no_risk`, `event_segments_n_no_risk` for arity 1..10, and `streaming_2_no_risk` fallback/parity comparator exist;
- no-risk full summary metrics are computed;
- self-check compares fast exact scoring with a slow generic reference on bounded candidates;
- Mac Studio evidence records `self_check` and `exact_scoring` versus canonical target for arity 1..7 x direction mode.

## Context / Current State

- Iteration 4.1 should have created the boundary. If it is absent, implement only the minimal boundary needed and say so.
- Canonical benchmark compares `exact_scoring` per `{arity, risk_mode, direction_mode, backend}`.
- Previous failed Iteration 4 attempt showed `exact_scoring` can pass; do not copy reverted code as-is, but use that evidence to focus the stage.

## Requirements (Must)

- Match canonical notebook semantics for no-risk trade execution:
  - `long_only`;
  - `long_short_reversal`;
  - fees, slippage, initial cash, and the currently supported sizing behavior needed by the canonical fixture.
- Implement metrics:
  - `total_return_pct`;
  - `max_drawdown_pct`;
  - `return_over_max_drawdown`;
  - `profit_factor`;
  - `trade_count`;
  - `sharpe_trades`;
  - `win_rate_pct`;
  - `avg_trade_ret_pct`;
  - `avg_trade_exec_bars`;
  - `exposure_pct`.
- Record exact backend display/logical name separately from implementation id where needed.
- Preserve `benchmark_top_k = 5` as telemetry only; exact scoring should not depend on public `top_n`.
- Add unit tests for backend dispatch, direction semantics, and self-check pass/fail.

## Requirements (Should)

- Keep kernels array-first and object-free.
- Keep Python orchestration outside jitted hot loops.
- Warm up JIT before benchmark timing.

## Requirements (Nice-to-have)

- Add a tiny parity fixture for arity 1, 2, and 3 if it stays maintainable.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest executor report/state if available
3. task entrypoints
4. conditional bundles required by implementation or failing checks
5. consult-if-needed references only for blockers

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Expand only for Numba typing blockers, benchmark threshold conflicts, or contract ambiguity.

# Reading manifest

Use front-matter `context_sources`. Do not create a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing:

- `numba`: use while implementing kernels.
- `backend-performance-evidence`: use before any performance or acceptance claim.
- `backend-quality-gates`: use for local checks.
- `contract-impact-analysis`: use if DTO/telemetry fields change.

1. Confirm 4.1 boundary and no public/persistence contract changes.
2. Port notebook exact scoring semantics for no-risk into service kernels.
3. Add self-check against slow reference.
4. Add/adjust tests.
5. Run local gates.
6. Run Mac Studio benchmark pipeline if corrected runner exists; otherwise stop with blocker.

Mac Studio benchmark pipeline:

1. Commit scoped local changes after local gates pass.
2. Push branch/commit to the remote repository.
3. SSH to `macstudio`.
4. In `/opt/roehub/app`, fetch and pull the same branch/commit with fast-forward semantics.
5. Verify Mac Studio checkout commit equals the pushed commit.
6. Run the Iteration 4.2 benchmark command/runner for `self_check` and `exact_scoring` only.
7. Save `benchmark_results.json` and `benchmark_summary.md` under `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_2_exact_scoring_self_check/`.
8. Copy/commit the evidence back in the repo.

# Acceptance criteria (Definition of Done)

- Local tests pass.
- Mac Studio evidence exists or final report explicitly says acceptance is pending due to runner/deploy blocker.
- `exact_scoring` ratio is `>= 0.9` for arity 1..7 x both direction modes.
- `self_check` ratio is `>= 0.9` for arity 1..7 x both direction modes where the canonical timer is present.
- Self-check passes for all measured tuples.
- No heap/proxy/persistence/API identity work is included.

# Implementation constraints

## Determinism & ordering

- Keep candidate order deterministic.
- Do not change combo planning order from Iteration 3.

## API / contracts

- Public API: none.
- DTO schema: compatible internal additions only.
- Persisted schema: none.
- Config schema: none.
- Request/cache identity: none.

## Benchmark

- Only Mac Studio benchmark is acceptance evidence.
- Compare equivalent stage boundaries only.
- Do not compare `service_total_without_warmup` to canonical `total_without_warmup`.

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- `src/trading/contexts/backtest/application/dto/no_risk_exact.py`
- `tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- benchmark evidence folder if Mac Studio run is completed

# Non-goals

- heap optimization;
- top result proxy fill;
- result hash normalization;
- persistence;
- public API endpoints.

# Quality gates (must run and pass)

- `uv run ruff check src/trading/contexts/backtest/application/services/v2 src/trading/contexts/backtest/application/dto tests/unit/contexts/backtest/application/services/v2`
- `uv run pyright`
- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- Mac Studio benchmark evidence for `exact_scoring` and `self_check`, if runner is available.

# Final output: report format (strict)

Report in Russian:

1. Что реализовано.
2. Stage boundaries touched.
3. Contract impact classification.
4. Local checks.
5. Mac Studio benchmark: commit, path, ratios, pass/fail.
6. Blockers or next sub-iteration.
