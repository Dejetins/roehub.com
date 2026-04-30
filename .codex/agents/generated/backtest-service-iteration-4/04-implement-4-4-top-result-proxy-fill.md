---
prompt_name: backtest_service_iteration_4_4_top_result_proxy_fill
repo: roehub.com
branch: current
scope: "Iteration 4.4: implement notebook-compatible top_result_proxy_fill with arity-2 proxy_for_two_rows fast path."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "proxy fill contract"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark manifest"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "no-risk service and proxy fill"
      inspect_symbols:
        - top_result_proxy_fill
        - proxy_for_indicator_rows
    - path: tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
      why: "canonical proxy_for_two_rows dispatch"
      inspect_symbols:
        - proxy_for_two_rows
        - proxy_for_indicator_rows
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      why: "proxy fill target values"
      inspect_symbols:
        - timers.top_result_proxy_fill
  conditional_bundles:
    tests:
      read_when: "updating proxy-fill tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
    previous_failure:
      read_when: "triaging arity-2 miss"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-04-27_iteration_4_no_risk_exact_scoring_notebook_topk/benchmark_summary.md
  consult_if_needed:
    - path: src/trading/contexts/backtest/application/dto/no_risk_exact.py
      read_when: "top result DTO shape is unclear"

style_references:
  - src/trading/contexts/backtest/application/services/v2/no_risk_exact.py

hard_requirements:
  arity_2_fast_path_required: true
  final_top_rows_only: true
  no_lazy_trades: true
  macstudio_acceptance_required: true
  max_implementation_attempts: 2

task_toggles:
  implement_top_result_proxy_fill: true
  implement_proxy_for_two_rows: true
  implement_exact_scoring: false
  implement_heap_update: false

skill_routing:
  - skill: numba
    use_when: "implementing proxy_for_two_rows fast path"
    timing: during implementation
    reason: "JIT scalar loop and warmup behavior"
  - skill: backend-performance-evidence
    use_when: "benchmarking top_result_proxy_fill"
    timing: during verification
    reason: "arity-specific benchmark comparison"
  - skill: backend-quality-gates
    use_when: "running local checks"
    timing: during verification
    reason: "lint/type/test gates"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "proxy_for_two_rows"
  - "proxy_for_indicator_rows"
  - "top_result_proxy_fill"
  - "arity 2"
  - "benchmark_top_k = 5"

non_goals:
  - "Do not recompute lazy trades."
  - "Do not repeat exact scoring."
  - "Do not batch-fill product top_n=100 in canonical benchmark."
  - "Do not add persisted rows or variant identity."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "Proxy-fill benchmark"
    - "Проверки"
    - "Scope guard"
    - "Следующий шаг"

quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/backtest/application/services/v2/no_risk_exact.py tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"
    expect: "passes"
  - cmd: "uv run pyright"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"
    expect: "passes"

expected_primary_touches:
  - "src/trading/contexts/backtest/application/services/v2/no_risk_exact.py"
  - "tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"

possible_secondary_touches:
  - "src/trading/contexts/backtest/application/dto/no_risk_exact.py"
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_4_top_result_proxy_fill/"

safety_notes:
  - "The previous fail was arity-2 proxy fill using generic NumPy path instead of notebook fast path."
  - "Median proxy-fill pass is insufficient; arity 2 must pass per tuple."
  - "The executor has only 2 implementation attempts; after the second failed corrective cycle, stop and report the blocker."
---

# Task

Implement Iteration 4.4: notebook-compatible `top_result_proxy_fill`, including the arity-2 compiled `proxy_for_two_rows(...)` fast path.

Done means:

- `proxy_for_indicator_rows(...)` dispatches to `proxy_for_two_rows(...)` when `len(eval_rows) == 2`;
- generic consensus path remains for arity 1 and 3..10;
- proxy recompute runs only for final heap rows with `_proxy_pending=true`;
- internal `_local_indices` and `_proxy_pending` are removed from returned top results;
- Mac Studio `top_result_proxy_fill` passes all arity 1..7 x direction mode tuples.

## Context / Current State

- Iteration 4.3 should have a passing low-allocation heap.
- Previous failed benchmark showed only arity 2 proxy-fill missed badly.
- The notebook has a special arity-2 path; the service must reproduce this dispatch.

## Requirements (Must)

- Preserve input boundary: heap already contains at most `benchmark_top_k` rows.
- Sort heap descending by heap key.
- Recompute proxy only for final rows where `_proxy_pending` is true.
- Use `proxy_for_two_rows` for arity 2.
- Do not include lazy trades, exact scoring, identity/hash assembly, persisted rows, or product `top_n=100` batch fill.
- Add tests proving:
  - only final top rows are recomputed;
  - arity 2 uses the fast path;
  - internal fields are removed from results.

## Requirements (Should)

- Keep generic path simple and close to notebook.
- Keep JIT warmup out of measured warm runtime.

## Requirements (Nice-to-have)

- Add a small direct parity test for `proxy_for_two_rows`.

# Context acquisition protocol

Read only in order: repo contract, latest state/report, task entrypoints, conditional bundles. Do not preload all docs.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Expand only for Numba typing failures, arity-2 benchmark miss, or output-shape ambiguity.

# Reading manifest

Use front-matter `context_sources`; do not duplicate it as a mandatory reading list.

# Work plan (agent should follow)

Skill routing:

- `numba`: use for `proxy_for_two_rows`.
- `backend-performance-evidence`: use before benchmark claims.
- `backend-quality-gates`: use for local gates.

1. Inspect notebook `proxy_for_two_rows` and `proxy_for_indicator_rows`.
2. Implement dispatch and cleanup of internal fields.
3. Add targeted tests.
4. Run local gates.
5. Run Mac Studio benchmark pipeline.

Mac Studio benchmark pipeline:

1. Commit local changes after gates pass.
2. Push branch/commit to remote.
3. SSH to `macstudio`.
4. In `/opt/roehub/app`, fetch and pull the pushed commit.
5. Verify commit SHA.
6. Run Iteration 4.4 benchmark for `top_result_proxy_fill`.
7. Save evidence under `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_4_top_result_proxy_fill/`.
8. Report per tuple proxy-fill ratios, especially arity 2.

# Acceptance criteria (Definition of Done)

- `top_result_proxy_fill` pass count is `14 / 14`.
- Arity 2 passes separately for `long_only` and `long_short_reversal`.
- Semantic/proxy metadata parity remains intact.
- No lazy trades or persistence logic enters measured stage.

# Implementation constraints

## Determinism & ordering

- Preserve heap ordering and top result order.

## API / contracts

- Public API: none.
- DTO schema: no breaking change.
- Persistence: none.
- Config: none.
- Request/cache identity: none.

## Benchmark

- Mac Studio only for acceptance.
- If corrected benchmark runner accounting is unavailable, do not mark accepted; route to Iteration 4.6.

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- `tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- benchmark evidence folder if run

# Non-goals

- exact scoring changes;
- heap changes except tiny adapter if required;
- result hash normalization;
- persistence/API identity;
- lazy trades.

# Quality gates (must run and pass)

- `uv run ruff check src/trading/contexts/backtest/application/services/v2/no_risk_exact.py tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- `uv run pyright`
- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- Mac Studio `top_result_proxy_fill` benchmark evidence.

# Final output: report format (strict)

Report in Russian:

1. Proxy-fill implementation summary.
2. Tests.
3. Mac Studio benchmark ratios and evidence path.
4. Contract impact classification.
5. Next step.
