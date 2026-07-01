---
prompt_name: backtest_service_iteration_4_3_low_allocation_heap_update
repo: roehub.com
branch: current
scope: "Iteration 4.3: implement notebook-compatible low-allocation heap_update for no-risk top-K."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "heap_update stage contract"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "benchmark manifest"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "no-risk exact service"
      inspect_symbols:
        - heap_update
        - BacktestNoRiskExactScoringService
    - path: tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
      why: "canonical heap_update algorithm"
      inspect_symbols:
        - search_topk_indicator_no_risk
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      why: "heap_update target values"
      inspect_symbols:
        - runs
        - timers.heap_update
  conditional_bundles:
    tests:
      read_when: "updating top-K tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
    previous_failure:
      read_when: "triaging a heap benchmark miss"
      paths:
        - docs/architecture/backtest/benchmark_iterations/2026-04-27_iteration_4_no_risk_exact_scoring_notebook_topk/benchmark_summary.md
  consult_if_needed:
    - path: src/trading/contexts/backtest/application/dto/no_risk_exact.py
      read_when: "telemetry or score DTO fields are unclear"

style_references:
  - src/trading/contexts/backtest/application/services/v2/no_risk_exact.py

hard_requirements:
  benchmark_top_k_5: true
  low_allocation_heap: true
  no_variant_identity: true
  macstudio_acceptance_required: true
  max_implementation_attempts: 2

task_toggles:
  implement_heap_update: true
  implement_exact_kernels: false
  implement_proxy_fill: false
  implement_persistence: false

skill_routing:
  - skill: backend-performance-evidence
    use_when: "benchmarking or changing hot Python top-K code"
    timing: before implementation
    reason: "stage boundary and target comparison"
  - skill: production-risk-review
    use_when: "reviewing diff before pushing benchmark candidate"
    timing: before ship
    reason: "catch scope creep into identity/persistence"
  - skill: backend-quality-gates
    use_when: "running local checks"
    timing: during verification
    reason: "lint/type/test gates"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "heap_update"
  - "benchmark_top_k = 5"
  - "request.top_n = 100"
  - "target_heap_update / service_heap_update >= 0.9"

non_goals:
  - "Do not change exact scoring kernels unless a test proves heap integration requires a tiny adapter."
  - "Do not implement proxy fill."
  - "Do not build public variant_key, variant_hash, DTO read models, or persisted rows."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "Heap benchmark"
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
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_3_heap_update/"

safety_notes:
  - "The previous miss was Python object churn in heap_update."
  - "Build full item metadata only for retained heap candidates."
  - "The executor has only 2 implementation attempts; after the second failed corrective cycle, stop and report the blocker."
---

# Task

Implement Iteration 4.3: low-allocation, notebook-compatible `heap_update` for no-risk top-K.

Done means:

- canonical heap capacity is `benchmark_top_k=5`, sample warmup uses `top_k=1`;
- public `request.top_n=100` is telemetry/input only and does not drive measured heap work;
- default ranking `total_return_pct desc` uses direct score-array reads in the hot loop;
- full item and metadata materialization happens only for candidates that enter/replace heap;
- Mac Studio heap benchmark passes `>=0.9` for all arity 1..7 x direction modes.

## Context / Current State

- Iteration 4.2 should provide exact score arrays and metrics.
- Previous failed benchmark showed `heap_update` failed `13 / 14`.
- The plan allows low-allocation implementation as long as ranking, tie-break, cardinality, and final top results match notebook.

## Requirements (Must)

- Preserve deterministic heap key `(rank_score, original_row_ids)`.
- Preserve top result ordering and identity versus notebook.
- Do not allocate full `item` dict or metadata mapping for rejected candidates.
- Do not call generic string-dispatch ranking in the default benchmark path.
- Keep non-default ranking support only if it does not slow default measured path.
- Add tests proving top-K capacity separation: `request.top_n=100`, `benchmark_top_k=5`.
- Add tests proving metadata is materialized only for retained heap rows, if practical.

## Requirements (Should)

- Keep Python top-K code readable and auditable.
- Prefer small helpers only if they do not add hot-loop overhead.

## Requirements (Nice-to-have)

- Add a small micro smoke for rejected-candidate metadata avoidance, marked as non-acceptance.

# Context acquisition protocol

Read only in order: `.codex/AGENTS.md`, latest state/report if available, task entrypoints, conditional tests, consult-if-needed references. Do not preload broad docs.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Expand only for benchmark miss triage or contract ambiguity.

# Reading manifest

Use front-matter `context_sources`; do not duplicate it as a required reading list.

# Work plan (agent should follow)

Skill routing:

- `backend-performance-evidence`: use before implementation and during benchmark reporting.
- `production-risk-review`: use before push to catch scope creep.
- `backend-quality-gates`: use for local verification.

1. Inspect current heap implementation and notebook heap loop.
2. Implement low-allocation heap admission path.
3. Preserve exact output shape expected by later proxy fill.
4. Add focused tests.
5. Run local gates.
6. Run Mac Studio benchmark pipeline.

Mac Studio benchmark pipeline:

1. Commit local changes after local gates pass.
2. Push branch/commit to remote.
3. SSH to `macstudio`.
4. In `/opt/roehub/app`, fetch and pull the pushed commit.
5. Verify commit SHA on Mac Studio.
6. Run Iteration 4.3 heap benchmark with canonical fixture and corrected runner accounting.
7. Save evidence under `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_3_heap_update/`.
8. Record `request.top_n`, `benchmark_top_k`, `top_results_count`, heap capacity, per tuple ratios.

# Acceptance criteria (Definition of Done)

- `heap_update` pass count is `14 / 14`.
- `target_heap_update / service_heap_update >= 0.9` for every arity 1..7 x direction mode.
- Top result identity/order remains notebook-compatible.
- No service-only identity/persistence/API work is inside `heap_update`.
- Local gates pass.

# Implementation constraints

## Determinism & ordering

- Keep deterministic ordering for equal scores via original row ids.

## API / contracts

- Public API: none.
- DTO schema: no breaking change.
- Persistence: none.
- Config: none.
- Request/cache identity: none.
- Benchmark gate: compatible measured-stage refinement.

## Benchmark

- Mac Studio only for acceptance.
- Do not compare service-only totals.
- If the runner still compares `service_total_without_warmup` to canonical total, stop and route to Iteration 4.6.

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- `tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- benchmark evidence folder if run

# Non-goals

- exact scoring changes;
- proxy fill;
- result hash normalization;
- persistence/API identity;
- `top_n=100` production budget.

# Quality gates (must run and pass)

- `uv run ruff check src/trading/contexts/backtest/application/services/v2/no_risk_exact.py tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- `uv run pyright`
- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- Mac Studio `heap_update` benchmark evidence.

# Final output: report format (strict)

Report in Russian:

1. Heap implementation summary.
2. Output/parity guarantees.
3. Local checks.
4. Mac Studio benchmark table summary and evidence path.
5. Contract impact classification.
6. Next step.
