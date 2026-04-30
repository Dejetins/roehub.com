---
prompt_name: backtest_service_iteration_4_5_result_shape_hash_parity
repo: roehub.com
branch: current
scope: "Iteration 4.5: align no-risk top result shape, ordering, serialization, and hash/parity with notebook evidence."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md
      why: "result shape/hash contract"
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      why: "evidence manifest"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "top result production"
      inspect_symbols:
        - top_result_proxy_fill
        - BacktestNoRiskExactScoringService
    - path: src/trading/contexts/backtest/application/dto/no_risk_exact.py
      why: "result DTO shape"
      inspect_symbols:
        - BacktestNoRiskExactResult
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-26_engine_test_btcusdt_15m/benchmark_results.json
      why: "canonical hash/result fields"
      inspect_symbols:
        - result_hash
        - top_results
  conditional_bundles:
    notebook_shape:
      read_when: "top result field order or float formatting is unclear"
      paths:
        - tests/notebook_tests/engine_test/btcusdt_15m_research_engine.ipynb
    tests:
      read_when: "updating parity tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-04-27_iteration_4_no_risk_exact_scoring_notebook_topk/benchmark_summary.md
      read_when: "investigating previous arity 1/2 hash drift"

style_references:
  - src/trading/contexts/backtest/application/services/v2/no_risk_exact.py

hard_requirements:
  no_semantic_drift: true
  strict_hash_or_documented_waiver: true
  no_performance_scope_creep: true
  max_implementation_attempts: 2

task_toggles:
  implement_hash_normalization: true
  implement_result_shape_alignment: true
  implement_kernel_changes: false
  implement_persistence: false

skill_routing:
  - skill: contract-impact-analysis
    use_when: "changing result DTO serialization or hashes"
    timing: before implementation
    reason: "protect benchmark and future API identity semantics"
  - skill: backend-performance-evidence
    use_when: "checking measured stage timings after shape changes"
    timing: during verification
    reason: "ensure shape work does not pollute measured stages"
  - skill: backend-quality-gates
    use_when: "running local checks"
    timing: during verification
    reason: "lint/type/test gates"

target_envs:
  - local-dev
  - macstudio

required_literals:
  - "strict result hash"
  - "semantic metric parity"
  - "proxy metadata parity"
  - "float representation drift"

non_goals:
  - "Do not change exact scoring semantics."
  - "Do not change heap or proxy-fill timing boundaries unless fixing serialization leak."
  - "Do not add public variant identity or persistence."

final_report_format:
  language: ru
  sections:
    - "Что сделано"
    - "Parity/hash evidence"
    - "Проверки"
    - "Contract impact"
    - "Следующий шаг"

quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/backtest/application/services/v2/no_risk_exact.py src/trading/contexts/backtest/application/dto/no_risk_exact.py tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"
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
  - "docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_5_result_shape_hash_parity/"

safety_notes:
  - "Hash parity is correctness evidence; do not hide metric or row identity drift as float drift."
  - "Serialization work must not enter heap_update or top_result_proxy_fill timers."
  - "The executor has only 2 implementation attempts; after the second failed corrective cycle, stop and report the blocker."
---

# Task

Implement Iteration 4.5: align no-risk top result shape, ordering, float normalization, and strict hash/parity behavior with canonical notebook evidence.

Done means:

- semantic metric parity passes for all top rows;
- proxy metadata parity passes for all top rows;
- strict result hash passes or a narrow documented waiver exists for non-semantic float representation only;
- serialization/normalization does not pollute measured `heap_update` or `top_result_proxy_fill`.

## Context / Current State

- Iterations 4.2-4.4 should provide exact scoring, heap, and proxy fill.
- Previous failed record had arity 1/2 strict hash drift while semantic parity passed.
- This prompt owns output shape and hash evidence, not hot-path algorithms.

## Requirements (Must)

- Preserve top row order and identity.
- Preserve metric names and types expected by canonical evidence.
- Normalize floats deterministically for hash calculation.
- Keep internal fields like `_local_indices` and `_proxy_pending` out of public/top result hash payload.
- Add tests for canonical serialization and arity 1/2 hash drift regression.
- If strict hash cannot be matched without changing semantics, document exact waived fields and prove metric/proxy parity.

## Requirements (Should)

- Keep hash helper small and isolated.
- Make serialization reusable by benchmark runner without importing API/persistence code.

## Requirements (Nice-to-have)

- Add a compact golden payload fixture for top result hash.

# Context acquisition protocol

Read only in order: repo contract, latest report, task entrypoints, conditional bundles. Do not preload all docs.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Expand only for hash ambiguity or failed parity gates.

# Reading manifest

Use front-matter `context_sources`; do not convert to broad reading.

# Work plan (agent should follow)

Skill routing:

- `contract-impact-analysis`: use before changing serialization/hash semantics.
- `backend-performance-evidence`: use if measured-stage timings move.
- `backend-quality-gates`: use for local gates.

1. Inspect current top result payload and canonical expected fields.
2. Implement/adjust canonical serialization and hash helper.
3. Add tests for row order, field shape, float normalization, and absence of internal fields.
4. Run local gates.
5. Run Mac Studio parity/hash benchmark if runner supports it.

Mac Studio benchmark pipeline:

1. Commit local changes after gates pass.
2. Push branch/commit to remote.
3. SSH to `macstudio`.
4. In `/opt/roehub/app`, fetch and pull the pushed commit.
5. Verify commit SHA.
6. Run Iteration 4.5 parity/hash benchmark.
7. Save evidence under `docs/architecture/backtest/benchmark_iterations/YYYY-MM-DD_iteration_4_5_result_shape_hash_parity/`.
8. Record strict hash pass count, semantic parity, proxy parity, and any waiver.

# Acceptance criteria (Definition of Done)

- Semantic metric parity is `14 / 14`.
- Proxy metadata parity is `14 / 14`.
- Strict result hash is `14 / 14`, or any remaining drift is explicitly waived as non-semantic with field-level proof.
- No measured-stage boundary pollution.
- Local gates pass.

# Implementation constraints

## Determinism & ordering

- Stable field ordering and deterministic JSON/canonical payload.

## API / contracts

- Public API: none.
- DTO schema: compatible internal serialization helper only.
- Persistence: none.
- Config: none.
- Request/cache identity: no change.
- Benchmark gate: compatible correctness evidence.

# Files to indicate (expected touched areas)

- `src/trading/contexts/backtest/application/services/v2/no_risk_exact.py`
- `src/trading/contexts/backtest/application/dto/no_risk_exact.py`
- `tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- benchmark evidence folder if run

# Non-goals

- exact scoring changes;
- heap optimization;
- proxy-fill performance work;
- persistence/API identity;
- lazy trades.

# Quality gates (must run and pass)

- `uv run ruff check src/trading/contexts/backtest/application/services/v2/no_risk_exact.py src/trading/contexts/backtest/application/dto/no_risk_exact.py tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- `uv run pyright`
- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- Mac Studio parity/hash evidence if runner supports it.

# Final output: report format (strict)

Report in Russian:

1. Result shape/hash changes.
2. Parity/hash results.
3. Any waiver and why it is non-semantic.
4. Checks.
5. Contract impact classification.
6. Next step.
