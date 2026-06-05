---
prompt_name: backtest_compute_acceleration_stage_11_lazy_detail_reuse
repo: roehub.com
branch: main
scope: "Reuse sparse trade tape backend for lazy selected-variant materialization without changing bulk top-N scoring."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 11 lazy scope"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 10 gate"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py
      why: "current lazy materialization reference"
      inspect_symbols:
        - "LazyTradesDetailServiceV2"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py
      why: "sparse trade tape candidate reuse"
      inspect_symbols:
        - "extract"
    - path: src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py
      why: "lazy cache identity and payload storage"
      inspect_symbols:
        - "LazyTradesCache"
  conditional_bundles:
    api_boundary:
      read_when: "lazy endpoint payload or cache identity might change"
      paths:
        - apps/api/routes/backtests.py
        - src/trading/contexts/backtest/application/ports/lazy_trades_materializations.py
    tests:
      read_when: "adding lazy detail parity tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py
        - tests/unit/contexts/backtest/adapters/outbound/cache_fs/test_lazy_trades_cache.py
  consult_if_needed:
    - path: scripts/backtest/run_iteration_9_lazy_trades_benchmark.py
      read_when: "historical lazy benchmark harness is needed"

style_references:
  - .codex/promt_template.md

hard_requirements:
  previous_stage_required: "10 accepted"
  no_bulk_topn_change: true
  lazy_identity_parity_required: true

task_toggles:
  implementation_allowed: true
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "lazy payload, cache key, API, or materialization identity may change"
    timing: before implementation
    reason: "protect persisted/lazy detail contracts"
  - skill: backend-performance-evidence
    use_when: "measuring selected-variant materialization latency"
    timing: during verification
    reason: "performance evidence for perceived latency"
  - skill: backend-quality-gates
    use_when: "Python gates fail"
    timing: during verification
    reason: "backend gate triage"

target_envs:
  - local
  - Mac Studio

required_literals:
  - "lazy materialization"
  - "sparse trade tape"
  - "cache identity parity"
  - "no bulk top-N scoring change"

non_goals:
  - "Change bulk backtest top-N scoring."
  - "Change lazy trades public payload shape."
  - "Change cache identity without explicit migration."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Implementation"
    - "Lazy parity"
    - "Benchmark"
    - "Checks and residual risks"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py tests/unit/contexts/backtest/adapters/outbound/cache_fs/test_lazy_trades_cache.py"
    expect: "passes or focused equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "lazy detail parity tests"
    - "lazy materialization benchmark"
    - "API/use-case smoke if route behavior is touched"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "11"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

possible_secondary_touches:
  - src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py
  - tests/unit/contexts/backtest/application/services/v2/
  - docs/architecture/backtest/benchmark_iterations/<stage11_dir>/

safety_notes:
  - "This stage improves selected-variant latency, not bulk scoring acceptance."
  - "Reject if cache identity or public lazy payload semantics drift."
---

# Task

Implement Stage 11 lazy detail reuse of the sparse trade tape backend for selected-variant materialization.

Done means:

- Lazy detail materialization can reuse sparse trade tape where safe.
- Lazy payload and cache identity parity are preserved.
- Selected-variant latency benchmark evidence is recorded.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 10 accepted exact-safe high-arity pruning or recorded a terminal blocker.
- open_items:
  - Improve perceived latency for selected variant details without changing bulk scoring.
- contract_changes:
  - None unless explicitly classified; lazy payload shape and cache identity must remain stable.
- touched_paths:
  - Lazy detail service, sparse trade tape adapter, tests, benchmark evidence.
- risks:
  - Improving latency while invalidating persisted lazy detail identity.
  - Accidentally changing public lazy trades response shape.
- next_focus:
  - Final report should summarize the whole compute-acceleration rollout status.

Additional context:

- This stage is separate from the main bulk compute speed gates.

## Requirements (Must)

- Verify Stage 10 accepted or explicitly allows Stage 11.
- Reuse sparse trade tape only for lazy selected-variant materialization.
- Do not change bulk top-N scoring.
- Preserve lazy payload shape, cache key, materialization identity, fees/slippage, sizing, and trade boundaries.
- Provide fallback to current lazy materialization.
- Run lazy materialization benchmark and focused tests.
- Update ledger with final stage status and residual risk.

## Requirements (Should)

- Keep adapter selection explicit and observable in telemetry.
- Record latency with cache miss and cache hit where feasible.

## Requirements (Nice-to-have)

- Add a compact final rollout summary note in the ledger.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. conditional bundles for API/cache/tests only when needed
6. consult-if-needed references only for blockers

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once lazy payload contract, cache identity, fallback, and benchmark path are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation for lazy payload/cache identity.
- `backend-performance-evidence`: use during verification for lazy latency benchmark.
- `backend-quality-gates`: use if Python gates fail.

1. Verify Stage 10 status and lazy contract.
2. Add sparse trade tape reuse behind explicit lazy adapter path.
3. Preserve fallback to current lazy materialization.
4. Add parity/cache identity tests.
5. Run local gates and lazy benchmark.
6. Update ledger with final status and residual risks.

# Acceptance criteria (Definition of Done)

- Lazy detail output parity passes.
- Cache identity and payload shape are unchanged.
- Selected-variant materialization latency improves or the stage records a blocker/rejection.
- Bulk top-N scoring is untouched.
- Ledger records final stage status.

# Implementation constraints

## Determinism & ordering

- Preserve trade ordering and payload determinism.

## API / contracts

- No public lazy payload or cache identity change unless explicitly classified and approved.

## Documentation

- Update ledger and evidence docs.

## Tests

- Add lazy parity and cache identity tests.

## Validation depth

- Tests plus lazy materialization benchmark are required; API/use-case smoke is required if route behavior is touched.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`
- `src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/backtest/adapters/outbound/cache_fs/lazy_trades_cache.py`
- `tests/unit/contexts/backtest/application/services/v2/`
- `docs/architecture/backtest/benchmark_iterations/<stage11_dir>/`

# Non-goals

- Bulk top-N scoring changes.
- Public lazy payload migration.
- Publisher changes.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_lazy_trades_detail_service.py tests/unit/contexts/backtest/adapters/outbound/cache_fs/test_lazy_trades_cache.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Lazy materialization benchmark evidence.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Implementation**

3) **Lazy parity**

4) **Benchmark**

5) **Checks and residual risks**
