---
prompt_name: backtest_compute_acceleration_stage_05_reversal_arity6
repo: roehub.com
branch: main
scope: "Extend matrix_bitset_no_risk_v1 to long_short_reversal and arity 6 heavy rows."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 05 reversal and heavy-row rules"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 04 gate"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/no_risk_score.py
      why: "Stage 04 matrix no-risk scorer"
      inspect_symbols:
        - "score"
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "current reversal semantics reference"
      inspect_symbols:
        - "score"
    - path: src/trading/contexts/backtest/application/services/v2/combo_planning.py
      why: "arity 6 candidate stream"
      inspect_symbols:
        - "BacktestComboPlanningServiceV2"
  conditional_bundles:
    correctness:
      read_when: "reversal transition or trade boundary semantics are unclear"
      paths:
        - src/trading/contexts/backtest/application/services/v2/result_series.py
        - tests/unit/contexts/backtest/golden/multi-trade.md
    tests:
      read_when: "adding reversal parity tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-06-03_matrix_bitset_stage_00_current_baseline/benchmark_results.json
      read_when: "Stage 00 heavy-row timing is needed"

style_references:
  - .codex/promt_template.md

hard_requirements:
  previous_stage_required: "04 accepted"
  reversal_cases_required: true
  heavy_rows_required: true

task_toggles:
  implementation_allowed: true
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: numba
    use_when: "JIT kernels are changed"
    timing: during implementation
    reason: "Numba correctness and performance"
  - skill: backend-performance-evidence
    use_when: "comparing arity 6 heavy rows"
    timing: during verification
    reason: "acceptance speedup discipline"
  - skill: backend-quality-gates
    use_when: "Python gates fail"
    timing: during verification
    reason: "focused backend triage"

target_envs:
  - local
  - Mac Studio

required_literals:
  - "long -> short"
  - "short -> long"
  - "none/arity_6/long_short_reversal"
  - "none/arity_6/long_only"

non_goals:
  - "TP/SL scoring."
  - "Consensus signature cache."
  - "High-arity pruning."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Implementation"
    - "Reversal parity"
    - "Heavy benchmark"
    - "Checks and next-stage notes"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"
    expect: "passes with reversal coverage"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "focused reversal parity tests"
    - "Mac Studio arity 6 heavy benchmark"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "05"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/no_risk_score.py
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

possible_secondary_touches:
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/consensus.py
  - tests/unit/contexts/backtest/application/services/v2/
  - docs/architecture/backtest/benchmark_iterations/<stage05_dir>/

safety_notes:
  - "Do not infer reversal semantics; prove each transition case."
  - "Heavy-row acceptance must compare against Stage 00 current baseline."
---

# Task

Implement Stage 05 by extending `matrix_bitset_no_risk_v1` to `long_short_reversal` and arity 6 heavy rows.

Done means:

- All reversal transitions match the current backend.
- Arity 6 heavy rows show accepted speedup or the stage is rejected/blocked.
- Service wall, memory, public result shape, and identity do not regress.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 04 implemented no-risk MVP for arity 2-3 long-only.
- open_items:
  - Add reversal semantics and heavy arity 6 coverage.
- contract_changes:
  - Internal backend scope expansion only.
- touched_paths:
  - Matrix no-risk scorer, trade tape, tests, benchmark evidence.
- risks:
  - Reversal transitions can silently change trade boundaries.
  - Heavy arity 6 can erase MVP speedup through service overhead.
- next_focus:
  - Stage 06 signature cache only starts after heavy no-risk acceptance.

Additional context:

- Required reversal transitions: `long -> short`, `short -> long`, `long -> flat`, `short -> flat`, `flat -> long`, `flat -> short`.

## Requirements (Must)

- Verify Stage 04 is accepted.
- Implement reversal and arity 6 support without changing current defaults.
- Prove all reversal transitions with focused tests.
- Preserve fees/slippage, sizing, `close_on_end`, trade boundary, top-N identity, and ranking.
- Run Mac Studio heavy rows for `none/arity_6/long_only` and `none/arity_6/long_short_reversal`.
- Update ledger with speedup, parity, memory, and next-stage decision.

## Requirements (Should)

- Keep arity handling generic enough for later high-arity pruning.
- Record exact candidates per second and average trades per candidate.

## Requirements (Nice-to-have)

- Include a small diagnostic comparing arity 3 vs arity 6 scaling.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. conditional bundles for correctness/tests only when needed
6. consult-if-needed references only for blockers

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once reversal semantics, heavy benchmark path, tests, and touched modules are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `numba`: use during implementation if JIT kernels are changed.
- `backend-performance-evidence`: use during verification for heavy-row benchmark acceptance.
- `backend-quality-gates`: use if Python gates fail.

1. Verify Stage 04 accepted and read its evidence summary.
2. Implement reversal transitions and arity 6 path.
3. Add focused tests for all transitions and top-N parity.
4. Run local gates.
5. Run Mac Studio heavy benchmark rows.
6. Update ledger with accepted, blocked, or rejected decision.

# Acceptance criteria (Definition of Done)

- Reversal transition tests pass.
- Heavy rows are faster than Stage 00 by the plan threshold.
- Service wall and memory do not regress.
- Public result shape and identity remain unchanged.
- Stage 06 is allowed only if ledger marks Stage 05 accepted.

# Implementation constraints

## Determinism & ordering

- Keep deterministic transition ordering and top-N merge.

## API / contracts

- Do not change public API, DB schema, request hash, or canonical artifacts.

## Documentation

- Update ledger and benchmark evidence.

## Tests

- Add focused reversal and arity tests.

## Validation depth

- Tests plus Mac Studio heavy benchmark are required.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/backtest/application/services/v2/matrix_backend/no_risk_score.py`
- `src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/backtest/application/services/v2/matrix_backend/consensus.py`
- `tests/unit/contexts/backtest/application/services/v2/`
- `docs/architecture/backtest/benchmark_iterations/<stage05_dir>/`

# Non-goals

- TP/SL scoring.
- Signature cache.
- Approximate search.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Mac Studio API-runner heavy benchmark for no-risk arity 6 rows.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Implementation**

3) **Reversal parity**

4) **Heavy benchmark**

5) **Checks and next-stage notes**
