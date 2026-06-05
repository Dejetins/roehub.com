---
prompt_name: backtest_compute_acceleration_stage_04_no_risk_mvp
repo: roehub.com
branch: main
scope: "Implement matrix_bitset_no_risk_v1 MVP for none/arity 2-3/long_only in shadow or gated mode."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 04 MVP and no-advantage rules"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 03 gate"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      why: "current exact scoring reference"
      inspect_symbols:
        - "score"
    - path: src/trading/contexts/backtest/application/services/v2/top_result_assembly.py
      why: "public top-N result shape"
      inspect_symbols:
        - "assemble"
    - path: src/trading/contexts/backtest/application/services/v2/combo_planning.py
      why: "candidate stream and arity handling"
      inspect_symbols:
        - "BacktestComboPlanningServiceV2"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/bitsets.py
      why: "Stage 03 bitset representation"
      inspect_symbols:
        - "pack"
  conditional_bundles:
    correctness:
      read_when: "trade boundary, fees, slippage, or sizing parity is unclear"
      paths:
        - src/trading/contexts/backtest/application/services/v2/result_series.py
        - src/trading/contexts/backtest/application/services/v2/execution_sizing.py
    tests:
      read_when: "adding no-risk matrix tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
        - tests/unit/contexts/backtest/golden/multi-trade.md
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      read_when: "benchmark comparability policy is unclear"

style_references:
  - .codex/promt_template.md

hard_requirements:
  previous_stage_required: "03 accepted_for_learning or accepted"
  exact_parity_required: true
  no_advantage_policy_required: true

task_toggles:
  implementation_allowed: true
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: numba
    use_when: "implementing or optimizing JIT kernels"
    timing: during implementation
    reason: "JIT and threading correctness"
  - skill: backend-performance-evidence
    use_when: "claiming speedup or comparing benchmark rows"
    timing: during verification
    reason: "performance acceptance discipline"
  - skill: contract-impact-analysis
    use_when: "backend selector or telemetry touches config/DTO/result identity"
    timing: before implementation
    reason: "avoid silent contract drift"
  - skill: backend-quality-gates
    use_when: "Python gates fail"
    timing: during verification
    reason: "backend gate triage"

target_envs:
  - local
  - Mac Studio

required_literals:
  - "matrix_bitset_no_risk_v1"
  - "none/arity_2/long_only"
  - "none/arity_3/long_only"
  - "No-Advantage Benchmark Policy"

non_goals:
  - "Support reversal in this stage."
  - "Support TP/SL in this stage."
  - "Enable production on mode without accepted benchmark evidence."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Implementation"
    - "Parity"
    - "Benchmark"
    - "Checks and next-stage notes"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"
    expect: "passes with matrix backend parity coverage"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "focused no-risk parity tests"
    - "Mac Studio API-runner MVP rows"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "04"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/consensus.py
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/no_risk_score.py
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

possible_secondary_touches:
  - src/trading/contexts/backtest/application/services/v2/job_orchestration.py
  - src/trading/contexts/backtest/application/services/v2/top_result_assembly.py
  - tests/unit/contexts/backtest/application/services/v2/

safety_notes:
  - "Accepted speedup must pass the no-advantage policy."
  - "If sidecar or warm-cache advantages appear, label evidence diagnostic only."
---

# Task

Implement Stage 04 `matrix_bitset_no_risk_v1` MVP for `risk.mode = none`, arity 2-3, `long_only`.

Done means:

- Matrix-bitset scoring matches current `no_risk_exact` semantics for scoped rows.
- Top-N shape/hash or bounded metric diff passes.
- Comparable benchmark evidence shows accepted speedup or the stage is rejected/blocked.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 03 proved runtime bitset pack parity in shadow mode.
- open_items:
  - First real compute speedup stage for variant scoring.
  - Start with no-risk, arity 2-3, long-only before reversal or TP/SL.
- contract_changes:
  - Optional internal backend selector/telemetry only.
- touched_paths:
  - New matrix backend modules and tests.
- risks:
  - Speedup can be invalid if measured outside the service pipeline.
  - Float accumulation or deterministic top merge can drift rankings.
- next_focus:
  - Stage 05 adds reversal and arity 6 only after Stage 04 accepted.

Additional context:

- No production `on` mode unless accepted evidence passes the no-advantage policy.

## Requirements (Must)

- Work from branch `main`; stop and report a blocker if the checkout is not on `main` unless the user explicitly approves another branch for this stage.
- After an `accepted` stage, update ledger/evidence/docs, run required gates, stage only scoped files, commit them to `main`, and report commit SHA and scoped paths. Do not push unless explicitly requested.
- For `accepted_for_learning`, commit scoped shadow/telemetry/docs/evidence only when that record is the durable handoff; keep the production-off limitation explicit.
- For `blocked` or `rejected`, do not commit production runtime changes; commit only ledger/evidence/docs documenting the blocker or rejection when needed, and report residual uncommitted changes.
- Verify Stage 03 accepted or accepted for learning.
- Implement only `none/arity_2..3/long_only`.
- Use the same candidate stream after current proxy.
- Preserve fees, slippage, scoped sizing, trade boundaries, `close_on_end`, result shape, `variant_key`, and `variant_hash`.
- Run parity against current backend for random sample candidates and top-N.
- Produce Mac Studio benchmark evidence for MVP rows.
- Update tests and stage ledger.

## Requirements (Should)

- Keep backend modules procedural and low-allocation.
- Reuse Stage 03 bitset representation.

## Requirements (Nice-to-have)

- Record exact candidates per second for current and matrix paths.

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

Stop reading once current scoring semantics, backend hook, parity tests, and benchmark path are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation for selector/config/result boundary review.
- `numba`: use during implementation if JIT kernels are added.
- `backend-performance-evidence`: use during verification for speedup claims.
- `backend-quality-gates`: use if Python gates fail.

1. Verify Stage 03 status and no-advantage policy.
2. Implement blockwise consensus, sparse trade tape extraction, and no-risk scorer for scoped MVP.
3. Wire backend in shadow/gated mode without altering production default.
4. Add parity and deterministic top merge tests.
5. Run local gates and Mac Studio MVP benchmark rows.
6. Update ledger with accepted, blocked, or rejected result.

# Acceptance criteria (Definition of Done)

- Exact parity passes for scoped semantics.
- Top-N identity/order is stable or bounded metric diffs are justified.
- Stage target hot path is faster than baseline by the plan threshold.
- Service wall and memory do not regress.
- Ledger records whether Stage 05 may start.

# Implementation constraints

## Determinism & ordering

- Use deterministic block merge and stable tie-breaks.

## API / contracts

- Do not change public API, DB schema, request hash, or persisted identity.

## Documentation

- Update ledger and benchmark evidence.

## Tests

- Add focused no-risk parity and top-N tests.

## Validation depth

- Tests plus Mac Studio API-runner benchmark are required.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/backtest/application/services/v2/matrix_backend/consensus.py`
- `src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py`
- `src/trading/contexts/backtest/application/services/v2/matrix_backend/no_risk_score.py`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/backtest/application/services/v2/job_orchestration.py`
- `src/trading/contexts/backtest/application/services/v2/top_result_assembly.py`
- `tests/unit/contexts/backtest/application/services/v2/`

# Non-goals

- Reversal support.
- TP/SL support.
- Publisher or sidecar changes.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Mac Studio API-runner benchmark for `none/arity_2/long_only` and `none/arity_3/long_only`.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Implementation**

3) **Parity**

4) **Benchmark**

5) **Checks and next-stage notes**
