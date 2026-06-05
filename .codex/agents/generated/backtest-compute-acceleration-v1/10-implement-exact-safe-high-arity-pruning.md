---
prompt_name: backtest_compute_acceleration_stage_10_high_arity_pruning
repo: roehub.com
branch: main
scope: "Implement exact-safe high-arity pruning for arity 7/10 without approximate beam search in the default path."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 10 pruning rules"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 09 gate"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/combo_planning.py
      why: "candidate enumeration and arity handling"
      inspect_symbols:
        - "BacktestComboPlanningServiceV2"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/consensus.py
      why: "monotonic consensus facts"
      inspect_symbols:
        - "consensus"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/no_risk_score.py
      why: "retained candidate scoring parity"
      inspect_symbols:
        - "score"
  conditional_bundles:
    tp_sl:
      read_when: "pruning also affects TP/SL candidate scoring"
      paths:
        - src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
        - src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
    tests:
      read_when: "adding pruning proof and parity tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      read_when: "bounded arity benchmark policy is unclear"

style_references:
  - .codex/promt_template.md

hard_requirements:
  previous_stage_required: "09 accepted"
  exact_safe_only: true
  approximate_beam_default_forbidden: true

task_toggles:
  implementation_allowed: true
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "pruning affects candidate identity, request hash, or ranking semantics"
    timing: before implementation
    reason: "prevent silent search-contract changes"
  - skill: backend-performance-evidence
    use_when: "measuring arity 7/10 bounded evidence"
    timing: during verification
    reason: "performance acceptance discipline"
  - skill: backend-quality-gates
    use_when: "Python gates fail"
    timing: during verification
    reason: "backend gate triage"

target_envs:
  - local
  - Mac Studio

required_literals:
  - "exact-safe"
  - "branch-and-bound"
  - "monotonic pruning"
  - "approximate beam remains off"

non_goals:
  - "Approximate beam search in default path."
  - "Product-level approximate ranking."
  - "GPU rewrite."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Pruning rule"
    - "Correctness proof"
    - "Benchmark"
    - "Checks and next-stage notes"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py"
    expect: "passes or focused equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "pruning proof tests"
    - "Mac Studio arity 7/10 bounded benchmark"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "10"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/combo_blocks.py
  - src/trading/contexts/backtest/application/services/v2/combo_planning.py
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

possible_secondary_touches:
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/consensus.py
  - tests/unit/contexts/backtest/application/services/v2/
  - docs/architecture/backtest/benchmark_iterations/<stage10_dir>/

safety_notes:
  - "Block if pruning can remove a valid top candidate."
  - "Approximate search requires separate product approval and plan update."
---

# Task

Implement Stage 10 exact-safe high-arity pruning for arity 7/10. The default path must remain exact; approximate beam search is forbidden unless a separate approved plan adds it.

Done means:

- Pruning rule has an explicit exact-safety proof.
- Retained candidates score with parity.
- Arity 7/10 bounded benchmark evidence is recorded.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 09 accepted full-grid TP/SL cell-block scoring.
- open_items:
  - Matrix scoring alone does not solve high-arity combinatorics.
- contract_changes:
  - Candidate planning internals only; public exact semantics unchanged.
- touched_paths:
  - Combo planning, matrix combo blocks, tests, benchmark evidence.
- risks:
  - Unsafe pruning can remove the true top candidate.
  - Approximate beam can silently change product semantics.
- next_focus:
  - Stage 11 lazy detail reuse follows after bulk scoring path is accepted.

Additional context:

- If exact-safe proof is not possible, stop and record a blocker. Do not substitute approximate search.

## Requirements (Must)

- Work from branch `main`; stop and report a blocker if the checkout is not on `main` unless the user explicitly approves another branch for this stage.
- After an `accepted` stage, update ledger/evidence/docs, run required gates, stage only scoped files, commit them to `main`, and report commit SHA and scoped paths. Do not push unless explicitly requested.
- For `accepted_for_learning`, commit scoped shadow/telemetry/docs/evidence only when that record is the durable handoff; keep the production-off limitation explicit.
- For `blocked` or `rejected`, do not commit production runtime changes; commit only ledger/evidence/docs documenting the blocker or rejection when needed, and report residual uncommitted changes.
- Verify Stage 09 accepted.
- Implement only exact-safe pruning or branch-and-bound.
- Prove pruning cannot remove a valid top candidate under current ranking semantics.
- Keep approximate beam off and non-default.
- Preserve request hash, result shape, `variant_hash`, ranking, fees/slippage, sizing, and TP/SL tie-breaking.
- Run bounded arity 7/10 evidence on Mac Studio.
- Update tests and ledger.

## Requirements (Should)

- Keep pruning planner isolated and reversible.
- Record candidate counts before/after pruning and exact candidates scored.

## Requirements (Nice-to-have)

- Include diagnostic evidence for why approximate beam remains out of scope.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. conditional bundles for TP/SL/tests only when needed
6. consult-if-needed references only for blockers

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once pruning proof, candidate stream boundary, tests, and benchmark plan are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation for candidate/ranking semantics.
- `backend-performance-evidence`: use during verification for arity benchmark evidence.
- `backend-quality-gates`: use if Python gates fail.

1. Verify Stage 09 accepted.
2. Define exact-safe pruning rule and proof.
3. Implement pruning planner behind explicit backend path.
4. Add proof/parity tests, including cases where a top candidate must be retained.
5. Run local gates and Mac Studio bounded arity benchmark.
6. Update ledger with accepted/blocked/rejected decision.

# Acceptance criteria (Definition of Done)

- Exact-safe proof is documented in tests or stage evidence.
- No valid top candidate can be pruned.
- Arity 7/10 evidence shows accepted speedup or records a blocker/rejection.
- Public result contract remains exact.
- Ledger is updated.

# Implementation constraints

## Determinism & ordering

- Pruning and retained candidate ordering must be deterministic.

## API / contracts

- Do not change public API, request hash, DB schema, or exact semantics.

## Documentation

- Update ledger and evidence docs.

## Tests

- Add focused pruning proof and parity tests.

## Validation depth

- Tests plus Mac Studio bounded arity benchmark are required.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/backtest/application/services/v2/matrix_backend/combo_blocks.py`
- `src/trading/contexts/backtest/application/services/v2/combo_planning.py`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/backtest/application/services/v2/matrix_backend/consensus.py`
- `tests/unit/contexts/backtest/application/services/v2/`
- `docs/architecture/backtest/benchmark_iterations/<stage10_dir>/`

# Non-goals

- Approximate beam search.
- GPU rewrite.
- Publisher artifacts.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Mac Studio arity 7/10 bounded benchmark evidence.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Pruning rule**

3) **Correctness proof**

4) **Benchmark**

5) **Checks and next-stage notes**
