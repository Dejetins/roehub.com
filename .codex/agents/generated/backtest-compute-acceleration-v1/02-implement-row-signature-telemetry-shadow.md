---
prompt_name: backtest_compute_acceleration_stage_02_row_signature_telemetry
repo: roehub.com
branch: main
scope: "Add shadow-only row/signature telemetry for dedup potential without pruning or scoring changes."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 02 requirements"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 01 gate and handoff"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/prepare_pools.py
      why: "signal rows and pool metadata source"
      inspect_symbols:
        - "prepare"
        - "PreparedBacktestPoolsV2"
    - path: src/trading/contexts/backtest/application/services/v2/combo_planning.py
      why: "candidate planning and row universe"
      inspect_symbols:
        - "BacktestComboPlanningServiceV2"
    - path: src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py
      why: "telemetry output path"
      inspect_symbols:
        - "BacktestBenchmarkAccounting"
  conditional_bundles:
    tests:
      read_when: "adding row signature tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py
        - tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py
  consult_if_needed:
    - path: src/trading/contexts/backtest/application/services/v2/no_risk_exact.py
      read_when: "variant identity expansion or scoring inputs are unclear"

style_references:
  - .codex/promt_template.md

hard_requirements:
  previous_stage_required: "01 accepted or accepted_for_learning"
  no_pruning: true
  no_topn_drift: true

task_toggles:
  implementation_allowed: true
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "dedup identifiers touch variant identity or report contracts"
    timing: before implementation
    reason: "prevent silent identity or cache-key drift"
  - skill: backend-performance-evidence
    use_when: "recording telemetry overhead and dedup potential"
    timing: during verification
    reason: "classify telemetry as learning evidence"
  - skill: backend-quality-gates
    use_when: "focused Python checks fail"
    timing: during verification
    reason: "backend gate triage"

target_envs:
  - local
  - Mac Studio

required_literals:
  - "unique_rows_after_dedup"
  - "consensus_signature_count"
  - "duplicate_signal_row_ids"
  - "accepted_for_learning"

non_goals:
  - "Remove duplicate rows from production scoring."
  - "Change candidate stream or top-N."
  - "Introduce sidecar files."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Telemetry added"
    - "Evidence"
    - "Checks"
    - "Next-stage notes"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py"
    expect: "passes or focused equivalent if touched tests differ"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "focused tests"
    - "API-runner no-drift benchmark evidence"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "02"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/row_signatures.py
  - src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

possible_secondary_touches:
  - src/trading/contexts/backtest/application/services/v2/prepare_pools.py
  - src/trading/contexts/backtest/application/services/v2/combo_planning.py
  - tests/unit/contexts/backtest/application/services/v2/

safety_notes:
  - "This is a shadow telemetry stage; no dedup pruning is allowed."
  - "Collision handling must be explicit before any later cache or dedup stage."
---

# Task

Implement Stage 02 row/signature telemetry in shadow mode. Measure duplicate row and consensus-signature potential without changing candidate selection, scoring, or top-N.

Done means:

- Telemetry reports duplicate/signature potential.
- No production pruning or scoring reuse occurs.
- Result shape and hash do not drift.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 01 added instrumentation counters or blocked with a recorded reason.
- open_items:
  - Quantify whether row/signature dedup is worth implementing.
  - Prove identity expansion and collision rules before pruning.
- contract_changes:
  - Additive benchmark telemetry only.
- touched_paths:
  - Row signatures helper, benchmark accounting, possible pool/planning hooks.
- risks:
  - Hash collisions or duplicate identity collapse can corrupt public `variant_key`/`variant_hash`.
  - Telemetry can accidentally become pruning if wired into candidate stream.
- next_focus:
  - Stage 03 runtime bitset pack shadow depends on knowing row universe shape.

Additional context:

- A later stage may use `signal_row_hashes.u64.npy`, `unique_signal_row_ids.u32.npy`, and `duplicate_signal_row_ids.u32.npy`, but Stage 02 must not create persistent sidecar artifacts.

## Requirements (Must)

- Work from branch `main`; stop and report a blocker if the checkout is not on `main` unless the user explicitly approves another branch for this stage.
- After an `accepted` stage, update ledger/evidence/docs, run required gates, stage only scoped files, commit them to `main`, and report commit SHA and scoped paths. Do not push unless explicitly requested.
- For `accepted_for_learning`, commit scoped shadow/telemetry/docs/evidence only when that record is the durable handoff; keep the production-off limitation explicit.
- For `blocked` or `rejected`, do not commit production runtime changes; commit only ledger/evidence/docs documenting the blocker or rejection when needed, and report residual uncommitted changes.
- Verify Stage 01 is accepted or explicitly accepted for learning.
- Add row/signature telemetry only.
- Do not prune, deduplicate, reorder, or skip candidates in current scoring.
- Record collision handling strategy and duplicate mapping semantics.
- Preserve public `variant_key`, stable `variant_hash`, ranking order, and persisted top-N shape.
- Update tests and stage ledger.

## Requirements (Should)

- Keep signature computation deterministic and cheap enough for Stage 02 telemetry.
- Prefer exact row content hashing over metadata-only hashing.

## Requirements (Nice-to-have)

- Report estimated theoretical savings for dedup/cache without using it as acceptance speedup.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. conditional bundles only for tests or identity ambiguity
6. consult-if-needed references only for blockers

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once row source, telemetry hook, tests, and no-drift validation are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation if identity or report boundaries are touched.
- `backend-performance-evidence`: use during verification for overhead/no-drift evidence.
- `backend-quality-gates`: use if Python gates fail.

1. Check Stage 01 status and existing telemetry shape.
2. Implement deterministic row signature telemetry in a scoped helper.
3. Wire telemetry without affecting candidate stream or scoring.
4. Add focused tests for signatures, duplicates, and no-pruning behavior.
5. Run local gates and API-runner no-drift benchmark.
6. Update ledger with telemetry results and Stage 03 decision.

# Acceptance criteria (Definition of Done)

- Duplicate/signature counts are recorded.
- Collision strategy is explicit.
- No candidates are removed or reordered.
- Top-N shape/hash and result semantics do not drift.
- Ledger is updated with `accepted_for_learning` or blocker status.

# Implementation constraints

## Determinism & ordering

- Keep duplicate maps deterministic and stable.

## API / contracts

- Do not change public API, persistence, request hash, or cache identity.

## Documentation

- Update only plan/ledger/evidence docs if needed.

## Tests

- Include focused signature/dedup tests and no-drift checks.

## Validation depth

- Tests plus API-runner no-drift evidence are required.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/backtest/application/services/v2/matrix_backend/row_signatures.py`
- `src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/backtest/application/services/v2/prepare_pools.py`
- `src/trading/contexts/backtest/application/services/v2/combo_planning.py`
- `tests/unit/contexts/backtest/application/services/v2/`

# Non-goals

- Dedup pruning.
- Consensus cache.
- Sidecar generation.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_prepare_pools_service.py tests/unit/contexts/backtest/application/services/v2/test_combo_planning_service.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- API-runner no-drift benchmark evidence.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Telemetry added**

3) **Evidence**

4) **Checks**

5) **Next-stage notes**
