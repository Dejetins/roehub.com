---
prompt_name: backtest_compute_acceleration_stage_01_instrumentation
repo: roehub.com
branch: main
scope: "Add backtest compute instrumentation counters without changing scoring behavior."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and safety rules"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "stage requirements and metrics list"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "prior stage gate and ledger update"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/job_orchestration.py
      why: "current full-job orchestration timers"
      inspect_symbols:
        - "BacktestJobOrchestrationServiceV2"
    - path: src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py
      why: "benchmark metric accounting and report fields"
      inspect_symbols:
        - "BacktestBenchmarkAccounting"
    - path: scripts/backtest/run_api_runner_benchmark_parity.py
      why: "API-runner benchmark output shape"
      inspect_symbols:
        - main
  conditional_bundles:
    dto_and_renderer:
      read_when: "telemetry fields require DTO or summary rendering updates"
      paths:
        - src/trading/contexts/backtest/application/dto/backtest_jobs.py
        - docs/architecture/backtest/benchmark_iterations/README.md
    tests:
      read_when: "adding or updating counter assertions"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py
        - tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py
  consult_if_needed:
    - path: apps/worker/backtest_job_runner/main/full_job_child.py
      read_when: "child-process evidence fields are unclear"

style_references:
  - .codex/promt_template.md

hard_requirements:
  stage_00_required: true
  behavior_change_allowed: false
  overhead_limit_percent: 1

task_toggles:
  implementation_allowed: true
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "adding or evaluating hot-path instrumentation"
    timing: during implementation
    reason: "ensure counters do not distort timing claims"
  - skill: backend-quality-gates
    use_when: "Python tests, lint, or type gates fail"
    timing: during verification
    reason: "focused backend gate triage"
  - skill: contract-impact-analysis
    use_when: "new fields cross DTO, report, or persisted boundaries"
    timing: before implementation
    reason: "classify telemetry/report compatibility"

target_envs:
  - local
  - Mac Studio

required_literals:
  - "artifact_load_ms"
  - "signals_pack_ms"
  - "exact_candidates_per_sec"
  - "trade_cell_evals_per_sec"
  - "accepted_for_learning"

non_goals:
  - "Change scoring results or top-N ordering."
  - "Introduce matrix backend scoring."
  - "Change public API payload semantics."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Implementation"
    - "Benchmark evidence"
    - "Checks"
    - "Next-stage notes"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py"
    expect: "passes"
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py"
    expect: "passes or justify if not touched"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if Markdown docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "focused unit tests"
    - "Mac Studio API-runner benchmark overhead check"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "01"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py
  - scripts/backtest/run_api_runner_benchmark_parity.py
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

possible_secondary_touches:
  - src/trading/contexts/backtest/application/services/v2/job_orchestration.py
  - src/trading/contexts/backtest/application/dto/backtest_jobs.py
  - tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py
  - docs/architecture/backtest/benchmark_iterations/README.md

safety_notes:
  - "Instrumentation must be additive and off the public result identity path."
  - "Reject the stage if result hashes or top-N shape drift."
---

# Task

Implement Stage 01 instrumentation counters for the backtest compute acceleration plan. Add only telemetry/reporting needed to measure future stages.

Done means:

- Required counters are emitted into benchmark evidence.
- Scoring behavior, ranking, top-N, request identity, and persistence semantics do not change.
- The stage ledger records overhead, correctness, memory, and next-stage decision.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 00 baseline is accepted in the stage ledger.
  - No matrix backend is implemented yet.
- open_items:
  - Add counters before optimizing to avoid tuning the wrong stage.
  - Keep instrumentation overhead <= 1%.
- contract_changes:
  - Additive benchmark/report telemetry only.
- touched_paths:
  - Benchmark accounting, benchmark renderer, possibly orchestration telemetry.
- risks:
  - Instrumentation can perturb hot-path timing.
  - New report fields can accidentally become public contract if added to API payloads.
- next_focus:
  - Stage 02 row/signature telemetry depends on these counters.

Additional context:

- Metrics must include artifact load, signal pack, combo/proxy/exact/top assembly, rows, candidates, trades, cells, and throughput fields when available.

## Requirements (Must)

- Verify Stage 00 is accepted before implementation.
- Add counters only; do not change scoring, candidate selection, ranking, top-N merge, request hash, or persistence identity.
- Keep telemetry additive and benchmark/report scoped unless a compatible DTO addition is explicitly needed.
- Measure overhead with comparable benchmark evidence; overhead must be <= 1%.
- Update tests for new accounting/rendering fields.
- Update the stage ledger after validation and before final report.

## Requirements (Should)

- Prefer existing timing/accounting structures over new cross-cutting abstractions.
- Keep missing counters explicit as `null` or absent according to existing report style, not silently zero.

## Requirements (Nice-to-have)

- Add a compact summary table for new counters in benchmark summaries if it follows existing report style.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. conditional bundles only for touched DTO/tests
6. consult-if-needed references only for blockers

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once counter insertion points, output shape, tests, and benchmark gate are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation if telemetry crosses DTO/report boundaries.
- `backend-performance-evidence`: use during implementation and verification for overhead and comparability.
- `backend-quality-gates`: use during verification if Python gates fail.

1. Verify Stage 00 accepted and Stage 01 not already completed.
2. Identify existing timing/accounting output path.
3. Add counters with minimal hot-path overhead.
4. Add or update focused tests.
5. Run local gates and Mac Studio overhead benchmark.
6. Update ledger with status, evidence, overhead, and Stage 02 allowance.

# Acceptance criteria (Definition of Done)

- Required counters appear in benchmark evidence or are explicitly marked not applicable.
- Result hash/top-N shape does not drift.
- Instrumentation overhead is <= 1% on comparable API-runner evidence.
- Memory cleanup does not regress.
- Ledger row for Stage 01 is updated with status and `next_iteration_allowed`.

# Implementation constraints

## Determinism & ordering

- Keep telemetry field ordering deterministic in generated summaries.

## API / contracts

- Do not add user-visible API fields unless unavoidable and classified as compatible.

## Documentation

- Update benchmark docs only if evidence shape changes.
- Update the stage ledger after validation.

## Tests

- Add focused tests for accounting/rendering changes.

## Validation depth

- Local tests are required but not sufficient.
- Mac Studio API-runner overhead evidence is required for acceptance.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/backtest/application/services/v2/benchmark_accounting.py`
- `scripts/backtest/run_api_runner_benchmark_parity.py`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/backtest/application/services/v2/job_orchestration.py`
- `src/trading/contexts/backtest/application/dto/backtest_jobs.py`
- `tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py`
- `docs/architecture/backtest/benchmark_iterations/README.md`

# Non-goals

- Matrix backend scoring.
- Sidecar artifact generation.
- Publisher or manifest changes.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_benchmark_accounting.py`
- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Mac Studio API-runner benchmark proving <= 1% overhead.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Implementation**

3) **Benchmark evidence**

4) **Checks**

5) **Next-stage notes**
