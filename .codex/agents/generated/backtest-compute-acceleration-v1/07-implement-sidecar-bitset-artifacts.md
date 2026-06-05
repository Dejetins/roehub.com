---
prompt_name: backtest_compute_acceleration_stage_07_sidecar_bitsets
repo: roehub.com
branch: main
scope: "Generate and load benchmark/test sidecar bitset artifacts without changing publisher or canonical manifests."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "sidecar and no-advantage rules"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 06 gate"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/bitsets.py
      why: "runtime bitset representation"
      inspect_symbols:
        - "pack"
    - path: src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py
      why: "runtime artifact array loading boundary"
      inspect_symbols:
        - "ArtifactArrayLoader"
    - path: scripts/backtest/run_api_runner_benchmark_parity.py
      why: "sidecar benchmark evidence path"
      inspect_symbols:
        - main
  conditional_bundles:
    artifact_contracts:
      read_when: "risk of accidentally touching canonical publisher/manifests appears"
      paths:
        - src/trading/contexts/backtest_artifacts/application/services/v2/artifact_precompute_runner.py
        - src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py
    tests:
      read_when: "adding sidecar generator or loader tests"
      paths:
        - tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py
        - tests/unit/contexts/backtest/application/services/v2/test_no_risk_exact_scoring_service.py
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/README.md
      read_when: "sidecar evidence placement is unclear"

style_references:
  - .codex/promt_template.md

hard_requirements:
  no_publisher_changes: true
  no_canonical_manifest_changes: true
  sidecar_only: true
  no_advantage_policy_required: true

task_toggles:
  implementation_allowed: true
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: contract-impact-analysis
    use_when: "sidecar path or metadata touches config, artifacts, or identity"
    timing: before implementation
    reason: "keep canonical artifact contract unchanged"
  - skill: backend-performance-evidence
    use_when: "comparing sidecar load versus runtime pack"
    timing: during verification
    reason: "fair sidecar benchmark evidence"
  - skill: backend-quality-gates
    use_when: "Python gates fail"
    timing: during verification
    reason: "backend gate triage"

target_envs:
  - local
  - Mac Studio

required_literals:
  - "signals_pos_bits.u64.npy"
  - "signals_neg_bits.u64.npy"
  - "signal_row_hashes.u64.npy"
  - "unique_signal_row_ids.u32.npy"
  - "duplicate_signal_row_ids.u32.npy"
  - "matrix_sidecar_manifest.json"
  - "sidecar_generate_ms"
  - "sidecar_load_ms"

non_goals:
  - "Modify artifact publisher/precompute."
  - "Modify canonical manifest.yaml/current.yaml."
  - "Enable production on mode based only on sidecar speedup."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Sidecar artifacts"
    - "Fairness and benchmark"
    - "Checks"
    - "Next-stage notes"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py"
    expect: "passes or focused sidecar loader equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "sidecar generator/loader tests"
    - "API-runner sidecar benchmark labeled by fairness status"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "07"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - scripts/backtest/generate_matrix_sidecar_artifacts.py
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/bitsets.py
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

possible_secondary_touches:
  - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py
  - tests/unit/contexts/backtest/application/services/v2/
  - docs/architecture/backtest/benchmark_iterations/<stage07_dir>/

safety_notes:
  - "Block the stage if canonical publisher/precompute or active manifests change."
  - "Sidecar-dependent speedup is accepted for learning only unless measured generation is included or a publisher plan is approved."
---

# Task

Implement Stage 07 sidecar/test bitset artifacts outside the canonical publisher. Generate and load sidecar `.npy` files for benchmark/testing, with explicit source hashes and fallback to runtime packing.

Done means:

- Sidecar files and `matrix_sidecar_manifest.json` can be generated from current canonical artifacts.
- Sidecar safety validation and fallback behavior are tested.
- Benchmark evidence records `sidecar_generate_ms`, `sidecar_load_ms`, fairness classification, and no canonical publisher changes.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 06 cache was accepted, rejected, or accepted for learning with explicit evidence.
- open_items:
  - Test whether pre-generated bitsets reduce pack/load cost without changing publisher.
- contract_changes:
  - Test-only sidecar metadata; canonical artifact schema remains unchanged.
- touched_paths:
  - Sidecar generator/helper, optional loader hook, tests, benchmark evidence.
- risks:
  - Hidden benchmark advantage from precomputed sidecars.
  - Accidental canonical manifest or publisher changes.
- next_focus:
  - Stage 08 TP/SL selected-cell shadow can use sidecar strategy only as diagnostic/test overlay.

Additional context:

- Stage 07 cannot enable production `on` by sidecar speedup alone.

## Requirements (Must)

- Work from branch `main`; stop and report a blocker if the checkout is not on `main` unless the user explicitly approves another branch for this stage.
- After an `accepted` stage, update ledger/evidence/docs, run required gates, stage only scoped files, commit them to `main`, and report commit SHA and scoped paths. Do not push unless explicitly requested.
- For `accepted_for_learning`, commit scoped shadow/telemetry/docs/evidence only when that record is the durable handoff; keep the production-off limitation explicit.
- For `blocked` or `rejected`, do not commit production runtime changes; commit only ledger/evidence/docs documenting the blocker or rejection when needed, and report residual uncommitted changes.
- Verify Stage 06 decision permits Stage 07.
- Do not modify `backtest_artifacts` publisher/precompute, canonical `manifest.yaml`, `current.yaml`, or active slots.
- Generate sidecar files outside the canonical artifact store or under explicitly recorded test/evidence overlay.
- Write `matrix_sidecar_manifest.json` with source manifest hash, source `signals.i8.npy` hash, schema version, shapes, dtypes, padding, timeframe/market/symbol identity.
- Validate dtype, shape, padding, duplicate map, source hashes before use.
- Record `sidecar_generate_ms` and `sidecar_load_ms`.
- Include `sidecar_load_ms` in service wall when sidecar is used.
- Label sidecar-dependent speedup as `accepted_for_learning` unless no-advantage production criteria are met.
- Update ledger.

## Requirements (Should)

- Keep sidecar generator deterministic and idempotent.
- Place generated benchmark sidecars under the stage evidence directory unless a test overlay path is explicitly recorded.

## Requirements (Nice-to-have)

- Include a command to verify sidecar files without running full scoring.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. conditional bundles only if canonical artifact boundaries are at risk
6. consult-if-needed references only for blockers

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once sidecar generation, safety validation, fallback, and evidence placement are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation for artifact/config boundary classification.
- `backend-performance-evidence`: use during verification for fair sidecar measurement.
- `backend-quality-gates`: use if Python gates fail.

1. Verify Stage 06 status and no-publisher constraint.
2. Implement sidecar generator/helper and metadata schema.
3. Implement sidecar validation/fallback path in the matrix backend test path.
4. Add tests for validation, fallback, duplicate map, and no canonical publisher changes.
5. Run local gates and API-runner sidecar benchmark.
6. Update ledger with fairness classification and Stage 08 decision.

# Acceptance criteria (Definition of Done)

- Required sidecar files are generated and validated.
- Missing or invalid sidecar falls back to runtime packing.
- No canonical publisher/precompute/manifest files are changed.
- Sidecar load/generation timing is recorded.
- Ledger marks result according to no-advantage policy.

# Implementation constraints

## Determinism & ordering

- Sidecar row order and duplicate maps must be deterministic.

## API / contracts

- Canonical artifact schema and public API remain unchanged.

## Documentation

- Update ledger and evidence docs.

## Tests

- Add generator/loader validation and fallback tests.

## Validation depth

- Tests plus API-runner sidecar benchmark are required.

# Files to indicate (expected touched areas)

Primary touches:

- `scripts/backtest/generate_matrix_sidecar_artifacts.py`
- `src/trading/contexts/backtest/application/services/v2/matrix_backend/bitsets.py`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py`
- `tests/unit/contexts/backtest/application/services/v2/`
- `docs/architecture/backtest/benchmark_iterations/<stage07_dir>/`

# Non-goals

- Publisher/precompute changes.
- Production `on` enablement.
- TP/SL scoring.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/adapters/outbound/artifacts_fs/test_artifact_array_loader.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- API-runner sidecar benchmark with no-advantage classification.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Sidecar artifacts**

3) **Fairness and benchmark**

4) **Checks**

5) **Next-stage notes**
