---
prompt_name: backtest_compute_acceleration_stage_08_tp_sl_selected_cells
repo: roehub.com
branch: main
scope: "Implement TP/SL selected-cell shadow validation and by-entry hit-times layout measurement."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 08 selected-cell requirements"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 07 gate"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
      why: "current TP/SL exact scoring reference"
      inspect_symbols:
        - "score"
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py
      why: "hit-times loading and current layout"
      inspect_symbols:
        - "load"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py
      why: "sparse trade tape input"
      inspect_symbols:
        - "extract"
  conditional_bundles:
    hit_times:
      read_when: "hit-times artifact semantics or layout are unclear"
      paths:
        - src/trading/contexts/backtest_artifacts/application/services/v2/hit_times_compute_v2.py
        - tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py
    tests:
      read_when: "adding selected-cell parity tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-05-01_iteration_5_tp_sl_hit_times_loading_validation/benchmark_results.json
      read_when: "historical hit-times benchmark context is needed"

style_references:
  - .codex/promt_template.md

hard_requirements:
  selected_cell_only: true
  no_production_topn_feed: true
  sl_wins_tie_required: true

task_toggles:
  implementation_allowed: true
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: backend-performance-evidence
    use_when: "measuring selected-cell TP/SL and layout counters"
    timing: during verification
    reason: "performance and comparability evidence"
  - skill: contract-impact-analysis
    use_when: "hit-times layout or sidecar metadata crosses artifact/config boundaries"
    timing: before implementation
    reason: "preserve canonical artifact contracts"
  - skill: backend-quality-gates
    use_when: "Python gates fail"
    timing: during verification
    reason: "backend gate triage"

target_envs:
  - local
  - Mac Studio

runtime_env_sources:
  mac_studio_native_env_file: /Users/daniildegtyarev/.config/roehub/roehub.env
  docker_env_file: /etc/roehub/roehub.env
  benchmark_env_file_arg: "--env-file"
  mac_studio_required_runtime_env:
    ROEHUB_ENV: prod
    ROEHUB_BACKTEST_ARTIFACTS_CONFIG: configs/prod/backtest_artifacts.yaml
  mac_studio_artifact_root: /opt/roehub/state/backtest_artifacts/v2
  benchmark_env_fallback_order:
    - "$ROEHUB_ENV_FILE"
    - /Users/daniildegtyarev/.config/roehub/roehub.env
    - /etc/roehub/roehub.env
  source_references:
    - infra/macos/launchd/com.roehub.api.plist
    - infra/macos/launchd/com.roehub.backtest-job-runner.plist
    - infra/docker/.env.example
    - infra/docker/docker-compose.backend.yml
  required_postgres_env:
    - "STRATEGY_PG_DSN or POSTGRES_DSN or IDENTITY_PG_DSN"
    - "or POSTGRES_DB + POSTGRES_USER + POSTGRES_PASSWORD"
  benchmark_report_contract:
    - "Report env file path, runtime key names, and artifact config path only."
    - "Never print DSN, password, token, API key, or secret values."
  secret_reporting_rule: "Report only key/path presence, never DSN or password values."

required_literals:
  - "tp_count <= 8"
  - "sl_count <= 8"
  - "long_tp_by_entry.u32.npy"
  - "short_sl_by_entry.u32.npy"
  - "SL wins"

non_goals:
  - "Full TP/SL grid production scoring."
  - "Publisher/manifest changes."
  - "Production top-N feed from selected-cell shadow path."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Implementation"
    - "TP/SL parity"
    - "Layout and benchmark"
    - "Checks and next-stage notes"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py"
    expect: "passes or focused equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "focused TP/SL selected-cell parity tests"
    - "API-runner selected-cell shadow evidence"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "08"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

possible_secondary_touches:
  - src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
  - src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py
  - tests/unit/contexts/backtest/application/services/v2/

safety_notes:
  - "Selected-cell shadow path must not feed production top-N."
  - "Tie-breaking must prove current SL-wins behavior."
---

# Task

Implement Stage 08 TP/SL selected-cell shadow validation. Limit the stage to `tp_count <= 8` and `sl_count <= 8`, prove parity, and measure by-entry hit-times layout implications.

Done means:

- Selected-cell TP/SL parity passes.
- SL-wins tie case is proven.
- Hit-times layout counters or selected by-entry arrays are recorded.
- Production top-N remains fed by the current path.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 07 sidecar bitset artifacts were accepted for learning or explicitly blocked/rejected.
- open_items:
  - Validate TP/SL cell scoring on selected cells before full grid.
  - Determine whether by-entry hit-times layout is worth pursuing.
- contract_changes:
  - None to canonical artifacts or public API.
- touched_paths:
  - TP/SL cell helper, hit-times layout adapter/counters, tests, evidence.
- risks:
  - TP/SL tie semantics can change results.
  - Selected-cell speedup may not generalize to full grid.
- next_focus:
  - Stage 09 full grid can start only after selected-cell parity and layout evidence.

Additional context:

- If by-entry hit-times are persisted, they must be sidecar/test-only unless a separate publisher plan is approved.

## Requirements (Must)

- Work from branch `main`; stop and report a blocker if the checkout is not on `main` unless the user explicitly approves another branch for this stage.
- After an `accepted` stage, update ledger/evidence/docs, run required gates, stage only scoped files, commit them to `main`, and report commit SHA and scoped paths. Do not push unless explicitly requested.
- For `accepted_for_learning`, commit scoped shadow/telemetry/docs/evidence only when that record is the durable handoff; keep the production-off limitation explicit.
- For `blocked` or `rejected`, do not commit production runtime changes; commit only ledger/evidence/docs documenting the blocker or rejection when needed, and report residual uncommitted changes.
- Verify Stage 07 status permits Stage 08.
- Implement selected-cell shadow only, not production top-N feed.
- Preserve current TP/SL tie-breaking: SL wins on same hit index.
- Preserve trade boundaries, fees/slippage, long/short formulas, and sizing semantics.
- Record `tp_count`, `sl_count`, `tp_sl_cells`, layout counters, memory, and timing.
- Use sidecar-only strategy for persisted test arrays; do not change publisher/manifests.
- Update tests and ledger.

## Requirements (Should)

- Structure code so Stage 09 can extend it to full grid blocks.
- Test selected cells with both long and short trades.

## Requirements (Nice-to-have)

- Compare multiple small cell block shapes as diagnostic evidence.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. latest state snapshot, if available
3. latest executor final report, if available
4. task entrypoints
5. conditional bundles for hit-times/tests only when needed
6. consult-if-needed references only for blockers

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once TP/SL reference semantics, selected-cell hook, and parity tests are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `contract-impact-analysis`: use before implementation if hit-times sidecar/config boundaries are touched.
- `backend-performance-evidence`: use during verification for selected-cell and layout evidence.
- `backend-quality-gates`: use if Python gates fail.

1. Verify Stage 07 status and no publisher-change rule.
2. Implement selected-cell TP/SL scorer in shadow path.
3. Add tie-breaking, trade-boundary, fees/slippage, and selected-cell tests.
4. Add layout counters or selected by-entry arrays as sidecar/test-only if needed.
5. Run local gates and API-runner selected-cell evidence.
6. Update ledger with Stage 09 decision.

# Acceptance criteria (Definition of Done)

- Selected-cell parity passes.
- SL-wins tie proof is recorded.
- Layout/timing/memory evidence is recorded.
- No production top-N or canonical artifact path changes.
- Ledger permits Stage 09 only if evidence is complete.

# Implementation constraints

## Determinism & ordering

- Cell block ordering must be deterministic.

## API / contracts

- No public API, DB, request hash, canonical manifest, or publisher changes.

## Documentation

- Update ledger and evidence docs.

## Tests

- Add selected-cell and tie-breaking tests.

## Validation depth

- Tests plus API-runner selected-cell evidence are required.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py`
- `src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py`
- `tests/unit/contexts/backtest/application/services/v2/`
- `docs/architecture/backtest/benchmark_iterations/<stage08_dir>/`

# Non-goals

- Full-grid production scoring.
- Publisher hit-times artifacts.
- High-arity pruning.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- API-runner selected-cell shadow evidence.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Implementation**

3) **TP/SL parity**

4) **Layout and benchmark**

5) **Checks and next-stage notes**
