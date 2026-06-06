---
prompt_name: backtest_compute_acceleration_stage_09_tp_sl_full_grid
repo: roehub.com
branch: main
scope: "Implement matrix_cell_tp_sl_v1 full-grid cell-block scoring with exact parity and benchmark evidence."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 09 full-grid requirements"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 08 gate"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
      why: "Stage 08 selected-cell implementation"
      inspect_symbols:
        - "score"
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
      why: "current full-grid reference"
      inspect_symbols:
        - "score"
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py
      why: "hit-times source and layout"
      inspect_symbols:
        - "load"
  conditional_bundles:
    sizing_and_fees:
      read_when: "fees, slippage, sizing, or execution semantics are unclear"
      paths:
        - src/trading/contexts/backtest/application/services/v2/execution_sizing.py
        - src/trading/contexts/backtest/application/services/v2/result_series.py
    tests:
      read_when: "adding full-grid TP/SL tests"
      paths:
        - tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py
        - tests/unit/contexts/backtest/golden/multi-trade.md
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-06-03_matrix_bitset_stage_00_current_baseline/benchmark_results.json
      read_when: "Stage 00 TP/SL heavy baseline is needed"

style_references:
  - .codex/promt_template.md

hard_requirements:
  previous_stage_required: "08 accepted_for_learning or accepted"
  full_grid_parity_required: true
  no_service_wall_regression: true

task_toggles:
  implementation_allowed: true
  benchmark_required: true
  docs_update_allowed: true

skill_routing:
  - skill: numba
    use_when: "cell-block kernels use Numba"
    timing: during implementation
    reason: "JIT performance and correctness"
  - skill: backend-performance-evidence
    use_when: "comparing TP/SL full-grid scoring"
    timing: during verification
    reason: "performance acceptance discipline"
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
  - "matrix_cell_tp_sl_v1"
  - "tp_sl_exact_scoring"
  - "trade_cell_evals_per_sec"
  - "16 x 16"
  - "SL wins"

non_goals:
  - "Approximate TP/SL ranking."
  - "Publisher hit-times artifacts."
  - "High-arity pruning."

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Implementation"
    - "Full-grid parity"
    - "Benchmark"
    - "Checks and next-stage notes"

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py"
    expect: "passes with full-grid matrix parity coverage"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "focused full-grid parity tests"
    - "Mac Studio TP/SL heavy benchmark"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  current_stage: "09"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md

expected_primary_touches:
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
  - src/trading/contexts/backtest/application/services/v2/matrix_backend/topn.py
  - docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md

possible_secondary_touches:
  - src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
  - tests/unit/contexts/backtest/application/services/v2/
  - docs/architecture/backtest/benchmark_iterations/<stage09_dir>/

safety_notes:
  - "Selected-cell speedup is not sufficient; full-grid evidence is required."
  - "Reject if memory or service wall dominates despite kernel speedup."
---

# Task

Implement Stage 09 `matrix_cell_tp_sl_v1` full-grid cell-block scoring with exact parity and Mac Studio benchmark evidence.

Done means:

- Full-grid TP/SL parity passes.
- Cell-block size, memory, and trade-cell throughput are recorded.
- `tp_sl_exact_scoring` and service wall pass stage gates.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 08 selected-cell shadow validated TP/SL semantics and layout.
- open_items:
  - Extend selected-cell scoring to full request grids.
- contract_changes:
  - Internal backend scope only.
- touched_paths:
  - TP/SL cell-block scorer, top-N merge, tests, benchmark evidence.
- risks:
  - Full grid can erase selected-cell speedup through memory or service overhead.
  - Tie-breaking or fees/slippage drift can reorder top variants.
- next_focus:
  - Stage 10 high-arity pruning starts only after full-grid acceptance.

Additional context:

- Record tested cell block shapes such as `16 x 16`, `32 x 8`, or `8 x 32`.

## Requirements (Must)

- Work from branch `main`; stop and report a blocker if the checkout is not on `main` unless the user explicitly approves another branch for this stage.
- After an `accepted` stage, update ledger/evidence/docs, run required gates, stage only scoped files, commit them to `main`, and report commit SHA and scoped paths. Do not push unless explicitly requested.
- For `accepted_for_learning`, commit scoped shadow/telemetry/docs/evidence only when that record is the durable handoff; keep the production-off limitation explicit.
- For `blocked` or `rejected`, do not commit production runtime changes; commit only ledger/evidence/docs documenting the blocker or rejection when needed, and report residual uncommitted changes.
- Verify Stage 08 has complete selected-cell parity and layout evidence.
- Implement full-grid cell-block scoring without approximation.
- Preserve SL-wins tie-breaking, trade boundaries, fees/slippage, sizing, ranking, and top-N identity.
- Record cell-block size, `tp_count`, `sl_count`, `tp_sl_cells`, `trade_cell_evals_per_sec`, memory, and timing.
- Run heavy TP/SL rows on Mac Studio.
- Update tests and ledger.

## Requirements (Should)

- Keep block sizes configurable in the internal backend config.
- Keep memory bounded per candidate/trade/cell block.

## Requirements (Nice-to-have)

- Include diagnostic block-size comparison if it does not slow acceptance.

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

Stop reading once full-grid reference, cell-block implementation path, tests, and benchmark rows are clear.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map. Do not convert it into a broad mandatory reading list.

# Work plan (agent should follow)

Skill routing for this task:

- `numba`: use during implementation if JIT kernels are changed.
- `backend-performance-evidence`: use during verification for full-grid benchmark claims.
- `backend-quality-gates`: use if Python gates fail.

1. Verify Stage 08 evidence and full-grid preconditions.
2. Implement full-grid cell-block scoring from selected-cell path.
3. Add parity tests for full grid, tie-breaking, fees/slippage, and deterministic top merge.
4. Run local gates.
5. Run Mac Studio TP/SL heavy benchmark rows.
6. Update ledger with accepted/rejected decision.

# Acceptance criteria (Definition of Done)

- Full-grid parity passes.
- `tp_sl_exact_scoring` speedup meets threshold.
- Service wall and memory do not regress.
- Public result shape and identity remain unchanged.
- Ledger records Stage 10 allowance.

# Implementation constraints

## Determinism & ordering

- Deterministic cell ordering and top merge are required.

## API / contracts

- No public API, DB, request hash, or canonical artifact changes.

## Documentation

- Update ledger and evidence docs.

## Tests

- Add full-grid TP/SL parity and tie tests.

## Validation depth

- Tests plus Mac Studio TP/SL heavy benchmark are required.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py`
- `src/trading/contexts/backtest/application/services/v2/matrix_backend/topn.py`
- `docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py`
- `tests/unit/contexts/backtest/application/services/v2/`
- `docs/architecture/backtest/benchmark_iterations/<stage09_dir>/`

# Non-goals

- Approximate ranking.
- Publisher hit-times artifacts.
- High-arity pruning.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`
- Mac Studio API-runner TP/SL heavy benchmark rows.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1) **Stage status**

2) **Implementation**

3) **Full-grid parity**

4) **Benchmark**

5) **Checks and next-stage notes**
