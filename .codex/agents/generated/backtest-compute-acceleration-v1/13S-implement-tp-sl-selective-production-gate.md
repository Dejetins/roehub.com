---
prompt_name: backtest_compute_acceleration_stage_13s_tp_sl_selective_production_gate
repo: roehub.com
branch: main
scope: "Implement a narrow TP/SL production selector only for the Stage 13 large-grid long-only winner, with reversal and smaller grids forced to current exact."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and delivery rules"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 13S scope, accepted baselines and stop-list"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 13 rejected handoff and current next-stage status"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "do not re-enable rejected TP/SL global block autotune"
    - path: docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_13_tp_sl_block_autotune/model_handoff_report.md
      why: "Stage 13 evidence interpretation and selector recommendation"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/job_orchestration.py
      why: "production backend selection/default routing"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
      why: "Stage 09/13 TP/SL matrix-cell backend and block shape"
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
      why: "current exact TP/SL fallback semantics"
    - path: scripts/backtest/run_stage_13_tp_sl_block_autotune_gate.py
      why: "existing Stage 13 TP/SL control/candidate aggregation harness"
    - path: scripts/backtest/run_api_runner_benchmark_parity.py
      why: "API-runner benchmark path and evidence writer"
  consult_if_needed:
    - path: tests/unit/contexts/backtest/application/services/v2/test_stage_13_tp_sl_block_autotune_gate.py
      read_when: "updating or extending Stage 13S selector/gate tests"
    - path: tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py
      read_when: "adding backend selection/default routing tests"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding a new TP/SL backend selector/default mode or telemetry fields"
    timing: before implementation
    reason: "selector/default changes affect config semantics and result identity guarantees"
  - skill: backend-performance-evidence
    use_when: "designing and interpreting Stage 13S A/B service-wall benchmark"
    timing: during verification
    reason: "acceptance depends on comparable Mac Studio service-wall evidence"
  - skill: backend-quality-gates
    use_when: "Python lint/type/test gates fail"
    timing: during verification
    reason: "focused backend gates must pass before stage acceptance"

runtime_env_sources:
  mac_studio_native_env_file: /Users/daniildegtyarev/.config/roehub/roehub.env
  benchmark_env_file_arg: "--env-file"
  mac_studio_required_runtime_env:
    ROEHUB_ENV: prod
    ROEHUB_BACKTEST_ARTIFACTS_CONFIG: configs/prod/backtest_artifacts.yaml
  mac_studio_artifact_root: /opt/roehub/state/backtest_artifacts/v2
  secret_reporting_rule: "Report paths and key names only; never report secret values."

mac_studio_test_execution:
  ssh_alias: macstudio
  repo_checkout: /Users/daniildegtyarev/Projects/roehub.com
  command_prefix: "ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && <command>'"
  acceptance_testing: "Run Stage 13S A/B benchmark over SSH on Mac Studio; local runs are preflight only."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_13s_tp_sl_selective_selector/
  write_policy: "Evidence only under evidence_output_dir; do not write canonical artifacts."

hard_requirements:
  previous_stage_required: "13 rejected with Stage 13S explicitly opened in the ledger"
  baseline_code_required: "Benchmark control and candidate must run from a checkout/runtime containing the Stage 05+12 production default and the preserved Stage 13 evidence commit."
  production_default_benchmark_command: "uv run python scripts/backtest/run_api_runner_benchmark_parity.py --env-file /Users/daniildegtyarev/.config/roehub/roehub.env --stage-05-12-production-default-rows --out-dir docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_05_12_production_default_stage13s_baseline"
  benchmark_claim_rule: "Acceptance benchmark evidence is valid only if measured heavy jobs are claimed by the benchmark harness process; if the live launchd backtest-job-runner claims a benchmark job, record the run as diagnostic and rerun with isolation or explicit claim verification."
  implementation_allowed: true
  benchmark_required: true
  docs_update_required: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted or rejected, update ledger/evidence/docs, stage only scoped files, and commit to main with a Stage 13S-specific message."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not accept Stage 13 global block autotune."
  - "Do not route long_short_reversal to matrix-cell backend."
  - "Do not change TP/SL result semantics, SL-wins tie behavior, best_tp/best_sl, ranking, sizing, fees or slippage."
  - "Do not change publisher/precompute artifacts."
  - "Do not implement reversal repair; Stage 13S only selects existing exact/matrix paths."

task:
  summary: "Add a deterministic TP/SL selector that uses matrix_cell_tp_sl_v1 shape 64x128 only for tp_sl_grid/arity_6/long_only when tp_count >= 64 and sl_count >= 32, and uses current exact for smaller grids and tp_sl_grid/arity_6/long_short_reversal."
  selector_policy:
    tp_sl_grid_arity6_long_only:
      min_tp_count: 64
      min_sl_count: 32
      backend: matrix_cell_tp_sl_v1
      tp_block_size: 64
      sl_block_size: 128
      reason: stage_13s_long_only_selector
    tp_sl_grid_arity6_long_only_small_grid:
      backend: current_exact
      reason: stage_13s_small_grid_exact_fallback
    tp_sl_grid_arity6_long_short_reversal:
      backend: current_exact
      reason: stage_13s_reversal_exact_fallback
  env_mode:
    name: ROEHUB_BACKTEST_TP_SL_BACKEND_MODE
    values:
      - off
      - stage_13s_selector
      - matrix_cell_force
    accepted_default_after_gate: stage_13s_selector
  required_telemetry:
    - tp_sl_backend_selected
    - tp_sl_backend_reason
    - tp_block_size
    - sl_block_size
    - fallback_backend

acceptance:
  correctness:
    - "Same top-N identity/order, best_tp, best_sl and metrics within accepted tolerance for both mandatory TP/SL rows."
    - "variant_hash, ranking, fees/slippage, sizing and TP/SL tie-breaking unchanged."
    - "Selector decision and rollback/override state are visible in telemetry/evidence."
  performance:
    - "long_only service wall improves by >=25% versus current exact."
    - "long_only service wall improves by >=15% versus Stage 09 64x64."
    - "long_short_reversal routes to current exact and service wall is no worse than current exact by >2%."
    - "Small TP/SL grids below the selector threshold route to current exact without service-wall regression."
    - "Combined service wall across mandatory TP/SL rows improves by >=20% versus current exact combined baseline."
    - "Memory peak does not worsen by >10% and cleanup does not regress."
  decision:
    - "If any gate fails, keep selector disabled/off and record rejected evidence; do not enable partial production default."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_stage_13_tp_sl_block_autotune_gate.py tests/unit/contexts/backtest/application/services/v2/test_job_orchestration.py"
    expect: "focused selector/gate tests pass or narrower justified equivalent"
  - cmd: "uv run ruff check scripts/backtest/run_stage_13_tp_sl_block_autotune_gate.py scripts/backtest/run_api_runner_benchmark_parity.py src/trading/contexts/backtest/application/services/v2"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "Mac Studio API-runner benchmark over ssh macstudio"
    - "selector telemetry and rollback env evidence"
    - "stage ledger and benchmark_summary.md"
  tests_only_allowed_reason: "not allowed; this is a production-affecting performance selector"
  evidence_target: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_13s_tp_sl_selective_selector/

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  required_update: true
  current_stage: "13S"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Selector policy"
    - "Parity"
    - "Benchmark"
    - "Rollback"
    - "Ledger and commit"
---

# Task

Implement Stage 13S selective TP/SL production gate. This is a narrow selector,
not a global TP/SL backend enablement. Done means the selector is accepted by
Mac Studio API-runner evidence or remains disabled with rejected evidence
preserved.
