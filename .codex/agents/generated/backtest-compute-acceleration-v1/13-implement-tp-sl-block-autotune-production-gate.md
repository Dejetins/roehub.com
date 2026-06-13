---
prompt_name: backtest_compute_acceleration_stage_13_tp_sl_block_autotune_production_gate
repo: roehub.com
branch: main
scope: "Turn the accepted opt-in TP/SL 64x64 cell-block backend into a production candidate only if block autotune proves service-wall speedup."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 13 plan and TP/SL gate"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 12 accepted-code baseline policy and Stage 09 TP/SL baseline"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "Stage 09 16x16 failure and stop-list"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
      why: "Stage 09 TP/SL cell-block backend"
    - path: scripts/backtest/run_api_runner_benchmark_parity.py
      why: "API-runner benchmark path"
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_09_tp_sl_full_grid_64x64_rerun/
      why: "accepted Stage 09 evidence"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "TP/SL result identity, ranking, tie-breaking or default mode may change"
    timing: before implementation
  - skill: backend-performance-evidence
    use_when: "comparing block shapes and service wall"
    timing: during verification
  - skill: backend-quality-gates
    use_when: "Python gates fail"
    timing: during verification

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
  acceptance_testing: "Run A/B and shape matrix benchmarks over SSH on Mac Studio; local runs are preflight only."
  sync_rule: "Before benchmark, verify the checkout/runtime contains the Stage 05+12 production default code and record measured commit SHA or dirty state."
  production_runtime_rule: "If using or claiming live production baseline, verify /opt/roehub/app contains the accepted code and records default/unset ROEHUB_BACKTEST_MATRIX_BACKEND_MODE as stage_05_and_12_no_risk, or records any explicit override from /Users/daniildegtyarev/.config/roehub/roehub.env."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_13_tp_sl_block_autotune/
  write_policy: "Write evidence only under evidence_output_dir; no publisher/current.yaml/active-slot writes."

hard_requirements:
  previous_stage_required: "12 accepted"
  baseline_code_required: "Benchmark control and candidate must run from a checkout/runtime containing the Stage 05+12 production default: Stage 05 for no-risk arity 6 and Stage 12 for no-risk arity 7. If live production runtime is not updated, label evidence as Mac Studio project-checkout evidence, not live-production baseline."
  production_default_benchmark_command: "uv run python scripts/backtest/run_api_runner_benchmark_parity.py --env-file /Users/daniildegtyarev/.config/roehub/roehub.env --stage-05-12-production-default-rows --out-dir docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_05_12_production_default_stage13_baseline"
  implementation_allowed: true
  benchmark_required: true
  docs_update_required: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted or accepted_for_learning, update ledger/evidence/docs, stage only scoped files, and commit to main with a stage-specific message."
  - "For blocked or rejected, do not commit production runtime changes; commit only docs/evidence needed to preserve the decision."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not use 16x16 as accepted shape."
  - "Do not change TP/SL hit-time semantics or SL-wins tie rule."
  - "Do not change publisher/precompute artifacts."
  - "Do not make TP/SL backend default without passing this gate."

task:
  summary: "Benchmark and, if justified, implement block-shape selection for the accepted Stage 09 full-grid TP/SL backend."
  shapes:
    - "64x64"
    - "128x32"
    - "32x128"
    - "128x64"
    - "64x128"
  required_rows:
    - "tp_sl_grid/arity_6/long_only"
    - "tp_sl_grid/arity_6/long_short_reversal"
  required_controls:
    - "Stage 05+12 production-default checkout/runtime"
    - "default/current exact path"
    - "Stage 09 accepted 64x64 opt-in path"
    - "candidate shape path"

acceptance:
  correctness:
    - "Same top-N identity/order, best_tp, best_sl and metrics within accepted tolerance."
    - "variant_hash, ranking, fees/slippage, sizing and TP/SL tie-breaking unchanged."
  performance:
    - "Best candidate service wall improves by >=15% versus the correct accepted comparison path."
    - "Memory peak does not worsen by >10% and cleanup does not regress."
    - "tp_sl_exact_scoring, trade_cell_evals_per_sec and block shape are recorded."
  decision:
    - "If no shape passes, record rejected/blocked evidence and keep TP/SL backend opt-in/internal."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "focused TP/SL cell and parity tests pass or narrower justified equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  required_update: true
  current_stage: "13"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Shape matrix"
    - "Parity"
    - "Benchmark"
    - "Default/rollback"
    - "Ledger and commit"
---

# Task

Implement Stage 13 TP/SL block autotune production gate. Keep the accepted
Stage 09 backend opt-in unless the full Mac Studio gate passes.
