---
prompt_name: backtest_compute_acceleration_stage_14_tp_sl_monotonic_cell_kernel
repo: roehub.com
branch: main
scope: "Implement an exact TP/SL monotonic cell classification kernel if it beats the best Stage 13 block shape."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 14 scope"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 13 gate"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "avoid failed TP/SL shapes and local-only wins"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
      why: "current TP/SL cell kernel"
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
      why: "reference TP/SL semantics"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "TP/SL exact semantics or tie-breaking could change"
    timing: before implementation
  - skill: numba
    use_when: "JIT kernel or monotonic boundary loop performance is involved"
    timing: during implementation
  - skill: backend-performance-evidence
    use_when: "measuring cell comparisons and service wall"
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
  secret_reporting_rule: "Report only key/path presence, never secret values."

mac_studio_test_execution:
  ssh_alias: macstudio
  repo_checkout: /Users/daniildegtyarev/Projects/roehub.com
  command_prefix: "ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && <command>'"
  acceptance_testing: "Run benchmark/testing evidence over SSH on Mac Studio; local runs are preflight only."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_14_tp_sl_monotonic_kernel/
  write_policy: "Evidence only under evidence_output_dir; do not write canonical artifacts."

hard_requirements:
  previous_stage_required: "13 accepted"
  baseline_code_required: "Benchmark control and candidate must run from a Stage-12-or-later accepted-code checkout/runtime. If live production runtime is used or claimed, verify /opt/roehub/app code state and ROEHUB_BACKTEST_MATRIX_BACKEND_MODE env state before benchmarking."
  implementation_allowed: true
  benchmark_required: true
  docs_update_required: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted or accepted_for_learning, update ledger/evidence/docs, stage only scoped files, and commit to main with a stage-specific message."
  - "For blocked or rejected, do not commit production runtime changes; commit only docs/evidence needed to preserve the decision."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not approximate TP/SL grid results."
  - "Do not change SL-wins tie behavior."
  - "Do not use selected-cell shadow evidence as production proof."
  - "Do not change publisher artifacts."

task:
  summary: "Use monotonic TP/SL hit-time structure to reduce branch and comparison cost inside trade x TP x SL classification."
  required_rows:
    - "tp_sl_grid/arity_6/long_only"
    - "tp_sl_grid/arity_6/long_short_reversal"
  required_counters:
    - "cell_comparisons_per_trade"
    - "cell_classification_ms"
    - "trade_cell_evals_per_sec"
    - "tp_sl_exact_scoring"
    - "service_wall_clock_s"

acceptance:
  correctness:
    - "Same top-N identity/order, best_tp, best_sl, exit reason and metrics within accepted tolerance."
    - "Explicit tests cover equal TP/SL hit index where SL wins."
  performance:
    - "cell comparisons per trade lower than Stage 13 winner."
    - "trade-cell/sec higher than Stage 13 winner."
    - "service wall better than Stage 13 winner with no memory cleanup regression."
  decision:
    - "If only local kernel timers improve, reject or keep disabled until API-runner service wall also improves."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "focused TP/SL monotonic kernel tests pass or narrower justified equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  required_update: true
  current_stage: "14"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Kernel"
    - "Parity"
    - "Benchmark"
    - "Ledger and commit"
---

# Task

Implement Stage 14 TP/SL monotonic cell kernel only if exact output and service
wall beat the Stage 13 winner.
