---
prompt_name: backtest_compute_acceleration_stage_14r_tp_sl_reversal_kernel_repair
repo: roehub.com
branch: main
scope: "Implement an exact-safe TP/SL long_short_reversal repair only after Stage 13R identifies the dominant cost center."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and hot-path safety"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 14R replacement scope and blocked predecessor state"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 13R telemetry decision and Stage 14R unblock condition"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "do not reintroduce rejected cache/block-shape patterns"
    - path: docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_13_tp_sl_block_autotune/model_handoff_report.md
      why: "repair candidates and exact-safe constraints"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
      why: "matrix-cell TP/SL kernel and possible split-by-side implementation"
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
      why: "current exact baseline and reversal semantics"
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py
      why: "hit-time access and signal-exit shortcut inputs"
    - path: scripts/backtest/run_api_runner_benchmark_parity.py
      why: "API-runner benchmark and report fields"
  conditional_bundles:
    signal_exit_shortcut:
      read_when: "Stage 13R reports signal_exit_dominant_cell_pct high enough to justify an exact-safe shortcut"
      paths:
        - src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py
        - tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py
    trade_window_grouping:
      read_when: "Stage 13R reports high weighted trade-window reuse"
      paths:
        - src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
        - docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md

skill_routing:
  - skill: contract-impact-analysis
    use_when: "candidate changes exit classification, exact semantics, selector/default or telemetry payload"
    timing: before implementation
    reason: "TP/SL repair must preserve exact mode contracts"
  - skill: numba
    use_when: "implementing split-by-side kernels, branch-free loops, or signal-exit shortcut in JIT code"
    timing: during implementation
    reason: "Stage 14R is a Numba/JIT hot-path task if implemented"
  - skill: backend-performance-evidence
    use_when: "measuring reversal service wall, kernel timers, CPU and memory"
    timing: during verification
    reason: "acceptance requires comparable Mac Studio speed evidence"
  - skill: backend-quality-gates
    use_when: "Python gates fail"
    timing: during verification
    reason: "focused tests and lint/type checks must pass"

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
  acceptance_testing: "Run Stage 14R A/B benchmark over SSH on Mac Studio; local runs are preflight only."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_14r_tp_sl_reversal_repair/
  write_policy: "Evidence only under evidence_output_dir; do not write canonical artifacts."

hard_requirements:
  previous_stage_required: "13R accepted_for_learning with a recorded dominant reversal cost center and recommended repair direction"
  baseline_code_required: "Benchmark control and candidate must run from a checkout/runtime containing Stage 05+12 production default and Stage 13R diagnostic counters."
  production_default_benchmark_command: "uv run python scripts/backtest/run_api_runner_benchmark_parity.py --env-file /Users/daniildegtyarev/.config/roehub/roehub.env --stage-05-12-production-default-rows --out-dir docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_05_12_production_default_stage14r_baseline"
  benchmark_claim_rule: "Acceptance benchmark evidence is valid only if measured heavy jobs are claimed by the benchmark harness process; if the live launchd backtest-job-runner claims a benchmark job, record the run as diagnostic and rerun with isolation or explicit claim verification."
  implementation_allowed: true
  benchmark_required: true
  docs_update_required: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "If Stage 13R is missing or did not identify a dominant cost center, stop and update the ledger as blocked; do not implement a speculative kernel."
  - "After accepted or rejected, update ledger/evidence/docs, stage only scoped files, and commit to main with a Stage 14R-specific message."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not implement the old Stage 14 monotonic-cell prompt as written."
  - "Do not approximate TP/SL grid results."
  - "Do not change SL-wins tie behavior."
  - "Do not change publisher/precompute artifacts."
  - "Do not introduce Python dict cache or unbounded cross-job cache."
  - "Do not change production default unless the Mac Studio gate accepts it and rollback is documented."

task:
  summary: "Use Stage 13R telemetry to implement exactly one reversal repair candidate."
  primary_candidate:
    name: split_by_side_reversal_kernel
    description: "Build side-specific trade tapes and score long and short TP/SL hit-time arrays in branch-reduced kernels while preserving exact aggregation."
  optional_candidate:
    name: signal_exit_dominated_shortcut
    allowed_only_if: "Stage 13R shows signal_exit_dominant_cell_pct >= 20-30% or otherwise clearly material."
    exact_rule: "Signal exit wins only when signal_exit_idx < tp_hit and signal_exit_idx < sl_hit; SL wins ties where sl_hit <= tp_hit and sl_hit <= signal_exit_idx; TP wins when tp_hit < sl_hit and tp_hit <= signal_exit_idx."
  required_rows:
    - "tp_sl_grid/arity_6/long_short_reversal"
    - "tp_sl_grid/arity_6/long_only no-regression row"

acceptance:
  correctness:
    - "Same top-N identity/order, best_tp, best_sl and metrics within accepted tolerance."
    - "SL-wins tie rule covered by tests, including equal TP/SL hit index and signal-exit comparisons."
    - "Unsupported modes fall back to current exact path."
  performance:
    - "long_short_reversal service wall improves by >=10-15% versus current exact on Mac Studio API-runner path."
    - "long_only service wall and memory do not regress."
    - "service_total_without_warmup, exact scoring timers, CPU and RSS are recorded separately."
  decision:
    - "Reject or keep disabled if the win is only local/kernel-level and not present in API-runner service wall."
    - "If multiple repair ideas look plausible, implement only the one supported by Stage 13R and leave the rest as future work."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py"
    expect: "focused exact semantics and tie-rule tests pass or narrower justified equivalent"
  - cmd: "uv run ruff check src/trading/contexts/backtest/application/services/v2 scripts/backtest/run_api_runner_benchmark_parity.py"
    expect: "passes"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

validation_strategy:
  depth: benchmark
  e2e_required: true
  acceptance_surfaces:
    - "Mac Studio API-runner A/B benchmark over ssh macstudio"
    - "exact TP/SL parity and SL-wins tie tests"
    - "stage ledger and benchmark_summary.md"
  tests_only_allowed_reason: "not allowed; this is a hot-path repair candidate"
  evidence_target: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_14r_tp_sl_reversal_repair/

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  required_update: true
  current_stage: "14R"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Stage 13R evidence used"
    - "Kernel/shortcut"
    - "Parity"
    - "Benchmark"
    - "Rollback"
    - "Ledger and commit"
---

# Task

Implement Stage 14R only after Stage 13R has identified a defensible reversal
cost center. The default answer to missing telemetry is to block, not to guess.
