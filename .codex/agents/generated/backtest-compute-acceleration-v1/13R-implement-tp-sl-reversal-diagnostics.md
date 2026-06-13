---
prompt_name: backtest_compute_acceleration_stage_13r_tp_sl_reversal_diagnostics
repo: roehub.com
branch: main
scope: "Add telemetry-only TP/SL reversal diagnostics to identify the long_short_reversal cost center before any repair implementation."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract and telemetry safety rules"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 13R scope and counters"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 13 rejection and Stage 13R acceptance rules"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "avoid Stage 06 cache and rejected TP/SL global autotune patterns"
    - path: docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_13_tp_sl_block_autotune/model_handoff_report.md
      why: "recommended reversal diagnostic counters and hypotheses"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
      why: "matrix-cell TP/SL work, cell blocks and trade-cell counters"
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_exact.py
      why: "current exact TP/SL reversal semantics and fallback path"
    - path: src/trading/contexts/backtest/application/services/v2/tp_sl_hit_times.py
      why: "hit-time arrays and signal-exit comparison source"
    - path: scripts/backtest/run_api_runner_benchmark_parity.py
      why: "API-runner benchmark/reporting fields"
  consult_if_needed:
    - path: tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py
      read_when: "adding exact-path diagnostics tests"
    - path: tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py
      read_when: "adding hit-time/signal-exit diagnostic tests"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "adding telemetry fields that may affect DTO/config/report semantics"
    timing: before implementation
    reason: "diagnostics must not change public result identity or request/cache keys"
  - skill: backend-performance-evidence
    use_when: "measuring telemetry overhead and interpreting cost attribution"
    timing: during verification
    reason: "Stage 13R acceptance is overhead-bounded and evidence-driven"
  - skill: backend-quality-gates
    use_when: "Python gates fail"
    timing: during verification
    reason: "focused service tests and lint/type checks must pass"

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
  acceptance_testing: "Run telemetry overhead and diagnostic benchmark over SSH on Mac Studio; local runs are preflight only."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_13r_tp_sl_reversal_diagnostics/
  write_policy: "Evidence only under evidence_output_dir; do not write canonical artifacts."

hard_requirements:
  previous_stage_required: "13 rejected with Stage 13R explicitly opened in the ledger"
  baseline_code_required: "Benchmark control and candidate must run from a checkout/runtime containing Stage 05+12 production default and preserved Stage 13 evidence."
  production_default_benchmark_command: "uv run python scripts/backtest/run_api_runner_benchmark_parity.py --env-file /Users/daniildegtyarev/.config/roehub/roehub.env --stage-05-12-production-default-rows --out-dir docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_05_12_production_default_stage13r_baseline"
  benchmark_claim_rule: "Acceptance benchmark evidence is valid only if measured heavy jobs are claimed by the benchmark harness process; if the live launchd backtest-job-runner claims a benchmark job, record the run as diagnostic and rerun with isolation or explicit claim verification."
  implementation_allowed: true
  benchmark_required: true
  docs_update_required: true
  telemetry_only: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted_for_learning or rejected, update ledger/evidence/docs, stage only scoped files, and commit to main with a Stage 13R-specific message."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not change scoring, candidate eligibility, top-N ordering, best_tp/best_sl or TP/SL tie behavior."
  - "Do not implement split-by-side kernel, signal-exit shortcut, runtime cache or grouped scoring."
  - "Do not enable TP/SL backend default or selector."
  - "Do not use Python dict caches in the hot path."
  - "Do not change publisher/precompute artifacts."

task:
  summary: "Add counters that explain why tp_sl_grid/arity_6/long_short_reversal does not improve with block-shape tuning."
  required_rows:
    - "tp_sl_grid/arity_6/long_short_reversal current exact"
    - "tp_sl_grid/arity_6/long_short_reversal matrix 64x64"
    - "tp_sl_grid/arity_6/long_short_reversal matrix 64x128"
    - "tp_sl_grid/arity_6/long_only matrix 64x128 as contrast"
  required_counter_groups:
    workload_shape:
      - tp_sl_candidates_total
      - tp_sl_candidates_scored
      - tp_sl_combo_count_planned
      - tp_sl_candidates_after_proxy
      - tp_sl_total_trades
      - tp_sl_long_trades
      - tp_sl_short_trades
      - tp_sl_reversal_transitions
      - tp_sl_signal_exit_trades
      - tp_sl_close_on_end_trades
      - tp_sl_avg_trades_per_candidate
      - tp_sl_p50_trades_per_candidate
      - tp_sl_p95_trades_per_candidate
      - tp_sl_max_trades_per_candidate
    cell_work:
      - tp_sl_tp_count
      - tp_sl_sl_count
      - tp_sl_total_cells
      - tp_sl_total_trade_cell_evals
      - tp_sl_trade_cell_evals_per_sec
      - tp_sl_cell_blocks_total
      - tp_sl_cell_blocks_long
      - tp_sl_cell_blocks_short
      - tp_sl_cell_blocks_mixed
      - tp_sl_cell_block_shape
    exit_reason_distribution:
      - tp_sl_exit_reason_tp_count
      - tp_sl_exit_reason_sl_count
      - tp_sl_exit_reason_signal_count
      - tp_sl_exit_reason_end_count
      - tp_sl_exit_reason_tie_sl_wins_count
      - tp_sl_signal_exit_before_any_hit_count
      - tp_sl_signal_exit_before_any_hit_cell_count
      - tp_sl_signal_exit_dominant_cell_pct
    time_breakdown:
      - tp_sl_prepare_ms
      - tp_sl_combo_iteration_ms
      - tp_sl_consensus_build_ms
      - tp_sl_trade_tape_extract_ms
      - tp_sl_cell_scoring_ms
      - tp_sl_best_cell_reduce_ms
      - tp_sl_heap_update_ms
      - tp_sl_top_result_assembly_ms
      - tp_sl_payload_build_ms
    hit_time_access:
      - tp_sl_long_hit_time_reads
      - tp_sl_short_hit_time_reads
      - tp_sl_no_hit_sentinel_reads
      - tp_sl_unique_entry_indices
      - tp_sl_total_trade_windows
      - tp_sl_unique_trade_windows
      - tp_sl_trade_window_reuse_ratio
      - tp_sl_weighted_reuse_by_cell_count
      - tp_sl_cache_candidate_savings_estimate
    allocation:
      - tp_sl_cell_metric_buffer_bytes
      - tp_sl_trade_tape_buffer_bytes
      - tp_sl_temp_alloc_count
      - tp_sl_temp_alloc_bytes

acceptance:
  correctness:
    - "No behavior change: top-N identity/order, best_tp, best_sl and metrics remain unchanged."
    - "Telemetry fields are additive and nullable where a backend cannot produce them."
  performance:
    - "Service wall overhead is <=3% versus the same row without diagnostics, or diagnostics stay default-off and are marked rejected for production use."
    - "Evidence records current exact and matrix-cell paths, not only one side."
  decision:
    - "Stage status may be accepted_for_learning only."
    - "Ledger must name the likely dominant reversal cost center and recommend exactly one next action: split-by-side kernel, signal-exit shortcut, compiled grouping, thread-scaling, allocation cleanup, or no-op."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2/test_tp_sl_exact_scoring_service.py tests/unit/contexts/backtest/application/services/v2/test_tp_sl_hit_times_service.py"
    expect: "focused diagnostic/semantics tests pass or narrower justified equivalent"
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
    - "Mac Studio API-runner benchmark over ssh macstudio"
    - "diagnostic counter overhead comparison"
    - "stage ledger and benchmark_summary.md"
  tests_only_allowed_reason: "not allowed; this stage must prove telemetry overhead and diagnostic completeness"
  evidence_target: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_13r_tp_sl_reversal_diagnostics/

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  plan_doc: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
  required_update: true
  current_stage: "13R"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Telemetry counters"
    - "Overhead"
    - "Dominant cost finding"
    - "Next action"
    - "Ledger and commit"
---

# Task

Implement Stage 13R TP/SL reversal diagnostics as telemetry only. Do not repair
or optimize reversal in this stage; produce evidence that makes the next repair
stage technically defensible.
