---
prompt_name: backtest_compute_acceleration_stage_14_tp_sl_monotonic_cell_kernel
repo: roehub.com
branch: main
scope: "Superseded: do not implement the original TP/SL monotonic cell kernel against the rejected Stage 13 winner."

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
    - path: docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_13_tp_sl_block_autotune/model_handoff_report.md
      why: "Stage 13 rejection interpretation and replacement-stage recommendation"
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
  previous_stage_required: "blocked: Stage 13 was rejected; use 13S, 13R and 14R prompts instead"
  baseline_code_required: "Benchmark control and candidate must run from a checkout/runtime containing the Stage 05+12 production default: Stage 05 for no-risk arity 6 and Stage 12 for no-risk arity 7. If live production runtime is used or claimed, verify /opt/roehub/app code state and ROEHUB_BACKTEST_MATRIX_BACKEND_MODE env state before benchmarking."
  production_default_benchmark_command: "uv run python scripts/backtest/run_api_runner_benchmark_parity.py --env-file /Users/daniildegtyarev/.config/roehub/roehub.env --stage-05-12-production-default-rows --out-dir docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_05_12_production_default_stage14_baseline"
  benchmark_claim_rule: "Acceptance benchmark evidence is valid only if measured heavy jobs are claimed by the benchmark harness process; if the live launchd backtest-job-runner claims a benchmark job, record the run as diagnostic and rerun with isolation or explicit claim verification."
  implementation_allowed: false
  benchmark_required: false
  docs_update_required: true
  superseded_by:
    - .codex/agents/generated/backtest-compute-acceleration-v1/13S-implement-tp-sl-selective-production-gate.md
    - .codex/agents/generated/backtest-compute-acceleration-v1/13R-implement-tp-sl-reversal-diagnostics.md
    - .codex/agents/generated/backtest-compute-acceleration-v1/14R-implement-tp-sl-reversal-kernel-repair.md

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
  summary: "Do not execute this original Stage 14 prompt. Stage 13 was rejected and the monotonic-kernel dependency is superseded by Stage 13R diagnostics and Stage 14R repair."
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
    - "No implementation is allowed from this prompt."
    - "If opened by mistake, update the ledger/report as blocked and point to Stage 13S/13R/14R."
  performance:
    - "N/A; this prompt is superseded."
  decision:
    - "Blocked/superseded only."

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

Do not implement this original Stage 14 prompt. Stage 13 was rejected, so there
is no accepted Stage 13 winner to beat. Use Stage 13S for the narrow selector
gate, Stage 13R for reversal diagnostics, and Stage 14R for any repair after
diagnostics.
