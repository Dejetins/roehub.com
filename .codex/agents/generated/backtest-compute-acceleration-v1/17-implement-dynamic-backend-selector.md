---
prompt_name: backtest_compute_acceleration_stage_17_dynamic_backend_selector
repo: roehub.com
branch: main
scope: "Later global backend selector only; do not use Stage 17 to bypass the Stage 13S narrow TP/SL selector gate."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 17 selector policy"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "prior accepted/rejected modes"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "arity 2/3 and sidecar/cache stop-list"
    - path: docs/architecture/backtest/benchmark_iterations/2026-06-13_matrix_bitset_stage_13_tp_sl_block_autotune/model_handoff_report.md
      why: "Stage 13S owns the narrow TP/SL selector policy"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/
      why: "backend selection and scoring orchestration"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/
      why: "matrix/cell backend ids and telemetry"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "backend default, request hash, result identity or config semantics may change"
    timing: before implementation
  - skill: backend-performance-evidence
    use_when: "A/B testing selector decisions and service wall"
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
  secret_reporting_rule: "Report only paths/key names, never secret values."

mac_studio_test_execution:
  ssh_alias: macstudio
  repo_checkout: /Users/daniildegtyarev/Projects/roehub.com
  command_prefix: "ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && <command>'"
  acceptance_testing: "Run selector A/B benchmark over SSH on Mac Studio; local runs are preflight only."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_17_dynamic_selector/
  write_policy: "Evidence only under evidence_output_dir; no canonical artifact writes."

hard_requirements:
  previous_stage_required: "Stage 13S decision complete for TP/SL selector policy, Stage 13R/14R decisions complete for reversal if TP/SL is in scope, and any Stage 16 reuse follow-up explicitly accepted if used"
  baseline_code_required: "Benchmark control and candidate must run from a checkout/runtime containing the Stage 05+12 production default: Stage 05 for no-risk arity 6 and Stage 12 for no-risk arity 7. If live production runtime is used or claimed, verify /opt/roehub/app code state and ROEHUB_BACKTEST_MATRIX_BACKEND_MODE env state before benchmarking."
  production_default_benchmark_command: "uv run python scripts/backtest/run_api_runner_benchmark_parity.py --env-file /Users/daniildegtyarev/.config/roehub/roehub.env --stage-05-12-production-default-rows --out-dir docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_05_12_production_default_stage17_baseline"
  benchmark_claim_rule: "Acceptance benchmark evidence is valid only if measured heavy jobs are claimed by the benchmark harness process; if the live launchd backtest-job-runner claims a benchmark job, record the run as diagnostic and rerun with isolation or explicit claim verification."
  implementation_allowed: true
  benchmark_required: true
  docs_update_required: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted or accepted_for_learning, update ledger/evidence/docs, stage only scoped files, and commit to main with a stage-specific message."
  - "For blocked or rejected, do not commit production runtime changes; commit only docs/evidence needed to preserve the decision."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not enable matrix_bitset_no_risk_v1 default for arity 2/3."
  - "Do not implement the Stage 13S TP/SL long-only selector here."
  - "Do not enable TP/SL backend default unless Stage 13S or a later TP/SL repair gate accepted that exact path."
  - "Do not use sidecar load as a default speedup."
  - "Do not change request hash or result semantics based on backend choice."

task:
  summary: "Implement a deterministic backend selector based on estimated work and fixed overhead, with explicit telemetry and env override."
  selector_inputs:
    - "risk_mode"
    - "arity"
    - "direction_mode"
    - "candidate_count"
    - "word_count"
    - "estimated_bit_ops"
    - "estimated_trade_cell_ops"
    - "estimated_fixed_overhead"
  required_rows:
    - "none/arity_1..3"
    - "none/arity_6/long_only"
    - "none/arity_6/long_short_reversal"
    - "none/arity_7/long_only and long_short_reversal when Stage 12 remains eligible"
    - "TP/SL rows only if a TP/SL backend is eligible from Stage 13S, Stage 14R, or a later accepted TP/SL stage"

acceptance:
  correctness:
    - "Top-N identity/order and result hashes unchanged on all selector rows."
    - "Selector decision, fallback reason and override state are visible in telemetry."
  performance:
    - "arity 1/2/3 service wall does not regress."
    - "arity 6/7 keeps accepted Stage 12 speed where the selector chooses compiled prefix traversal."
    - "Stage 05 remains available as rollback/default comparison for arity 6."
    - "Any TP/SL selector path must compare against its accepted baseline."
  decision:
    - "If selector cannot avoid known losing rows, reject or keep default-off."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "focused selector and backend parity tests pass or narrower justified equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  required_update: true
  current_stage: "17"

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

Implement Stage 17 only as a later global selector. The narrow TP/SL long-only
selector belongs to Stage 13S and must not be reimplemented or weakened here.
The selector must protect small workloads, preserve accepted Stage 12 no-risk
behavior where eligible, and keep Stage 05 as rollback/default comparison.
