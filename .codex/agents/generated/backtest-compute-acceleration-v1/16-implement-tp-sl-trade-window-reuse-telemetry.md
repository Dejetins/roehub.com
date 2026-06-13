---
prompt_name: backtest_compute_acceleration_stage_16_tp_sl_trade_window_reuse_telemetry
repo: roehub.com
branch: main
scope: "Add TP/SL trade-window reuse telemetry before any cache or grouped implementation."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 16 telemetry scope"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 15 gate"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "avoid Stage 06 cache pattern"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
      why: "TP/SL trade windows are produced/scored here"
    - path: scripts/backtest/run_api_runner_benchmark_parity.py
      why: "benchmark report fields"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "telemetry risks changing cache identity, request hash or result payload"
    timing: before implementation
  - skill: backend-performance-evidence
    use_when: "measuring telemetry overhead and reuse ratios"
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
  acceptance_testing: "Run telemetry overhead benchmark over SSH on Mac Studio; local runs are preflight only."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_16_trade_window_reuse_telemetry/
  write_policy: "Evidence only under evidence_output_dir; no source artifact writes."

hard_requirements:
  previous_stage_required: "15 accepted"
  baseline_code_required: "Benchmark control and candidate must run from a checkout/runtime containing the Stage 05+12 production default: Stage 05 for no-risk arity 6 and Stage 12 for no-risk arity 7. If live production runtime is used or claimed, verify /opt/roehub/app code state and ROEHUB_BACKTEST_MATRIX_BACKEND_MODE env state before benchmarking."
  implementation_allowed: true
  benchmark_required: true
  docs_update_required: true
  telemetry_only: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted or accepted_for_learning, update ledger/evidence/docs, stage only scoped files, and commit to main with a stage-specific message."
  - "For blocked or rejected, do not commit production runtime changes; commit only docs/evidence needed to preserve the decision."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not implement a runtime cache."
  - "Do not use Python dict cache in hot path."
  - "Do not group or reorder scoring results."
  - "Do not change top-N, metrics, request hash, cache identity or payload shape."

task:
  summary: "Add counters that quantify reusable TP/SL work keyed by side, entry_1m_idx and signal_exit_1m_idx."
  required_counters:
    - "tp_sl_total_trade_windows"
    - "tp_sl_unique_trade_windows"
    - "tp_sl_trade_window_reuse_ratio"
    - "tp_sl_weighted_reuse_by_cell_count"
    - "tp_sl_top_reused_window_count"
    - "tp_sl_cache_candidate_savings_estimate"
  decision_thresholds:
    - "If weighted reuse is low or telemetry overhead is material, close the topic as rejected/learning-only."
    - "If weighted reuse is high, update the ledger with a proposed future compiled grouping stage; do not implement that grouped stage here."

acceptance:
  correctness:
    - "Top-N identity/order and metrics unchanged."
    - "Telemetry fields are additive and nullable when unavailable."
  performance:
    - "Telemetry overhead is <=1% service wall or separately justified as diagnostics-only and disabled by default."
    - "Evidence records reuse ratios for both TP/SL arity-6 rows."
  decision:
    - "Accepted_for_learning only; this stage cannot enable production cache or grouped scoring."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "focused telemetry/reporting tests pass or narrower justified equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  required_update: true
  current_stage: "16"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Telemetry"
    - "Overhead"
    - "Reuse decision"
    - "Ledger and commit"
---

# Task

Implement Stage 16 TP/SL trade-window reuse telemetry only. Do not implement a
cache or grouped scorer in this stage.
