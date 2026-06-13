---
prompt_name: backtest_compute_acceleration_stage_20_allocation_reuse_scratch_buffers
repo: roehub.com
branch: main
scope: "Measure allocation churn and add per-child scratch buffers only if they reduce service wall or RSS without cleanup regression."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 20 allocation scope"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 19 gate"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "fixed overhead pattern"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/
      why: "matrix/cell buffers"
    - path: src/trading/contexts/backtest/application/services/v2/
      why: "child job lifecycle and cleanup"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "buffer reuse could affect job isolation, cache identity or cleanup semantics"
    timing: before implementation
  - skill: backend-performance-evidence
    use_when: "measuring allocations, RSS and service wall"
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
  acceptance_testing: "Run allocation telemetry and A/B benchmarks over SSH on Mac Studio; local runs are preflight only."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_20_allocation_reuse/
  write_policy: "Evidence only under evidence_output_dir; do not write canonical artifacts."

hard_requirements:
  previous_stage_required: "19 accepted_for_learning with thread policy decision, or accepted if worker config changed"
  implementation_allowed: true
  benchmark_required: true
  docs_update_required: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted or accepted_for_learning, update ledger/evidence/docs, stage only scoped files, and commit to main with a stage-specific message."
  - "For blocked or rejected, do not commit production runtime changes; commit only docs/evidence needed to preserve the decision."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not add global cross-job cache."
  - "Do not keep buffers alive across child process lifetime in a way that breaks cleanup."
  - "Do not accept allocation telemetry as speedup without service-wall/RSS evidence."

task:
  summary: "Add lightweight allocation telemetry, then implement per-child scratch buffers only if telemetry shows material allocation churn."
  required_counters:
    - "matrix_buffers_allocated"
    - "matrix_buffer_bytes"
    - "cell_metric_buffer_bytes"
    - "trade_tape_buffer_bytes"
    - "temporary_array_count"
    - "scratch_buffer_reuse_count"
  candidate_buffers:
    - "long_bits_buffer"
    - "short_bits_buffer"
    - "trade_tape_buffer"
    - "tp_sl_cell_metrics_buffer"
    - "candidate_best_buffer"
    - "block_top_buffer"

acceptance:
  correctness:
    - "Top-N identity/order and metrics unchanged."
    - "Buffer reuse is per-child/job-safe and reset between uses."
  performance:
    - "Allocation/RSS or service wall improves materially on target rows."
    - "Memory cleanup and RSS peak do not regress."
  decision:
    - "If allocation churn is not material, keep telemetry only and do not implement scratch complexity."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "focused buffer lifecycle/parity tests pass or narrower justified equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  required_update: true
  current_stage: "20"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Allocation telemetry"
    - "Implementation"
    - "Benchmark"
    - "Cleanup"
    - "Ledger and commit"
---

# Task

Implement Stage 20 allocation telemetry and per-child scratch buffers only if the
Mac Studio evidence shows material allocation or RSS benefit.
