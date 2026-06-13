---
prompt_name: backtest_compute_acceleration_stage_19_thread_scaling_benchmark
repo: roehub.com
branch: main
scope: "Benchmark NUMBA_NUM_THREADS by workload and update worker policy only if service-wall evidence supports it."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 19 thread scaling plan"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 18 gate"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "fixed-overhead pattern from failed stages"
  task_entrypoints:
    - path: scripts/backtest/run_api_runner_benchmark_parity.py
      why: "API-runner benchmark"
    - path: infra/macos/launchd/
      why: "Mac Studio worker environment references"
    - path: src/trading/contexts/backtest/
      why: "worker/thread configuration if accepted"

skill_routing:
  - skill: backend-performance-evidence
    use_when: "benchmarking thread count, service wall, CPU and memory"
    timing: during verification
  - skill: contract-impact-analysis
    use_when: "worker config/default thread policy may change"
    timing: before any config change
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
  acceptance_testing: "Run all thread-count benchmarks over SSH on Mac Studio."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_19_thread_scaling/
  write_policy: "Evidence only under evidence_output_dir; do not write canonical artifacts."

hard_requirements:
  previous_stage_required: "18 accepted, or accepted_for_learning with assembly-not-hot decision and no runtime merge"
  implementation_allowed: true
  benchmark_required: true
  docs_update_required: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted or accepted_for_learning, update ledger/evidence/docs, stage only scoped files, and commit to main with a stage-specific message."
  - "For blocked or rejected, do not commit production runtime changes; commit only docs/evidence needed to preserve the decision."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not change worker thread defaults based on microbenchmarks only."
  - "Do not generalize Mac Studio thread policy to other hardware without new evidence."
  - "Do not ignore oversubscription or memory pressure."

task:
  summary: "Run a thread-scaling matrix for accepted no-risk, TP/SL and high-arity workloads."
  thread_counts:
    - 1
    - 2
    - 4
    - 6
    - 8
    - 12
  required_rows:
    - "none/arity_6/long_only"
    - "none/arity_6/long_short_reversal"
    - "tp_sl_grid/arity_6/long_only"
    - "tp_sl_grid/arity_6/long_short_reversal"
    - "none/arity_7 if Stage 12 exists or current exact high-arity fixture is available"
  required_metrics:
    - "service wall"
    - "exact_scoring"
    - "combo_iteration"
    - "RSS peak"
    - "CPU utilization"
    - "candidate/sec"
    - "trade-cell/sec"

acceptance:
  correctness:
    - "Requests, artifacts, warmup and result identity remain unchanged for every thread count."
  performance:
    - "Select a best thread count table by workload only when variance is controlled."
    - "Worker config update is allowed only if service wall improves without memory/oversubscription regression."
  decision:
    - "If best thread count is workload-specific, implement a safe selector or record evidence only."

quality_gates:
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  required_update: true
  current_stage: "19"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Thread matrix"
    - "Policy decision"
    - "Checks"
    - "Ledger and commit"
---

# Task

Run Stage 19 thread scaling benchmarks. Do not change worker defaults unless the
service-wall evidence is clear and the ledger accepts the policy.
