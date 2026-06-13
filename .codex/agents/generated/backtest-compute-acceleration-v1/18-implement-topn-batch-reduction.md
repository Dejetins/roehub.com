---
prompt_name: backtest_compute_acceleration_stage_18_topn_batch_reduction
repo: roehub.com
branch: main
scope: "Measure top-N/result assembly cost and implement stable batch reduction only if assembly is a proven hot path."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 18 top-N assembly plan"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 17 gate"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "notebook top-k failure and strict hash drift warning"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/top_result_assembly.py
      why: "current heap update, result assembly and persistence shape"
    - path: src/trading/contexts/backtest/application/services/v2/job_orchestration.py
      why: "service pipeline boundary and persisted top-N handoff"
    - path: scripts/backtest/run_api_runner_benchmark_parity.py
      why: "reporting assembly timers"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "top-N ordering, variant_hash, persisted payload or tie-break may change"
    timing: before implementation
  - skill: backend-performance-evidence
    use_when: "measuring assembly timers and batch merge impact"
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
  acceptance_testing: "Run assembly telemetry and any batch-reduction A/B over SSH on Mac Studio."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_18_topn_batch_reduction/
  write_policy: "Evidence only under evidence_output_dir; no canonical artifact writes."

hard_requirements:
  previous_stage_required: "17 accepted"
  baseline_code_required: "Benchmark control and candidate must run from a checkout/runtime containing the Stage 05+12 production default: Stage 05 for no-risk arity 6 and Stage 12 for no-risk arity 7. If live production runtime is used or claimed, verify /opt/roehub/app code state and ROEHUB_BACKTEST_MATRIX_BACKEND_MODE env state before benchmarking."
  implementation_allowed: true
  benchmark_required: true
  docs_update_required: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted or accepted_for_learning, update ledger/evidence/docs, stage only scoped files, and commit to main with a stage-specific message."
  - "For blocked or rejected, do not commit production runtime changes; commit only docs/evidence needed to preserve the decision."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not repeat notebook top-k accepted-timing mistake."
  - "Do not change stable tie-break, variant_hash or persisted top-N payload shape."
  - "Do not accept heap/local timer win without API-runner service-wall improvement."

task:
  summary: "First expose top-N/result assembly timers; if they are hot enough, implement block-local top-M reduction with deterministic global merge."
  required_timers:
    - "heap_update_ms"
    - "top_result_proxy_fill_ms"
    - "variant_hash_ms"
    - "canonical_params_build_ms"
    - "payload_json_ms"
    - "db_persist_ms"
  optional_implementation:
    - "candidate metrics block -> stable local top-M -> deterministic global merge"
    - "tie-break order: ranking metric, variant_hash, combo ordinal"

acceptance:
  correctness:
    - "Top-50 identity/order, variant_hash and persisted payload shape unchanged."
    - "Tie-break tests cover equal ranking metric cases."
  performance:
    - "If assembly is not a hot path, stop after telemetry and mark accepted_for_learning only."
    - "If batch merge is implemented, service wall improves materially with no memory regression."
  decision:
    - "Reject runtime batch merge if assembly timers are not material or top-N identity drifts."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "focused top-N/tie-break/result assembly tests pass or narrower justified equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  required_update: true
  current_stage: "18"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Assembly timers"
    - "Implementation"
    - "Parity"
    - "Benchmark"
    - "Ledger and commit"
---

# Task

Implement Stage 18 top-N/result assembly telemetry first. Add batch reduction
only if the evidence proves assembly is a material cost center.
