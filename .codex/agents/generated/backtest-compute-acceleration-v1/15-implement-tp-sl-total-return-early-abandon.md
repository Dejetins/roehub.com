---
prompt_name: backtest_compute_acceleration_stage_15_tp_sl_total_return_early_abandon
repo: roehub.com
branch: main
scope: "Closed: Stage 15 runtime candidate was rejected and accepted for learning only; do not execute this implementation prompt."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 15 exact-safe bound scope"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 15 gate and removed Stage 13/14 guardrails"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "avoid confirm_prefilter and non-service-wall wins"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/tp_sl_cells.py
      why: "TP/SL scoring loop"
    - path: src/trading/contexts/backtest/application/services/v2/top_result_assembly.py
      why: "ranking metric and top-N assembly semantics"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "ranking, pruning or candidate eligibility may change"
    timing: before implementation
  - skill: backend-performance-evidence
    use_when: "measuring early-abandon speedup"
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
  acceptance_testing: "Run A/B benchmark/testing evidence over SSH on Mac Studio; local runs are preflight only."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_15_tp_sl_early_abandon/
  write_policy: "Evidence only under evidence_output_dir; no canonical artifact writes."

hard_requirements:
  previous_stage_required: "Stage 15 has already been benchmarked and closed as accepted_for_learning; Stage 16 telemetry is the next executable prompt."
  baseline_code_required: "Benchmark control and candidate must run from a checkout/runtime containing the Stage 05+12 production default for no-risk rows, and current exact TP/SL scoring for TP/SL rows. Do not restore or benchmark against removed Stage 13/13S/13R/14/14R runtime, prompts or harnesses."
  production_default_benchmark_command: "uv run python scripts/backtest/run_api_runner_benchmark_parity.py --env-file /Users/daniildegtyarev/.config/roehub/roehub.env --stage-05-12-production-default-rows --out-dir docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_05_12_production_default_stage15_preflight"
  stage_15_benchmark_command: "Run Mac Studio A/B through scripts/backtest/run_api_runner_benchmark_parity.py using current exact TP/SL control vs Stage 15 default-on candidate on the same TP/SL heavy rows. If the harness has no dedicated Stage 15 row selector, add one in the scoped implementation diff rather than reusing removed Stage 13/14 harnesses."
  benchmark_claim_rule: "Acceptance benchmark evidence is valid only if measured heavy jobs are claimed by the benchmark harness process; if the live launchd backtest-job-runner claims a benchmark job, record the run as diagnostic and rerun with isolation or explicit claim verification."
  implementation_allowed: false
  benchmark_required: false
  docs_update_required: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted or accepted_for_learning, update ledger/evidence/docs, stage only scoped files, and commit to main with a stage-specific message."
  - "For blocked or rejected, do not commit production runtime changes; commit only docs/evidence needed to preserve the decision."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not implement confirm_prefilter or a second full pass."
  - "Do not apply early abandon to profit_factor, sharpe, return_over_max_drawdown or unsupported rankings."
  - "Do not use an approximate score bound."
  - "Do not remove candidates unless exact-safe proof applies."
  - "Do not use this stage to revive or bypass the removed Stage 13/14 TP/SL branch."

task:
  summary: "Do not implement this prompt. Stage 15 evidence is recorded in the ledger and benchmark_iterations; continue with Stage 16 telemetry."
  required_design:
    - "Represent the bound in log-return or another proven upper-bound domain."
    - "Include fees/slippage in the maximum possible trade gain bound."
    - "Fallback to current exact path when ranking/sizing/mode is unsupported."
    - "Expose an env/config override for A/B default-off vs candidate."
  required_counters:
    - "early_abandon_candidates"
    - "early_abandon_cells"
    - "early_abandon_bound_ms"
    - "tp_sl_exact_scoring"
    - "service_wall_clock_s"

acceptance:
  correctness:
    - "Top-N identity/order and metrics match the baseline within accepted tolerance."
    - "A test proves the bound cannot prune a later top candidate for the supported mode."
  performance:
    - "Mac Studio API-runner service wall improves materially versus the current exact TP/SL baseline on both TP/SL Stage 15 rows."
    - "service_total_without_warmup and memory cleanup do not regress."
  decision:
    - "If the proof is incomplete or speedup is only local, keep disabled and mark rejected/accepted_for_learning as appropriate."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "focused ranking/bound/TP-SL tests pass or narrower justified equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  required_update: true
  current_stage: "15"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Exact-safe proof"
    - "Parity"
    - "Benchmark"
    - "Fallbacks"
    - "Ledger and commit"
---

# Task

Do not execute Stage 15. The runtime candidate was rejected after Mac Studio A/B
evidence showed `0` pruned candidates and service-wall regression. Treat this
prompt as closed/superseded and continue with Stage 16 trade-window reuse
telemetry.
