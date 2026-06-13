---
prompt_name: backtest_compute_acceleration_stage_15_tp_sl_total_return_early_abandon
repo: roehub.com
branch: main
scope: "Implement exact-safe TP/SL early abandon for total_return_pct ranking only."

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
      why: "Stage 14 gate"
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
  previous_stage_required: "14 accepted"
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
  - "Do not implement confirm_prefilter or a second full pass."
  - "Do not apply early abandon to profit_factor, sharpe, return_over_max_drawdown or unsupported rankings."
  - "Do not use an approximate score bound."
  - "Do not remove candidates unless exact-safe proof applies."

task:
  summary: "Use an exact-safe optimistic remaining-return bound to abandon candidate/cell scoring only for ranking=total_return_pct desc and supported sizing."
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
    - "Mac Studio API-runner service wall improves materially versus Stage 14/baseline on both TP/SL Stage 15 rows."
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

Implement Stage 15 TP/SL total-return early abandon only for proven exact-safe
surfaces. Unsupported rankings must stay on the current exact path.
