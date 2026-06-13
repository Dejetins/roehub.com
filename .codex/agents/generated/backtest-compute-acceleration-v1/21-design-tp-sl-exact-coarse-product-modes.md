---
prompt_name: backtest_compute_acceleration_stage_21_tp_sl_exact_coarse_product_modes
repo: roehub.com
branch: main
scope: "Prepare an architecture/product decision for exact full-grid TP/SL versus clearly marked approximate coarse-grid search."

language:
  implementation: none
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 21 product-mode scope"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "Stage 20 gate and accepted exact baselines"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "approximate/default stop-list"
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/
      why: "existing cost model evidence for TP/SL full grid"
    - path: apps/
      why: "only if UI/API mode naming needs impact analysis"

skill_routing:
  - skill: architecture-design
    use_when: "drafting exact/coarse mode architecture and rollout"
    timing: during design
  - skill: contract-impact-analysis
    use_when: "API, request hash, cache identity, UI or persisted semantics may change"
    timing: before recommendations
  - skill: backend-performance-evidence
    use_when: "summarizing benchmark cost model and admission thresholds"
    timing: during evidence review

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
  acceptance_testing: "No runtime benchmark is required unless the prompt explicitly opens a cost-model refresh; any refresh must run over SSH on Mac Studio."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_21_tp_sl_exact_coarse_modes/
  write_policy: "If cost-model evidence is generated, write it under evidence_output_dir; do not write canonical artifacts."

hard_requirements:
  previous_stage_required: "20 accepted"
  baseline_code_required: "Architecture and any optional benchmark references must treat the Stage 05+12 production default as the minimum accepted-code baseline; live-production claims require /opt/roehub/app code/env state verification."
  production_default_benchmark_command: "Optional cost-model refresh command: uv run python scripts/backtest/run_api_runner_benchmark_parity.py --env-file /Users/daniildegtyarev/.config/roehub/roehub.env --stage-05-12-production-default-rows --out-dir docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_05_12_production_default_stage21_reference"
  benchmark_claim_rule: "Any optional live-production benchmark reference is valid only if measured heavy jobs are claimed by the benchmark harness process; if the live launchd backtest-job-runner claims a benchmark job, record the run as diagnostic and rerun with isolation or explicit claim verification."
  implementation_allowed: false
  benchmark_required: false
  docs_update_required: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted_for_learning, update ledger/docs, stage only scoped files, and commit to main with a stage-specific message."
  - "For blocked, commit only docs needed to preserve the decision."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not implement approximate/coarse TP/SL runtime."
  - "Do not change exact full-grid default semantics."
  - "Do not change API/request hash/cache identity without an explicit approved follow-up plan."

task:
  summary: "Create an ADR or architecture note that separates exact full-grid TP/SL from a possible approximate coarse-grid UX mode."
  required_content:
    - "Exact full-grid mode remains the default exact semantics."
    - "Approximate coarse-grid mode must be visibly labeled and product-approved."
    - "Optional exact refine for selected candidates must be described separately from approximate search."
    - "Cost-aware admission policy must use candidate_count, expected_trade_count, tp_count, sl_count and service-wall evidence."
    - "Contract impact must classify API, request hash, cache identity, persisted schema, UI and benchmark dimensions."
    - "Follow-up implementation prompts are allowed only after product approval."

acceptance:
  correctness:
    - "No runtime code changes."
    - "No exact-default behavior changes."
  documentation:
    - "ADR/note links back to plan, ledger and negative-results stop-list."
    - "Ledger records Stage 21 as design-only accepted_for_learning or blocked."

quality_gates:
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  required_update: true
  current_stage: "21"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Decision"
    - "Contract impact"
    - "Follow-up prompts"
    - "Checks"
    - "Ledger and commit"
---

# Task

Prepare Stage 21 exact/coarse TP/SL mode architecture decision. Do not implement
approximate runtime behavior in this prompt.
