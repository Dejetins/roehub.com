---
prompt_name: backtest_compute_acceleration_stage_12_compiled_prefix_product_traversal
repo: roehub.com
branch: main
scope: "Implement fused compiled prefix product traversal for high-arity no-risk rows without using rejected Python traversal."

language:
  implementation: python
  agent_report: ru

context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/backtest/backtest-compute-acceleration-plan-v1.md
      why: "Stage 12 plan, baselines, benchmark policy"
    - path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
      why: "current stage gate and accepted/rejected history"
    - path: docs/architecture/backtest/backtest-compute-acceleration-negative-results-v1.md
      why: "stop-list for rejected acceleration methods"
  task_entrypoints:
    - path: src/trading/contexts/backtest/application/services/v2/combo_planning.py
      why: "current product combo enumeration"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/bitsets.py
      why: "bitset packing and consensus semantics"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/no_risk_score.py
      why: "accepted Stage 05 scorer"
    - path: src/trading/contexts/backtest/application/services/v2/matrix_backend/trade_tape.py
      why: "sparse trade tape semantics"
  consult_if_needed:
    - path: docs/architecture/backtest/benchmark_iterations/2026-06-10_matrix_bitset_stage_10_high_arity_pruning_arity7_partial/
      why: "Stage 10 high-arity traversal and exact-scoring evidence"

skill_routing:
  - skill: contract-impact-analysis
    use_when: "candidate enumeration, variant identity, request hash, pruning, or ranking semantics may change"
    timing: before implementation
  - skill: numba
    use_when: "implementing compiled CPU loops or troubleshooting JIT typing/performance"
    timing: during implementation
  - skill: backend-performance-evidence
    use_when: "benchmarking API-runner service wall, combo_iteration, exact_scoring, CPU and memory"
    timing: during verification
  - skill: backend-quality-gates
    use_when: "focused Python gates fail"
    timing: during verification

runtime_env_sources:
  mac_studio_native_env_file: /Users/daniildegtyarev/.config/roehub/roehub.env
  benchmark_env_file_arg: "--env-file"
  mac_studio_required_runtime_env:
    ROEHUB_ENV: prod
    ROEHUB_BACKTEST_ARTIFACTS_CONFIG: configs/prod/backtest_artifacts.yaml
  mac_studio_artifact_root: /opt/roehub/state/backtest_artifacts/v2
  secret_reporting_rule: "Report only key/path presence, never DSN, password, token, API key, or secret values."

mac_studio_test_execution:
  ssh_alias: macstudio
  repo_checkout: /Users/daniildegtyarev/Projects/roehub.com
  command_prefix: "ssh macstudio 'cd /Users/daniildegtyarev/Projects/roehub.com && <command>'"
  acceptance_testing: "Run acceptance benchmark/testing evidence over SSH on Mac Studio; local runs are preflight only."
  sync_rule: "Before SSH testing, ensure the Mac Studio checkout contains the exact candidate code being measured and record commit SHA or dirty state."
  source_artifacts:
    root: /opt/roehub/state/backtest_artifacts/v2
    symbol_current: /opt/roehub/state/backtest_artifacts/v2/BTCUSDT/current.yaml
    active_manifest: "resolve from BTCUSDT/current.yaml; read-only"
  evidence_output_dir: docs/architecture/backtest/benchmark_iterations/<date>_matrix_bitset_stage_12_compiled_prefix_traversal/
  write_policy: "Save evidence under evidence_output_dir; never write to source_artifacts.root, current.yaml, active slots, or publisher outputs."

hard_requirements:
  previous_stage_required: "2026-06-13 continuation plan opens Stage 12"
  implementation_allowed: true
  benchmark_required: true
  docs_update_required: true
  default_path_allowed_only_after_gate: true

delivery_requirements:
  - "Work from branch main; stop and report a blocker if the checkout is not main."
  - "After accepted or accepted_for_learning, update ledger/evidence/docs, stage only scoped files, and commit to main with a stage-specific message."
  - "For blocked or rejected, do not commit production runtime changes; commit only docs/evidence needed to preserve the decision."
  - "Do not push or deploy unless the user explicitly asks."

non_goals:
  - "Do not reuse Stage 10 Python branch traversal."
  - "Do not revive Stage 06 consensus signature cache."
  - "Do not add sidecar .npy publisher or canonical manifest changes."
  - "Do not use approximate beam search or any score/ranking upper bound without a separate proof."
  - "Do not enable arity 2/3 matrix defaults."

task:
  summary: "Implement a compiled/iterative prefix product traversal for product-form indicator pools and fuse prefix consensus, selectivity ordering, exact-safe eligibility pruning, and Stage 05 scoring handoff."
  required_scope:
    - "risk.mode=none"
    - "arity=6 and arity=7 where the fixture/request is available"
    - "direction_mode=long_only and long_short_reversal"
    - "Stage 05 matrix_bitset_no_risk_v1 remains the scorer or baseline fallback"
  required_design:
    - "Hot traversal must be compiled, e.g. Numba/Cython/Rust/C++; no Python recursion/object allocation in the hot path."
    - "Prefix state must reuse AND consensus across levels."
    - "Selectivity order may change compute order only; output variant order, variant_hash, combo ordinal tie-break and persisted top-N stay canonical."
    - "Exact-safe pruning may use active bars, possible closed trades and exposure eligibility upper bounds only."
    - "Provide an env/config override to force current traversal for A/B."

acceptance:
  correctness:
    - "Top-50 identity/order matches the baseline for comparable rows, or the existing accepted bounded metric tolerance is documented where exact identity is not applicable."
    - "variant_key, variant_hash, ranking, fees/slippage, sizing and close_on_end semantics are unchanged."
  performance:
    - "Mac Studio API-runner A/B uses the same env file, artifacts, request semantics and warmup policy."
    - "none/arity_7 service wall improves by >=20% and combo_iteration is materially lower."
    - "none/arity_6 long_only and long_short_reversal do not regress versus Stage 05/default/current accepted path."
    - "service_total_without_warmup, API wall and memory cleanup do not regress."
  decision:
    - "If the speed gate fails, remove or keep disabled any runtime candidate, record rejected evidence, and do not proceed to Stage 13."

quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/backtest/application/services/v2"
    expect: "focused tests covering combo traversal, signal semantics and no-risk parity pass or narrower justified equivalent"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes if docs changed"
  - cmd: "git diff --check"
    expect: "passes"

stage_execution_ledger:
  path: docs/architecture/backtest/backtest-compute-acceleration-v1-stage-ledger.md
  required_update: true
  current_stage: "12"
  record_required_fields:
    - "baseline/current/candidate timings"
    - "combo_iteration, exact_scoring, service_wall_clock_s, memory"
    - "prefix counters and selectivity order"
    - "top-N/variant_hash parity"
    - "contract impact"
    - "commit SHA on main if accepted or accepted_for_learning"

final_report_format:
  language: ru
  sections:
    - "Stage status"
    - "Implementation"
    - "Parity"
    - "Benchmark"
    - "Ledger and commit"
    - "Residual risks"
---

# Task

Implement Stage 12 compiled prefix product traversal. Do not move forward unless
Mac Studio API-runner evidence proves the required service-wall improvement and
the ledger records the decision.
