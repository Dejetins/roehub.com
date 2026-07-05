---
prompt_name: 08n-candidate-quality-reclassification
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
prompt_pack_execution:
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  prompt_pack_dir: .codex/agents/generated/rl-trading-agent-platform-v1
  stage_ledger: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  mode: manual_sequential
  execution_mode: manual_sequential
  goal_md_policy: "GOAL.md is optional, not required by default"
  goal_driven_mode: "optional only over the same plan_doc/prompt_pack_dir/stage_ledger; no separate GOAL.md required"
  stage_gate: "read ledger before edits; run only when current_stage is 08N and prerequisites match"
  file_manifest_required: true
goal_mode_optional: true
goal_artifact_required: false
proof_boundary:
  label: target_host_non_production_quality_reclassification_pre_main
  changed_code_production_claim_allowed: false
  production_proof_requires:
    - not applicable because this prompt must not change runtime code or activate runtime behavior
  pre_main_not_production_proof:
    - target_host_*_pre_main
    - local browser harness
    - read-only host check
    - artifact-only check under `/opt/roehub/state/rl_trading/`
browser_auth:
  status: "N/A; this prompt must not read browser credentials or perform authenticated UI work"
  smoke_username: smoke_e2e_keycloak
  host_local_password_source: "/Users/daniildegtyarev/.config/roehub/roehub.env key ROEHUB_SMOKE_E2E_PASSWORD"
  redaction_rule: "do not read or print the password; browser/auth work is out of scope"
change_ownership:
  allowed_files:
    - scripts/rl_trading
    - tests/unit/scripts/rl_trading
    - docs/architecture/ml/rl-trading-agent-platform-v1.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08n-candidate-quality-reclassification.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
    - docs/architecture/README.md
  forbidden_without_user_approval:
    - exchange execution paths
    - live_execution behavior
    - API/UI behavior
    - model registry state mutation
    - runtime config activation
    - branch/worktree/stash/local-folder workflow changes
scope: "Reclassify accepted Stage 08M candidate quality before any Stage 17+ runtime, soak, mainnet-readiness or rollout progression."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and updated quality gate"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08m-supervised-warm-start-candidate-scorecard.md
      why: "accepted 08M candidate evidence being reclassified"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/10-per-ticker-calibration.md
      why: "per-ticker actionable/fail-closed calibration evidence"
  task_entrypoints:
    - path: /opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_scorecard_summary.json
      why: "08M final scorecard summary artifact"
    - path: /opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55
      why: "08M candidate scorecard artifact directory"
    - path: /opt/roehub/state/rl_trading/calibration_packs/stage10_per_ticker_calibration_v1
      why: "Stage 10 calibration pack directory"
    - path: scripts/rl_trading
      why: "reuse existing scorecard/calibration readers if metric extraction needs a tiny helper"
  consult_if_needed:
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08k-article-demo-profile-training-evaluation.md
      read_when: "comparing 08M against the blocked DQN article candidate"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08l-reward-warm-start-research.md
      read_when: "checking the research-only status and proxy lineage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/13-monitor-only-inference-producer.md
      read_when: "ensuring runtime proof is not confused with model quality"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/15-paper-rl-integration.md
      read_when: "ensuring paper proof is not confused with model quality"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/16-testnet-rl-integration.md
      read_when: "ensuring testnet proof is not confused with model quality"
    - path: docs/architecture/README.md
      read_when: "Markdown docs are added or changed and docs index must be verified"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  prompt_path_and_sha256_in_report: true
  no_secret_or_raw_provider_payload_in_artifacts: true
  explicit_delivery_state: true
  no_runtime_activation: true
  real_boundary_evidence_for_non_trivial_stage: true
task_toggles:
  allow_training: false
  allow_mainnet_submit: false
  allow_exchange_side_effects: false
  allow_browser_runtime_verification: false
  allow_tests_only_acceptance: false
skill_routing:
  - skill: data-analytics-methodology
    use_when: "recomputing or interpreting trading quality metrics, stability and per-ticker scorecards"
    timing: "during analysis"
    reason: "keeps conclusions evidence-backed and avoids treating weak aggregate PnL as product quality"
  - skill: backend-quality-gates
    use_when: "Python helper code/tests are changed"
    timing: "verification"
    reason: "owns focused ruff, pyright and pytest gates"
  - skill: contract-impact-analysis
    use_when: "classifying the effect of reclassifying 08M on runtime, registry, UI, mainnet and product rollout contracts"
    timing: "before final report"
    reason: "prevents hidden compatibility or rollout claims"
target_envs:
  - "local checkout"
  - "Mac Studio local artifacts under /opt/roehub/state/rl_trading/"
required_literals:
  - "08N"
  - "stage08m_a3823cbd01143878_fd7c614b"
  - "accepted_for_research_only"
  - "stage17_infrastructure_only_allowed"
  - "stage18_soak_allowed"
  - "stage19_mainnet_readiness_allowed"
  - "stage20_mainnet_canary_allowed"
  - "stage21_product_rollout_allowed"
non_goals:
  - "Do not train or tune a new model."
  - "Do not mutate model registry state, activation state, runtime config, user entitlements, API/UI behavior, paper/testnet/live execution, or exchange state."
  - "Do not use Stage 13/15/16 runtime plumbing success as model-quality evidence."
  - "Do not open Stage 19, Stage 20 or Stage 21 unless this stage explicitly classifies the candidate as promotion-grade."
quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
  - cmd: "git diff --check"
    expect: "passes for changed docs/prompts"
  - cmd: "uv run ruff check scripts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes if Python helper code/tests changed"
  - cmd: "uv run pyright scripts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes if Python helper code/tests changed"
  - cmd: "uv run pytest -q tests/unit/scripts/rl_trading"
    expect: "passes if Python helper code/tests changed"
validation_strategy:
  depth: candidate_quality_reclassification
  e2e_required: true
  proof_boundary: target_host_non_production_quality_reclassification_pre_main
  acceptance_surfaces:
    - "08M aggregate scorecard and manifest hash validation"
    - "per-trade expectancy, cost efficiency and turnover"
    - "per-ticker and per-month PnL/trade/drawdown/win-rate stability"
    - "Stage 10 actionable vs fail-closed ticker calibration"
    - "fee/funding/slippage stress or explicit unavailable-with-reason classification"
    - "runtime-stage gating decision for 17, 18, 19, 20 and 21"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08n-candidate-quality-reclassification.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "08N"
  required_update: true
expected_primary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08n-candidate-quality-reclassification.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
  - "docs/architecture/README.md"
possible_secondary_touches:
  - "scripts/rl_trading"
  - "tests/unit/scripts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
safety_notes:
  - "This is a quality/governance reclassification stage, not a runtime activation stage."
  - "If evidence is incomplete, fail closed and mark 08M as accepted_for_research_only or blocked_for_product, not promotion-grade."
---

# Task

Implement Stage `08N` candidate quality reclassification for the accepted Stage `08M`
candidate `stage08m_a3823cbd01143878_fd7c614b`.

The purpose is to stop the plan from automatically moving from technically accepted
runtime plumbing into Stage `17+` and eventually mainnet when the current candidate
may be economically weak. Stage `08M` remains historical evidence and may remain valid
for registry/plumbing history, but Stage `08N` must decide what it is valid for now.

## Prompt-Pack Execution Anchor

- `plan_doc`: `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/rl-trading-agent-platform-v1`
- `stage_ledger`: `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `execution_mode`: `manual_sequential`
- `GOAL.md`: optional, not required by default.
- Stage gate: read the ledger before edits; run only when `current_stage=08N`.
- Manifest gate: every created/modified/deleted file and every runtime artifact path must be recorded in the stage report and ledger.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Verify prerequisites before analysis:
  - `08M` is `accepted`;
  - `09`, `09B`, `10`, `10A`, `11`, `12`, `13`, `14`, `15`, and `16` are accepted technical/plumbing stages;
  - `17` is still pending and has no accepted stage report;
  - `current_stage=08N`.
  If not true, write/update the `08N` report as blocked, update the ledger, and stop.
- Compute and record this prompt hash:
  - `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08n-candidate-quality-reclassification.md`
- Verify the accepted `08M` summary artifact and candidate manifest hash before any conclusions:
  - summary path `/opt/roehub/state/rl_trading/evaluation_runs/stage08m_supervised_warm_start_candidate_scorecard_v1/stage08m_supervised_warm_start_fe2fe3c5257fd9992c55/stage08m_supervised_warm_start_candidate_scorecard_summary.json`
  - summary sha256 `ff518bf3134670a0e814db7bfff45a3112e40f2169f9110c0b352cc77f044ab7`
  - candidate manifest sha256 `9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c`
- Read the Stage `10` calibration summary/pack and reconcile the `323` ticker rows, `65` actionable rows and `258` fail-closed/blocked rows with the candidate quality decision.
- Produce a quality reclassification matrix with at least these rows:
  - aggregate return: final PnL, return percent, starting equity, closed trades, average and median PnL per trade;
  - per-ticker scorecard: PnL, trades, win-rate, drawdown if available, actionability state and fail-closed reason;
  - per-month or time-bucket stability;
  - volatility-bucket stability;
  - fee/funding/slippage sensitivity or a fail-closed explanation if not available from artifacts;
  - turnover and cost efficiency;
  - action distribution and one-sided bias;
  - comparison against hold/no-trade, random, simple threshold and Stage `08K` blocked native DQN result;
  - Stage `10` calibration impact: which tickers are product-eligible, research-only, or fail-closed.
- Classify the candidate into exactly one `quality_status`:
  - `promotion_grade_candidate`
  - `paper_research_candidate`
  - `runtime_infrastructure_only_candidate`
  - `accepted_for_research_only`
  - `blocked_for_product`
  - `insufficient_evidence`
- Write explicit downstream gate booleans in the report and ledger:
  - `stage17_infrastructure_only_allowed`
  - `stage17_full_runtime_allowed`
  - `stage18_monitor_only_technical_soak_allowed`
  - `stage18_soak_allowed`
  - `stage19_mainnet_readiness_allowed`
  - `stage20_mainnet_canary_allowed`
  - `stage21_product_rollout_allowed`
- Default fail-closed rules:
  - `stage19_mainnet_readiness_allowed=false`, `stage20_mainnet_canary_allowed=false`, and `stage21_product_rollout_allowed=false` unless `quality_status=promotion_grade_candidate` and the report explains why the weak aggregate concern is resolved.
  - If `quality_status` is not `promotion_grade_candidate`, Stage `17` may be allowed only as `stage17_infrastructure_only_allowed=true`, with a clear warning that it is not product/trading-quality proof.
  - Full Stage `18` soak may not run for trade-readiness if the candidate is only research/infrastructure grade; only `stage18_monitor_only_technical_soak_allowed` may be true when useful for runtime proof.
- Explicitly state whether `08M` should be downgraded from `stage09_allowed=true` to a narrower current classification for future stages. Do not rewrite historical `08M` facts; append the new `08N` decision.
- Do not train, tune, activate, register, promote, paper/testnet/live trade, mainnet submit, or read browser credentials.

## Acceptance Criteria

- Stage report exists at `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08n-candidate-quality-reclassification.md`.
- Report includes prompt path/hash, prerequisite check, verified artifact paths/hashes, quality matrix, downstream gate booleans, file manifest, contract impact, proof boundary, residual risks and next-stage handoff.
- Ledger `current_stage` moves according to the `08N` decision:
  - if only infrastructure benchmarking is allowed, next executable stage is `17` with `infrastructure_only`;
  - if evidence is insufficient, next executable stage remains a research/repair prompt and `17` stays closed;
  - `18`, `19`, `20`, and `21` stay closed unless explicitly allowed by the booleans above.
- No runtime code, API/UI, registry state, exchange behavior or mainnet behavior is changed unless separately justified and tested.

## Final Output

Respond in Russian with:

- `08N` status and `quality_status`;
- what the `08M` PnL/trade result means in practical terms;
- exact downstream gates for `17`, `18`, `19`, `20`, `21`;
- changed files and checks;
- residual risks and the next prompt to run.
