---
prompt_name: 08l-reward-warm-start-research
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
  stage_gate: "read ledger before edits; run only when current_stage is 08L and prerequisites match"
  file_manifest_required: true
goal_mode_optional: true
goal_artifact_required: false
proof_boundary:
  label: target_host_readiness_pre_main
  changed_code_production_claim_allowed: false
browser_auth:
  status: "N/A unless this prompt is explicitly expanded into browser-visible UI/auth work"
  smoke_username: smoke_e2e_keycloak
  host_local_password_source: "/Users/daniildegtyarev/.config/roehub/roehub.env key ROEHUB_SMOKE_E2E_PASSWORD"
  redaction_rule: "do not read or print the password unless browser/auth work is explicitly in scope; never write credentials to prompts, docs, logs, traces, reports, screenshots, or ledgers"
change_ownership:
  allowed_files:
    - src/trading/contexts/rl_trading
    - scripts/rl_trading
    - tests/unit/contexts/rl_trading
    - tests/unit/scripts/rl_trading
    - docs/architecture/ml/rl-trading-agent-platform-v1.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08l-reward-warm-start-research.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
    - docs/architecture/README.md
  forbidden_without_user_approval:
    - exchange execution paths
    - live_execution behavior
    - API/UI behavior
    - branch/worktree/stash/local-folder workflow changes
scope: "Fail-closed research after article parity/dataset path fails: reward shaping, supervised warm-start, behavior cloning, or contextual bandit sanity."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and reward/action contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md
      why: "blocked first-diff evidence retained as history"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i2-exhaustive-methodology-discrepancy-audit.md
      why: "complete blocked discrepancy matrix that was repaired/rechecked by 08I3/08I4"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i3-evaluator-action-reward-parity-repair.md
      why: "accepted evaluator/action/reward-reporting repair prerequisite"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i4-post-repair-methodology-recheck.md
      why: "accepted post-repair matrix recheck prerequisite"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08j-article-session-extractor-dataset.md
      why: "accepted article-selector dataset"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08k-article-demo-profile-training-evaluation.md
      why: "blocked article/demo run that motivates fallback research"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  prompt_path_and_sha256_in_report: true
  no_secret_or_raw_provider_payload_in_artifacts: true
  explicit_delivery_state: true
  real_boundary_evidence_for_non_trivial_stage: true
task_toggles:
  allow_mainnet_submit: false
  allow_exchange_side_effects: false
  allow_browser_runtime_verification: false
  allow_tests_only_acceptance: false
skill_routing:
  - skill: data-analytics-methodology
    use_when: "reward, supervised labels, bandit/RL comparison, overfit and baseline interpretation"
    timing: "entire stage"
    reason: "keeps research design and conclusions evidence-backed"
  - skill: backend-quality-gates
    use_when: "Python code/tests are changed"
    timing: "verification"
    reason: "owns focused gates"
target_envs:
  - "local checkout"
  - "macstudio for bounded research runs"
required_literals:
  - "08L"
  - "reward_research_not_contract_replacement"
non_goals:
  - "Do not silently replace the Stage 02C action/reward contract."
  - "Do not register, promote, activate, paper/testnet/live trade, or mainnet submit."
  - "Do not run an unbounded full training search without an explicit bounded experiment matrix."
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes for changed Python paths"
  - cmd: "uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes for changed Python paths"
  - cmd: "uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "focused tests pass"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: research
  e2e_required: true
  proof_boundary: target_host_non_production_research_pre_main
  acceptance_surfaces:
    - "bounded experiment matrix"
    - "reward/warm-start/bandit comparison"
    - "decision whether a new candidate stage is justified"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08l-reward-warm-start-research.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "08L"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "tests/unit/contexts/rl_trading"
  - "tests/unit/scripts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08l-reward-warm-start-research.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Research artifacts live under /opt/roehub/state/rl_trading/ and are not committed."
  - "Any reward/action contract change must be proposed as a new explicit accepted stage, not hidden in 08L."
---

# Task

Implement Stage `08L` reward and warm-start research fallback only if `08K` is blocked after accepted `08I3`/`08I4`/`08J`.

This stage is for controlled research, not registry activation. It should determine whether a new candidate path is justified, using bounded experiments such as dense mark-to-market reward, realized plus unrealized delta, transaction-cost-aware shaping, supervised warm-start/behavior cloning from oracle labels, or contextual-bandit sanity.

## Prompt-Pack Execution Anchor

- `plan_doc`: `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/rl-trading-agent-platform-v1`
- `stage_ledger`: `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `execution_mode`: `manual_sequential`
- `GOAL.md`: optional, not required by default.
- GOAL.md is optional, not required by default.
- Stage gate: read the ledger before edits; run only when `current_stage=08L`.
- Manifest gate: every created/modified/deleted file and every runtime artifact path must be recorded in the stage report and ledger.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: verify `08I3`, `08I4`, and `08J` are `accepted`, `08K` is `blocked`, `current_stage=08L`, and `09` is blocked. If not true, write/update `08L` as blocked, update the ledger, and stop.
- Before any reward/warm-start experiment, read the blocked `08I2` methodology discrepancy matrix and the accepted `08I4` recheck matrix. If any mandatory row is missing or any reward/action/evaluator/session row is unresolved without an explicit accepted repair/recheck path, block `08L` instead of running new research.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for `08L`. Do not use username `smoke_e2e_keycloak` and do not read `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Compute and record this prompt hash: `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08l-reward-warm-start-research.md`.
- Define a bounded experiment matrix before running anything: hypotheses, dataset branch, profile, max runtime, metrics, stop conditions, and expected artifact paths.
- Keep Stage `02C` realized-PnL reward as the baseline. Any dense/shaped reward or supervised warm-start is research-only unless a later plan explicitly changes the contract.
- Include technical baselines: hold/no-trade, deterministic random, simple threshold, supervised oracle-label sanity, and contextual-bandit sanity where applicable.
- Do not accept calibration-only or baseline-losing results as a candidate path.
- Do not open Stage `09` directly unless this stage creates a fully accepted research candidate with the same strict `08K` scorecard and explicitly records `stage09_allowed=true`; otherwise create/update the next corrective prompt and keep `09` blocked.

## Acceptance Criteria

- Stage report includes bounded experiment matrix, runtime/resource evidence, comparison table, failure/success interpretation, proposed next-stage decision, file manifest, contract impact, proof boundary, and ledger handoff.
- If a new reward/warm-start candidate path is justified, the report and ledger create a new explicit next stage; no hidden promotion or registry jump.
- If no path is justified, keep the plan blocked with a clear stop condition and required user/product decision.

## Final Output

Respond in Russian with result/status, experiment evidence, candidate-path decision, file manifest, quality gates, residual risks, and next-stage handoff.
