---
prompt_name: 08k-article-demo-profile-training-evaluation
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
  stage_gate: "read ledger before edits; run only when current_stage is 08K and prerequisites match"
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
    - tests/perf_smoke/contexts/rl_trading
    - docs/architecture/ml/rl-trading-agent-platform-v1.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08k-article-demo-profile-training-evaluation.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
    - docs/architecture/README.md
  forbidden_without_user_approval:
    - exchange execution paths
    - live_execution behavior
    - API/UI behavior
    - branch/worktree/stash/local-folder workflow changes
scope: "Run article/demo 30/10 full training/evaluation on HF-original control and Roehub-native article-selector data."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and strict corrective gates"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md
      why: "blocked first-diff evidence retained as history"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i2-exhaustive-methodology-discrepancy-audit.md
      why: "complete blocked discrepancy matrix that was repaired/rechecked by 08I3/08I4"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i3-evaluator-action-reward-parity-repair.md
      why: "accepted evaluator/action/reward-reporting parity prerequisite"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i4-post-repair-methodology-recheck.md
      why: "accepted post-repair matrix recheck and downstream row ownership"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08j-article-session-extractor-dataset.md
      why: "article-selector dataset manifest"
  task_entrypoints:
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04-hf-reproducibility.md
      why: "HF original dataset paths/hashes"
    - path: src/trading/contexts/rl_trading
      why: "training/evaluation implementation"
    - path: scripts/rl_trading
      why: "training, Optuna and evaluation CLIs"
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
  - skill: backend-quality-gates
    use_when: "Python code/tests are changed"
    timing: "verification"
    reason: "owns focused gates"
  - skill: data-analytics-methodology
    use_when: "scorecard, baselines, calibration/final split and overfit interpretation"
    timing: "evaluation"
    reason: "keeps ML evaluation methodology explicit"
  - skill: backend-performance-evidence
    use_when: "reporting CPU/MPS throughput or resource claims"
    timing: "runtime evidence"
    reason: "requires comparable measurement"
target_envs:
  - "local checkout"
  - "macstudio for full training/evaluation"
required_literals:
  - "08K"
  - "agent_history_len=30"
  - "agent_session_len=10"
  - "article_future_10m_5pct_contrast_v1"
non_goals:
  - "Do not use Stage 06 current-selector data as the native candidate source for this stage."
  - "Do not use 90/60 as the primary article-reproduction profile."
  - "Do not register, promote, activate, paper/testnet/live trade, or mainnet submit."
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading"
    expect: "passes for changed Python paths"
  - cmd: "uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading"
    expect: "passes for changed Python paths"
  - cmd: "uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading tests/perf_smoke/contexts/rl_trading"
    expect: "focused tests pass"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: integration
  e2e_required: true
  proof_boundary: target_host_non_production_training_and_evaluation_pre_main
  acceptance_surfaces:
    - "HF-original control training/evaluation"
    - "Roehub-native article-selector training/evaluation"
    - "Optuna calibration separated from untouched final holdout"
    - "strict native baseline-beating research gate"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08k-article-demo-profile-training-evaluation.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "08K"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "tests/unit/contexts/rl_trading"
  - "tests/unit/scripts/rl_trading"
  - "tests/perf_smoke/contexts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08k-article-demo-profile-training-evaluation.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Training/evaluation artifacts live under /opt/roehub/state/rl_trading/ and are not committed."
  - "Progress must be durable: progress.jsonl plus latest status; background launch alone is not acceptance."
---

# Task

Implement Stage `08K` article demo-profile training/evaluation.

Run the source/demo `30/10` workflow after accepted evaluator/action/reward repair, accepted post-repair methodology recheck and article-selector dataset materialization. This stage is the next planned path that can reopen Stage `09`; if it fails, preserve the failure and hand off to `08L`.

## Prompt-Pack Execution Anchor

- `plan_doc`: `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/rl-trading-agent-platform-v1`
- `stage_ledger`: `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `execution_mode`: `manual_sequential`
- `GOAL.md`: optional, not required by default.
- GOAL.md is optional, not required by default.
- Stage gate: read the ledger before edits; run only when `current_stage=08K`.
- Manifest gate: every created/modified/deleted file and every runtime artifact path must be recorded in the stage report and ledger.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: verify `08I3`, `08I4`, and `08J` are `accepted`, `current_stage=08K`, and `09` is blocked. If not true, write/update `08K` as blocked, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for `08K`. Do not use username `smoke_e2e_keycloak` and do not read `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Compute and record this prompt hash: `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08k-article-demo-profile-training-evaluation.md`.
- Train/evaluate two branches:
  - `hf_original_control_30_10`: original HF splits, source/demo `agent_history_len=30`, `agent_session_len=10`;
  - `roehub_native_article_selector_30_10`: accepted `08J` article-selector dataset.
- Use accepted `08I3` evaluator/action/reward parity fixtures and accepted `08I4` methodology recheck matrix. If parity is later found broken, any mandatory matrix row is missing, or any `08K`-owned row lacks a recheck plan, stop and block instead of continuing.
- Run `Optuna` only on calibration split; final holdout must remain untouched by tuning.
- Persist progress and resource evidence for Mac Studio CPU/MPS. Use measured device policy; do not claim MPS/CPU speed without comparable evidence.
- Native branch acceptance is strict:
  - final PnL after fees/slippage/funding policy is positive;
  - final PnL beats the best sanity baseline on the same surface;
  - closed trades are sufficient for the report-defined minimum;
  - no single symbol/month/session bucket dominates the result;
  - monthly/ticker stability is not obviously broken;
  - action distribution is not pathologically one-sided, such as the prior short-bias pattern;
  - no zero-trade or calibration-only `Optuna` overfit candidate is selected.
- If HF control fails on execution/parity grounds, block native training. If HF control is weak only as quality signal but parity is accepted, native can run with the warning carried forward.

## Acceptance Criteria

- Stage report records branch manifests, checkpoint hashes, normalization hashes, progress/resource evidence, Optuna summaries, final holdout scorecards, baselines, action/filter distributions, and strict research decision.
- Ledger advances to `09` only when native article-selector branch is accepted and records `stage09_allowed=true`.
- If native fails, ledger advances to `08L` only as a research fallback; Stage `09` remains blocked.

## Final Output

Respond in Russian with result/status, branch evidence, strict native candidate decision, file manifest, quality gates, residual risks, and next-stage handoff.
