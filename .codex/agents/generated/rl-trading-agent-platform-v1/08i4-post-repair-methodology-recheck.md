---
prompt_name: 08i4-post-repair-methodology-recheck
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
  stage_gate: "read ledger before edits; run only when current_stage is 08I4 and prerequisites match"
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
    - scripts/rl_trading
    - tests/unit/scripts/rl_trading
    - docs/architecture/ml/rl-trading-agent-platform-v1.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i4-post-repair-methodology-recheck.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
    - docs/architecture/README.md
  forbidden_without_user_approval:
    - exchange execution paths
    - live_execution behavior
    - API/UI behavior
    - database migrations
    - branch/worktree/stash/local-folder workflow changes
scope: "Recheck the complete 08I2 methodology matrix after 08I3 repair and decide whether 08J may start."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, proof boundaries, branch policy"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and 08I4 recheck rules"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i2-exhaustive-methodology-discrepancy-audit.md
      why: "original complete discrepancy matrix"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i3-evaluator-action-reward-parity-repair.md
      why: "accepted repair evidence and remaining differences"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md
      why: "historical first-diff evidence to verify closed"
  task_entrypoints:
    - path: scripts/rl_trading/stage08i_upstream_evaluator_session_parity_forensic.py
      why: "forensic trace runner to reuse or validate"
    - path: src/trading/contexts/rl_trading/domain/hf_original_evaluation.py
      why: "post-repair evaluator source"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md
      why: "Stage 06 selector gap source"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08h-oracle-supervised-selector-reward-90-60-research.md
      why: "signal/reward/action/Optuna diagnostic evidence"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  prompt_path_and_sha256_in_report: true
  no_secret_or_raw_provider_payload_in_artifacts: true
  explicit_delivery_state: true
  real_boundary_evidence_for_non_trivial_stage: true
task_toggles:
  allow_training: false
  allow_optuna: false
  allow_mainnet_submit: false
  allow_exchange_side_effects: false
  allow_browser_runtime_verification: false
  allow_tests_only_acceptance: false
skill_routing:
  - skill: architecture-review
    use_when: "checking whether 08I2 rows are closed, downstream-assigned, or still blocking"
    timing: "entire stage"
    reason: "owns stage readiness and fact-vs-inference discipline"
  - skill: data-analytics-methodology
    use_when: "classifying dataset geometry, signal, reward sparsity, baselines and overfit rows"
    timing: "analysis"
    reason: "keeps ML/research conclusions evidence-backed"
  - skill: backend-quality-gates
    use_when: "Python helper/tests are changed"
    timing: "verification"
    reason: "owns focused gates"
target_envs:
  - "local checkout"
  - "macstudio for existing runtime artifacts and optional non-production recheck artifacts"
required_literals:
  - "08I4"
  - "methodology_recheck_matrix"
  - "08j_allowed"
  - "stage09_allowed=false"
  - "target_host_non_production_forensic_pre_main"
non_goals:
  - "Do not implement the article selector; that belongs to 08J."
  - "Do not train, tune, register, promote, activate, paper/testnet/live trade, or mainnet submit."
  - "Do not declare all 08I2 gaps solved if a row is only assigned to 08J or 08K."
  - "Do not create branches, worktrees, stashes, temporary repo checkouts, or auxiliary workflow folders."
quality_gates:
  - cmd: "uv run ruff check scripts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes if Python helper/tests are changed; otherwise record N/A"
  - cmd: "uv run pyright scripts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes if Python helper/tests are changed; otherwise record N/A"
  - cmd: "uv run pytest -q tests/unit/scripts/rl_trading"
    expect: "passes if Python helper/tests are changed; otherwise record N/A"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: integration
  e2e_required: true
  proof_boundary: target_host_non_production_forensic_pre_main
  acceptance_surfaces:
    - "08I2 matrix row-by-row recheck"
    - "08I3 repair evidence verification"
    - "explicit 08J allow/block decision"
    - "explicit stage09_allowed=false decision"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i4-post-repair-methodology-recheck.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "08I4"
  required_update: true
expected_primary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i4-post-repair-methodology-recheck.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "scripts/rl_trading"
  - "tests/unit/scripts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Runtime matrices live under /opt/roehub/state/rl_trading/ and are not committed."
  - "Reports contain sanitized scalar metrics, statuses, hashes and paths only."
  - "Mac Studio git commands must use /Users/daniildegtyarev/Projects/roehub.com; /opt/roehub/app is runtime state only."
---

# Task

Implement Stage `08I4` post-repair methodology recheck.

Stage `08I3` is expected to repair the evaluator/action/reward-reporting parity blockers. This stage must not add a new model-quality claim. It must re-open the `08I2` matrix and decide, row by row, whether each discrepancy is:

- closed by `08I3`;
- explicitly assigned to `08J` article-selector dataset work;
- explicitly assigned to `08K` article-demo training/evaluation gates;
- still blocking;
- superseded with a source-backed reason;
- not applicable with a source-backed reason.

The purpose is to prevent another loop where one gap is fixed, training restarts, and a different known gap is rediscovered later.

## Prompt-Pack Execution Anchor

- `plan_doc`: `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/rl-trading-agent-platform-v1`
- `stage_ledger`: `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `execution_mode`: `manual_sequential`
- `GOAL.md`: optional, not required by default.
- Stage gate: read the ledger before edits; run only when `current_stage=08I4`.
- Manifest gate: every created/modified/deleted file and every runtime artifact path must be recorded in the stage report and ledger.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: verify `08I3` is `accepted`, `current_stage=08I4`, and `08J`/`08K`/`09` are blocked. If not true, write/update the `08I4` report as blocked, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for `08I4`. Do not use username `smoke_e2e_keycloak` and do not read `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Compute and record this prompt hash: `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08i4-post-repair-methodology-recheck.md`.
- Read the `08I2` matrix and verify all eight original rows are present. If the matrix artifact itself is unavailable, use the sanitized `08I2` report as evidence and record the missing runtime artifact.
- Produce a durable `methodology_recheck_matrix` artifact under `/opt/roehub/state/rl_trading/` and summarize it in the report.
- For each original `08I2` row, include:
  - `surface`;
  - `08i2_status`;
  - `08i2_severity`;
  - `08i3_evidence`;
  - `recheck_disposition`;
  - `source_backed_reason`;
  - `remaining_blocker`;
  - `owner_next_stage`;
  - `recheck_required`;
  - `08j_allowed_for_row`;
  - `stage09_allowed_for_row`.
- Allowed `recheck_disposition` values are only:
  - `closed_by_08i3`;
  - `assigned_to_08j`;
  - `assigned_to_08k`;
  - `still_blocking`;
  - `superseded_with_source_reason`;
  - `not_applicable_with_source_reason`.
- Pre-`08J` blockers must be closed or superseded before `08J` is allowed:
  - full evaluator/backtest parity;
  - rolling `open_sessions` scheduling;
  - shared balance and `position_fraction` sizing;
  - action/Q mask/filter order semantics;
  - `training_reward` vs `backtest_reporting_reward` trace semantics.
- Dataset/model-quality rows may remain open only if they are explicitly assigned:
  - `session_extractor_policy` and `dataset_geometry_and_distribution` may be assigned to `08J`;
  - `past_only_signal_strength`, reward sparsity diagnostics, final action distribution, `Optuna` overfit and sanity baselines may be assigned to `08K`;
  - none of these assignments may open `09`.
- Set top-level booleans:
  - `08j_allowed`: true only when no pre-`08J` blocker remains and every row has a disposition;
  - `08k_allowed`: false unless `08J` is already accepted, which should not be true in this stage;
  - `stage09_allowed`: false.
- Do not use `08I4` to hide a gap. If a row is unresolved and not clearly owned by `08J` or `08K`, mark `still_blocking` and keep `08J` closed.
- Write a stage report at `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i4-post-repair-methodology-recheck.md`.
- Update the stage ledger after validation and before final output.

## Acceptance Criteria

- Stage report includes prompt hash, `08I2` matrix reference/hash, `08I3` repair reference/hash, `methodology_recheck_matrix` path/hash, row summary, `08j_allowed`, `stage09_allowed=false`, file manifest, contract impact, proof boundary, and next-stage handoff.
- If `08j_allowed=true`, ledger advances to `08J` and records exactly which rows `08J` and `08K` must close later.
- If `08j_allowed=false`, ledger remains blocked and names the next repair prompt/stage required.
- Stage `09` remains blocked in all `08I4` outcomes.

## Final Output

Respond in Russian with result/status, matrix disposition summary, `08J` allow/block decision, file manifest, quality gates, residual risks, and the next prompt to run.
