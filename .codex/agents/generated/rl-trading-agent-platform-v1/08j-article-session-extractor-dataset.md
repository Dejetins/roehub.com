---
prompt_name: 08j-article-session-extractor-dataset
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
  stage_gate: "read ledger before edits; run only when current_stage is 08J and prerequisites match"
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
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08j-article-session-extractor-dataset.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
    - docs/architecture/README.md
  forbidden_without_user_approval:
    - Stage 06 artifact overwrite
    - exchange execution paths
    - live_execution behavior
    - branch/worktree/stash/local-folder workflow changes
scope: "Add article-style session extraction as a new dataset policy beside historical Stage 06 selector."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and article selector contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md
      why: "accepted evaluator/session parity prerequisite"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md
      why: "historical Stage 06 selector and split/leakage contract"
  task_entrypoints:
    - path: src/trading/contexts/rl_trading
      why: "dataset/session extraction implementation"
    - path: scripts/rl_trading
      why: "dataset materialization CLIs"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04c-dataset-refresh-manifest.md
      why: "accepted raw data refresh inputs"
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  prompt_path_and_sha256_in_report: true
  no_secret_or_raw_provider_payload_in_artifacts: true
  explicit_delivery_state: true
  real_boundary_evidence_for_non_trivial_stage: true
task_toggles:
  allow_training: false
  allow_mainnet_submit: false
  allow_exchange_side_effects: false
  allow_browser_runtime_verification: false
  allow_tests_only_acceptance: false
skill_routing:
  - skill: data-analytics-methodology
    use_when: "dataset distribution, leakage, split and selector comparison"
    timing: "analysis"
    reason: "owns dataset-quality methodology"
  - skill: backend-quality-gates
    use_when: "Python code/tests are changed"
    timing: "verification"
    reason: "owns focused gates"
target_envs:
  - "local checkout"
  - "macstudio for dataset materialization"
required_literals:
  - "08J"
  - "article_future_10m_5pct_contrast_v1"
  - "target_host_non_production_dataset_pre_main"
non_goals:
  - "Do not overwrite or mutate accepted Stage 06 artifacts."
  - "Do not train/evaluate a model."
  - "Do not change live feed, API, UI, exchange execution, or mainnet behavior."
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes for changed Python paths"
  - cmd: "uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes for changed Python paths"
  - cmd: "uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "focused selector/leakage tests pass"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: integration
  e2e_required: true
  proof_boundary: target_host_non_production_dataset_pre_main
  acceptance_surfaces:
    - "article selector materialized dataset manifest"
    - "split/leakage/lifecycle proof"
    - "distribution comparison against HF-original and Stage 06 current selector"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08j-article-session-extractor-dataset.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "08J"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "tests/unit/contexts/rl_trading"
  - "tests/unit/scripts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08j-article-session-extractor-dataset.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Large dataset artifacts live under /opt/roehub/state/rl_trading/ and are not committed."
  - "No synthetic candles and no lookahead leakage are allowed in accepted training windows."
---

# Task

Implement Stage `08J` article session extractor dataset.

Create a new Roehub-native dataset variant using an article-style event selector. This is not a replacement for historical Stage `06`; it is a new policy/materialization path that lets Stage `08K` test the article methodology on Roehub data.

## Prompt-Pack Execution Anchor

- `plan_doc`: `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/rl-trading-agent-platform-v1`
- `stage_ledger`: `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `execution_mode`: `manual_sequential`
- `GOAL.md`: optional, not required by default.
- GOAL.md is optional, not required by default.
- Stage gate: read the ledger before edits; run only when `current_stage=08J`.
- Manifest gate: every created/modified/deleted file and every runtime artifact path must be recorded in the stage report and ledger.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: verify `08I` is `accepted` and `current_stage=08J`. If not true, write/update `08J` as blocked, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for `08J`. Do not use username `smoke_e2e_keycloak` and do not read `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Compute and record this prompt hash: `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08j-article-session-extractor-dataset.md`.
- Implement selector id `article_future_10m_5pct_contrast_v1` as a separate policy.
- Selector semantics:
  - event move: absolute price move over `10m` is at least `5%`;
  - contrast rule: previous `90m` must not already contain a similar impulse according to the accepted stage report definition;
  - `event_end_t` becomes `signal_ts_open`;
  - `pre_window=[signal_ts_open-90m, signal_ts_open)`;
  - `post_window=[signal_ts_open, signal_ts_open+60m)`;
  - materialized session length remains `150` minutes, with source/demo training later using `agent_history_len=30`, `agent_session_len=10`.
- Preserve split boundaries, lifecycle availability, gap handling, listing/delisting exclusions, overlap policy and embargo. Overlap is allowed only within a split; split-boundary leakage is a blocker.
- Materialize under `/opt/roehub/state/rl_trading/` with manifest/hash, counts by split/symbol/month, rejected-window reasons, overlap rate, gap rate, and lifecycle exclusions.
- Compare distributions for HF-original, Stage `06` current selector, and article selector: session counts, long/short oracle labels, oracle best-return, volatility/range, symbol/month coverage, and supervised past-only sanity.
- Do not run training or `Optuna` in `08J`.

## Acceptance Criteria

- Article-selector dataset manifest exists with deterministic hash and clean leakage/gap evidence.
- Stage report records exact selector formula, accepted deviations from article/repo, distribution comparison, file manifest, proof boundary, and `08K` handoff.
- Ledger advances to `08K` only if the dataset variant is accepted. If the selector produces too few or invalid sessions, block with exact evidence.
- Stage `09` remains blocked.

## Final Output

Respond in Russian with result/status, dataset evidence, selector comparison, file manifest, quality gates, residual risks, and the next prompt to run.
