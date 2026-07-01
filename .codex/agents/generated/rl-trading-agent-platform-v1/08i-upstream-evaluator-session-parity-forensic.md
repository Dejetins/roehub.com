---
prompt_name: 08i-upstream-evaluator-session-parity-forensic
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
  stage_gate: "read ledger before edits; run only when current_stage is 08I and prerequisites match"
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
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
    - docs/architecture/README.md
  forbidden_without_user_approval:
    - exchange execution paths
    - live_execution behavior
    - API/UI behavior
    - database migrations
    - branch/worktree/stash/local-folder workflow changes
scope: "Forensic parity between upstream backtest/session semantics and Roehub evaluator before any new training."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, branch policy, proof boundaries"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and corrective stage sequence"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08h-oracle-supervised-selector-reward-90-60-research.md
      why: "latest blocked evidence and failure diagnosis"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-upstream-methodology-parity-audit.md
      why: "pinned upstream source map and parity surface"
  task_entrypoints:
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04-hf-reproducibility.md
      why: "HF dataset/checkpoint paths and hashes"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08d-original-hf-backtest-evaluation.md
      why: "Roehub HF evaluation mechanics and warnings"
    - path: src/trading/contexts/rl_trading
      why: "current evaluator, state builder, normalization, Q-cache and backtest policies"
    - path: scripts/rl_trading
      why: "current evaluation CLI patterns"
  external_sources:
    - repo: https://github.com/YuriyKolesnikov/rl-trading-binance
      required_commit: f71130903f8237351164f4b875494185465bf1ea
      required_files:
        - configs/alpha.py
        - config.py
        - utils.py
        - trading_environment.py
        - model.py
        - agent.py
        - backtest_engine.py
        - optimize_cfg.py
      rule: "Use read-only source inspection or an existing local clone; do not vendor upstream code into Roehub."
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
  - skill: architecture-review
    use_when: "classifying upstream-vs-Roehub parity gaps"
    timing: "before and after implementation"
    reason: "keeps source-backed fact/inference separation"
  - skill: backend-quality-gates
    use_when: "Python code/tests are changed"
    timing: "verification"
    reason: "owns focused ruff, pyright, pytest gate triage"
  - skill: data-analytics-methodology
    use_when: "interpreting trace diffs and scorecard comparability"
    timing: "analysis"
    reason: "keeps evaluation methodology defensible"
target_envs:
  - "local checkout"
  - "macstudio for HF artifacts and target-host forensic evidence"
required_literals:
  - "08I"
  - "backtest_engine.py"
  - "target_host_non_production_forensic_pre_main"
non_goals:
  - "Do not train, tune, register, promote, activate, paper trade, testnet trade, live trade, or mainnet submit."
  - "Do not replace Stage 06 selector."
  - "Do not change reward/action semantics except to fix proven parity bugs."
  - "Do not create branches, worktrees, stashes, temporary repo checkouts, or auxiliary workflow folders."
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes for changed Python paths; narrow if unchanged paths are unrelated"
  - cmd: "uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes for changed Python paths"
  - cmd: "uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "focused parity tests pass"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: integration
  e2e_required: true
  proof_boundary: target_host_non_production_forensic_pre_main
  acceptance_surfaces:
    - "original-vs-Roehub step-level trace on the same HF checkpoint/config/data"
    - "first material diff report or accepted parity summary"
    - "session extraction semantic comparison"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "08I"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "tests/unit/contexts/rl_trading"
  - "tests/unit/scripts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Runtime traces live under /opt/roehub/state/rl_trading/ and are not committed to git."
  - "Trace files must contain hashes/counts/metrics and sanitized scalar decisions only; never raw checkpoint tensors or credentials."
  - "Mac Studio git commands must use /Users/daniildegtyarev/Projects/roehub.com; /opt/roehub/app is runtime state only."
---

# Task

Implement Stage `08I` upstream evaluator/session parity forensic.

The current native path is blocked by `08F`, `08G`, and `08H`. Do not launch new training. First prove whether Roehub evaluation/backtest behavior is equivalent to the original repository on the same HF checkpoint/config/data, or record the exact first material diff that must be fixed.

## Prompt-Pack Execution Anchor

- `plan_doc`: `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/rl-trading-agent-platform-v1`
- `stage_ledger`: `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `execution_mode`: `manual_sequential`
- `GOAL.md`: optional, not required by default.
- GOAL.md is optional, not required by default.
- Stage gate: read the ledger before edits; run only when `current_stage=08I`.
- Manifest gate: every created/modified/deleted file and every runtime artifact path must be recorded in the stage report and ledger.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: verify `08H` is `blocked`, `current_stage=08I`, and `09` is blocked. If not true, write/update the `08I` report as blocked, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for `08I`. Do not use username `smoke_e2e_keycloak` and do not read `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Compute and record this prompt hash: `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08i-upstream-evaluator-session-parity-forensic.md`.
- Use upstream commit `f71130903f8237351164f4b875494185465bf1ea`. If the source cannot be read or run, record a blocked state with the exact missing dependency/source/artifact.
- Compare original `backtest_engine.py` and Roehub evaluation on the same HF `backtest_data.npz`, same `configs/alpha.py`-family config, same checkpoint, and 20-50 identical sessions.
- Produce step-level traces with at least: `session_idx`, `symbol`, `signal_time`, `step_idx`, `price`, `state_hash`, `q_values_hash` or sanitized q-values, `raw_argmax_action`, `masked_q_values_hash`, `selected_action`, `effective_action`, `position_side`, `entry_price`, `pnl_change`, `reward`, `balance_or_equity`, and `audit_reason`.
- Identify the first material diff by step. Check shared balance vs independent aggregation, signal group ordering, close/open price index semantics, last-step action mask, commission/slippage application, action filter thresholds, and risk-management timing.
- Separately document the session extraction semantic gap: original article/repo event selection vs Stage `06` `pre_signal_realized_volatility_plus_range_v1`.
- Do not accept `08I` if a material evaluator diff remains unexplained and could change final PnL/action selection.
- If a narrow parity bug is fixed, add regression tests and rerun focused gates. If the diff requires broader redesign, block `08I` and do not implement speculative fixes.

## Acceptance Criteria

- Stage report exists at `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md`.
- Report includes prompt hash, upstream commit, input artifact hashes, trace artifact paths/hashes, first-diff table, fixed/remaining parity gaps, file manifest, contract impact, proof boundary, delivery state, and next-stage handoff.
- Ledger advances to `08J` only if evaluator parity is accepted or all material diffs are fixed with tests.
- Stage `09` remains blocked.

## Final Output

Respond in Russian with result/status, file manifest, parity evidence, blockers, quality gates, residual risks, and the next prompt to run.
