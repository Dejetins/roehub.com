---
prompt_name: 08i3-evaluator-action-reward-parity-repair
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
  stage_gate: "read ledger before edits; run only when current_stage is 08I3 and prerequisites match"
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
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i3-evaluator-action-reward-parity-repair.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
    - docs/architecture/README.md
  forbidden_without_user_approval:
    - exchange execution paths
    - live_execution behavior
    - API/UI behavior
    - database migrations
    - branch/worktree/stash/local-folder workflow changes
scope: "Repair the pre-08J upstream-vs-Roehub evaluator/action/reward-reporting parity blockers found by 08I and 08I2."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, branch policy, proof boundaries, Mac Studio rules"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and corrected 08I3/08I4 gates"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md
      why: "source-derived first-diff evidence to repair"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i2-exhaustive-methodology-discrepancy-audit.md
      why: "complete discrepancy matrix and row-level repair requirements"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-upstream-methodology-parity-audit.md
      why: "pinned upstream file/function map"
  task_entrypoints:
    - path: scripts/rl_trading/stage08i_upstream_evaluator_session_parity_forensic.py
      why: "current forensic trace generator and source-vs-Roehub comparator"
    - path: src/trading/contexts/rl_trading/domain/hf_original_evaluation.py
      why: "current grouped filtered backtest, risk-management, Q masking and scorecard implementation"
    - path: src/trading/contexts/rl_trading/domain/roehub_native_evaluation.py
      why: "native evaluator path that must inherit fixed semantics"
    - path: src/trading/contexts/rl_trading/domain/upstream_methodology.py
      why: "state builder, Q-value cache, filtered policy, model/agent helpers"
    - path: src/trading/contexts/rl_trading/domain/action_state_reward_contract.py
      why: "training reward contract and action semantics"
    - path: tests/unit/scripts/rl_trading/test_stage08i_upstream_evaluator_session_parity_forensic.py
      why: "existing regression tests for the found mismatch"
  external_sources:
    - repo: https://github.com/YuriyKolesnikov/rl-trading-binance
      required_commit: f71130903f8237351164f4b875494185465bf1ea
      required_files:
        - backtest_engine.py
        - trading_environment.py
        - agent.py
        - config.py
        - configs/alpha.py
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
  allow_optuna: false
  allow_mainnet_submit: false
  allow_exchange_side_effects: false
  allow_browser_runtime_verification: false
  allow_tests_only_acceptance: false
skill_routing:
  - skill: root-cause-debugging
    use_when: "repairing the concrete parity regressions found by 08I and 08I2"
    timing: "during implementation"
    reason: "owns evidence-first repair of observed behavior drift"
  - skill: backend-quality-gates
    use_when: "Python code/tests are changed"
    timing: "verification"
    reason: "owns focused ruff, pyright and pytest gate triage"
  - skill: architecture-review
    use_when: "classifying whether a source-vs-Roehub difference is repaired, accepted deviation, or still blocking"
    timing: "before final report"
    reason: "keeps fact/inference/source separation"
target_envs:
  - "local checkout"
  - "macstudio for target-host non-production forensic parity evidence"
required_literals:
  - "08I3"
  - "rolling_open_sessions"
  - "shared_balance_position_fraction"
  - "training_reward"
  - "backtest_reporting_reward"
  - "target_host_non_production_forensic_pre_main"
non_goals:
  - "Do not start Stage 08J, materialize article-selector datasets, run training, run Optuna, register models, or open Stage 09."
  - "Do not redesign the Stage 02C training reward; only separate training reward from backtest reporting reward where needed for parity."
  - "Do not tune max_parallel_sessions, position_fraction, thresholds, or risk parameters in this stage."
  - "Do not create branches, worktrees, stashes, temporary repo checkouts, or auxiliary workflow folders."
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes for changed Python paths; narrow only with a documented unrelated failure"
  - cmd: "uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes for changed Python paths"
  - cmd: "uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "focused evaluator/action/reward parity tests pass"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: integration
  e2e_required: true
  proof_boundary: target_host_non_production_forensic_pre_main
  acceptance_surfaces:
    - "source-derived evaluator/session trace parity"
    - "rolling open_sessions scheduling fixture"
    - "shared balance and position_fraction sizing fixture"
    - "unmasked-Q filter/order fixture"
    - "training_reward vs backtest_reporting_reward trace fixture"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i3-evaluator-action-reward-parity-repair.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "08I3"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "tests/unit/contexts/rl_trading"
  - "tests/unit/scripts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i3-evaluator-action-reward-parity-repair.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Runtime traces and matrices live under /opt/roehub/state/rl_trading/ and are not committed."
  - "Trace files must contain sanitized scalar decisions, hashes, counts and metrics only; never raw checkpoint tensors or credentials."
  - "Mac Studio git commands must use /Users/daniildegtyarev/Projects/roehub.com; /opt/roehub/app is runtime state only."
---

# Task

Implement Stage `08I3` evaluator/action/reward parity repair.

`08I` found a material upstream-vs-Roehub mismatch, and `08I2` proved the mismatch is part of a larger methodology gap set. This stage fixes only the pre-`08J` blockers that can invalidate any later dataset or training conclusion:

- rolling `open_sessions` session scheduling;
- shared balance and `balance * position_fraction` sizing semantics;
- Q/action filtering order and invalid/no-op action accounting;
- separation between `training_reward` and `backtest_reporting_reward`;
- source-derived trace parity against pinned upstream `backtest_engine.py`.

## Prompt-Pack Execution Anchor

- `plan_doc`: `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/rl-trading-agent-platform-v1`
- `stage_ledger`: `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `execution_mode`: `manual_sequential`
- `GOAL.md`: optional, not required by default.
- Stage gate: read the ledger before edits; run only when `current_stage=08I3`.
- Manifest gate: every created/modified/deleted file and every runtime artifact path must be recorded in the stage report and ledger.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: verify `08I` and `08I2` are `blocked`, `current_stage=08I3`, and `08J`/`08K`/`09` are blocked. If not true, write/update the `08I3` report as blocked, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for `08I3`. Do not use username `smoke_e2e_keycloak` and do not read `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Compute and record this prompt hash: `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08i3-evaluator-action-reward-parity-repair.md`.
- Use upstream commit `f71130903f8237351164f4b875494185465bf1ea`. If source behavior cannot be read or proven, block with exact missing source/evidence instead of guessing.
- Implement source-derived evaluator parity, not a new trading idea. If upstream behavior differs from this prompt wording, upstream source wins and the report must explain the correction.
- Port/repair rolling `open_sessions` scheduling:
  - do not cap only exact `signal_time` groups;
  - maintain rolling active sessions until their `agent_session_len` window completes;
  - preserve source ordering and selected/skipped session accounting.
- Port/repair shared-balance sizing:
  - selected sessions must be sized from source-derived shared account balance and `position_fraction`;
  - independent per-session `initial_balance * position_fraction` aggregation is not accepted for article-parity backtest;
  - record the exact balance/equity update order used by the source.
- Port/repair Q/action filtering order:
  - compare raw Q-values, unmasked action, filter-selected action, masked/effective action, and invalid/no-op reason as separate trace fields;
  - do not mask invalid Q-values before the filter if upstream filters unmasked Q advantages;
  - preserve invalid action/no-op handling in environment/backtest semantics rather than hiding it in a pre-filter mask.
- Port/repair reward reporting semantics:
  - keep Stage `02C` training reward unchanged unless upstream parity proves an implementation bug;
  - report `training_reward` separately from `backtest_reporting_reward`;
  - source `backtest_step()` reporting reward `0.0` with PnL in `info` must not be compared as if it were the training reward.
- Cover close/open price index semantics, last-step forced close, commission/slippage application, risk-management override timing, Q-cache/state normalization, and trace field names enough to remove pre-`08J` evaluator blockers.
- Add regression tests for every repaired behavior. Tests must fail on the old exact-group/independent-sizing/masked-before-filter/reporting-reward behavior.
- Re-run or extend the `08I` forensic trace on the same HF checkpoint/config/backtest split when Mac Studio artifacts are available. If full dynamic parity cannot be run, block with exact unavailable artifact and keep `08J` closed.
- Write a stage report at `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i3-evaluator-action-reward-parity-repair.md`.
- Update the stage ledger after validation and before final output.

## Acceptance Criteria

- The report includes prompt hash, upstream commit, source files/hashes used, code/test changes, local gates, Mac Studio/runtime artifact paths/hashes, parity verdict, file manifest, contract impact, proof boundary, and next-stage handoff.
- `08I3` can be accepted only if the repaired Roehub evaluator has no material first diff against the source-derived upstream trace for the required surface, or if any remaining difference is explicitly classified as a non-material accepted deviation with source-backed reason.
- `08I3` acceptance does not open `09` and does not start training. It may only unlock `08I4` recheck.
- If any pre-`08J` blocker remains, ledger stays blocked and `08J`/`08K`/`09` remain closed.

## Final Output

Respond in Russian with result/status, repaired semantics, parity evidence, file manifest, quality gates, residual risks, and the next prompt to run.
