---
prompt_name: 08i2-exhaustive-methodology-discrepancy-audit
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
  stage_gate: "read ledger before edits; run only when current_stage is 08I2 and prerequisites match"
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
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i2-exhaustive-methodology-discrepancy-audit.md
    - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
    - docs/architecture/README.md
  forbidden_without_user_approval:
    - exchange execution paths
    - live_execution behavior
    - API/UI behavior
    - database migrations
    - branch/worktree/stash/local-folder workflow changes
scope: "Exhaustively audit source-vs-Roehub methodology discrepancies before any repair conclusion, article-selector dataset work, or new training."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, branch policy, proof boundaries"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and mandatory 08I2 discrepancy matrix"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i-upstream-evaluator-session-parity-forensic.md
      why: "blocked first-diff evidence from Stage 08I"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08h-oracle-supervised-selector-reward-90-60-research.md
      why: "diagnostics for oracle opportunity, reward sparsity, action/policy and Optuna failure"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-upstream-methodology-parity-audit.md
      why: "pinned upstream source map and parity surface"
  task_entrypoints:
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04-hf-reproducibility.md
      why: "HF dataset paths and hashes"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md
      why: "Stage 06 selector, split and leakage contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08d-original-hf-backtest-evaluation.md
      why: "HF evaluation mechanics and warnings"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08f-roehub-native-backtest-evaluation.md
      why: "native scorecard, action distribution and baseline evidence"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08g-dual-branch-cpu-optuna-training-evaluation.md
      why: "Optuna calibration/final evidence"
    - path: src/trading/contexts/rl_trading
      why: "current dataset, training, evaluator and scorecard code"
    - path: scripts/rl_trading
      why: "current training/evaluation/forensic CLIs"
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
        - replay_buffer.py
        - train.py
        - test_agent.py
        - backtest_engine.py
        - optimize_cfg.py
        - baseline_cnn_classifier.py
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
  - skill: architecture-review
    use_when: "classifying source-vs-Roehub methodology gaps and plan readiness"
    timing: "before final report"
    reason: "keeps fact/inference/evidence separation"
  - skill: data-analytics-methodology
    use_when: "dataset geometry, signal sanity, reward sparsity, baselines and overfit interpretation"
    timing: "analysis"
    reason: "keeps evaluation methodology defensible"
  - skill: backend-quality-gates
    use_when: "Python diagnostics/tests are changed"
    timing: "verification"
    reason: "owns focused ruff, pyright, pytest gate triage"
target_envs:
  - "local checkout"
  - "macstudio for HF/native artifacts and target-host forensic evidence"
required_literals:
  - "08I2"
  - "methodology_discrepancy_matrix"
  - "target_host_non_production_forensic_pre_main"
non_goals:
  - "Do not train, tune, register, promote, activate, paper trade, testnet trade, live trade, or mainnet submit."
  - "Do not start Stage 08J or materialize the article-selector dataset."
  - "Do not stop after the first material diff."
  - "Do not silently classify any diagnosis point as out of scope."
  - "Do not create branches, worktrees, stashes, temporary repo checkouts, or auxiliary workflow folders."
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes for changed Python paths; narrow only with a documented unrelated failure"
  - cmd: "uv run pyright src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "passes for changed Python paths"
  - cmd: "uv run pytest -q tests/unit/contexts/rl_trading tests/unit/scripts/rl_trading"
    expect: "focused discrepancy/forensic tests pass when Python changes"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: integration
  e2e_required: true
  proof_boundary: target_host_non_production_forensic_pre_main
  acceptance_surfaces:
    - "complete methodology discrepancy matrix with source-backed status for every mandatory row"
    - "runtime/source artifacts for any dynamic row that can be checked without training"
    - "repair backlog and recheck contract for every material gap"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i2-exhaustive-methodology-discrepancy-audit.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "08I2"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "tests/unit/contexts/rl_trading"
  - "tests/unit/scripts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i2-exhaustive-methodology-discrepancy-audit.md"
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

Implement Stage `08I2` exhaustive methodology discrepancy audit.

Stage `08I` already found an important material diff, but that does not prove all source-vs-Roehub methodology drift is understood. This stage must check every diagnosis surface before any repair conclusion, `08J`, `08K`, `09`, or new training.

## Prompt-Pack Execution Anchor

- `plan_doc`: `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `prompt_pack_dir`: `.codex/agents/generated/rl-trading-agent-platform-v1`
- `stage_ledger`: `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`
- `execution_mode`: `manual_sequential`
- `GOAL.md`: optional, not required by default.
- Stage gate: read the ledger before edits; run only when `current_stage=08I2`.
- Manifest gate: every created/modified/deleted file and every runtime artifact path must be recorded in the stage report and ledger.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: verify `08I` is `blocked`, `current_stage=08I2`, and `08J`/`09` are blocked. If not true, write/update the `08I2` report as blocked, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for `08I2`. Do not use username `smoke_e2e_keycloak` and do not read `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Compute and record this prompt hash: `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08i2-exhaustive-methodology-discrepancy-audit.md`.
- Use upstream commit `f71130903f8237351164f4b875494185465bf1ea`. If the source cannot be read or run, record a blocked state with the exact missing dependency/source/artifact.
- Produce a durable `methodology_discrepancy_matrix` artifact and summarize it in the stage report. Each mandatory row must have: `surface`, `source_expected_behavior`, `roehub_current_behavior`, `status`, `evidence_paths`, `severity`, `repair_required`, `recheck_required`, and `next_stage_allowed`.
- Allowed row statuses are only: `checked_no_gap`, `gap`, `blocked_by_prior_gap`, `not_applicable_with_source_reason`.
- Do not stop at the first material diff. If a dynamic check is blocked by the known `08I` scheduler/sizing mismatch, still complete source review, code review and any controlled/static checks for that row, then mark it `blocked_by_prior_gap` with the exact recheck after repair.

## Mandatory Discrepancy Matrix

Check every row below before any conclusion:

1. **Session extractor policy**
   - Compare article/repo event selection with Stage `06` `pre_signal_realized_volatility_plus_range_v1`.
   - Verify the intended `article_future_10m_5pct_contrast_v1` semantics: `10m` move at least `5%`, previous `90m` contrast rule, `event_end_t` as `signal_ts_open`, `pre_window=[signal_ts_open-90m, signal_ts_open)`, `post_window=[signal_ts_open, signal_ts_open+60m)`.
   - Check overlap, stride, split embargo, lifecycle/listing/delisting, leakage and rejected-window reasons required for `08J`.

2. **Dataset geometry and distribution**
   - Compare HF-original vs Stage `06` current selector vs planned article-selector requirements.
   - Include counts by split, ticker, month, volatility/range bucket, session density, train/validation/test/backtest ratio, symbol coverage and lifecycle exclusions.

3. **Past-only signal strength**
   - Check oracle labels, oracle best-return and supervised sanity evidence by split/profile.
   - Decide whether past-only features predict direction or only select volatile/noisy windows.

4. **Reward sparsity and semantics**
   - Compare source training reward, source `backtest_step()` reporting reward, Roehub training reward, Roehub backtest/reporting reward, hold penalty and close timing.
   - Cover both `30/10` article/demo and failed `90/60` evidence.

5. **Action/Q policy distribution**
   - Check raw argmax, masked action, selected action, effective action, long/short/hold/close distribution, Q-value scale, action mask order and pathological one-sided bias.
   - Include the prior short-bias evidence from `08F`/`08H` and state what must be rechecked after evaluator repair.

6. **`Optuna` and calibration overfit**
   - Compare upstream `optimize_cfg.py`, `configs/alpha.py`, Roehub search spaces, calibration/final split isolation, trade-sufficient selection, zero-trade prevention and final-holdout stability.
   - Confirm `Optuna` cannot open `09` without untouched final holdout and strict native gate.

7. **Sanity baselines**
   - Check hold/no-trade/random/simple-threshold and any source-relevant baseline on the same evaluation surface.
   - Confirm baseline beating is a hard blocker for native research candidate acceptance.

8. **Full evaluator/backtest parity**
   - Continue beyond the first `08I` diff and check: shared balance vs independent aggregation, rolling `open_sessions`, signal group ordering, close/open price index semantics, last-step action mask, commission/slippage application, action filter thresholds, risk-management timing, Q-cache/state hashes, train-only normalization and trace field semantics.

## Acceptance Criteria

- Stage report exists at `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08i2-exhaustive-methodology-discrepancy-audit.md`.
- Report includes prompt hash, upstream commit, source files/hashes, input artifact references, discrepancy matrix summary, runtime/source artifact paths/hashes, repair backlog, file manifest, contract impact, proof boundary, delivery state, and next-stage handoff.
- Every mandatory matrix row is present and has an allowed row status. Missing rows are a blocker.
- Ledger advances to `08J` only if `08I2` is accepted and no unresolved material evaluator/session blocker remains. If material gaps remain, keep `08J`, `08K`, and `09` blocked and record the exact repair/recheck path.
- Stage `09` remains blocked unless a later accepted corrective candidate explicitly records `stage09_allowed=true`; `08I2` itself must not open `09`.

## Final Output

Respond in Russian with result/status, discrepancy matrix verdict, material gaps, repair backlog, file manifest, quality gates, residual risks, and the next prompt to run.
