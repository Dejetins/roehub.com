---
prompt_name: 08f-roehub-native-backtest-evaluation
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Evaluate roehub_native_candidate with Roehub-native scorecard and decide whether Stage 09 may start."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08e-roehub-native-full-training-run.md
      why: "Roehub-native candidate manifest and hashes"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08d-original-hf-backtest-evaluation.md
      why: "HF methodology-execution evidence and warning register"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-upstream-methodology-parity-audit.md
      why: "frozen upstream grouped-backtest and acceptance checklist"
  task_entrypoints:
    - path: src/trading/contexts/rl_trading
      why: "evaluation/backtest code"
    - path: scripts/rl_trading
      why: "evaluation CLIs"
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
    use_when: "scorecard, baselines, overfit and stability interpretation"
    timing: "implementation/evaluation"
    reason: "owns evaluation methodology"
  - skill: backend-quality-gates
    use_when: "Python code/tests are changed"
    timing: "verification"
    reason: "owns ruff, pyright, pytest triage"
target_envs:
  - "local checkout"
  - "macstudio for Roehub-native evaluation/backtest"
required_literals:
  - "roehub_native_candidate"
  - "08F"
non_goals:
  - "Do not train a new model in Stage 08F."
  - "Do not register, promote, activate, paper/testnet/live trade, or mainnet submit."
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/rl_trading apps tests"
    expect: "passes for changed paths"
  - cmd: "uv run pyright src/trading/contexts/rl_trading apps tests"
    expect: "passes for changed paths"
  - cmd: "uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps"
    expect: "passes focused tests"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: integration
  e2e_required: true
  proof_boundary: target_host_non_production_evaluation_pre_main
  acceptance_surfaces:
    - "Roehub-native held-out evaluation"
    - "Roehub-native grouped backtest with action filters"
    - "sanity baselines, simulator/accounting parity and research candidate decision"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08f-roehub-native-backtest-evaluation.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  current_stage: "08F"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "tests/unit/contexts/rl_trading"
  - "tests/perf_smoke/contexts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08f-roehub-native-backtest-evaluation.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
---

# Task

Implement Stage `08F` Roehub-native evaluation/backtest.

This is the new research-save gate for the platform. Stage `09` remains blocked unless `08F` accepts a Roehub-native research candidate.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: before any edits or evaluation launch, read the ledger and verify Stage `08E` is `accepted`, provides `roehub_native_candidate`, and `current_stage` allows Stage `08F`. If not true, write/update the Stage `08F` report as `blocked`, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for this offline evaluation stage. Do not use the Roehub smoke Keycloak username `smoke_e2e_keycloak` and do not read the host-local password source `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Credential redaction rule: never write secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, signed requests, raw checkpoint tensors, or credentials into prompts, docs, ledgers, traces, screenshots, logs, reports, or runtime artifacts.
- File manifest gate: every created, modified, deleted, and runtime artifact path must be listed in the Stage `08F` report and ledger update; any file outside expected paths requires explicit outside-manifest justification and must not be changed speculatively.
- Evaluate `best` checkpoint by default and use train-only normalization stats.
- Evaluate held-out test/backtest splits from the accepted Stage `06` dataset. Do not evaluate train split as acceptance.
- Use the same Stage `08A`/`08D` upstream-compatible filtered backtest lifecycle: grouped signals by timestamp, `max_parallel_sessions`, `position_fraction`, Q-value cache, advantage threshold filter and/or MC-dropout ensemble filter, action rejection/skipped-signal counts, optional risk-management diagnostics, balance curve, fees/slippage/funding policy and cache statistics. Raw argmax-only evaluation is diagnostic only and cannot accept the research candidate.
- Run sanity baselines: hold/no-trade, deterministic random, simple threshold and any accepted HF baseline diagnostics.
- Include simulator/accounting parity fixture and make divergence a blocker.
- Candidate may be saved as `research_candidate` only if net PnL after costs is positive and execution/parity checks do not block. Baseline dominance, low positive-session ratio, missing tuning, or weak demo-HF carryover are warning signals unless they reveal simulator/accounting leakage, reward-shaping failure, invalid split/normalization, or no actionable native research signal.
- Record whether stronger `90/60` or larger-profile training, multiple seeds, or Optuna/tuned backtest calibration are required before promotion-grade review. Do not run those inside Stage `08F` unless the prompt is explicitly superseded.
- Positive research candidate does not grant promotion-grade or runtime activation; Stage `09` may only register candidate metadata and Stage `10A` owns promotion thresholds.

## Acceptance Criteria

- Stage report records scorecard by split/ticker/month/volatility bucket, trade count, action distribution, skipped/rejected actions, drawdown, stability, overfit warnings, Stage `08D` carried-forward warnings, latency/resource notes and cost/funding policy.
- Evaluation artifacts are written under `/opt/roehub/state/rl_trading/` with hashes only in docs.
- Ledger advances to `09` only if `08F` is accepted. Otherwise it remains blocked with exact repair instructions.

## Final Output

Respond in Russian with result/status, file manifest, Roehub-native evaluation evidence, research candidate decision, gates, residual risks and `09` handoff.
