---
prompt_name: 08a-original-hf-backtest-evaluation
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Evaluate hf_original_candidate on the original HF test/backtest splits using the upstream-compatible lifecycle."
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
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07e-original-hf-full-training-run.md
      why: "HF candidate manifest and hashes"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07c-upstream-methodology-parity-audit.md
      why: "methodology parity checklist"
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
  - "macstudio for HF evaluation/backtest"
required_literals:
  - "hf_original_candidate"
  - "08A"
non_goals:
  - "Do not train a new model in Stage 08A."
  - "Do not train on Roehub-native Stage 06 data."
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
    - "HF test evaluation"
    - "HF grouped backtest with action filters"
    - "sanity baselines and methodology-parity verdict"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-original-hf-backtest-evaluation.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  current_stage: "08A"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "tests/unit/contexts/rl_trading"
  - "tests/perf_smoke/contexts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-original-hf-backtest-evaluation.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
---

# Task

Implement Stage `08A` original HF evaluation/backtest.

This stage proves whether the Roehub port can run the original dataset lifecycle coherently before the platform spends time training on the Roehub-native dataset.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: before any edits or evaluation launch, read the ledger and verify Stage `07E` is `accepted`, provides `hf_original_candidate`, and `current_stage` allows Stage `08A`. If not true, write/update the Stage `08A` report as `blocked`, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for this offline evaluation stage. Do not use the Roehub smoke Keycloak username `smoke_e2e_keycloak` and do not read the host-local password source `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Credential redaction rule: never write secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, signed requests, raw checkpoint tensors, or credentials into prompts, docs, ledgers, traces, screenshots, logs, reports, or runtime artifacts.
- Evaluate the `best` checkpoint by default; `final` may be diagnostic only unless the report explains otherwise.
- Use train-only normalization stats from `07E`.
- Run test-set episode evaluation with reward/PnL/win-rate metrics.
- Run grouped backtest using upstream-compatible `backtest_engine.py` semantics: signal grouping by timestamp, `max_parallel_sessions`, `position_fraction`, Q-value cache, advantage filter and/or MC-dropout ensemble filter.
- Record raw-argmax diagnostics separately, but raw argmax-only cannot be the acceptance backtest.
- Run technical baselines/sanity checks and action-filter/rejection counts.
- Produce a methodology-parity verdict:
  - `accepted` only if the HF lifecycle runs end to end with no unresolved implementation gap and scorecard is internally coherent;
  - `blocked` if metrics are negative or materially inconsistent without an explained dataset/config reason.
- `07F` may start only if `08A` is accepted.

## Acceptance Criteria

- Stage report records HF test and backtest scorecards, action filter thresholds, rejection counts, baselines, metrics by period/ticker and overfit warnings.
- Evaluation artifacts are written under `/opt/roehub/state/rl_trading/` with hashes only in docs.
- Ledger advances to `07F` only after accepted HF methodology-parity evidence.

## Final Output

Respond in Russian with result/status, file manifest, HF evaluation evidence, parity verdict, gates, residual risks and `07F` handoff.
