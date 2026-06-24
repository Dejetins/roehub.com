---
prompt_name: 08e-roehub-native-full-training-run
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Run full upstream-compatible training on the accepted Roehub-native Stage 06 dataset and produce roehub_native_candidate."
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
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08d-original-hf-backtest-evaluation.md
      why: "accepted HF methodology-execution gate and warning register"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-upstream-methodology-parity-audit.md
      why: "frozen upstream source-profile and accepted deviations"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08c-original-hf-full-training-run.md
      why: "HF-original training profile and checkpoint/progress contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md
      why: "accepted Roehub-native dataset manifest"
  task_entrypoints:
    - path: src/trading/contexts/rl_trading
      why: "upstream-compatible trainer"
    - path: scripts/rl_trading
      why: "training CLIs"
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
  - skill: backend-performance-evidence
    use_when: "measuring full training runtime, CPU/MPS/RSS and throughput"
    timing: "verification"
    reason: "owns resource evidence"
  - skill: backend-quality-gates
    use_when: "Python code/tests are changed"
    timing: "verification"
    reason: "owns ruff, pyright, pytest triage"
target_envs:
  - "local checkout"
  - "macstudio for full Roehub-native training"
required_literals:
  - "roehub_native_candidate"
  - "/opt/roehub/state/rl_trading/"
  - "08E"
non_goals:
  - "Do not train on the HF original dataset in Stage 08E."
  - "Do not run Stage 08F evaluation inside Stage 08E."
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
  depth: benchmark
  e2e_required: true
  proof_boundary: target_host_non_production_training_pre_main
  acceptance_surfaces:
    - "completed full Roehub-native training run"
    - "best/final checkpoint hashes"
    - "episode/step progress and resource evidence"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08e-roehub-native-full-training-run.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  current_stage: "08E"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "apps/worker/rl_trading_trainer"
  - "tests/unit/contexts/rl_trading"
  - "tests/perf_smoke/contexts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08e-roehub-native-full-training-run.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "configs/dev"
  - "configs/test"
  - "configs/prod"
  - "docs/architecture/README.md"
---

# Task

Implement Stage `08E` Roehub-native full training run.

Train the upstream-compatible Roehub agent on the accepted Stage `06` Roehub-native Binance Futures dataset. This is the platform-native research branch after HF methodology execution has been accepted; Stage `08D` score weakness is warning context, not a native-training blocker.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: before any edits or training launch, read the ledger and verify Stage `08D` is `accepted` for methodology execution, Stage `06` is `accepted`, and `current_stage` allows Stage `08E`. If not true, write/update the Stage `08E` report as `blocked`, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for this offline training stage. Do not use the Roehub smoke Keycloak username `smoke_e2e_keycloak` and do not read the host-local password source `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Credential redaction rule: never write secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, signed requests, raw checkpoint tensors, or credentials into prompts, docs, ledgers, traces, screenshots, logs, reports, or runtime artifacts.
- File manifest gate: every created, modified, deleted, and runtime artifact path must be listed in the Stage `08E` report and ledger update; any file outside expected paths requires explicit outside-manifest justification and must not be changed speculatively.
- Use accepted Stage `06` sessionized dataset only; do not use six-symbol fallback, old 215-symbol subset, or external HF data.
- Use the same Stage `08A`/`08C` methodology/config family: environment rollout through agent actions, train-only normalization stats, CNN dueling D3QN with dropout, epsilon/PER lifecycle, target sync, gradient clipping, validation-selected `best` plus `final`, and durable episode/env-step progress. Any adaptation for Stage `06` dataset size, symbol count, train/val split, cost/funding policy or resource budget must be listed in an adaptation diff.
- Carry forward Stage `08D` warnings explicitly: weak untuned HF-demo profitability, simple baseline outperforming the HF candidate, low positive-session ratio, missing Optuna/tuning, and demo `30/10` profile. These warnings do not block Stage `08E`, but they must inform Stage `08F` scorecard interpretation and later stronger-training/tuning follow-up.
- Progress must be durable and episode/environment-step based.
- If full training cannot complete in the active session, launch only through a managed resumable/background path, prove a fresh `running` event and leave Stage `08E` `in_progress`. Do not mark accepted until full training completes.
- On completion, write `roehub_native_candidate` manifest with Stage `06` dataset hash, config hash, code state, train-only normalization stats hash, `best` checkpoint hash, `final` checkpoint hash, validation curves, resource metrics and progress hash.
- Do not evaluate candidate quality here; Stage `08F` owns evaluation.

## Acceptance Criteria

- Completed full Roehub-native training manifest exists under `/opt/roehub/state/rl_trading/`.
- Candidate path uses environment rollout and validation-selected `best` checkpoint.
- Stage report includes adaptation diff from `hf_original_candidate` training and a carried-forward warning register from Stage `08D`.
- Ledger advances to `08F` only after completed `roehub_native_candidate` evidence.

## Final Output

Respond in Russian with result/status, file manifest, training evidence, adaptation diff, progress/resource evidence, gates, residual risks and `08F` handoff.
