---
prompt_name: 08c-original-hf-full-training-run
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Run full upstream-compatible training on the original HF dataset and produce hf_original_candidate."
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
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08b-upstream-methodology-core-port.md
      why: "accepted upstream-compatible core"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04-hf-reproducibility.md
      why: "HF dataset manifests, hashes and local paths"
  task_entrypoints:
    - path: src/trading/contexts/rl_trading
      why: "upstream-compatible trainer core"
    - path: scripts/rl_trading
      why: "training CLIs"
    - path: apps/worker/rl_trading_trainer
      why: "trainer app"
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
  - "macstudio for full HF training"
required_literals:
  - "hf_original_candidate"
  - "/opt/roehub/state/rl_trading/"
  - "08C"
non_goals:
  - "Do not train on Roehub-native Stage 06 data in Stage 08C."
  - "Do not run Stage 08D evaluation inside Stage 08C."
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
    - "completed full HF-original training run"
    - "best/final checkpoint hashes"
    - "episode/step progress and resource evidence"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08c-original-hf-full-training-run.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  current_stage: "08C"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "apps/worker/rl_trading_trainer"
  - "tests/unit/contexts/rl_trading"
  - "tests/perf_smoke/contexts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08c-original-hf-full-training-run.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "configs/dev"
  - "configs/test"
  - "configs/prod"
  - "docs/architecture/README.md"
safety_notes:
  - "HF datasets/checkpoints/logs stay under /opt/roehub/state/rl_trading/ and are not committed."
---

# Task

Implement Stage `08C` original HF full training run.

Train the upstream-compatible Roehub agent on the original external HF dataset splits using the accepted Stage `08B` core. This is the reproducibility gate before Roehub-native training.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: before any edits or training launch, read the ledger and verify Stage `08B` is `accepted`, Stage `04` is `accepted`, and `current_stage` allows Stage `08C`. If not true, write/update the Stage `08C` report as `blocked`, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for this offline training stage. Do not use the Roehub smoke Keycloak username `smoke_e2e_keycloak` and do not read the host-local password source `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Credential redaction rule: never write secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, signed requests, raw checkpoint tensors, or credentials into prompts, docs, ledgers, traces, screenshots, logs, reports, or runtime artifacts.
- File manifest gate: every created, modified, deleted, and runtime artifact path must be listed in the Stage `08C` report and ledger update; any file outside expected paths requires explicit outside-manifest justification and must not be changed speculatively.
- Use the original HF dataset files and hashes recorded by Stage `04`; do not use Stage `06` Roehub-native data.
- Use upstream `configs/alpha.py` as the default full-training profile: CNN maps/kernels/strides, dense layers, dropout, action-history length, `episodes=55_000`, `batch_size=16`, `learning_rate=1e-4`, `train_start=10_000`, PER capacity `230_000`, validation cadence and advantage thresholds. Any deviation must be explicit and justified.
- Progress must be durable and episode/environment-step based: `progress.jsonl`, latest status, completed episodes, planned episodes, completed env steps, elapsed, ETA, device/resource snapshot and status.
- If full training cannot complete in the active session, launch only through a managed resumable/background path, prove a fresh `running` event and leave Stage `08C` `in_progress`. Do not mark accepted until full training completes.
- On completion, write `hf_original_candidate` manifest with train dataset hashes, config hash, code state, train-only normalization stats hash, `best` checkpoint hash, `final` checkpoint hash, validation curves, resource metrics and progress hash.
- Do not run test/backtest acceptance here; Stage `08D` owns evaluation.

## Acceptance Criteria

- Completed full HF training manifest exists under `/opt/roehub/state/rl_trading/`.
- `best` checkpoint selection is by validation metric, not final step by default.
- Training used environment rollout and agent actions; scripted offline transition generation is absent from candidate path.
- Ledger advances to `08D` only after completed `hf_original_candidate` evidence.

## Final Output

Respond in Russian with result/status, file manifest, training evidence, progress/resource evidence, quality gates, residual risks and `08D` handoff.
