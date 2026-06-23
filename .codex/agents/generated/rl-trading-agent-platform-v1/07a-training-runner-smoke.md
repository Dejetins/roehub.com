---
prompt_name: 07a-training-runner-smoke
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Implement the RL D3QN/PER trainer mechanics and prove a deterministic small Mac Studio training smoke without claiming a full candidate model."
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
      why: "stage ledger and current stage"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "optional compact state; ignore if unrelated"
  task_entrypoints:
    - path: src/trading/contexts/rl_trading/domain/action_state_reward_contract.py
      why: "Stage 02C action/reward/state compatibility"
    - path: src/trading/contexts/rl_trading/domain/sessionized_dataset.py
      why: "accepted Stage 06 dataset contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md
      why: "accepted Stage 06 report and runtime manifest path"
    - path: pyproject.toml
      why: "optional rl-ml dependency/runtime isolation"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md
      why: "stage report shape"
  conditional_bundles:
    docs_to_update:
      read_when: "stage changes shared docs, contracts, runbooks, or operator workflows"
      paths:
        - docs/architecture/README.md
        - docs/runbooks/mac-studio-native-backend-operations.md
    performance:
      read_when: "benchmarking CPU/MPS/RSS or resource limits"
      paths:
        - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/03-mac-studio-ml-environment.md
  consult_if_needed:
    - path: docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md
      read_when: "unexpected source-event, intent, risk gate, paper/testnet/live, or idempotency scope appears"
    - path: docs/architecture/README.md
      read_when: "Markdown docs are added or changed and docs index must be verified"
style_references:
  - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md
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
    use_when: "benchmarking CPU/MPS/RSS/training-smoke time"
    timing: "during verification"
    reason: "requires comparable resource evidence"
  - skill: backend-quality-gates
    use_when: "backend Python code, tests, lint, or type gates are changed"
    timing: "during verification"
    reason: "owns focused ruff, pyright, pytest gate triage"
  - skill: contract-impact-analysis
    use_when: "API, DTO, persistence, config, cache identity, or browser-visible defaults change"
    timing: "during implementation"
    reason: "classifies compatibility and rollout impact"
  - skill: publish-ci-deploy
    use_when: "publishing accepted stage changes through the default direct-main delivery workflow"
    timing: "before ship"
    reason: "owns scoped staging, main delivery, CI/deploy follow-up, and Mac Studio verification; branch/PR only when explicitly requested by the user"
target_envs:
  - "local checkout"
  - "macstudio for small training-smoke/resource evidence"
required_literals:
  - "rl-trading-agent-platform-v1"
  - "/opt/roehub/state/rl_trading/"
  - "07A"
non_goals:
  - "Do not run the full candidate training run in Stage 07A."
  - "Do not register, promote, activate, paper trade, testnet trade, live trade, or mainnet submit."
  - "Do not train user-owned custom models."
  - "Do not add cloud/S3/model hosting."
  - "Do not bypass live_execution or exchange-execution."
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/rl_trading apps tests"
    expect: "passes for changed backend/app/test paths"
  - cmd: "uv run pyright src/trading/contexts/rl_trading apps tests"
    expect: "passes for changed backend/app/test paths"
  - cmd: "uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps"
    expect: "focused tests pass; narrow target if directories do not exist yet"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: benchmark
  e2e_required: true
  proof_boundary: target_host_non_production_sample_pre_main
  acceptance_surfaces:
    - "small deterministic Mac Studio training smoke"
    - "CPU/MPS/RSS/resource evidence"
    - "action/reward/state fixture compatibility"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07a-training-runner-smoke.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "07A"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "apps/worker/rl_trading_trainer"
  - "tests/unit/contexts/rl_trading"
  - "tests/perf_smoke/contexts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07a-training-runner-smoke.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "pyproject.toml"
  - "uv.lock"
  - "configs/dev"
  - "configs/test"
  - "configs/prod"
  - "docs/runbooks/mac-studio-native-backend-operations.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Never write secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, raw signed requests, or raw checkpoint tensors into prompts, docs, ledgers, traces, screenshots, or logs."
  - "Runtime ML artifacts live under /opt/roehub/state/rl_trading/ and are not committed to git."
  - "Mac Studio git commands must use /Users/daniildegtyarev/Projects/roehub.com; /opt/roehub/app is runtime state only."
  - "ML trainer code must not call exchange SDKs or resolve exchange secrets directly."
  - "Browser/auth is N/A for this trainer/data stage. Do not use the Roehub smoke Keycloak username `smoke_e2e_keycloak` and do not read the host-local password source `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD` unless a later browser-visible task explicitly requires it; never print or persist that secret."
---

# Task

Implement Stage `07A` D3QN/PER training runner smoke.

Build the Roehub trainer mechanics: D3QN model surface, PER replay buffer, training-loop skeleton, environment/action-reward fixtures, run-record schema, and a deterministic small training smoke on Mac Studio. This stage proves that training works technically. It must not claim a full trained candidate model.

Done means:

- trainer/replay/environment fixtures exist and pass focused tests;
- Stage `02C` action/reward/state compatibility is tested before smoke acceptance;
- a small deterministic Mac Studio smoke runs from accepted Stage `06` sessionized data or a tiny fixture derived from it;
- run records include seed, config hash, dataset manifest hash, model architecture hash, metrics, resource usage, and artifact hashes;
- Stage `07B` receives an exact handoff for full candidate training.

## Context / Current State

- Stage `06` is accepted and materialized under `/opt/roehub/state/rl_trading/datasets/stage06_sessionized_dataset_v1/`.
- Stage `07` has been split: `07A` is trainer-smoke capability; `07B` is the full candidate training run.
- V1 training source remains `binance:futures` only. Binance spot, Bybit spot, and Bybit futures stay `blocked_not_training_source_v1`.
- Classic producer blockers do not block this offline trainer stage, but they still block later RL paper/testnet/live stages.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`. If that statement is not true after reading the ledger, stop and record the blocker.
- Verify prerequisites before implementation. Required accepted prerequisites: Stage `04`, Stage `06`. If any required prerequisite is not accepted, write/update the Stage `07A` report as blocked, update the ledger, and do not implement dependent work.
- Compute this prompt hash with `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/07a-training-runner-smoke.md` and record path/hash in the stage report.
- Before editing, narrow expected directories to a concrete file list and record it in the stage report.
- Keep the change bounded to Stage `07A`. Do not run full candidate training, evaluate candidate quality, register model metadata, promote, activate, or touch execution paths.
- Keep PyTorch isolated from default API runtime.
- Use accepted Stage `06` `binance:futures` dataset metadata and hashes; do not train/evaluate spot or Bybit branches.
- Store smoke artifacts under `/opt/roehub/state/rl_trading/`, not git. Commit only code, tests, and sanitized summaries/hashes.
- Bound CPU threads/RSS/MPS usage and record observed resource evidence.
- Update/create `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07a-training-runner-smoke.md`.
- Update the stage ledger after validation and before final response. Stage `07B` may start only if `07A` is accepted.
- The ledger update must record `current_stage`, prerequisite status, prompt path/hash, final file manifest, delivery state, validation evidence, blockers/residual risks, and whether Stage `07B` is allowed.

## Requirements (Should)

- Prefer small pure-domain components and application orchestration over framework coupling.
- Make run records deterministic and hashable.
- Keep evidence compact: hashes, counts, timings, resource metrics, and sanitized excerpts.

# Context Acquisition Protocol

Read `.codex/AGENTS.md`, the RL plan, ledger, Stage `06` report, then the listed task entrypoints. Expand only for blockers, failing checks, contract ambiguity, resource measurement, or dependency/runtime isolation concerns.

# Work Plan

1. Verify ledger current stage and prerequisites.
2. Record `User required before start`, prompt path/hash, planned file list, and blockers in the Stage `07A` report.
3. Implement trainer primitives, replay buffer, environment fixtures, run-record hashing, and smoke CLI/worker entrypoint.
4. Add focused tests for replay buffer sampling/update, D3QN/PER update shapes, reward/action fixtures, run-record hash determinism, and rejected-run behavior.
5. Run focused local gates.
6. Run a small deterministic Mac Studio smoke with CPU/MPS policy/resource evidence.
7. Update report and ledger with status, evidence, delivery state, residual risks, and Stage `07B` handoff.

# Acceptance Criteria

- Focused tests cover replay buffer, D3QN/PER update shape, action/reward fixtures, run-record hashing, and rejected-run behavior.
- Mac Studio smoke records wall-clock, RSS, CPU/MPS policy, seed, dataset manifest hash, config hash, model architecture hash, and artifact hashes.
- No full candidate checkpoint is reported as production/research candidate in Stage `07A`.
- Stage report includes prompt path/hash, file manifest, evidence, contract impact, and next-stage handoff.
- Ledger `current_stage` advances to `07B` only after accepted `07A` evidence.

# Implementation Constraints

- Determinism: sort inputs, seed all random sources, hash canonical JSON, and record nondeterministic backend limitations.
- Contracts: no public API, persistence, browser, exchange, or live execution behavior changes unless explicitly justified as additive and tested.
- Artifacts: large tensors/checkpoints/logs stay under `/opt/roehub/state/rl_trading/`.
- Git: default branch is `main`; do not create branches/worktrees/stashes unless explicitly requested by the user.
- Validation: tests are local gates; Mac Studio smoke/resource evidence is required for acceptance.

# Files To Indicate

Primary touches:

- `src/trading/contexts/rl_trading`
- `scripts/rl_trading`
- `apps/worker/rl_trading_trainer`
- `tests/unit/contexts/rl_trading`
- `tests/perf_smoke/contexts/rl_trading`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07a-training-runner-smoke.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`

Final report must list `created`, `modified`, `deleted`, and `outside_expected_paths`.

# Quality Gates

- `uv run ruff check src/trading/contexts/rl_trading apps tests`
- `uv run pyright src/trading/contexts/rl_trading apps tests`
- `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps`
- `python -m tools.docs.generate_docs_index --check`

# Final Output

Respond in Russian with:

1. **Result and stage status**
2. **File manifest**
3. **Evidence and quality gates**
4. **Contract, safety, and delivery state**
5. **Blockers and next-stage handoff**
