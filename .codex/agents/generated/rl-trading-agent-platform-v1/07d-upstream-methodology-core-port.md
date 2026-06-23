---
prompt_name: 07d-upstream-methodology-core-port
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Port the upstream CNN D3QN/PER environment-rollout methodology into Roehub without running full training."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and parity target"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07c-upstream-methodology-parity-audit.md
      why: "authoritative parity checklist"
  task_entrypoints:
    - path: src/trading/contexts/rl_trading/domain
      why: "RL domain implementation"
    - path: scripts/rl_trading
      why: "operator/training/evaluation CLIs"
    - path: apps/worker/rl_trading_trainer
      why: "trainer app entrypoint"
    - path: tests/unit/contexts/rl_trading
      why: "focused tests"
  external_sources:
    - repo: https://github.com/YuriyKolesnikov/rl-trading-binance
      required_files:
        - config.py
        - configs/alpha.py
        - model.py
        - agent.py
        - replay_buffer.py
        - trading_environment.py
        - train.py
        - test_agent.py
        - backtest_engine.py
        - utils.py
      rule: "Use the same pinned source SHA recorded by Stage 07C."
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
  - skill: backend-quality-gates
    use_when: "Python implementation/tests are changed"
    timing: "verification"
    reason: "owns ruff, pyright, pytest triage"
  - skill: backend-performance-evidence
    use_when: "CPU/MPS/RSS or smoke runtime is measured"
    timing: "verification"
    reason: "owns resource evidence"
  - skill: contract-impact-analysis
    use_when: "feature/model artifact/config contracts are changed"
    timing: "implementation"
    reason: "classifies compatibility"
target_envs:
  - "local checkout"
  - "macstudio for small non-production parity smoke when needed"
required_literals:
  - "roehub_d3qn_cnn_dueling_v1"
  - "upstream_methodology_parity"
  - "07D"
non_goals:
  - "Do not run full HF or Roehub-native training in Stage 07D."
  - "Do not evaluate candidate quality or advance registry/promotion."
  - "Do not remove historical Stage 07A/07B artifacts; mark old MLP path smoke/debug only."
quality_gates:
  - cmd: "uv run ruff check src/trading/contexts/rl_trading apps tests"
    expect: "passes for changed backend/app/test paths"
  - cmd: "uv run pyright src/trading/contexts/rl_trading apps tests"
    expect: "passes for changed backend/app/test paths"
  - cmd: "uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps"
    expect: "focused tests pass"
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: integration
  e2e_required: true
  proof_boundary: target_host_non_production_sample_pre_main
  acceptance_surfaces:
    - "upstream-compatible unit/golden tests"
    - "tiny environment-rollout training smoke"
    - "filtered backtest policy fixture"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07d-upstream-methodology-core-port.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  current_stage: "07D"
  required_update: true
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "apps/worker/rl_trading_trainer"
  - "tests/unit/contexts/rl_trading"
  - "tests/perf_smoke/contexts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07d-upstream-methodology-core-port.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "pyproject.toml"
  - "uv.lock"
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Do not log or commit raw checkpoint tensors, datasets, secrets, provider payloads, or exchange credentials."
  - "Large runtime artifacts stay under /opt/roehub/state/rl_trading/."
---

# Task

Implement Stage `07D` upstream-compatible RL core port.

This stage changes the core methodology, but it must remain a small non-production port/smoke stage. Full trainings happen later in `07E` and `07F`.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: before any edits, read the ledger and verify Stage `07C` is `accepted` and `current_stage` allows Stage `07D`. If not true, write/update the Stage `07D` report as `blocked`, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for this offline ML core stage. Do not use the Roehub smoke Keycloak username `smoke_e2e_keycloak` and do not read the host-local password source `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Credential redaction rule: never write secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, signed requests, raw checkpoint tensors, or credentials into prompts, docs, ledgers, traces, screenshots, logs, reports, or runtime artifacts.
- Compute and record this prompt hash.
- Implement upstream-compatible components in Roehub style:
  - CNN dueling Q-network with dropout and target network support.
  - State builder matching upstream normalized history + extras + action-history one-hot.
  - Train-only normalization stats and application.
  - Trading environment rollout with action masking, no pyramiding, last-step forced close, realized-PnL reward and flat-hold penalty.
  - D3QN/PER agent with epsilon-greedy exploration, train_start, target sync, gradient clipping, batch size/config fields and deterministic seeds.
  - Best/final checkpoint policy interfaces.
  - Q-value cache and filtered backtest policy: advantage thresholds and MC-dropout ensemble uncertainty.
- Preserve Stage `02C` action/reward/money-boundary semantics.
- Keep historical MLP/scripted-transition functions only as smoke/debug helpers; they must not be the candidate-training default.
- Add focused golden tests using tiny synthetic fixtures and, where available, tiny HF-like session fixtures.
- Run a bounded Mac Studio smoke only for core capability/resource evidence, not full training.
- Write Stage `07D` report and update ledger. `07E` may start only if the port has complete parity fixtures and no candidate-quality claim.

## Acceptance Criteria

- Code exposes an upstream-compatible architecture id such as `roehub_d3qn_cnn_dueling_v1`.
- Training rollout tests prove actions are selected by epsilon-greedy agent interaction, not a prebuilt scripted transition sequence.
- Normalization tests prove train-only stats and no validation/test leakage.
- Backtest policy tests prove weak actions can be rejected to hold and rejection counts are observable.
- Stage report lists any deliberate deviations from upstream `alpha.py` and why they are safe.

## Final Output

Respond in Russian with result/status, file manifest, parity evidence, quality gates, contract impact, residual risks and `07E` handoff.
