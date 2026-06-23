---
prompt_name: 08a-upstream-methodology-parity-audit
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Freeze the full upstream rl-trading-binance methodology parity contract after the rejected Stage 08 candidate."
language:
  implementation: docs
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract, branch policy, cold-head/proof boundaries"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and methodology parity target"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger and current stage"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08-roehub-backtest-evaluation.md
      why: "rejected historical candidate evidence"
  task_entrypoints:
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04-hf-reproducibility.md
      why: "HF dataset paths, hashes, attribution and external baseline"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md
      why: "accepted Roehub-native session dataset contract"
    - path: src/trading/contexts/rl_trading/domain/training_runner.py
      why: "current Roehub trainer implementation to compare"
    - path: src/trading/contexts/rl_trading/domain/backtest_evaluation.py
      why: "current Roehub evaluator implementation to compare"
  external_sources:
    - repo: https://github.com/YuriyKolesnikov/rl-trading-binance
      required_files:
        - README.md
        - config.py
        - configs/alpha.py
        - model.py
        - agent.py
        - replay_buffer.py
        - trading_environment.py
        - train.py
        - test_agent.py
        - backtest_engine.py
        - optimize_cfg.py
        - baseline_cnn_classifier.py
      rule: "Read the exact source files from a pinned commit or recorded clone SHA. Do not vendor source code in this stage."
    - url: https://habr.com/ru/articles/934258/
      rule: "Use only to cross-check methodology claims against the source repo; repo code is stronger evidence when they differ."
hard_requirements:
  update_stage_ledger: true
  stage_report_file_manifest: true
  prompt_path_and_sha256_in_report: true
  no_secret_or_raw_provider_payload_in_artifacts: true
  explicit_delivery_state: true
task_toggles:
  allow_mainnet_submit: false
  allow_exchange_side_effects: false
  allow_browser_runtime_verification: false
  allow_tests_only_acceptance: false
skill_routing:
  - skill: architecture-review
    use_when: "reviewing the plan/prompt/code-methodology gap"
    timing: "entire stage"
    reason: "owns evidence discipline and gap classification"
  - skill: prompt-manager
    use_when: "updating downstream prompt files from the parity audit"
    timing: "after the parity matrix is frozen"
    reason: "owns executable prompt readiness"
  - skill: backend-quality-gates
    use_when: "only if helper scripts/tests are changed"
    timing: "verification"
    reason: "owns Python gate triage"
target_envs:
  - "local checkout"
  - "read-only external source clone or GitHub source read"
required_literals:
  - "rl-trading-agent-platform-v1"
  - "08A"
  - "methodology_parity_not_yet_ported"
non_goals:
  - "Do not implement the upstream core in this stage."
  - "Do not train or evaluate a new model in this stage."
  - "Do not register, promote, activate, paper trade, testnet trade, live trade, or mainnet submit."
  - "Do not create branches, worktrees, stashes, temporary repo checkouts, or auxiliary workflow files."
quality_gates:
  - cmd: "uv run python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates; regenerate first if needed"
validation_strategy:
  depth: architecture_review
  e2e_required: true
  acceptance_surfaces:
    - "source-to-Roehub parity matrix"
    - "explicit downstream stage/prompt checklist"
    - "ledger current_stage handoff"
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-upstream-methodology-parity-audit.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "08A"
  required_update: true
expected_primary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-upstream-methodology-parity-audit.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
  - ".codex/agents/generated/rl-trading-agent-platform-v1/08b-upstream-methodology-core-port.md"
  - ".codex/agents/generated/rl-trading-agent-platform-v1/08c-original-hf-full-training-run.md"
  - ".codex/agents/generated/rl-trading-agent-platform-v1/08d-original-hf-backtest-evaluation.md"
  - ".codex/agents/generated/rl-trading-agent-platform-v1/08e-roehub-native-full-training-run.md"
  - ".codex/agents/generated/rl-trading-agent-platform-v1/08f-roehub-native-backtest-evaluation.md"
  - ".codex/agents/generated/rl-trading-agent-platform-v1/09-model-registry-activation.md"
  - ".codex/agents/generated/rl-trading-agent-platform-v1/10-per-ticker-calibration.md"
  - ".codex/agents/generated/rl-trading-agent-platform-v1/10a-retraining-promotion-lifecycle.md"
possible_secondary_touches:
  - ".codex/agents/generated/rl-trading-agent-platform-v1/07b-full-candidate-training-run.md"
  - ".codex/agents/generated/rl-trading-agent-platform-v1/08-roehub-backtest-evaluation.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Runtime ML artifacts live under /opt/roehub/state/rl_trading/ and are not committed to git."
  - "External source code may be summarized and mapped, but not vendored in this audit stage."
  - "Keep source citations to file names/functions/commit SHA; do not paste large source bodies into docs."
---

# Task

Implement Stage `08A` upstream methodology parity audit.

The Stage `08` result proved that the historical Stage `07B` candidate is not good enough and is not a full methodology port. This stage must freeze exactly what "fully port the original methodology" means before implementation resumes.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`.
- Previous-stage ledger gate: before any edits, read the ledger and verify Stage `08` is `blocked` with rejected-candidate evidence and `current_stage` allows corrective Stage `08A`. If not true, write/update the Stage `08A` report as `blocked`, update the ledger, and stop.
- Browser/auth anchor: browser QA and authenticated Roehub UI are N/A for this docs/review stage. Do not use the Roehub smoke Keycloak username `smoke_e2e_keycloak` and do not read the host-local password source `/Users/daniildegtyarev/.config/roehub/roehub.env` key `ROEHUB_SMOKE_E2E_PASSWORD`; if a browser/auth surface unexpectedly appears, stop and record a scope blocker.
- Credential redaction rule: never write secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, signed requests, raw checkpoint tensors, or credentials into prompts, docs, ledgers, traces, screenshots, logs, reports, or runtime artifacts.
- Compute and record this prompt hash: `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/08a-upstream-methodology-parity-audit.md`.
- File manifest gate: every created, modified, deleted, and runtime artifact path must be listed in the Stage `08A` report and ledger update; any prompt file outside the exact primary/secondary touch lists above requires explicit outside-manifest justification and must not be changed speculatively.
- Read the upstream repo files listed in front matter from a pinned commit or record the observed clone/source SHA. Repo source is stronger than README/Habr when there is conflict.
- Produce `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/08a-upstream-methodology-parity-audit.md`.
- Update the plan and ledger only where they need to reflect the frozen parity contract and next-stage handoff.
- Update downstream prompts if the parity audit finds missing instructions.
- Do not implement code, run training, or evaluate new candidates in `08A`.

## Required Methodology Matrix

The report must map each upstream component to Roehub implementation requirements:

- `config.py` and `configs/alpha.py`: feature order, sequence lengths, action history, fees/slippage, episodes, validation cadence, PER, epsilon, model and backtest thresholds.
- `utils.py`: dataset loading, key structure, channel arrangement, train-only normalization, grouped signal creation.
- `trading_environment.py`: state extras, action semantics, last-step action masking, reward, backtest step and optional risk management.
- `model.py`: CNN encoder and dueling value/advantage heads with dropout.
- `agent.py` and `replay_buffer.py`: D3QN, PER, epsilon-greedy, train_start, target sync, gradient clipping, MC dropout and q-value cache.
- `train.py`: environment rollout training, validation-selected `best.pth`, `final.pth`, progress metrics and test evaluation.
- `test_agent.py`: test-set evaluation and session/action visualization contract.
- `backtest_engine.py`: signal grouping, max parallel sessions, position fraction, Q-value filters, action rejection, metrics and balance curve.
- `optimize_cfg.py`: threshold/config tuning scope and non-goals for the current stage chain.
- `baseline_cnn_classifier.py`: optional sanity baseline only, not a business benchmark.

## Acceptance Criteria

- Stage `08A` report exists with prompt hash, upstream source SHA/URL, parity matrix, gap list, accepted deviations, and downstream checklist.
- The report explicitly states that MLP-D3QN, offline scripted transitions and raw argmax-only evaluation are not accepted for candidate quality.
- Ledger current stage advances to `08B` only if the parity matrix is complete and downstream prompts have exact implementation requirements.
- Stage `09` remains blocked until `08F` is accepted.

## Final Output

Respond in Russian with:

1. **Result and stage status**
2. **File manifest**
3. **Methodology parity findings**
4. **Prompt/plan/ledger updates**
5. **Quality gates**
6. **Next-stage handoff**
