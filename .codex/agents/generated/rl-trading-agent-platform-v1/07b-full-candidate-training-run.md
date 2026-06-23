---
prompt_name: 07b-full-candidate-training-run
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Run a full RL candidate training job on the accepted Stage 06 binance:futures dataset and produce a candidate artifact for Stage 08 evaluation."
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
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07a-training-runner-smoke.md
      why: "accepted trainer/smoke report"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md
      why: "accepted Stage 06 dataset report and manifest"
    - path: src/trading/contexts/rl_trading
      why: "trainer/model/dataset code from Stage 07A"
    - path: scripts/rl_trading
      why: "trainer CLI/job entrypoints"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md
      why: "stage report shape"
  conditional_bundles:
    performance:
      read_when: "benchmarking CPU/MPS/RSS/training time or deciding resume/background behavior"
      paths:
        - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/03-mac-studio-ml-environment.md
    docs_to_update:
      read_when: "stage changes shared docs, contracts, runbooks, or operator workflows"
      paths:
        - docs/architecture/README.md
        - docs/runbooks/mac-studio-native-backend-operations.md
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
    use_when: "measuring full training runtime, CPU/MPS/RSS, throughput, or resource contention"
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
  - "macstudio for full candidate training"
required_literals:
  - "rl-trading-agent-platform-v1"
  - "/opt/roehub/state/rl_trading/"
  - "07B"
non_goals:
  - "Do not register, promote, activate, paper trade, testnet trade, live trade, or mainnet submit."
  - "Do not run Stage 08 evaluation inside Stage 07B."
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
  proof_boundary: target_host_non_production_training_pre_main
  acceptance_surfaces:
    - "completed full candidate training run"
    - "candidate checkpoint/report hashes"
    - "CPU/MPS/RSS/resource evidence"
    - "resume/failure behavior evidence"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07b-full-candidate-training-run.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "07B"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "apps/worker/rl_trading_trainer"
  - "tests/unit/contexts/rl_trading"
  - "tests/perf_smoke/contexts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07b-full-candidate-training-run.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
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

Implement Stage `07B` full candidate training run.

Use the accepted Stage `07A` trainer and accepted Stage `06` sessionized `binance:futures` dataset to run a full candidate training job. This stage must produce a concrete candidate artifact/report for Stage `08` evaluation. It still must not register, promote, activate, or trade the model.

Done means:

- full candidate training completes on the accepted Stage `06` dataset, not a tiny smoke subset;
- candidate checkpoint/report/config artifacts are written under `/opt/roehub/state/rl_trading/`;
- durable progress artifacts exist and show step-based progress, percentage, elapsed time, ETA, and run status;
- hashes and resource evidence are recorded;
- Stage `08` receives one explicit candidate manifest path/hash;
- if full training cannot finish, Stage `07B` remains `in_progress` with resume instructions, latest progress event/status, and Stage `08` remains blocked.

## Context / Current State

- Stage `07A` owns trainer mechanics and small smoke. Do not repeat `07A` as the full-training claim.
- Stage `06` owns accepted sessionized train/val/test/backtest artifacts.
- Stage `08` evaluates a concrete candidate from `07B`; it must not rediscover hidden training state or use a smoke checkpoint.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`. If that statement is not true after reading the ledger, stop and record the blocker.
- Verify prerequisites before implementation. Required accepted prerequisites: Stage `07A`, Stage `06`. If any required prerequisite is not accepted, write/update the Stage `07B` report as blocked, update the ledger, and do not implement dependent work.
- Compute this prompt hash with `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/07b-full-candidate-training-run.md` and record path/hash in the stage report.
- Before editing, narrow expected directories to a concrete file list and record it in the stage report.
- Use the accepted Stage `06` `binance:futures` sessionized dataset manifest exactly. Do not train on spot, Bybit, six-symbol fallback, old 215-symbol subset, or smoke fixtures.
- Freeze and hash the training config, dataset manifest, model architecture, seed, code version, device policy, and resource limits.
- Define a frozen training plan before launch with `planned_training_steps`, `progress_emit_every_steps`, `progress_emit_every_sec`, checkpoint cadence, validation cadence, and selected device policy. Do not report progress as raw rows/slabs read; progress is step-based: `progress_pct = completed_training_steps / planned_training_steps * 100`.
- Write durable progress under the run directory, including at minimum `progress.jsonl` and a latest status snapshot. Every progress event must include `run_id`, `stage`, `status`, `completed_training_steps`, `planned_training_steps`, `progress_pct`, `elapsed_sec`, `eta_sec`, `device`, resource snapshot or resource summary reference, and timestamp. Status values must cover `starting`, `running`, `completed`, `failed`, and `interrupted`.
- Emit progress at least every configured step interval or configured time interval, whichever comes first. Recommended defaults for the full run are every `10_000` training steps or every `300` seconds unless Stage `07A` evidence or Mac Studio constraints justify a different value.
- Run full candidate training on Mac Studio. If it is too long for the interactive session, launch only through a managed resumable/background path with durable manifest/logs; prove the job state, record `in_progress`, and stop instead of keeping the agent session open just to watch training. Do not mark `07B` accepted until the full candidate run completes.
- If launching background/resumable training, prove not only that the process started but also that the durable progress file exists and has at least one fresh `running` progress event from the current run. Record where to inspect progress and the exact resume/status command in the stage report and ledger.
- On completion, write candidate checkpoint/report/manifest under `/opt/roehub/state/rl_trading/` and record sha256 hashes. Do not commit checkpoints or tensor dumps.
- Record wall-clock, RSS, CPU/MPS usage, throughput, train/validation curves, selected checkpoint policy, and rejected/failed-run behavior.
- Do not register or promote the model. Stage `09` owns registry/checkpoint security; Stage `08` owns evaluation.
- Update/create `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07b-full-candidate-training-run.md`.
- Update the stage ledger after validation and before final response. Stage `08` is allowed only if `07B` has a completed candidate manifest.
- The ledger update must record `current_stage`, prerequisite status, prompt path/hash, final file manifest, delivery state, validation evidence, blockers/residual risks, candidate manifest path/hash, and whether Stage `08` is allowed.

## Requirements (Should)

- Prefer resumable and hash-addressed run directories.
- Keep resource usage bounded so Mac Studio remains usable for other Roehub jobs.
- Keep evidence compact: hashes, counts, curves summaries, resource metrics, and sanitized excerpts.

# Context Acquisition Protocol

Read `.codex/AGENTS.md`, the RL plan, ledger, Stage `07A` report, Stage `06` report, and trainer entrypoints. Expand only for blockers, failing checks, resource issues, or ambiguous run/candidate contracts.

# Work Plan

1. Verify ledger current stage and prerequisites.
2. Record `User required before start`, prompt path/hash, planned file list, and blockers in the Stage `07B` report.
3. Resolve the exact Stage `06` manifest path/hash and accepted train/validation split artifacts.
4. Freeze the full candidate training config and create a deterministic run directory under `/opt/roehub/state/rl_trading/`.
5. Freeze the progress contract for the run: planned steps, emission cadence, `progress.jsonl` path, latest status snapshot path, status/resume command, and ETA calculation rule.
6. Run full candidate training on Mac Studio or start a managed resumable job if the run cannot complete within the session.
7. If the run is incomplete, prove a fresh `running` progress event, then update report/ledger as `in_progress` with resume command, job state, progress path, latest progress event, logs, and blockers; do not start Stage `08`.
8. If the run completes, validate checkpoint/report hashes, final `completed` progress event at `100%`, collect resource evidence, run focused gates, update report/ledger, and hand Stage `08` the candidate manifest path/hash.

# Acceptance Criteria

- A completed candidate training manifest exists under `/opt/roehub/state/rl_trading/` and references the accepted Stage `06` dataset hash.
- Candidate artifacts include checkpoint hash, config hash, dataset hash, model architecture hash, seed, code/delivery state, training curves, validation metrics, resource metrics, `progress.jsonl`, and latest status snapshot.
- Progress evidence is step-based and durable: final accepted run has a `completed` event with `completed_training_steps == planned_training_steps` and `progress_pct == 100.0`; incomplete/background runs have a fresh `running` event, ETA, status/resume command, and remain `in_progress`.
- Resume/failure behavior is deterministic and documented.
- Stage report includes prompt path/hash, file manifest, evidence, contract impact, and next-stage handoff.
- Ledger `current_stage` advances to `08` only after completed `07B` candidate evidence. Start-only or partial training evidence leaves Stage `07B` `in_progress`.

# Implementation Constraints

- Determinism: seed all random sources, record nondeterministic backend limitations, and hash canonical JSON.
- Contracts: no public API, browser, exchange, paper/testnet/live, registry activation, or promotion behavior changes.
- Artifacts: large checkpoints/logs/tensors stay under `/opt/roehub/state/rl_trading/`.
- Git: default branch is `main`; do not create branches/worktrees/stashes unless explicitly requested by the user.
- Validation: tests are local gates; completed Mac Studio full training evidence is required for acceptance.

# Files To Indicate

Primary touches:

- `src/trading/contexts/rl_trading`
- `scripts/rl_trading`
- `apps/worker/rl_trading_trainer`
- `tests/unit/contexts/rl_trading`
- `tests/perf_smoke/contexts/rl_trading`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/07b-full-candidate-training-run.md`
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
