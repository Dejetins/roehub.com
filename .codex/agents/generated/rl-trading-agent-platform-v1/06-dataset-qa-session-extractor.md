---
prompt_name: 06-dataset-qa-session-extractor
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Implement data QA and high-volatility session extraction with leakage, overlap, embargo, and lifecycle controls."
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
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04c-dataset-refresh-manifest.md
      why: "accepted Stage 04C refresh manifest and source universe"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/05-roehub-dataset-builder-v1.md
      why: "accepted Stage 05 feature slabs, manifests, and golden parity fixtures"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "optional compact state; ignore if unrelated"
  task_entrypoints:
    - path: src/trading/contexts/backtest_artifacts/application/services/v2/artifact_manifest_loader.py
      why: "artifact manifest shape"
      inspect_symbols:
        - "manifest loading"
        - "hash validation"
    - path: src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py
      why: "lifecycle/gap evidence"
      inspect_symbols:
        - "canonical rows"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md
      why: "machine-readable report shape"
      inspect_symbols:
        - "File Manifest"
        - "Evidence"
  conditional_bundles:
    execution_boundary:
      read_when: "source events, intents, paper, testnet, live, or idempotency are touched"
      paths:
        - docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md
        - src/trading/contexts/live_execution/domain/execution_source.py
        - src/trading/contexts/live_execution/domain/risk_gate.py
        - apps/api/routes/ui_execution.py
    docs_to_update:
      read_when: "stage changes shared docs, contracts, runbooks, or operator workflows"
      paths:
        - docs/architecture/README.md
        - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md
  consult_if_needed:
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md
      read_when: "paper, testnet, live, or classic producer dependency affects this stage"
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      read_when: "checking classic producer Stage 05/07/09 prerequisites"
    - path: docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md
      read_when: "source events, intents, risk gate, paper/testnet/live, or idempotency are touched"
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
  - skill: publish-ci-deploy
    use_when: "publishing accepted stage changes through the default direct-main delivery workflow"
    timing: "before ship"
    reason: "owns scoped staging, main delivery, CI/deploy follow-up, and Mac Studio verification; branch/PR only when explicitly requested by the user"
  - skill: data-analytics-methodology
    use_when: "selecting session extraction and leakage QA rules"
    timing: "during implementation"
    reason: "owns missing-data and split methodology"
  - skill: backend-quality-gates
    use_when: "backend Python code, tests, lint, or type gates are changed"
    timing: "during verification"
    reason: "owns focused ruff, pyright, pytest gate triage"
  - skill: contract-impact-analysis
    use_when: "API, DTO, persistence, config, cache identity, or browser-visible defaults change"
    timing: "during implementation"
    reason: "classifies compatibility and rollout impact"
target_envs:
  - "local checkout"
  - "macstudio when runtime or data evidence is required"
required_literals:
  - "rl-trading-agent-platform-v1"
  - "/opt/roehub/state/rl_trading/"
  - "ml_agent_decision"
non_goals:
  - "Do not train user-owned custom models."
  - "Do not add cloud/S3/model hosting."
  - "Do not bypass live_execution or exchange-execution."
  - "Do not open mainnet execution before Stage 19 approval and Stage 20 prompt conditions."
final_report_format:
  language: ru
  sections:
    - "Result and stage status"
    - "File manifest"
    - "Evidence and quality gates"
    - "Contract, safety, and delivery state"
    - "Blockers and next-stage handoff"
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
  depth: integration
  e2e_required: true
  acceptance_surfaces:
    - "sessionized dataset manifests"
    - "leakage report"
    - "gap report"
  tests_only_allowed_reason: ""
  evidence_target: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "06"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - "src/trading/contexts/rl_trading"
  - "scripts/rl_trading"
  - "tests/unit/contexts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/runbooks/backtest-artifacts-rebuild.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Never write secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, or raw checkpoint tensors into prompts, docs, ledgers, traces, screenshots, or logs."
  - "Runtime ML artifacts live under /opt/roehub/state/rl_trading/ and are not committed to git."
  - "Mac Studio git commands must use /Users/daniildegtyarev/Projects/roehub.com; /opt/roehub/app is runtime state only."
  - "ML trainer/inference code must not call exchange SDKs or resolve exchange secrets directly."
---

# Task

Implement Stage 06 dataset QA and Binance Futures high-volatility session extractor. Emit accepted `binance:futures` sessionized train/val/test/backtest datasets with gap, lifecycle, overlap, embargo, and leakage evidence.

Done means:

- Session extraction policy is implemented and documented for `binance:futures` only.
- Accepted Binance Futures sessionized datasets have hashes, counts, split manifest, and leakage/gap reports.
- Overlap is allowed only within a split; split boundaries have time embargo and lifecycle-aware proof.

## Context / Current State

Context ledger from the previous iteration:

- completed:
  - Stage 01 plan and ledger are accepted.
  - This prompt pack is the execution handoff for the current stage.
  - No code for this stage is assumed implemented before the executor runs it.
- open_items:
  - Turn raw feature slabs into accepted trainable/evaluable datasets without leakage.
  - Stage report, ledger status, prompt path/hash, and evidence still need executor updates.
  - Delivery state must be recorded explicitly.
- contract_changes:
  - Do not silently change public, persistence, execution, or market-data contracts.
  - Any additive contract change must be documented, tested, and classified.
  - No exchange or money-moving side effect is allowed unless this stage explicitly permits it.
- touched_paths:
  - Use expected_primary_touches as the initial scope, then narrow to concrete files before editing.
  - Justify every touched path outside expected_primary_touches or possible_secondary_touches.
  - Large datasets, checkpoints, and local logs stay outside git under /opt/roehub/state/rl_trading/.
- risks:
  - Train/live feature drift, stage sequencing drift, and stale ledger state are acceptance blockers.
  - Secrets and provider payload leakage into docs/logs is forbidden.
  - Tests-only acceptance is not enough for non-trivial runtime, API, persistence, browser, performance, or exchange surfaces.
- next_focus:
  - Deliver this stage only; do not start later dependent stages.
  - Update the ledger after validation and before final response.
  - Record whether the next stage is allowed and what it must know.

Additional context:

- Stage 01 is accepted: the RL architecture plan, stage ledger, ClickHouse/data snapshot, and docs index evidence exist.
- Prompt generation snapshot: the ledger current_stage was 02A when this pack was authored; always trust the ledger value read during execution.
- Classic strategy producer Stage 05 is currently blocked on Binance Futures Testnet account funding/config (`insufficient_balance`, `margin_mode_mismatch`, `leverage_mismatch`); RL paper/testnet execution remains gated until classic Stage 05 repair and downstream classic Stage 07/09 acceptance.
- Stage 04C defines accepted refresh manifests for HF-compatible and post-HF Binance Futures datasets. Do not choose a different universe or fallback to the old six-symbol Roehub futures reference set unless Stage 04C explicitly marks that as the only accepted dataset version.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`. If that statement is not true after reading the ledger, stop and record the blocker instead of guessing.
- Verify the stage prerequisites before implementation. Required accepted prerequisites: Stage 05. If any required prerequisite is not accepted in the ledger, stop, write/update the stage report as blocked, update the ledger, and do not implement dependent work.
- Compute this prompt hash with `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/06-dataset-qa-session-extractor.md` and record the prompt path and hash in the stage report.
- Before editing, narrow broad expected directories to a concrete file list or planned new files and record that list in the stage report.
- Keep the change bounded to Stage `06`. Do not start later stages or silently repair unrelated legacy issues.
- Update or create the stage report at `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md` using the Stage 01 report shape: scope, file manifest, observed state, contract impact, quality gates, evidence, blockers, and next-stage handoff.
- Update `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md` after validation and before final response with status, evidence, blockers, touched contracts/files, delivery state, and whether the next stage is allowed.
- Include a final file manifest with `created`, `modified`, `deleted`, `outside_expected_paths`, and justification for every file outside expected paths.
- Preserve dependency direction: RL/ML code may produce decisions and source events, but exchange submission and secret custody stay in existing execution/exchange contexts.
- Keep all large runtime artifacts outside git under `/opt/roehub/state/rl_trading/`; commit only sanitized summaries, manifests, hashes, and tests.
- Do not log or document secrets, tokens, cookies, passphrases, ciphertext, raw provider payloads, raw signed requests, or model checkpoint contents.
- Define high-volatility window selection, stride/overlap, `pre_signal_len=90`, `post_signal_len=60`, Binance Futures ticker universe, and split policy.
- Use the accepted Stage 04C refresh manifest, including its source windows, symbol universe, residual-gap policy and latest post-HF endpoint. Do not recompute a different universe inside Stage 06.
- Do not include Binance spot, Bybit spot, or Bybit futures data in accepted v1 training/evaluation datasets.
- Prove no look-ahead, no survivorship leakage, and no overlap leakage across train/val/test/backtest boundaries.
- Handle listing/delisting and missing windows explicitly.
- Produce machine-readable QA reports plus sanitized summary.
- Do not train a model in this stage.

## Requirements (Should)

- Prefer additive, fail-closed contracts over replacing existing public behavior.
- Reuse existing Roehub ports, DTO style, metrics, and docs/report shapes before adding new abstractions.
- Keep evidence compact: hashes, counts, metrics, and sanitized excerpts instead of large logs.

## Requirements (Nice-to-have)

- Add small operator-facing notes when they reduce next-stage ambiguity.
- Capture reusable test fixtures for later stages when they are cheap and clearly scoped.

# Context acquisition protocol

Read only in this order and stop once sufficient:

1. `.codex/AGENTS.md`
2. `.codex/agents/.context/promt_manager_state.yaml` only if it clearly refers to `rl-trading-agent-platform-v1`; ignore unrelated snapshots and state that you ignored them
3. latest executor final report for the immediate prerequisite stage, if available
4. task entrypoints from front matter
5. only the conditional bundle(s) required by touched contracts, runtime evidence, or failing checks
6. consult-if-needed references only for blockers, ambiguity, or conflicts

Do not eagerly preload all listed sources.

Pre-implementation reading target:

- `<= 8 files`
- `<= ~35k-50k tokens`

Stop reading once all of the following are true:

- changed contracts are identified,
- touched files are bounded,
- acceptance criteria are implementable without ambiguity,
- no unresolved public API, persistence, execution, or market-data contract ambiguity remains.

Expand context only for:

- blockers,
- failing quality gates,
- unclear contracts,
- benchmark threshold conflicts,
- architecture conflicts that affect correctness.

# Reading manifest

Use the front-matter `context_sources` as the canonical reading map.

Read with this intent:

- `always_read`: repository rules, RL plan, stage ledger, and relevant compact state only when topical
- `task_entrypoints`: canonical code/doc entrypoints for this stage
- `conditional_bundles`: read only when the stated condition applies
- `consult_if_needed`: read only for blockers, ambiguity, or conflict resolution

Do not convert this manifest into a broad mandatory reading list. If a listed planned file does not exist, treat it as an expected artifact only, not as a failed read, unless it is listed under `always_read` or `task_entrypoints`.

# Work plan (agent should follow)

Skill routing for this task:

- `publish-ci-deploy`: use before ship when publishing accepted stage changes through the default direct-main delivery workflow; owns scoped staging, main delivery, CI/deploy/Mac Studio follow-up, and branch hygiene only when the user explicitly requested branch/PR delivery.

- `data-analytics-methodology`: use during implementation when selecting session extraction and leakage QA rules; owns missing-data and split methodology.
- `backend-quality-gates`: use during verification when backend Python code, tests, lint, or type gates are changed; owns focused ruff, pyright, pytest gate triage.
- `contract-impact-analysis`: use during implementation when API, DTO, persistence, config, cache identity, or browser-visible defaults change; owns classifies compatibility and rollout impact.

1. Read the required context using the bounded protocol and verify Stage `06` prerequisites in the ledger.
2. Record `User required before start`, prompt path/hash, planned concrete file list, and any initial blockers in `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md`.
3. Implement only this stage's scoped artifacts and tests. Preserve existing public contracts unless this prompt explicitly requires an additive change.
4. Run focused local gates first, then collect the nearest real-boundary evidence listed in `validation_strategy.acceptance_surfaces`.
5. Update directly affected docs/runbooks and run the docs index check if any Markdown doc changes.
6. Update the stage report and ledger with evidence, contract impact, delivery state, residual risk, and next-stage allowance.
7. Final response must use the strict Russian report format and must not claim acceptance if required evidence is missing.

# Acceptance criteria (Definition of Done)

- Binance Futures sessionized dataset manifests and hashes are derived from the accepted Stage 04C refresh manifest and exist under the local artifact store.
- Leakage/embargo/gap/lifecycle reports are machine-readable and summarized in the stage report.
- Stage 07 is allowed only for accepted `binance:futures` dataset branches.
- The stage report exists at `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md` and includes prompt path/hash plus a strict file manifest.
- The stage ledger is updated after validation and before final response.
- Contract impact is classified for public API, ports, DTOs, persistence, config/defaults, external side effects, browser-visible behavior, performance, and docs/runbooks as applicable.
- Non-trivial acceptance is backed by the selected real-boundary evidence; tests-only acceptance is not used.
- Delivery state is explicit: `local-only`, `delivered-to-main`, `deployed-on-macstudio`, and `published-to-branch/draft-pr` only when the user explicitly requested branch/PR delivery, with evidence appropriate to the stage.

# Implementation constraints

## Determinism & ordering

- Keep file ordering, dataset manifests, hashes, and generated reports deterministic.
- Preserve stable identity, idempotency, cache, and hash semantics unless this prompt explicitly changes them.
- Avoid hidden ordering changes from iteration order, import side effects, async scheduling, or non-seeded sampling.

## API / contracts

- Do not change public or persisted contracts unless explicitly allowed in this stage.
- Prefer additive DTO/schema/config changes with fail-closed defaults.
- For exchange or money-moving paths, never retry blindly after unknown provider or durable-write state; reconcile by idempotency key, durable ledger, or provider state first.

## GitHub delivery and branch hygiene

- Default delivery during stage work is `local-only` until an explicit publish step is required.
- Default publish target is `main`. Do not create a branch, draft PR, worktree, temporary checkout, local folder, stash, or auxiliary workflow artifact unless the user explicitly requests that exact workflow.
- If publishing is required, use `publish-ci-deploy` direct-main discipline: inspect `git status -sb` and the diff, stage only intended files, run required gates, commit on `main`, push `origin main`, and follow CI/deploy/Mac Studio verification when required by the stage.
- Do not use `git add -A` in a mixed working tree unless the user explicitly confirms that the whole working tree belongs to this stage.
- A stage may report `delivered-to-main` only after evidence proves the delivered SHA is on `origin/main`; local-only work remains `local-only`.
- If the user explicitly requests branch/PR delivery, use at most one branch for the entire prompt pack, never per-stage branches, and record the explicit request plus branch evidence in the report and ledger.

## Documentation

- Update only directly relevant docs.
- Keep the stage report and ledger aligned with delivered facts.
- If this stage creates or changes a runbook, use existing runbook style and keep secrets out of examples.
- Run `python -m tools.docs.generate_docs_index --check` when Markdown docs change; regenerate first if the check reports drift.

## Tests

- Add or update deterministic focused tests for changed behavior.
- If config, DTO, API, persistence, or feature contracts change, include compatibility and fail-closed tests.
- If a test directory does not exist yet, create the narrowest matching test path under the established `tests/unit/...` pattern.

## Validation depth

- Treat lint/type/unit tests as local gates, not stage acceptance for non-trivial work.
- Validate through the nearest changed boundary: ClickHouse/artifact manifest, API/use-case, database/migration, browser runtime, target runtime, external adapter, benchmark/profile, CI/deploy, or production-safe smoke.
- If real-boundary evidence cannot be collected, mark the stage blocked unless this prompt explicitly allows accepted-with-residual-risk.
- Record validation evidence in the stage report and ledger.

## Runtime, Mac Studio, and artifacts

- SSH alias is `macstudio`.
- Remote git checkout is `/Users/daniildegtyarev/Projects/roehub.com`.
- Runtime tree is `/opt/roehub/app`; do not run git commands there.
- ML artifact root is `/opt/roehub/state/rl_trading/`.
- Do not commit datasets, checkpoints, tensor dumps, or large local logs.

# Files to indicate (expected touched areas)

Primary touches:

- `src/trading/contexts/rl_trading`
- `scripts/rl_trading`
- `tests/unit/contexts/rl_trading`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/06-dataset-qa-session-extractor.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`

Possible secondary touches:

- `docs/runbooks/backtest-artifacts-rebuild.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `docs/architecture/README.md`

The final report must list `created`, `modified`, `deleted`, and `outside_expected_paths`. Any outside path needs a concrete justification and contract-impact note.

# Non-goals

- Do not train user-owned custom models.
- Do not add cloud/S3/model hosting.
- Do not bypass live_execution or exchange-execution.
- Do not open mainnet execution before Stage 19 approval and Stage 20 prompt conditions.

# Quality gates (must run and pass)

- `uv run ruff check src/trading/contexts/rl_trading apps tests` - passes for changed backend/app/test paths
- `uv run pyright src/trading/contexts/rl_trading apps tests` - passes for changed backend/app/test paths
- `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/apps` - focused tests pass; narrow target if directories do not exist yet
- `python -m tools.docs.generate_docs_index --check` - passes after Markdown updates

If a listed command needs a narrower path because the stage creates a new module, run the nearest focused equivalent and explain the substitution. Do not broaden to expensive unrelated gates unless focused gates pass or the failure requires wider context.

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1. **Result and stage status**

2. **File manifest**

3. **Evidence and quality gates**

4. **Contract, safety, and delivery state**

5. **Blockers and next-stage handoff**
