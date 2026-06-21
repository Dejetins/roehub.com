---
prompt_name: 04c-dataset-refresh-manifest
repo: roehub.com
branch: main
scope: "Freeze Binance Futures dataset refresh versions and manifests before raw feature slab construction."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and dataset refresh policy"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/02a-data-source-inventory.md
      why: "HF windows, channel order and source bounds"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04a-binance-futures-universe-whitelist.md
      why: "accepted universe and exclusions"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04b-binance-futures-history-backfill.md
      why: "coverage and residual gap evidence"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "optional compact state; ignore if unrelated"
  task_entrypoints:
    - path: src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py
      why: "canonical source metadata used by future feature slabs"
      inspect_symbols:
        - "canonical candles"
    - path: src/trading/contexts/backtest/adapters/outbound/artifacts_fs/artifact_array_loader.py
      why: "existing artifact array shape boundary"
      inspect_symbols:
        - "ohlcv shape"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md
      why: "stage report evidence style"
      inspect_symbols:
        - "File Manifest"
        - "Contract Impact"
        - "Next-Stage Handoff"
  conditional_bundles:
    dataset_artifact_design:
      read_when: "writing manifest schema or a small manifest helper"
      paths:
        - docs/runbooks/backtest-artifacts-rebuild.md
        - docs/architecture/market_data/market-data-rest-historical-catchup-1m-v2.md
    docs_to_update:
      read_when: "stage changes shared docs, contracts, runbooks, or operator workflows"
      paths:
        - docs/architecture/README.md
        - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md
  consult_if_needed:
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
  allow_market_data_writes: false
  allow_browser_runtime_verification: false
  allow_tests_only_acceptance: false
skill_routing:
  - skill: github:yeet
    use_when: "publishing accepted stage changes to GitHub through a scoped branch and draft PR"
    timing: "before ship"
    reason: "owns scoped staging, branch push, draft PR creation, and branch hygiene"
  - skill: data-analytics-methodology
    use_when: "freezing dataset-version methodology and residual-gap decisions"
    timing: "during investigation"
    reason: "keeps dataset lineage and coverage assumptions explicit"
  - skill: backend-quality-gates
    use_when: "backend Python code, tests, lint, or type gates are changed"
    timing: "during verification"
    reason: "owns focused ruff, pyright, pytest gate triage"
  - skill: contract-impact-analysis
    use_when: "dataset manifest schema, config, or persisted metadata contracts change"
    timing: "during implementation"
    reason: "classifies compatibility and rollout impact"
target_envs:
  - "local checkout"
  - "macstudio for runtime manifest storage and ClickHouse evidence"
required_literals:
  - "hf_period_rebuild_current_trading"
  - "post_hf_extension_current_trading"
  - "binance:futures"
  - "/opt/roehub/state/rl_trading/"
non_goals:
  - "Do not backfill candles."
  - "Do not build raw feature slabs or sessionized NPZ arrays."
  - "Do not overwrite the external HF baseline."
  - "Do not train models."
final_report_format:
  language: ru
  sections:
    - "Result and stage status"
    - "File manifest"
    - "Evidence and quality gates"
    - "Contract, safety, and delivery state"
    - "Blockers and next-stage handoff"
quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/rl_trading tests/unit/contexts/market_data"
    expect: "passes if manifest helper code is changed; narrow when directories do not exist"
  - cmd: "uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading"
    expect: "passes for changed backend/test paths"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: integration
  e2e_required: true
  acceptance_surfaces:
    - "Stage 04B coverage manifest"
    - "runtime dataset refresh manifests under /opt/roehub/state/rl_trading/"
    - "manifest hash/lineage validation"
  tests_only_allowed_reason: ""
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "04C"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - "scripts/rl_trading"
  - "src/trading/contexts/rl_trading"
  - "tests/unit/contexts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04c-dataset-refresh-manifest.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/runbooks/backtest-artifacts-rebuild.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Never write secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, raw candle dumps, or raw checkpoint tensors into prompts, docs, ledgers, traces, screenshots, or logs."
  - "Runtime dataset refresh manifests may live under /opt/roehub/state/rl_trading/; docs contain sanitized summaries and hashes only."
  - "Mac Studio git commands must use /Users/daniildegtyarev/Projects/roehub.com; /opt/roehub/app is runtime state only."
  - "Browser/auth is N/A for this manifest stage: do not use the smoke_e2e_keycloak account and do not read ROEHUB_SMOKE_E2E_PASSWORD from /Users/daniildegtyarev/.config/roehub/roehub.env unless a later browser-visible task explicitly requires it."
---

# Task

Implement Stage 04C dataset refresh manifest. Freeze the exact Binance Futures dataset refresh versions that Stage 05 will consume, based on the corrected full-current Binance `TRADING` `USDT` `PERPETUAL` universe and accepted Stage 04B coverage.

Done means:

- `hf_period_rebuild_current_trading` is defined for HF-compatible train/val/test/backtest signal/source windows over the full current Binance USDT perpetual universe; HF membership is not a symbol filter.
- `post_hf_extension_current_trading` is defined for post-HF data after `2025-06-01`, using the latest accepted coverage endpoint.
- Each dataset version records symbols, source windows, excluded symbols, residual gaps, coverage thresholds, feature contract dependency, source hashes/query hashes, and acceptance status.
- Stage 05 receives a single accepted refresh manifest path/hash and does not rediscover universe/backfill scope.

## Context / Current State

- Stage 04A historically onboarded a 215-symbol HF-intersection subset. Stage 04B owns the 2026-06-21 repair to all current Binance USDT perpetual symbols and must provide the accepted full-universe coverage manifest.
- Stage 04B backfills/repairs source candle windows and records coverage for the corrected full current USDT perpetual universe.
- Stage 05 builds raw feature slabs; it must not own universe resolution or historical data loading.
- External HF baseline is immutable reference input and must not be overwritten by Roehub refresh versions.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`. If that statement is not true after reading the ledger, stop and record the blocker instead of guessing.
- Verify prerequisites before implementation. Required accepted prerequisites: Stage 04B. If Stage 04B is not accepted, write/update the Stage 04C report as blocked, update the ledger, and do not implement dependent work.
- Treat Stage 04B `in_progress` start-proof evidence as insufficient for this stage. If Stage 04B only proved that backfill started and data began entering ClickHouse, stop and record Stage 04C as blocked/pending on the follow-up full coverage check; do not build refresh manifests from partial in-flight data.
- Compute this prompt hash with `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/04c-dataset-refresh-manifest.md` and record path/hash in the stage report.
- Before editing, narrow expected paths to a concrete file list and record it in the stage report.
- Keep the change bounded to Stage 04C. Do not backfill candles, build feature slabs, sessionize windows, train models, or change live execution behavior.
- Read Stage 04A accepted/excluded universe only as partial historical evidence. Read Stage 04B full-current-USDT universe, supplement, coverage and residual gaps as the source of truth; do not recompute a different universe inside Stage 04C.
- Create deterministic refresh manifest(s) under `/opt/roehub/state/rl_trading/` or another accepted runtime path. Commit only schema/helper code and sanitized docs/hashes.
- Define at minimum two dataset versions when coverage permits: `hf_period_rebuild_current_trading` and `post_hf_extension_current_trading`.
- If coverage is insufficient, mark the affected dataset version `blocked` or `partial_rejected`; do not let Stage 05 consume it silently.
- Record exact source windows, split windows, symbol list hash, excluded-symbol hash, coverage report hash, feature contract dependency and intended Stage 05 input path/hash.
- Update/create `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04c-dataset-refresh-manifest.md` and update the ledger before final response.

## Requirements (Should)

- Prefer a simple JSON manifest schema with stable ordering and version field.
- Keep runtime artifact paths relocatable under `/opt/roehub/state/rl_trading/`.
- Make the accepted/blocked status machine-compatible for Stage 05.

# Context acquisition protocol

Read `.codex/AGENTS.md`, RL plan, ledger, Stage 02A, Stage 04A and Stage 04B reports, then task entrypoints. Do not read execution/live docs unless a blocker unexpectedly crosses execution boundaries.

# Work plan (agent should follow)

Skill routing:

- `github:yeet`: use before ship when publishing accepted stage changes through a scoped branch and draft PR; do not push directly to `main`.
- `data-analytics-methodology`: use for dataset version methodology and coverage/residual-gap interpretation.
- `backend-quality-gates`: use for focused tests/lint/type gates when code changes.
- `contract-impact-analysis`: use for manifest schema or metadata contract changes.

1. Verify Stage 04B acceptance and load its full-current-USDT coverage manifest; `in_progress` start-only evidence or 215-symbol-only coverage is not accepted coverage.
2. Define deterministic manifest schema and exact runtime artifact paths.
3. Create refresh manifests for HF-period rebuild and post-HF extension, or mark versions blocked with explicit evidence.
4. Validate hashes, schema, symbol ordering, source-window coverage and residual-gap semantics.
5. Update stage report and ledger with exact Stage 05 handoff: accepted manifest path/hash or blocker.

# Acceptance criteria (Definition of Done)

- Stage 04C report records prompt path/hash, file manifest, manifest schema, dataset version statuses, runtime manifest paths/hashes, source windows, universe/exclusion hashes, coverage dependencies, and delivery state.
- Stage 05 has one explicit accepted input manifest over the corrected full current USDT perpetual universe or is blocked; no implicit six-symbol fallback or 215-symbol HF-intersection fallback is allowed.
- External HF baseline remains immutable and separate from Roehub-native refresh versions.
- Stage ledger is updated after validation and before final response.
- Delivery state is explicit: `local-only`, `published-to-branch/draft-pr`, `delivered-to-main`, and/or `deployed-on-macstudio`, with evidence appropriate to the stage.

# Implementation constraints

## Determinism & ordering

- Sort all symbols, exclusions and windows deterministically.
- Use UTC ISO timestamps and half-open windows `[start, end)`.
- Hash canonical JSON with stable key ordering.

## API / contracts

- Manifest schema is a dataset-builder contract; version it and classify compatibility.
- Do not change public API, exchange execution, model registry, or live trading behavior in this stage.

## GitHub delivery and branch hygiene

- Default delivery during stage work is `local-only` until an explicit publish step is required.
- Do not push directly to `main` and do not run `git push origin main` from this prompt.
- If publishing is required, use the `github:yeet` workflow: inspect `git status -sb` and the diff, stage only intended files, commit tersely, push a branch with tracking, and open a draft PR.
- Branch creation rule: create one `codex/<stage-scope>` branch only when currently on `main`, `master`, or another default branch. If already on a suitable task branch, stay on it and do not create another branch.
- Do not use `git add -A` in a mixed worktree unless the user explicitly confirms that the whole worktree belongs to this stage.
- A stage may report `published-to-branch/draft-pr` while the PR is open. It may report `delivered-to-main` only after the PR has been merged and evidence proves the delivered SHA is on `origin/main`.
- After successful PR merge and required tests/deploy checks, delete only the temporary `codex/*` branch created for that stage from local and remote. Never delete user-owned or pre-existing branches.
- `publish-ci-deploy` may be used for CI/deploy/Mac Studio verification after GitHub publication or merge, but it does not replace `github:yeet` for branch/PR publishing.
- If a user explicitly requests a direct main push, stop and record the request as a delivery-policy deviation before doing anything; do not infer direct-main push permission from `delivered-to-main` wording.
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
- Validate through the nearest changed boundary: Binance public metadata, whitelist/ref sync, ClickHouse market-data coverage, runtime artifact manifests, API/use-case, database/migration, target runtime, benchmark/profile, CI/deploy, or production-safe smoke.
- If real-boundary evidence cannot be collected, mark the stage blocked unless this prompt explicitly allows accepted-with-residual-risk.
- Record validation evidence in the stage report and ledger.

## Runtime, Mac Studio, and artifacts

- SSH alias is `macstudio`.
- Remote git checkout is `/Users/daniildegtyarev/Projects/roehub.com`.
- Runtime tree is `/opt/roehub/app`; do not run git commands there.
- ML artifact root is `/opt/roehub/state/rl_trading/`.
- Do not commit datasets, checkpoints, tensor dumps, raw candle dumps, or large local logs.
- Runtime reports may store sanitized JSON summaries and hashes; raw arrays and full backfill logs stay under `/opt/roehub/state/rl_trading/` or another accepted runtime path.
# Files to indicate (expected touched areas)

Primary touches:

- `scripts/rl_trading`
- `src/trading/contexts/rl_trading`
- `tests/unit/contexts/rl_trading`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04c-dataset-refresh-manifest.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`

Possible secondary touches:

- `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `docs/runbooks/backtest-artifacts-rebuild.md`
- `docs/architecture/README.md`

# Non-goals

- Do not backfill candles.
- Do not build raw feature slabs or sessionized NPZ arrays.
- Do not train models.
- Do not overwrite or mutate the external HF baseline.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/rl_trading tests/unit/contexts/market_data` - if manifest helper code is changed; narrow when directories do not exist
- `uv run ruff check src/trading/contexts/rl_trading scripts/rl_trading tests/unit/contexts/rl_trading` - focused lint for changed backend/test paths
- `python -m tools.docs.generate_docs_index --check` - passes after Markdown updates

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1. **Result and stage status**
2. **File manifest**
3. **Evidence and quality gates**
4. **Contract, safety, and delivery state**
5. **Blockers and next-stage handoff**
