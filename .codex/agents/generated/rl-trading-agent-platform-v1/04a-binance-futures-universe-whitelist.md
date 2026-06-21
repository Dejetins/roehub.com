---
prompt_name: 04a-binance-futures-universe-whitelist
repo: roehub.com
branch: main
scope: "Resolve the current Binance Futures training universe and onboard it through whitelist/ref/enrichment before historical backfill."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and refresh policy"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/02a-data-source-inventory.md
      why: "HF NPZ symbols, Binance Futures source windows, current Roehub coverage"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "optional compact state; ignore if unrelated"
  task_entrypoints:
    - path: configs/prod/whitelist.csv
      why: "production market-data whitelist to update for binance:futures only"
      inspect_symbols:
        - "market_id,symbol,is_enabled"
    - path: src/trading/contexts/market_data/application/use_cases/sync_whitelist_to_ref_instruments.py
      why: "idempotent whitelist -> ref_instruments sync"
      inspect_symbols:
        - "SyncWhitelistToRefInstrumentsUseCase"
    - path: src/trading/contexts/market_data/application/use_cases/enrich_ref_instruments_from_exchange.py
      why: "exchange metadata enrichment for filters/steps/min notional"
      inspect_symbols:
        - "EnrichRefInstrumentsFromExchangeUseCase"
    - path: src/trading/contexts/market_data/adapters/outbound/clients/rest_instrument_metadata_source.py
      why: "Binance Futures exchangeInfo metadata source"
      inspect_symbols:
        - "list_for_market"
        - "exchangeInfo"
    - path: docs/architecture/market_data/market-data-reference-data-sync-v2.md
      why: "reference-data sync contract"
      inspect_symbols:
        - "whitelist"
        - "ref_instruments"
  conditional_bundles:
    market_data_runtime:
      read_when: "running Mac Studio whitelist/ref/enrichment evidence"
      paths:
        - configs/prod/market_data.yaml
        - docs/architecture/market_data/market-data-runtime-config-invariants-v2.md
        - docs/architecture/market_data/market-data-reference-api-v1.md
    docs_to_update:
      read_when: "stage changes shared docs, contracts, runbooks, or operator workflows"
      paths:
        - docs/architecture/README.md
        - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md
  consult_if_needed:
    - path: docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md
      read_when: "checking classic producer status for ledger context only"
    - path: docs/architecture/market_data/market-data-rest-historical-catchup-1m-v2.md
      read_when: "universe onboarding exposes REST metadata/backfill ambiguity"
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
  allow_market_data_writes: true
  allow_browser_runtime_verification: false
  allow_tests_only_acceptance: false
skill_routing:
  - skill: github:yeet
    use_when: "publishing accepted stage changes to GitHub through a scoped branch and draft PR"
    timing: "before ship"
    reason: "owns scoped staging, branch push, draft PR creation, and branch hygiene"
  - skill: data-analytics-methodology
    use_when: "building the current-trading universe and exclusion methodology"
    timing: "during investigation"
    reason: "keeps sample-universe and missing-data decisions explicit"
  - skill: backend-quality-gates
    use_when: "backend Python code, tests, lint, or type gates are changed"
    timing: "during verification"
    reason: "owns focused ruff, pyright, pytest gate triage"
  - skill: contract-impact-analysis
    use_when: "config, ref-data, persisted schema, or runtime defaults change"
    timing: "during implementation"
    reason: "classifies compatibility and rollout impact"
target_envs:
  - "local checkout"
  - "macstudio for exchangeInfo/ref_instruments evidence"
required_literals:
  - "binance:futures"
  - "excluded_not_currently_trading_or_not_usdt_perpetual"
  - "/opt/roehub/state/rl_trading/"
non_goals:
  - "Do not backfill candles in this stage."
  - "Do not add Binance spot, Bybit spot, or Bybit futures to the training universe."
  - "Do not attempt synthetic mapping for delisted, renamed, non-USDT, BUSD/USDC, or quarterly contracts."
  - "Do not train models or build feature slabs."
final_report_format:
  language: ru
  sections:
    - "Result and stage status"
    - "File manifest"
    - "Evidence and quality gates"
    - "Contract, safety, and delivery state"
    - "Blockers and next-stage handoff"
quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/market_data/adapters/test_whitelist_csv_loader.py tests/unit/contexts/market_data/application/use_cases/test_enrich_ref_instruments_from_exchange.py"
    expect: "passes if whitelist/ref/enrichment code or config parsing changes"
  - cmd: "uv run ruff check configs src/trading/contexts/market_data scripts/rl_trading tests/unit/contexts/market_data"
    expect: "passes for changed backend/config/test paths; narrow if only docs/config changed"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: integration
  e2e_required: true
  acceptance_surfaces:
    - "Binance Futures exchangeInfo current-trading filter"
    - "whitelist diff"
    - "ref_instruments sync/enrichment evidence"
  tests_only_allowed_reason: ""
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "04A"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - "configs/prod/whitelist.csv"
  - "scripts/rl_trading"
  - "tests/unit/contexts/market_data"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04a-binance-futures-universe-whitelist.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "src/trading/contexts/market_data"
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/market_data"
  - "docs/architecture/README.md"
safety_notes:
  - "Never write secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, or raw signed requests into prompts, docs, ledgers, traces, screenshots, or logs."
  - "Binance public market metadata is allowed; private/account endpoints are not needed and must not be used."
  - "Mac Studio git commands must use /Users/daniildegtyarev/Projects/roehub.com; /opt/roehub/app is runtime state only."
---

# Task

Implement Stage 04A Binance Futures universe and whitelist. Resolve the full current Binance USD-M Futures `USDT` perpetual trading universe, update only the accepted `binance:futures` whitelist/ref/enrichment path, and produce a durable exclusion/onboarding manifest for Stage 04B.

Done means:

- Target universe starts from current Binance Futures `exchangeInfo`, not from HF train membership. HF counts remain reproducibility baseline evidence only.
- Only symbols with `status=TRADING`, `contractType=PERPETUAL`, and `quoteAsset=USDT` are accepted for this v1 training universe.
- Missing/delisted/renamed/non-USDT/BUSD/USDC/USD1/quarterly/dated/`TRADIFI_PERPETUAL` symbols are recorded with explicit exclusion reasons and are not added to whitelist or scheduled for backfill.
- `configs/prod/whitelist.csv`, `market_data.ref_instruments`, and exchange metadata enrichment are aligned for accepted `binance:futures` symbols.
- Stage 04B receives an exact symbol list and source-window lower-bound policy; no candle backfill is performed here.

## Context / Current State

- Stage 02A is accepted and amended after full HF NPZ inspection, but HF symbols are no longer the ceiling for Roehub-native training.
- Training-source v1 is `binance:futures` only.
- Current Roehub Binance Futures reference universe originally had only six tradable symbols, and the previous Stage 04A implementation accepted a 215-symbol HF-intersection subset. That subset is partial progress, not the final target universe.
- The last thread clarified that dataset refresh must not try to load symbols Binance no longer trades.
- `configs/prod/whitelist.csv` currently maps market ids to symbols; executor must confirm that `binance:futures` market id is still `2` from current config before editing.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`. If that statement is not true after reading the ledger, stop and record the blocker instead of guessing.
- Verify prerequisites before implementation. Required accepted prerequisites: Stage 04. If Stage 04 is not accepted, write/update the Stage 04A report as blocked, update the ledger, and do not implement dependent work.
- Compute this prompt hash with `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/04a-binance-futures-universe-whitelist.md` and record path/hash in the stage report.
- Before editing, narrow expected paths to a concrete file list and record it in the stage report.
- Keep the change bounded to Stage 04A. Do not backfill candles, build feature slabs, train models, or change non-`binance:futures` training scope.
- Build a deterministic universe resolver or report that records: current Binance metadata count, accepted full-current USDT perpetual count, excluded count by reason, accepted symbols, excluded symbols and reasons, and optional overlap with HF train/all-split symbols for analysis only.
- Confirm current Binance Futures metadata from public REST or existing metadata source; do not use private/account endpoints.
- Update only the `binance:futures` whitelist entries needed for accepted symbols, preserving unrelated whitelist rows and disabled rows.
- Run or document the operator-safe whitelist -> `ref_instruments` sync and enrichment evidence on Mac Studio. If this write is not allowed in the current environment, mark Stage 04A blocked rather than pretending the universe is onboarded.
- Record a machine-readable stage artifact summary under `/opt/roehub/state/rl_trading/` or a documented local runtime path; commit only sanitized summaries/hashes.
- Update/create `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04a-binance-futures-universe-whitelist.md` and update the ledger before final response.
- Include final file manifest with `created`, `modified`, `deleted`, `outside_expected_paths` and justification.

## Requirements (Should)

- Prefer append/update logic that keeps `configs/prod/whitelist.csv` stable and deterministic.
- Preserve existing market-data configuration conventions and avoid adding a separate RL-only whitelist format unless the existing whitelist cannot represent the contract.
- Keep evidence compact: counts, hashes, sanitized symbol lists and command summaries, not raw provider payload dumps.

# Context acquisition protocol

Read in order: `.codex/AGENTS.md`, RL plan, RL ledger, Stage 02A report, task entrypoints, then conditional market-data docs only when needed. Do not preload unrelated execution/live docs unless a blocker involves execution stages.

# Work plan (agent should follow)

Skill routing:

- `github:yeet`: use before ship when publishing accepted stage changes through a scoped branch and draft PR; do not push directly to `main`.
- `data-analytics-methodology`: use during universe/exclusion methodology.
- `backend-quality-gates`: use for focused tests/lint/type gates when backend/config parsing changes.
- `contract-impact-analysis`: use when whitelist/config/ref-data behavior changes.

1. Verify ledger prerequisites and record prompt path/hash plus planned concrete file list.
2. Read current Binance Futures metadata and filter to `status=TRADING`, `contractType=PERPETUAL`, `quoteAsset=USDT`. Do not filter by HF membership.
3. Compare accepted current symbols to current Roehub whitelist/ref state and write accepted/excluded manifests plus an optional HF-overlap summary.
4. Record the live Binance metadata snapshot/hash and accepted count; if it differs from the previous observed `528`, use the live count and explain the drift.
5. Update whitelist/ref/enrichment only for accepted `binance:futures` symbols.
6. Run focused local checks and collect Mac Studio ref/enrichment evidence.
7. Update stage report and ledger with accepted/blocked status and exact handoff to Stage 04B.

# Acceptance criteria (Definition of Done)

- Stage 04A report records prompt path/hash, file manifest, current Binance metadata evidence, accepted symbols, excluded symbols with reasons, whitelist diff, ref sync/enrichment evidence, and delivery state.
- No non-current, non-USDT, non-perpetual, quarterly/dated, `TRADIFI_PERPETUAL`, BUSD/USDC/USD1, or unmapped symbol is scheduled for backfill in this USDT-pair v1 scope.
- Binance spot, Bybit spot, and Bybit futures remain `blocked_not_training_source_v1` for training.
- Stage ledger is updated after validation and before final response.
- Stage 04B is allowed only if accepted symbol/ref/enrichment evidence exists.
- Delivery state is explicit: `local-only`, `published-to-branch/draft-pr`, `delivered-to-main`, and/or `deployed-on-macstudio`, with evidence appropriate to the stage.

# Implementation constraints

## Determinism & ordering

- Sort accepted/excluded symbol lists deterministically.
- Preserve stable CSV ordering rules or document the chosen deterministic order.
- Do not introduce hidden random sampling.

## API / contracts

- Whitelist/ref/enrichment changes are market-data contract changes; classify them.
- Do not change public API, exchange execution, model registry, or live trading behavior in this stage.
- Private exchange credentials are not needed.

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

- `configs/prod/whitelist.csv`
- `scripts/rl_trading`
- `tests/unit/contexts/market_data`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04a-binance-futures-universe-whitelist.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`

Possible secondary touches:

- `src/trading/contexts/market_data`
- `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `docs/architecture/market_data`
- `docs/architecture/README.md`

# Non-goals

- Do not backfill candles.
- Do not train models.
- Do not change spot/Bybit training scope.
- Do not use private Binance account endpoints.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/market_data/adapters/test_whitelist_csv_loader.py tests/unit/contexts/market_data/application/use_cases/test_enrich_ref_instruments_from_exchange.py` - if whitelist/ref/enrichment code or config parsing changes
- `uv run ruff check configs src/trading/contexts/market_data scripts/rl_trading tests/unit/contexts/market_data` - focused lint for changed paths; narrow if only docs/config changed
- `python -m tools.docs.generate_docs_index --check` - passes after Markdown updates

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1. **Result and stage status**
2. **File manifest**
3. **Evidence and quality gates**
4. **Contract, safety, and delivery state**
5. **Blockers and next-stage handoff**
