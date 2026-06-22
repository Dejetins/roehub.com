---
prompt_name: 04b-binance-futures-history-backfill
repo: roehub.com
branch: main
branch_policy:
  default_branch: main
  separate_branch_allowed: false
  stage_specific_branches_forbidden: true
scope: "Repair to the full current Binance Futures USDT perpetual universe, backfill 1m source windows, and prove coverage before dataset refresh manifests."
language:
  implementation: python
  agent_report: ru
context_sources:
  always_read:
    - path: .codex/AGENTS.md
      why: "repo contract"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1.md
      why: "RL plan and source-window policy"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
      why: "stage ledger"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/02a-data-source-inventory.md
      why: "HF windows, current coverage, channel and universe facts"
    - path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04a-binance-futures-universe-whitelist.md
      why: "previous 215-symbol partial universe and whitelist/ref evidence from Stage 04A"
    - path: .codex/agents/.context/promt_manager_state.yaml
      why: "optional compact state; ignore if unrelated"
  task_entrypoints:
    - path: src/trading/contexts/market_data/application/use_cases/rest_fill_range_1m.py
      why: "explicit REST range fill use case"
      inspect_symbols:
        - "RestFillRange1mUseCase"
    - path: src/trading/contexts/market_data/application/use_cases/rest_catchup_1m.py
      why: "seeded tail/gap catchup behavior"
      inspect_symbols:
        - "RestCatchUp1mUseCase"
    - path: src/trading/contexts/market_data/application/services/scheduler_backfill_planner.py
      why: "bootstrap/historical/tail task planning"
      inspect_symbols:
        - "SchedulerBackfillPlanner"
    - path: src/trading/contexts/market_data/adapters/outbound/clients/rest_candle_ingest_source.py
      why: "Binance Futures klines source"
      inspect_symbols:
        - "RestCandleIngestSource"
        - "_stream_binance_klines_1m"
    - path: src/trading/contexts/market_data/adapters/outbound/clients/rest_instrument_history_start_source.py
      why: "Binance Futures onboardDate/listing lower bound"
      inspect_symbols:
        - "RestInstrumentHistoryStartSource"
    - path: src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_index_reader.py
      why: "coverage/gap readback"
      inspect_symbols:
        - "bounds"
        - "distinct_ts_opens"
  conditional_bundles:
    market_data_runtime:
      read_when: "running Mac Studio backfill or writing a runner"
      paths:
        - configs/prod/market_data.yaml
        - apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py
        - docs/architecture/market_data/market-data-rest-historical-catchup-1m-v2.md
        - docs/architecture/market_data/market-data-runtime-config-invariants-v2.md
    parquet_path:
      read_when: "choosing parquet staging instead of REST range fill"
      paths:
        - apps/cli/commands/backfill_1m.py
        - docs/architecture/market_data/market-data-real-adapters-clickhouse-parquet.md
    docs_to_update:
      read_when: "stage changes shared docs, contracts, runbooks, or operator workflows"
      paths:
        - docs/architecture/README.md
        - docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/01-baseline-plan-freeze.md
  consult_if_needed:
    - path: tests/unit/contexts/market_data/application/use_cases/test_rest_fill_range_1m.py
      read_when: "implementing or changing explicit REST range fill behavior"
    - path: tests/unit/contexts/market_data/application/use_cases/test_backfill_1m_candles.py
      read_when: "changing generic backfill behavior"
    - path: tests/unit/apps/cli/commands/test_rest_catchup_1m_cli.py
      read_when: "changing CLI behavior"
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
  allow_public_market_data_rest: true
  allow_market_data_writes: true
  allow_browser_runtime_verification: false
  allow_tests_only_acceptance: false
skill_routing:
  - skill: publish-ci-deploy
    use_when: "publishing accepted stage changes through the default direct-main delivery workflow"
    timing: "before ship"
    reason: "owns scoped staging, main delivery, CI/deploy follow-up, and Mac Studio verification; branch/PR only when explicitly requested by the user"
  - skill: backend-quality-gates
    use_when: "backend Python code, CLI, tests, lint, or type gates are changed"
    timing: "during verification"
    reason: "owns focused ruff, pyright, pytest gate triage"
  - skill: data-analytics-methodology
    use_when: "designing coverage/gap criteria and interpreting missing minutes"
    timing: "during validation"
    reason: "keeps coverage evidence and exclusion policy explicit"
  - skill: contract-impact-analysis
    use_when: "market-data ingestion, config, persisted rows, or CLI behavior changes"
    timing: "during implementation"
    reason: "classifies compatibility and rollout impact"
target_envs:
  - "local checkout"
  - "macstudio for ClickHouse/public REST backfill evidence"
required_literals:
  - "binance:futures"
  - "hf_period_rebuild_current_trading"
  - "post_hf_extension_current_trading"
  - "/opt/roehub/state/rl_trading/"
non_goals:
  - "Do not add non-USDT, non-perpetual, quarterly/dated, TRADIFI_PERPETUAL, BUSD/USDC/USD1, or non-current symbols to this v1 USDT-pair training universe."
  - "Do not synthesize missing candles."
  - "Do not build feature slabs or sessionized datasets."
  - "Do not use Binance private/account endpoints."
final_report_format:
  language: ru
  sections:
    - "Result and stage status"
    - "File manifest"
    - "Evidence and quality gates"
    - "Contract, safety, and delivery state"
    - "Blockers and next-stage handoff"
quality_gates:
  - cmd: "uv run pytest -q tests/unit/contexts/market_data/application/use_cases/test_rest_fill_range_1m.py tests/unit/contexts/market_data/application/use_cases/test_backfill_1m_candles.py tests/unit/contexts/market_data/application/use_cases/test_rest_catchup_1m.py"
    expect: "passes when REST/backfill behavior changes"
  - cmd: "uv run pytest -q tests/unit/apps/cli/commands/test_rest_catchup_1m_cli.py apps/cli/test_backfill_1m_parsing.py"
    expect: "passes when CLI paths change; narrow if no CLI changes"
  - cmd: "uv run ruff check apps/cli src/trading/contexts/market_data scripts/rl_trading tests/unit/contexts/market_data tests/unit/apps/cli"
    expect: "passes for changed backend/CLI/test paths"
  - cmd: "python -m tools.docs.generate_docs_index --check"
    expect: "passes after Markdown updates"
validation_strategy:
  depth: target-runtime
  e2e_required: true
  acceptance_surfaces:
    - "Mac Studio ClickHouse raw/canonical coverage"
    - "public Binance Futures REST range fill or accepted parquet staging path"
    - "per-symbol backfill resume manifest"
    - "gap/duplicate/feature coverage report"
  tests_only_allowed_reason: ""
stage_execution_ledger:
  path: docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md
  plan_doc: docs/architecture/ml/rl-trading-agent-platform-v1.md
  current_stage: "04B"
  required_update: true
  template: .codex/agents/stage_execution_ledger_template.md
expected_primary_touches:
  - "configs/prod/whitelist.csv"
  - "scripts/rl_trading"
  - "src/trading/contexts/market_data"
  - "tests/unit/contexts/market_data"
  - "tests/unit/scripts/rl_trading"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04b-binance-futures-history-backfill.md"
  - "docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md"
possible_secondary_touches:
  - "apps/cli/commands"
  - "tests/unit/apps/cli"
  - "apps/scheduler/market_data_scheduler"
  - "docs/architecture/market_data"
  - "docs/architecture/ml/rl-trading-agent-platform-v1.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Never write secrets, tokens, cookies, passphrases, ciphertext, API keys, raw provider payloads, or raw signed requests into prompts, docs, ledgers, traces, screenshots, or logs."
  - "Binance public kline/metadata endpoints are allowed; private/account endpoints are not needed and must not be used."
  - "Backfill writes market_data raw/canonical rows only through existing writers/use cases; no exchange trading side effects."
  - "Mac Studio git commands must use /Users/daniildegtyarev/Projects/roehub.com; /opt/roehub/app is runtime state only."
  - "Browser/auth is N/A for this data backfill stage: do not use the smoke_e2e_keycloak account and do not read ROEHUB_SMOKE_E2E_PASSWORD from /Users/daniildegtyarev/.config/roehub/roehub.env unless a later browser-visible task explicitly requires it."
---

# Task

Implement Stage 04B Binance Futures historical backfill and coverage repair. The previous Stage 04A/04B path covered only the HF-intersection subset (`215` symbols). The corrected target is all current Binance USD-M Futures symbols where `status=TRADING`, `contractType=PERPETUAL`, and `quoteAsset=USDT` (`528` observed on 2026-06-21; use live metadata at execution). Supplement whitelist/ref/enrichment and backfill coverage to that full current USDT perpetual universe before any feature slabs are built.

Done means:

- The executor resolves the live full-current Binance `TRADING` `USDT` `PERPETUAL` universe and records count/hash; HF membership is not a filter.
- Existing 215-symbol Stage 04A/04B artifacts are reused as partial progress, not treated as final coverage.
- Every accepted full-current USDT perpetual symbol has a source-window task plan using `max(required_source_start, exchange onboard/history start)`.
- Missing symbols compared with the previous Stage 04A subset are added to `configs/prod/whitelist.csv`, synced to `market_data.ref_instruments`, and enriched before backfill.
- The executor uses an existing safe REST/scheduler/fill path when present; if no operator-safe explicit range runner exists, it implements the narrowest wrapper around existing market-data ports/use cases or blocks with evidence.
- Backfill runs are resumable and recorded with per-symbol status.
- Long-running backfill is launched as a managed resumable/background job, then the agent verifies within a bounded observation window that new rows/high-watermarks started appearing in ClickHouse and stops active work; it must not sit in an active session waiting for the entire historical load to finish.
- Full coverage report proves first/last candle, missing minutes, duplicates, `volume_quote`, `trades_count`, and `vwap` computability for required windows only when the backfill has actually completed.
- Stage 04C receives a clean coverage manifest and exact residual gaps only after full completion evidence; a start-only proof leaves Stage 04B `in_progress` and does not unlock Stage 04C.

## Context / Current State

- Stage 04A historically accepted a 215-symbol HF-intersection subset. User correction on 2026-06-21 supersedes that as final target: Stage 04B must repair/supplement to the full current Binance USDT perpetual universe.
- Existing `backfill-1m` CLI may be parquet-oriented; `rest-catchup` requires seed history for an instrument. Verify actual current code before choosing a path.
- Market-data docs describe REST catch-up, scheduler bootstrap, symbol-specific history starts, and `RestInstrumentHistoryStartSource` for Binance Futures `onboardDate`.
- This stage may write raw market data through existing market-data ingestion paths, but it must never place raw candle dumps in git/docs.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`. If that statement is not true after reading the ledger, stop and record the blocker instead of guessing.
- Verify prerequisites before implementation. Required accepted prerequisites: Stage 04A. If Stage 04A is not accepted, write/update the Stage 04B report as blocked, update the ledger, and do not implement dependent work.
- Compute this prompt hash with `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/04b-binance-futures-history-backfill.md` and record path/hash in the stage report.
- Before editing, narrow expected paths to a concrete file list and record it in the stage report.
- Keep the change bounded to Stage 04B. Do not change training-source scope beyond `binance:futures` USDT perpetuals, build feature slabs, sessionize windows, train models, or change live execution behavior.
- Resolve the target universe from current public Binance Futures metadata at execution time: accept only `status=TRADING`, `contractType=PERPETUAL`, `quoteAsset=USDT`. Do not require symbols to exist in the HF train/all-split lists.
- Treat the previous Stage 04A accepted `215` symbols as a partial subset. Compute the supplement set as `current_full_usdt_perpetual_universe - existing_stage04a_symbols`; update whitelist/ref/enrichment for the supplement before scheduling their backfill.
- If a previously backfilled Stage 04A symbol disappears from current Binance Futures metadata, mark it stale for future versions but keep its historical evidence separate; do not use it to block supplement scheduling for still-current symbols.
- Respect source windows from the plan and recompute the current post-HF endpoint from the latest Binance Futures candle snapshot before running extension tasks.
- For each symbol, compute safe start as `max(required_source_window_start, onboard/listing/history start)`. Do not request pre-listing candles.
- Use chunked/rate-limited fills with resume manifest and read-back coverage. Interrupted or unknown-state backfill must read canonical/raw coverage before retry.
- If the required historical load is long-running, run it only through a managed resumable/background path with durable logs/resume manifest. After launching, observe ClickHouse for a bounded window (for example 5-15 minutes or a small fixed number of completed chunks), prove that data started arriving by row-count/high-watermark deltas, record job id/process/service state and log/resume paths, then stop the agent work.
- Do not keep the agent session open to wait for the full historical backfill to complete. Do not mark Stage 04B `accepted` from start-only evidence; leave it `in_progress` with a clear follow-up coverage-check handoff unless full coverage already completed inside the bounded window.
- If no safe managed/background execution path exists, do not run a foreground long backfill just to keep the agent busy; mark Stage 04B blocked with the missing capability.
- Do not synthesize candles. Gaps that Binance does not return remain explicit residual gaps and may block Stage 04C or mark a dataset version partial/rejected.
- Store raw operational run logs and large reports under `/opt/roehub/state/rl_trading/`; docs receive only sanitized summaries, hashes, counts and paths.
- Update/create `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04b-binance-futures-history-backfill.md` and update the ledger before final response.

## Requirements (Should)

- Prefer existing market-data REST/scheduler services over new one-off ingestion logic.
- Add an explicit dry-run/plan mode before writes if a new operator wrapper is needed.
- Keep per-symbol tasks small enough to resume and audit.

# Context acquisition protocol

Read `.codex/AGENTS.md`, RL plan, ledger, Stage 02A report, Stage 04A report, then current market-data backfill code. Treat Stage 04A as partial historical evidence only; the current target universe is full current Binance USDT perpetuals. Expand into scheduler/runtime docs only if the safe path is ambiguous.

# Work plan (agent should follow)

Skill routing:

- `publish-ci-deploy`: use before ship when publishing accepted stage changes through the default direct-main delivery workflow; owns scoped staging, main delivery, CI/deploy/Mac Studio follow-up, and branch hygiene only when the user explicitly requested branch/PR delivery.
- `backend-quality-gates`: use during backend/CLI verification.
- `data-analytics-methodology`: use for coverage/gap methodology.
- `contract-impact-analysis`: use for market-data write/config/CLI impact.

1. Verify Stage 04A acceptance and load its previous 215-symbol manifest as partial progress.
2. Fetch current Binance Futures public metadata and resolve the full `TRADING` `USDT` `PERPETUAL` symbol list. Record count/hash and compare with the 2026-06-21 observed count `528`; live count wins if it changed.
3. Compute supplement symbols missing from the previous Stage 04A subset, update `configs/prod/whitelist.csv`, sync `market_data.ref_instruments`, and enrich exchange filters/steps/min-notional for the supplement. If this write path is unavailable, block instead of silently backfilling unregistered symbols.
4. Verify the current available backfill path: explicit REST range use case, scheduler bootstrap, rest-catchup, or parquet staging. Record the selected path and why alternatives were rejected.
5. Generate a dry-run task manifest for the full current USDT perpetual universe, reusing completed 215-symbol chunk state where valid and scheduling only missing/stale windows for supplement symbols.
6. Execute a bounded start of the supplement/full backfill plan on Mac Studio only after dry-run evidence is reviewed by the stage executor; use a managed resumable/background path, not a foreground command that requires the agent to wait for the whole history.
7. Read back ClickHouse canonical/raw state during a short observation window and prove supplement ingestion has started via new rows, changed high-watermarks, or completed chunk markers. If ingestion does not start, stop and record the blocker.
8. If the full backfill has not completed, update the stage report and ledger as `in_progress`, include job/process/service state, log/resume paths, start-proof evidence, and the exact follow-up coverage-check command. Stop there.
9. Only when the full job is already complete, read back ClickHouse canonical/raw coverage and compute gaps/duplicates/feature coverage for the full current USDT perpetual universe.
10. Update stage report and ledger with accepted/blocked/in_progress status and handoff to Stage 04C.

# Acceptance criteria (Definition of Done)

- Stage 04B report records prompt path/hash, file manifest, live Binance metadata count/hash, previous 215-symbol subset hash, supplement symbol count/hash, whitelist/ref/enrichment evidence for the supplement, selected backfill path, dry-run task manifest hash, execution/resume evidence, coverage report, residual gaps, and delivery state.
- If a long-running backfill was started but not completed, the report records start-proof evidence and Stage 04B remains `in_progress`; this is a valid stop point for the executor, not an accepted handoff to Stage 04C.
- No non-current, non-USDT, non-perpetual, quarterly/dated, `TRADIFI_PERPETUAL`, BUSD/USDC/USD1, or unmapped symbol is backfilled in this v1 USDT-pair scope.
- Coverage is computed per symbol and per dataset source window; summary includes first/last candle, missing minutes, duplicates, `volume_quote`, `trades_count`, zero-volume rows and `vwap` computability.
- Backfill unknown-state retry is guarded by read-back coverage/resume manifest, not by assuming no write happened.
- Stage ledger is updated after validation and before final response.
- Stage 04C is allowed only if at least one dataset refresh version can be built from accepted coverage; otherwise Stage 04C remains blocked.
- Delivery state is explicit: `local-only`, `delivered-to-main`, `deployed-on-macstudio`, and `published-to-branch/draft-pr` only when the user explicitly requested branch/PR delivery, with evidence appropriate to the stage.

# Implementation constraints

## Determinism & ordering

- Sort symbols and ranges deterministically.
- Use half-open UTC ranges `[start, end)` and minute-aligned timestamps.
- Keep manifest hashes stable across equivalent runs.

## API / contracts

- Any new runner/CLI must wrap existing market-data ports/use cases and keep source-of-truth in ClickHouse raw/canonical pipeline.
- Do not change public API, exchange execution, model registry, or live trading behavior.
- Public REST market-data calls are allowed; private endpoints and secrets are not.

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
- `src/trading/contexts/market_data`
- `tests/unit/contexts/market_data`
- `tests/unit/scripts/rl_trading`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/04b-binance-futures-history-backfill.md`
- `docs/architecture/ml/rl-trading-agent-platform-v1-stage-reports/rl-trading-agent-platform-v1-stage-ledger.md`

Possible secondary touches:

- `apps/cli/commands`
- `tests/unit/apps/cli`
- `apps/scheduler/market_data_scheduler`
- `docs/architecture/market_data`
- `docs/architecture/ml/rl-trading-agent-platform-v1.md`
- `docs/architecture/README.md`

# Non-goals

- Do not change whitelist/universe except to add, sync, and enrich the full-current Binance USDT perpetual supplement and mark stale/non-current symbols as blocked in the report.
- Do not build RL feature slabs or session windows.
- Do not train models.
- Do not use private exchange/account endpoints.

# Quality gates (must run and pass)

- `uv run pytest -q tests/unit/contexts/market_data/application/use_cases/test_rest_fill_range_1m.py tests/unit/contexts/market_data/application/use_cases/test_backfill_1m_candles.py tests/unit/contexts/market_data/application/use_cases/test_rest_catchup_1m.py` - if REST/backfill behavior changes
- `uv run pytest -q tests/unit/apps/cli/commands/test_rest_catchup_1m_cli.py apps/cli/test_backfill_1m_parsing.py` - if CLI paths change
- `uv run ruff check apps/cli src/trading/contexts/market_data scripts/rl_trading tests/unit/contexts/market_data tests/unit/apps/cli` - focused lint for changed paths
- `python -m tools.docs.generate_docs_index --check` - passes after Markdown updates

# Final output: report format (strict)

Your final message MUST be in Russian and follow exactly:

1. **Result and stage status**
2. **File manifest**
3. **Evidence and quality gates**
4. **Contract, safety, and delivery state**
5. **Blockers and next-stage handoff**
