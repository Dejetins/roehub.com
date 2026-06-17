---
prompt_name: 04b-binance-futures-history-backfill
repo: roehub.com
branch: main
scope: "Backfill accepted Binance Futures 1m source windows and prove coverage before dataset refresh manifests."
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
      why: "accepted symbol universe and exclusions from Stage 04A"
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
  - skill: github:yeet
    use_when: "publishing accepted stage changes to GitHub through a scoped branch and draft PR"
    timing: "before ship"
    reason: "owns scoped staging, branch push, draft PR creation, and branch hygiene"
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
  - "Do not add symbols outside the accepted Stage 04A universe."
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
  - "scripts/rl_trading"
  - "src/trading/contexts/market_data"
  - "tests/unit/contexts/market_data"
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
---

# Task

Implement Stage 04B Binance Futures historical backfill and coverage. Using the accepted Stage 04A universe, load or repair required 1m source candle windows for current-trading Binance Futures USDT perpetual symbols and prove coverage before any feature slabs are built.

Done means:

- Every accepted Stage 04A symbol has a source-window task plan using `max(required_source_start, exchange onboard/history start)`.
- The executor uses an existing safe REST/scheduler/fill path when present; if no operator-safe explicit range runner exists, it implements the narrowest wrapper around existing market-data ports/use cases or blocks with evidence.
- Backfill runs are resumable and recorded with per-symbol status.
- Long-running backfill is launched as a managed resumable/background job, then the agent verifies within a bounded observation window that new rows/high-watermarks started appearing in ClickHouse and stops active work; it must not sit in an active session waiting for the entire historical load to finish.
- Full coverage report proves first/last candle, missing minutes, duplicates, `volume_quote`, `trades_count`, and `vwap` computability for required windows only when the backfill has actually completed.
- Stage 04C receives a clean coverage manifest and exact residual gaps only after full completion evidence; a start-only proof leaves Stage 04B `in_progress` and does not unlock Stage 04C.

## Context / Current State

- Stage 04A owns universe and whitelist/ref/enrichment. Do not rediscover or expand the universe here.
- Existing `backfill-1m` CLI may be parquet-oriented; `rest-catchup` requires seed history for an instrument. Verify actual current code before choosing a path.
- Market-data docs describe REST catch-up, scheduler bootstrap, symbol-specific history starts, and `RestInstrumentHistoryStartSource` for Binance Futures `onboardDate`.
- This stage may write raw market data through existing market-data ingestion paths, but it must never place raw candle dumps in git/docs.

## Requirements (Must)

- Start by stating exactly: `User required before start: nothing unless a listed prerequisite is not accepted or a required credential/dataset/runtime source is unavailable; never ask for secrets in chat`. If that statement is not true after reading the ledger, stop and record the blocker instead of guessing.
- Verify prerequisites before implementation. Required accepted prerequisites: Stage 04A. If Stage 04A is not accepted, write/update the Stage 04B report as blocked, update the ledger, and do not implement dependent work.
- Compute this prompt hash with `shasum -a 256 .codex/agents/generated/rl-trading-agent-platform-v1/04b-binance-futures-history-backfill.md` and record path/hash in the stage report.
- Before editing, narrow expected paths to a concrete file list and record it in the stage report.
- Keep the change bounded to Stage 04B. Do not update whitelist, change training-source scope, build feature slabs, sessionize windows, train models, or change live execution behavior.
- Use only accepted Stage 04A symbols. If a symbol disappears from current Binance Futures metadata between Stage 04A and 04B, mark it excluded/stale and do not backfill it.
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

Read `.codex/AGENTS.md`, RL plan, ledger, Stage 02A report, Stage 04A report, then current market-data backfill code. Expand into scheduler/runtime docs only if the safe path is ambiguous.

# Work plan (agent should follow)

Skill routing:

- `github:yeet`: use before ship when publishing accepted stage changes through a scoped branch and draft PR; do not push directly to `main`.
- `backend-quality-gates`: use during backend/CLI verification.
- `data-analytics-methodology`: use for coverage/gap methodology.
- `contract-impact-analysis`: use for market-data write/config/CLI impact.

1. Verify Stage 04A acceptance and load its accepted symbol manifest.
2. Verify the current available backfill path: explicit REST range use case, scheduler bootstrap, rest-catchup, or parquet staging. Record the selected path and why alternatives were rejected.
3. Generate a dry-run task manifest for all accepted symbols and source windows.
4. Execute a bounded start of the backfill plan on Mac Studio only after dry-run evidence is reviewed by the stage executor; use a managed resumable/background path, not a foreground command that requires the agent to wait for the whole history.
5. Read back ClickHouse canonical/raw state during a short observation window and prove ingestion has started via new rows, changed high-watermarks, or completed chunk markers. If ingestion does not start, stop and record the blocker.
6. If the full backfill has not completed, update the stage report and ledger as `in_progress`, include job/process/service state, log/resume paths, start-proof evidence, and the exact follow-up coverage-check command. Stop there.
7. Only when the full job is already complete, read back ClickHouse canonical/raw coverage and compute gaps/duplicates/feature coverage.
8. Update stage report and ledger with accepted/blocked/in_progress status and handoff to Stage 04C.

# Acceptance criteria (Definition of Done)

- Stage 04B report records prompt path/hash, file manifest, selected backfill path, dry-run task manifest hash, execution/resume evidence, coverage report, residual gaps, and delivery state.
- If a long-running backfill was started but not completed, the report records start-proof evidence and Stage 04B remains `in_progress`; this is a valid stop point for the executor, not an accepted handoff to Stage 04C.
- No excluded Stage 04A symbols are backfilled.
- Coverage is computed per symbol and per dataset source window; summary includes first/last candle, missing minutes, duplicates, `volume_quote`, `trades_count`, zero-volume rows and `vwap` computability.
- Backfill unknown-state retry is guarded by read-back coverage/resume manifest, not by assuming no write happened.
- Stage ledger is updated after validation and before final response.
- Stage 04C is allowed only if at least one dataset refresh version can be built from accepted coverage; otherwise Stage 04C remains blocked.
- Delivery state is explicit: `local-only`, `published-to-branch/draft-pr`, `delivered-to-main`, and/or `deployed-on-macstudio`, with evidence appropriate to the stage.

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
- `src/trading/contexts/market_data`
- `tests/unit/contexts/market_data`
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

- Do not change whitelist/universe except to mark stale symbols as blocked in the report.
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
