---
prompt_name: "Backtest Futures Funding v1 Stage 00 - Baseline And Contract Freeze"
repo: "roehub.com"
branch: "codex/backtest-futures-funding-v1-stage-00"
scope: "Architecture and implementation baseline freeze before code changes"
language: "en"
context_sources:
  - ".codex/AGENTS.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/backtest/backtest-service-artifact-runtime-v1.ru.md"
  - "docs/architecture/market_data/market-data-real-adapters-clickhouse-parquet.md"
  - "docs/architecture/market_data/market-data-reference-data-sync-v2.md"
  - "docs/architecture/market_data/market-data-rest-historical-catchup-1m-v2.md"
  - "docs/runbooks/market-data-metrics-reference-ru.md"
hard_requirements:
  - "Record `User required before start: nothing` before edits."
  - "Previous-stage ledger gate: N/A for Stage 00 because this stage creates/repairs the baseline ledger; still read and update the ledger before final report."
  - "Previous stage: N/A; there is no previous required stage for Stage 00."
  - "Do not implement production code in this stage."
  - "Freeze exact file boundaries for stages 01-08."
  - "Re-check official Binance and Bybit funding docs or perform provider smoke before relying on API details."
  - "Verify existing market-data-scheduler topology, all-enabled-instruments pattern and Prometheus /metrics baseline."
  - "Do not reopen the closed backtest-compute-acceleration-v1 stage family."
task_toggles:
  implementation: false
  docs_only: true
  browser_qa: false
skill_routing:
  - "architecture-review"
  - "contract-impact-analysis"
  - "backend-quality-gates"
target_envs:
  - "local repository"
  - "Mac Studio target host via ssh macstudio for runtime-boundary availability notes"
required_literals:
  - "User required before start: nothing"
  - "short_direction_requires_futures_market"
  - "funding_manifest_hash"
non_goals:
  - "No ClickHouse migrations."
  - "No runtime code changes."
  - "No UI changes."
final_report_format:
  - "Scope"
  - "File manifest: created/modified/deleted"
  - "Baseline facts"
  - "Contract impact"
  - "Validation"
  - "Residual risks"
quality_gates:
  - "python -m tools.docs.generate_docs_index --check"
validation_strategy:
  - "Docs index check."
  - "Official provider docs re-check or explicit provider-smoke blocker."
  - "When recording runtime-boundary availability, distinguish Codex local loopback from Mac Studio target-host loopback."
stage_execution_ledger: "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
expected_primary_touches:
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/00-baseline-and-contract-freeze.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/README.md"
possible_secondary_touches:
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md"
safety_notes:
  - "Do not print secret-like environment values."
  - "If a later stage needs unavailable credentials or infrastructure, mark that future stage blocked rather than weakening acceptance."
  - "For Roehub runtime checks, 127.0.0.1 means Mac Studio loopback only after ssh macstudio; Codex local loopback is not acceptance evidence."
---

# Task

Freeze the current implementation baseline for Backtest Futures Funding And Short Direction Policy v1 before any production code changes.

## Stage Gate

Previous stage: N/A. Previous-stage ledger gate: N/A for Stage `00`. This is the baseline freeze
stage and it creates or repairs the stage execution ledger. The executor must
still read the existing ledger if present, avoid overwriting accepted facts, and
update it before the final report.

## Context / Current State

The repository already has an artifact-backed backtest runtime, market_data ClickHouse ingestion, strategy launch APIs and browser backtest UI. The architecture document for this plan lists specific gaps: funding storage is absent, funding artifacts are absent, runtime preflight lacks standalone `short`, and the web default currently combines `spot` with `long_short_reversal`.

This stage is documentation and verification only. It must produce a stage report and update the ledger.

## Requirements (Must)

- Start by writing down `User required before start: nothing` in the stage report.
- Record the Stage `00` gate as `N/A - baseline stage` and update the stage ledger before final output.
- Previous stage is N/A because Stage `00` is the baseline stage.
- Read the context sources listed in frontmatter and only the code entrypoints needed to verify current state.
- Verify official Binance/Bybit funding API details from primary sources or perform a provider REST smoke. Record exact date and source.
- Verify the current `market-data-scheduler` topology, all-enabled-instruments scan pattern and Prometheus `/metrics` baseline. If probing runtime availability, do it through `ssh macstudio`; local Codex loopback probes are development diagnostics only.
- Produce a narrow file manifest for each future stage.
- Reclassify all contract impacts if repository facts differ from the architecture document.
- Update the stage ledger with status, evidence and blockers.
- Keep this stage docs-only unless a docs index regeneration is required.

## Requirements (Should)

- Identify tests likely to need updates in each implementation stage.
- Identify any old docs that could mislead future implementation agents.
- Record whether ClickHouse, API, browser and scheduler smoke are available on the Mac Studio target host. If you also probe Codex-local loopback, label it explicitly as non-acceptance diagnostic evidence.

## Requirements (Nice-to-have)

- Add a short traceability table mapping source-plan bullets to final stage numbers.

# Context acquisition protocol

Read `.codex/AGENTS.md` first. Then read the architecture document, ledger and only the current code/doc entrypoints that directly affect funding, backtest direction modes, strategy launch and browser backtest form behavior.

# Reading manifest

Minimum entrypoints:

- `src/trading/contexts/backtest/application/services/v2/preflight.py`
- `src/trading/contexts/backtest/application/services/v2/top_result_assembly.py`
- `src/trading/contexts/backtest/application/services/v2/lazy_trades_detail.py`
- `src/trading/contexts/backtest_artifacts/application/services/v2/contracts.py`
- `src/trading/contexts/strategy/application/use_cases/scenario_matrix.py`
- `apps/api/routes/backtests.py`
- `apps/api/routes/strategies.py`
- `apps/web/templates/pages/backtests.html`
- `apps/web/dist/js/pages/backtests.js`
- `migrations/clickhouse/market_data_ddl.sql`
- `apps/scheduler/market_data_scheduler/main/main.py`
- `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`
- `src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py`
- `docs/runbooks/market-data-metrics-reference-ru.md`
- `infra/macos/prometheus/prometheus.prod.yml`

# Work plan (agent should follow)

1. Confirm the current git status and list unrelated dirty files without touching them.
2. Create `00-baseline-and-contract-freeze.md` under the stage reports folder.
3. Record baseline facts, file manifests, contract impact and target-host real-boundary availability.
4. Update the stage ledger.
5. Run docs index generation if needed, then run the docs index check.
6. Perform a cold-head self-review and record the receipt.

# Acceptance criteria (Definition of Done)

- Stage report exists and is linked from the ledger.
- The report records current facts, not assumptions, for all baseline areas.
- Future stage file manifests are narrow enough to guide implementation.
- Official API facts are refreshed or the reason they could not be refreshed is explicit.
- Docs index check passes.

# Implementation constraints

- Do not change production code.
- Do not edit unrelated dirty files.
- Do not use broad scans as a substitute for reading the listed entrypoints.

# Files to indicate (expected touched areas)

- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/00-baseline-and-contract-freeze.md`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md`
- `docs/architecture/README.md` if regenerated

# Non-goals

- No migrations, API changes, UI changes or runtime behavior changes.

# Quality gates (must run and pass)

```bash
python -m tools.docs.generate_docs_index --check
```

# Final output: report format (strict)

- Scope
- File manifest: created/modified/deleted
- Baseline facts
- Contract impact
- Validation
- Cold-head review receipt
- Residual risks
