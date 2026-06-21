---
prompt_name: "Backtest Futures Funding v1 Stage 01 - Funding Storage Automatic Catchup And Observability"
repo: "roehub.com"
branch: "codex/backtest-futures-funding-v1-stage-01"
scope: "Market-data funding-rate storage, provider source, writer, manual CLI, automatic scheduler catch-up and Prometheus observability"
language: "en"
context_sources:
  - ".codex/AGENTS.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/market_data/market-data-reference-data-sync-v2.md"
  - "docs/architecture/market_data/market-data-rest-historical-catchup-1m-v2.md"
  - "docs/runbooks/market-data-metrics-reference-ru.md"
  - "docs/runbooks/market-data-metrics.md"
hard_requirements:
  - "Record `User required before start: nothing` before edits."
  - "Previous-stage ledger gate: confirm Stage 00 is accepted in the stage execution ledger before implementation; if it is not accepted, stop and record Stage 01 as blocked unless the user explicitly supersedes Stage 00 in the current turn."
  - "Keep funding ingestion in market_data context, not backtest context."
  - "Use internal market_type=futures to Bybit category=linear mapping for v1."
  - "Extend existing market-data-scheduler with funding_rate_catchup; do not create a separate unmanaged daemon."
  - "Automatic scheduler catch-up must cover all exchange-discovered tradable Binance and Bybit futures instruments, not only whitelist-enabled instruments, BTCUSDT or UI-selected symbols."
  - "Do not mark every exchange futures symbol as ENABLED in whitelist-driven ref_instruments; use a dedicated funding universe table/reader or equivalent isolated contract."
  - "Scheduler wake-up cadence is not provider download cadence; provider funding history calls must be funding-interval aligned per symbol."
  - "Do not fetch the full futures universe every minute or every scheduler wake."
  - "Funding interval metadata is mandatory for due selection: Bybit requires instruments-info fundingInterval; Binance requires fundingInfo plus explicit standard 8h source only for symbols absent from the adjusted-only response."
  - "Do not silently treat Binance fundingInfo endpoint failure as ready 8h coverage."
  - "Add Prometheus metrics, alert rules and runbook coverage for funding freshness."
  - "Do not use symbol as a Prometheus label."
  - "Do not print secret-like config values."
  - "Credential redaction rule: never write DSNs, API keys, bearer tokens, ClickHouse passwords or smoke passwords into repo files, prompt artifacts, screenshots, traces, logs, stage ledgers or final reports."
  - "Browser/auth smoke is N/A for this stage. If an authenticated Roehub smoke unexpectedly becomes necessary, use username smoke_e2e_keycloak and source ROEHUB_SMOKE_E2E_PASSWORD from /Users/daniildegtyarev/.config/roehub/roehub.env on macstudio or a securely exported local env var; if unavailable, mark auth smoke blocked."
  - "All writes must be idempotent across repeated windows."
task_toggles:
  implementation: true
  docs_only: false
  browser_qa: false
skill_routing:
  - "contract-impact-analysis"
  - "backend-quality-gates"
target_envs:
  - "local"
  - "ClickHouse when configured; otherwise record boundary-unavailable blocker/evidence"
  - "market-data-scheduler metrics endpoint on 127.0.0.1:9202"
required_literals:
  - "User required before start: nothing"
  - "category=linear"
  - "canonical_funding_rates"
  - "funding_rate_catchup"
  - "scheduler_funding_catchup_"
non_goals:
  - "No backtest scoring changes."
  - "No artifact publisher changes."
  - "No browser changes."
final_report_format:
  - "Scope"
  - "File manifest: created/modified/deleted"
  - "Files changed"
  - "Storage contract"
  - "Provider contract"
  - "Scheduler contract"
  - "Prometheus and alerts"
  - "Validation"
  - "Real-boundary evidence"
  - "Residual risks"
quality_gates:
  - "uv run ruff check src/trading/contexts/market_data apps/cli apps/scheduler tests"
  - "uv run pyright src/trading/contexts/market_data apps/cli apps/scheduler tests"
  - "uv run pytest -q tests/unit/contexts/market_data tests/unit/apps/cli tests/unit/apps/scheduler tests/unit/infra"
  - "python -m tools.docs.generate_docs_index --check"
validation_strategy:
  - "Focused unit tests for Binance and Bybit response parsing, pagination/windowing and idempotent writes."
  - "Scheduler tests proving funding_rate_catchup enumerates all exchange-discovered tradable futures instruments for market_id 2 and 4, independent of whitelist-enabled candle instruments."
  - "Scheduler tests proving non-due symbols do not trigger provider funding-history calls."
  - "Tests proving Bybit missing fundingInterval degrades/skips the symbol, Binance adjusted fundingInfo rows override the standard interval, Binance symbols absent from fundingInfo use explicit binance_standard_8h_no_adjustment_row, and global Binance fundingInfo failure blocks/degrades readiness."
  - "Prometheus metrics tests proving funding metrics are registered without symbol labels."
  - "ClickHouse DDL apply/query smoke when ClickHouse is configured; otherwise record boundary-unavailable blocker/evidence."
  - "Provider REST contract smoke for one Binance futures symbol and one Bybit linear symbol, or mark stage blocked if network/provider access is unavailable."
  - "Runtime metrics proof: curl -fsS http://127.0.0.1:9202/metrics | rg '^scheduler_funding_catchup_'."
stage_execution_ledger: "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
expected_primary_touches:
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md"
  - "docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/01-funding-storage-and-catchup.md"
  - "migrations/clickhouse/funding_rates_ddl.sql"
  - "src/trading/contexts/market_data/application/ports/sources/funding_rate_history_source.py"
  - "src/trading/contexts/market_data/application/ports/sources/funding_instrument_universe_source.py"
  - "src/trading/contexts/market_data/application/ports/stores/funding_rate_writer.py"
  - "src/trading/contexts/market_data/application/ports/stores/funding_instrument_universe_store.py"
  - "src/trading/contexts/market_data/application/use_cases/backfill_funding_rates.py"
  - "src/trading/contexts/market_data/application/use_cases/sync_futures_funding_universe.py"
  - "src/trading/contexts/market_data/adapters/outbound/clients/funding_rate_history_source.py"
  - "src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py"
  - "apps/cli/commands/funding_rate_catchup.py"
  - "apps/cli/main/main.py"
  - "apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py"
  - "src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py"
  - "configs/dev/market_data.yaml"
  - "configs/test/market_data.yaml"
  - "configs/prod/market_data.yaml"
  - "infra/macos/prometheus/prometheus.prod.yml"
  - "infra/macos/prometheus/rules/market-data-funding.rules.yml"
  - "docs/runbooks/market-data-metrics-reference-ru.md"
possible_secondary_touches:
  - "tests/unit/contexts/market_data/"
  - "tests/unit/apps/cli/"
  - "tests/unit/apps/scheduler/"
  - "tests/unit/infra/test_monitoring_assets.py"
  - "docs/runbooks/market-data-metrics.md"
  - "docs/architecture/README.md"
safety_notes:
  - "Use env var names only in logs and reports."
  - "Do not add authenticated exchange clients for public funding history endpoints."
  - "Do not persist ROEHUB_SMOKE_E2E_PASSWORD or any derived browser credentials; browser auth smoke is N/A unless a later user request expands this stage."
---

# Task

Implement market_data funding-rate storage, manual catch-up and automatic scheduler catch-up for all futures symbols, with Prometheus observability.

## Stage Gate

Previous stage ledger gate: before any implementation edits, read the stage
execution ledger and verify Stage `00` is accepted in the ledger. If Stage `00` is not
accepted, do not implement Stage `01`; update the Stage `01` report/ledger as
blocked unless the user explicitly supersedes this gate in the current turn.

## Context / Current State

The repository has market_data ClickHouse candle ingestion, whitelist-driven reference instrument sync and an existing `market-data-scheduler` process with `/metrics` on port `9202`. Funding rates are absent. The architecture decision is to ingest funding in market_data, keep it fresh automatically for all exchange-discovered tradable Binance and Bybit futures instruments, and let backtest consume normalized data later through artifacts. Do not confuse the existing `EnabledInstrumentReader` whitelist with the full futures funding universe.

## Requirements (Must)

- Record `User required before start: nothing` in the stage report before edits.
- Previous-stage gate: Stage `00` must be `accepted` in the stage ledger before Stage `01` implementation starts; otherwise stop and mark this stage blocked unless the user explicitly supersedes Stage `00`.
- Add idempotent ClickHouse DDL for raw Binance, raw Bybit and canonical funding rates using `market_id UInt16`; include `funding_interval_minutes` and `funding_interval_source`.
- Add a dedicated funding instrument universe table/reader or equivalent isolated contract for all exchange-discovered tradable Binance/Bybit futures symbols.
- Implement provider source parsing for Binance USD-M funding history and Bybit v5 funding history.
- Map Roehub `market_type=futures` to Bybit `category=linear` for v1.
- Add writer/store and use case for single-symbol and all-funding-universe catch-up.
- Register the CLI command in `apps/cli/main/main.py`.
- Extend existing `market-data-scheduler` with a periodic `funding_rate_catchup` job.
- Add runtime config for `scheduler.jobs.funding_rate_catchup`, including scheduler wake-up interval, due mode, settlement lag, `binance_standard_interval_hours`, interval-metadata failure fallback controls, startup bootstrap, tail lookback and gap-audit controls.
- Treat `interval_seconds` as wake-up cadence only, default `1800`; actual provider funding-history downloads must be aligned to each symbol's funding interval.
- Funding interval metadata is mandatory for due calculation.
- Bybit must read instruments-info `fundingInterval` for each `category=linear` tradable instrument; missing interval metadata marks that symbol degraded/skipped until metadata is available.
- Binance must call `/fapi/v1/fundingInfo`; if a symbol has an adjusted row, use `fundingIntervalHours`; if a tradable USD-M symbol is absent from that adjusted-only response, use standard `8h` with `funding_interval_source=binance_standard_8h_no_adjustment_row`.
- If Binance `fundingInfo` fails globally, do not silently mark Binance funding ready from the `8h` default; fail/degrade the run via metrics and stage evidence unless an explicit emergency fallback is enabled.
- Define due work as `next_funding_time = last_funding_time + funding_interval`; fetch history only when `now >= next_funding_time + settlement_lag_minutes`; skip non-due symbols without provider history calls.
- Automatic scheduler mode must refresh and enumerate all exchange-discovered tradable futures instruments for Binance futures (`market_id=2`) and Bybit futures (`market_id=4`), using Binance exchange info and Bybit `category=linear` instruments info.
- Automatic scheduler mode must not require a canonical candle seed and must not be limited to BTCUSDT.
- Automatic scheduler mode must not mark all discovered futures symbols as `ENABLED` in the existing whitelist-driven `ref_instruments`, because that would expand candle ingestion/backtest symbol universe.
- Add Prometheus metrics with prefix `scheduler_funding_catchup_`; labels may include `exchange`, `market_type`, `status`, but must not include `symbol`.
- Add Prometheus alert rules for funding catch-up errors, no recent success, high lag and missing instruments.
- Update market-data metrics runbook(s).
- Add tests for parsing, Bybit window restrictions, mapping and idempotency.
- Prove real boundaries: ClickHouse apply/query, provider REST smoke, and scheduler `/metrics` proof on `127.0.0.1:9202`, or mark the stage blocked if the boundary is unavailable.

## Requirements (Should)

- Persist funding interval/cap/floor metadata and interval source where the provider contract exposes it.
- Support dry-run summary.
- Keep retry/backoff bounded and visible in reports.
- Keep scheduler job status and funding freshness visible without high-cardinality labels.
- Keep default steady-state provider load around one funding-history fetch per symbol per funding interval, usually every `8h`, plus bounded tail repair.

## Requirements (Nice-to-have)

- Add a small fixture pack with representative Binance and Bybit funding responses.

# Context acquisition protocol

Read current market_data DDL, rest catch-up command, market-data scheduler, runtime config, Prometheus assets and reference instrument sync code before designing new ports.

# Reading manifest

- `migrations/clickhouse/market_data_ddl.sql`
- `apps/cli/commands/rest_catchup_1m.py`
- `apps/cli/main/main.py`
- `apps/scheduler/market_data_scheduler/main/main.py`
- `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`
- `src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py`
- `configs/prod/market_data.yaml`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py`
- `src/trading/contexts/market_data/application/ports/`
- `src/trading/contexts/market_data/application/use_cases/`
- `docs/runbooks/market-data-metrics-reference-ru.md`
- `infra/macos/prometheus/prometheus.prod.yml`
- `tests/unit/infra/test_monitoring_assets.py`

# Work plan (agent should follow)

1. Confirm git status and unrelated local changes.
2. Create the stage report with file manifest and contract classification.
3. Add DDL and market_data ports/use case/adapters.
4. Add CLI command and dispatcher wiring.
5. Extend `market-data-scheduler` with `funding_rate_catchup`, config parsing and all-futures enumeration.
6. Add Prometheus metrics, alert rules, monitoring tests and runbook updates.
7. Add focused tests and fixtures.
8. Run focused checks, then real-boundary smoke including scheduler `/metrics`.
9. Update stage report and ledger.

# Acceptance criteria (Definition of Done)

- Funding tables can be created idempotently.
- Binance and Bybit funding responses normalize into canonical rows.
- Bybit `futures -> linear` behavior is covered by tests.
- CLI can dry-run and run for a bounded symbol/window.
- `market-data-scheduler` can run automatic `funding_rate_catchup` over all exchange-discovered tradable Binance and Bybit futures pairs from the dedicated funding universe.
- Existing whitelist-driven candle/backtest universe is not silently expanded to every futures symbol.
- Prometheus funding metrics are exposed without `symbol` labels.
- Prometheus alert rule assets and runbook docs are updated.
- Funding freshness alerts are interval-aware; they must not page between normal 8h settlement windows.
- Real-boundary evidence is recorded for ClickHouse, provider REST and scheduler `/metrics`.

# Implementation constraints

- Keep exchange REST code unauthenticated for these public endpoints.
- Do not couple backtest code to provider clients.
- Keep rows idempotent for reruns over the same window.
- Reuse the existing `market-data-scheduler` topology on port `9202`; do not introduce a second scheduler/daemon unless the stage report records a blocker and user decision.
- Do not implement minute-level full-market polling.
- Per-symbol funding diagnostics must be structured logs or ClickHouse queries, not Prometheus labels.

# Files to indicate (expected touched areas)

Use the frontmatter file list as the starting manifest and narrow it before edits.

# Non-goals

- No funding artifact publishing.
- No backtest preflight/scoring changes.

# Quality gates (must run and pass)

```bash
uv run ruff check src/trading/contexts/market_data apps/cli apps/scheduler tests
uv run pyright src/trading/contexts/market_data apps/cli apps/scheduler tests
uv run pytest -q tests/unit/contexts/market_data tests/unit/apps/cli tests/unit/apps/scheduler tests/unit/infra
python -m tools.docs.generate_docs_index --check
```

# Final output: report format (strict)

- Scope
- Files changed
- Storage contract
- Provider contract
- Scheduler contract
- Prometheus and alerts
- Validation
- Real-boundary evidence
- Cold-head review receipt
- Residual risks
