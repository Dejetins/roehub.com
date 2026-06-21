# Backtest Futures Funding And Short Direction Policy v1 - Stage 01 Funding Storage And Catchup

Funding-rate storage, automatic catch-up and observability implementation
record.

Date: 2026-06-22

Status: blocked on Mac Studio target-runtime proof; local implementation complete.

Branch: `codex/backtest-futures-funding-v1-stage-01`

`User required before start: nothing`

## Scope

Requested scope:

- market-data funding-rate storage for Binance futures and Bybit futures;
- dedicated exchange-discovered futures funding universe, separate from the
  whitelist-driven candle universe;
- manual funding catch-up CLI;
- automatic `market-data-scheduler` `funding_rate_catchup` job;
- Prometheus metrics and alerting for funding freshness.

Current scope for this run:

- read the stage prompt and repository engineering contract;
- checked current git branch and cleanliness;
- checked the previous-stage gate in the stage ledger before implementation;
- implement market_data funding storage, provider source, manual catch-up,
  automatic scheduler catch-up and observability;
- record local validation and Mac Studio target-runtime boundary state without
  promoting the stage to accepted.

## Business Narrative

Stage `01` добавляет базовый слой funding market data для futures: где хранить
funding ставки, как автоматически догонять историю и как видеть свежесть через
Prometheus. Это еще не меняет backtest ranking, API или UI, но создает
обязательный источник данных для следующих stages, где gross `total_return_pct`
должен остаться видимым, а net-of-funding метрики должны показывать реальную
прибыльность futures-сценариев.

Бизнес-эффект: после acceptance этого stage futures backtests смогут полагаться
на exchange-discovered funding coverage вместо ручного whitelist набора. Если
интервал funding не известен или scheduler не догнал историю, downstream stages
должны видеть degraded readiness, а не запускать расчет с выдуманной частотой.

Conditional service-call coverage:

- provider REST: applicable and partially proven by Binance and Bybit public
  funding-history smoke checks;
- ClickHouse: applicable; Mac Studio `127.0.0.1:8123` is reachable and HTTP
  query works, but Stage `01` DDL/query proof against the delivered runtime is
  still pending;
- scheduler `/metrics`: applicable; Mac Studio `127.0.0.1:9202/metrics` is
  reachable, but `scheduler_funding_catchup_*` is absent because this Stage
  `01` branch is not delivered to the Mac Studio checkout/runtime yet;
- browser/API route evidence: N/A for this stage because no browser-visible API
  or UI contract is changed.

## Previous-Stage Gate

The Stage `01` prompt requires Stage `00` to be `accepted` in the stage
execution ledger before implementation starts.

Observed ledger state after the acceptance repair:

- Stage `00` status: `accepted`.
- Stage `00` report status: `accepted, docs-only, remote branch delivered`.
- Delivery ledger for Stage `00`:
  `origin/codex/backtest-futures-funding-v1-stage-00@7dc0e726fc6babe8c101369a40a4119d5d23fd03`;
  runtime boundaries unavailable and N/A for docs-only Stage `00`.

Decision: previous-stage gate was satisfied before implementation. Required
ClickHouse DDL/query, provider REST and Mac Studio scheduler `/metrics` evidence
remain acceptance requirements.

## File Manifest

Created:

- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/01-funding-storage-and-catchup.md`
- `migrations/clickhouse/funding_rates_ddl.sql`
- `src/trading/contexts/market_data/application/dto/funding.py`
- `src/trading/contexts/market_data/application/ports/sources/funding_rate_history_source.py`
- `src/trading/contexts/market_data/application/ports/sources/funding_instrument_universe_source.py`
- `src/trading/contexts/market_data/application/ports/stores/funding_rate_writer.py`
- `src/trading/contexts/market_data/application/ports/stores/funding_instrument_universe_store.py`
- `src/trading/contexts/market_data/application/use_cases/backfill_funding_rates.py`
- `src/trading/contexts/market_data/application/use_cases/sync_futures_funding_universe.py`
- `src/trading/contexts/market_data/adapters/outbound/clients/funding_rate_history_source.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py`
- `apps/cli/commands/funding_rate_catchup.py`
- `infra/macos/prometheus/rules/market-data-funding.rules.yml`
- `tests/unit/apps/cli/commands/test_funding_rate_catchup_cli.py`
- `tests/unit/apps/scheduler/test_market_data_scheduler_funding.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_funding_rate_store.py`
- `tests/unit/contexts/market_data/adapters/test_funding_rate_history_source.py`
- `tests/unit/contexts/market_data/application/use_cases/test_backfill_funding_rates.py`
- `tests/unit/contexts/market_data/application/use_cases/test_sync_futures_funding_universe.py`

Modified:

- `apps/cli/main/main.py`
- `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`
- `src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py`
- `src/trading/contexts/market_data/application/dto/__init__.py`
- `src/trading/contexts/market_data/application/ports/sources/__init__.py`
- `src/trading/contexts/market_data/application/ports/stores/__init__.py`
- `src/trading/contexts/market_data/application/use_cases/__init__.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/__init__.py`
- `configs/dev/market_data.yaml`
- `configs/test/market_data.yaml`
- `configs/prod/market_data.yaml`
- `infra/macos/prometheus/prometheus.prod.yml`
- `scripts/macos/bootstrap_native_prod.sh`
- `docs/runbooks/market-data-metrics-reference-ru.md`
- `docs/runbooks/market-data-metrics.md`
- `docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md`
- `docs/architecture/README.md`
- `tests/unit/contexts/market_data/adapters/test_market_data_runtime_config.py`
- `tests/unit/infra/test_monitoring_assets.py`

Deleted:

- none

Tests / validation:

- `tests/unit/contexts/market_data/adapters/test_funding_rate_history_source.py`
- `tests/unit/contexts/market_data/adapters/test_clickhouse_funding_rate_store.py`
- `tests/unit/contexts/market_data/adapters/test_market_data_runtime_config.py`
- `tests/unit/contexts/market_data/application/use_cases/test_backfill_funding_rates.py`
- `tests/unit/contexts/market_data/application/use_cases/test_sync_futures_funding_universe.py`
- `tests/unit/apps/cli/commands/test_funding_rate_catchup_cli.py`
- `tests/unit/apps/scheduler/test_market_data_scheduler_funding.py`
- `tests/unit/infra/test_monitoring_assets.py`

## Storage Contract

Implemented as additive ClickHouse schema under `market_data`:

- `funding_instrument_universe` stores the exchange-discovered futures funding
  universe separately from the whitelist-driven candle universe.
- `raw_binance_funding_rates` stores Binance funding history rows with optional
  `mark_price`.
- `raw_bybit_funding_rates` stores Bybit funding history rows with explicit
  `category`, currently `linear` for futures.
- `canonical_funding_rates` stores normalized funding rows keyed by
  `(market_id, symbol, funding_time)`.

The tables use `ReplacingMergeTree` with stable ordering keys, so repeat
catch-up writes for the same `(market_id, symbol, funding_time)` remain
idempotent after ClickHouse replacement/merge. No existing `ref_instruments`
schema or whitelist candle ingestion contract was changed.

## Provider Contract

Implemented via additive market-data ports and REST adapter:

- Binance futures universe comes from `/fapi/v1/exchangeInfo`, filtered to
  tradable perpetual contracts.
- Binance interval metadata is read from `/fapi/v1/fundingInfo`; symbols absent
  from that adjusted-only response use the explicit
  `binance_standard_8h_no_adjustment_row` interval source. A global
  `fundingInfo` failure raises by default and only uses
  `binance_standard_8h_emergency_fallback` when the emergency config flag is
  explicitly enabled.
- Binance history comes from `/fapi/v1/fundingRate`.
- Bybit futures universe comes from `/v5/market/instruments-info` with
  `category=linear` and pagination.
- Bybit history comes from `/v5/market/funding/history` with
  `category=linear`.
- Instruments with missing interval metadata are persisted as degraded universe
  rows and skipped by catch-up instead of being fetched with invented cadence.

## Scheduler Contract

Implemented as an additive `market-data-scheduler` job named
`funding_rate_catchup`.

- The job is configured under `scheduler.jobs.funding_rate_catchup`.
- The funding universe refresh is throttled by
  `universe_refresh_interval_seconds`, so the scheduler does not fetch the full
  futures universe on every wake-up.
- Funding history catch-up runs in strict `due_only` mode over
  exchange-discovered Binance and Bybit futures instruments, currently market
  ids `2` and `4`.
- Non-due instruments are skipped before provider history calls.
- The manual CLI command `funding-rate-catchup` supports universe sync,
  all-due catch-up, single-symbol catch-up, dry-run and JSON/text reports.

## Prometheus And Alerts

Implemented aggregate scheduler metrics without a `symbol` label:

- `scheduler_funding_catchup_instruments_total{exchange,market_type,status}`
- `scheduler_funding_catchup_rows_written_total{exchange,market_type}`
- `scheduler_funding_catchup_lag_seconds{exchange,market_type,status}`
- `scheduler_funding_catchup_last_success_timestamp_seconds{exchange,market_type}`
- `scheduler_funding_catchup_universe_instruments{exchange,market_type,status}`

Added Prometheus rules in
`infra/macos/prometheus/rules/market-data-funding.rules.yml` and installed them
through the native production bootstrap path:

- `MarketDataFundingCatchupErrors`
- `MarketDataFundingNoRecentSuccess`
- `MarketDataFundingLagHigh`
- `MarketDataFundingMissingIntervals`

Updated the market-data metrics runbooks with funding metric meaning, expected
cadence and first-response checks.

## Contract Impact

| Dimension | Impact | Notes |
| --- | --- | --- |
| Public API contract | `none` | No API behavior changed. |
| Port contract | `compatible-change` | New market_data funding source/store ports are additive. |
| DTO schema | `compatible-change` | New market_data funding DTOs are additive. |
| Persisted schema | `compatible-change` | New ClickHouse funding tables are additive. |
| Config schema | `compatible-change` | Optional `scheduler.jobs.funding_rate_catchup` config is additive. |
| Request hash, cache key or persistence identity | `none` | No identity semantics changed. |
| Service-call auth, timeout, retry or error semantics | `compatible-change` | Adds unauthenticated Binance/Bybit public market-data REST calls with existing runtime timeout/retry settings. |
| External side-effect idempotency and unknown-state semantics | `compatible-change` | Funding writes use stable `(market_id, symbol, funding_time)` replacement identity. |
| Logs, metrics, traces, audit, ledger, report or redaction semantics | `compatible-change` | Adds aggregate funding scheduler metrics and stage evidence. No secrets may be recorded. |
| Alert or runbook semantics | `compatible-change` | Adds funding freshness alerts and runbook actions. |
| Browser-visible behavior | `none` | Browser/auth smoke is N/A for this stage and no browser files changed. |

## Validation

Focused implementation tests:

- `uv run pytest -q tests/unit/contexts/market_data/adapters/test_funding_rate_history_source.py tests/unit/contexts/market_data/adapters/test_clickhouse_funding_rate_store.py tests/unit/contexts/market_data/adapters/test_market_data_runtime_config.py tests/unit/contexts/market_data/application/use_cases/test_backfill_funding_rates.py tests/unit/contexts/market_data/application/use_cases/test_sync_futures_funding_universe.py tests/unit/apps/cli/commands/test_funding_rate_catchup_cli.py tests/unit/apps/scheduler/test_market_data_scheduler_funding.py tests/unit/infra/test_monitoring_assets.py`
  - result: passed, `30 passed in 1.28s`.

Required local gates:

- `uv run ruff check src/trading/contexts/market_data apps/cli apps/scheduler tests`
  - result: passed.
- `uv run pyright src/trading/contexts/market_data apps/cli apps/scheduler tests`
  - result: passed, `0 errors, 0 warnings, 0 informations`.
- `uv run pytest -q tests/unit/contexts/market_data tests/unit/apps/cli tests/unit/apps/scheduler tests/unit/infra`
  - result: passed, `163 passed in 2.49s`.
- `uv run python -m tools.docs.generate_docs_index --check`
  - result: passed after final report/ledger edits.
- `git diff --check`
  - result: passed after final report/ledger edits.

## Real-Boundary Evidence

Provider REST boundary:

- Binance `/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1` returned a funding
  row for `BTCUSDT`.
- Bybit `/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=1`
  returned `retCode=0`, `category=linear` and a funding row for `BTCUSDT`.

Mac Studio target-host boundary:

- SSH host `macstudio` is reachable; remote hostname reported
  `MacStudioDaniil`.
- Remote checkout `/Users/daniildegtyarev/Projects/roehub.com` is currently on
  `main`, not this Stage `01` branch.
- ClickHouse HTTP endpoint on Mac Studio is reachable:
  `curl http://127.0.0.1:8123/ping` returned `Ok.`, and
  `SELECT 1 FORMAT TabSeparated` through HTTP stdin returned `1`.
- Scheduler `/metrics` endpoint on Mac Studio is reachable and exposes existing
  `scheduler_` metrics.
- Funding scheduler metrics are not present on Mac Studio yet:
  `grep -E '^scheduler_funding_catchup_'` returned zero lines. This is a
  delivery/runtime-state blocker, not proof that the scheduler service is down.

Superseded diagnostic: earlier Codex-local probes to `127.0.0.1:8123` and
`127.0.0.1:9202` refused connections, but those probes were against the Codex
host, not the Mac Studio target runtime, and are not acceptance evidence.

Decision: Stage `01` is not accepted. It is locally implemented and locally
tested, but blocked until this branch is delivered to Mac Studio target runtime
and the Stage `01` DDL/query plus live `scheduler_funding_catchup_*` metrics are
proven there.

## Cold-Head Review Receipt

Cold-head review: completed  
Mode: cold self-review fallback  
Review scope: Stage `01` funding storage implementation, provider contract,
scheduler contract, Prometheus/runbook artifacts, stage report, stage ledger and
docs index state.  
Review instructions:
`architecture-review/references/cold-head-plan-prompt-pack-review.md`  
Verdict: block acceptance until Mac Studio target-runtime DDL/query and
scheduler `scheduler_funding_catchup_*` metrics are proven.  
Blockers fixed: report no longer treats Codex-local loopback refusal as target
runtime unavailability; Mac Studio path contract is recorded; report status no
longer says accepted or ready; ledger marks Stage `01` blocked; DTO contract
impact corrected to additive compatible-change; business impact and conditional
service-call coverage added.  
Local follow-up check: `uv run python -m tools.docs.generate_docs_index --check`
passed; `git diff --check` passed; Mac Studio ClickHouse and scheduler baseline
probes were run; scheduler funding metric labels were checked for `symbol`
cardinality.  
Residual risks: Stage `01` is not delivered to Mac Studio runtime yet, so DDL
application, canonical query behavior and exported funding metrics remain
unproven there.

## Residual Risks

- Real-boundary acceptance is blocked by missing Stage `01` deployment/runtime
  proof on Mac Studio, not by missing Mac Studio ClickHouse availability.
- ClickHouse DDL/query and scheduler `scheduler_funding_catchup_*` proof on Mac
  Studio are still required before Stage `01` can move from blocked to accepted.
