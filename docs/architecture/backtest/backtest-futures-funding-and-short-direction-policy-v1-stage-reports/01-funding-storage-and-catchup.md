# Backtest Futures Funding And Short Direction Policy v1 - Stage 01 Funding Storage And Catchup

Funding-rate storage, automatic catch-up and observability implementation
record.

Date: 2026-06-22

Status: blocked on post-main production runtime proof; implementation is on
`main` and backend deploy workflow is green, but Mac Studio SSH/Tailscale
reachability is currently unavailable for the final runtime proof.

Branch: `codex/backtest-futures-funding-v1`

Delivered main revision: `d14050b235807d60ae1d8cbf951bb651e40f1f45`

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
- record local validation, Mac Studio branch-worktree sync and read-only
  existing runtime state without promoting the stage to accepted.

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

Conditional service-call coverage and proof boundary:

- provider REST: applicable and partially proven by Binance and Bybit public
  funding-history smoke checks;
- `target_host_readiness_pre_main`: Mac Studio SSH, primary checkout, prompt-pack
  branch worktree, ClickHouse ping/query and scheduler `/metrics` are reachable;
- `read_only_existing_runtime_smoke`: the currently deployed `/opt/roehub/app`
  exposes API auth `401`, ClickHouse funding tables exist, scheduler job
  counters show one `funding_rate_catchup` run and one error, and no
  `scheduler_funding_catchup_*` sample lines are exported;
- `post_main_production_runtime_proof`: still blocked. The target revision is on
  `main`, GitHub Actions/CI is green and backend deploy workflow is green, but
  direct Mac Studio SSH/Tailscale reachability is unavailable from the operator
  machine, so checkout parity, ClickHouse counts, scheduler metrics and live
  funding writes cannot yet be verified;
- browser/API route evidence: N/A for this stage because no browser-visible API
  or UI contract is changed.

## Previous-Stage Gate

The Stage `01` prompt requires Stage `00` to be `accepted` in the stage
execution ledger before implementation starts.

Observed ledger state after the acceptance repair:

- Stage `00` status: `accepted`.
- Stage `00` report status: `accepted, docs-only baseline evidence`.
- Delivery ledger for Stage `00`: evidence commit
  `7dc0e726fc6babe8c101369a40a4119d5d23fd03` is retained in the unified
  prompt-pack branch history; historical
  `origin/codex/backtest-futures-funding-v1-stage-00` is superseded and not the
  branch model for later stages; runtime boundaries unavailable and N/A for
  docs-only Stage `00`.

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
  - result: passed, `164 passed in 2.89s` after the Bybit interval regression
    test was added.
- `uv run python -m tools.docs.generate_docs_index --check`
  - result: passed.

Additional follow-up checks after the Mac Studio runtime failure was localized:

- `uv run ruff check src/trading/contexts/market_data/adapters/outbound/clients/funding_rate_history_source.py tests/unit/contexts/market_data/adapters/test_funding_rate_history_source.py`
  - result: passed.
- `uv run pyright src/trading/contexts/market_data/adapters/outbound/clients/funding_rate_history_source.py tests/unit/contexts/market_data/adapters/test_funding_rate_history_source.py`
  - result: passed, `0 errors, 0 warnings, 0 informations`.
- `uv run pytest -q tests/unit/contexts/market_data/adapters/test_funding_rate_history_source.py`
  - result: passed, `4 passed`.
- `/usr/bin/python3 .codex/hooks/tests/run_tests.py`
  - result: passed, all `45` hook fixtures.
- `uv run ruff check .codex/hooks/validators/runtime_proof_boundary_guard.py`
  - result: passed.

Post-main production fix and delivery checks:

- `uv run ruff check .`
  - result: passed.
- `uv run pyright`
  - result: passed, `0 errors, 0 warnings, 0 informations`.
- `uv run pytest -q -ra`
  - result: passed, `1283 passed, 3 warnings in 56.21s`.
- `uv run python -m tools.docs.generate_docs_index --check`
  - result: passed.
- `git diff --check`
  - result: passed.
- GitHub Actions CI for `main` commit
  `d14050b235807d60ae1d8cbf951bb651e40f1f45`
  - result: passed, run `27944489784`.
- GitHub Actions Deploy Backend after that CI
  - result: passed, run `27944549900`.
- GitHub Actions Deploy Web and Publish App Image after that CI
  - result: passed, runs `27944549893` and `27944549896`.

## Real-Boundary Evidence

Provider REST boundary:

- Binance `/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1` returned a funding
  row for `BTCUSDT`.
- Bybit `/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=1`
  returned `retCode=0`, `category=linear` and a funding row for `BTCUSDT`.

Mac Studio `target_host_readiness_pre_main` boundary:

- SSH host `macstudio` is reachable when the local shell uses
  `SSH_AUTH_SOCK=$(launchctl getenv SSH_AUTH_SOCK)`.
- Primary checkout `/Users/daniildegtyarev/Projects/roehub.com` is clean on
  `main` at `d85ef43ccd4bc3b1717d396bae1bdcaf65d48b79`.
- Prompt-pack branch worktree exists at
  `/Users/daniildegtyarev/Projects/roehub-worktrees/codex__backtest-futures-funding-v1`,
  is clean on `codex/backtest-futures-funding-v1`, and is synced to
  `f94c8fa4a197626d45b3f2190d229d5cd9f9544f`.
- ClickHouse HTTP endpoint on Mac Studio is reachable:
  `curl http://127.0.0.1:8123/ping` returned `Ok.`.
- ClickHouse funding tables exist on the current runtime:
  `funding_tables = 4`; `canonical_funding_rates` currently has
  `canonical_count = 0`.
- Scheduler `/metrics` endpoint on Mac Studio is reachable. Existing runtime
  counters show `scheduler_job_runs_total{job="funding_rate_catchup"} 1.0` and
  `scheduler_job_errors_total{job="funding_rate_catchup"} 1.0`.
- `grep -E '^scheduler_funding_catchup_'` returned zero sample lines on the
  current runtime.

Earlier Mac Studio `read_only_existing_runtime_smoke` finding before the
`d14050b2` main delivery:

- `/opt/roehub/app` matches the branch for scheduler wiring and funding DDL, but
  differs for
  `src/trading/contexts/market_data/adapters/outbound/clients/funding_rate_history_source.py`,
  so the `f94c8fa4` Bybit non-positive interval fix is not deployed there.
- The scheduler log at that time still contained
  `ValueError: FundingInstrument.funding_interval_minutes must be a positive integer`
  from `funding_rate_catchup`.

Post-main production runtime proof attempt:

- The Bybit non-positive interval parser failure was fixed and delivered to
  `main` in `d14050b235807d60ae1d8cbf951bb651e40f1f45`.
- A follow-up ClickHouse runtime failure was localized to
  `ILLEGAL_AGGREGATION` in
  `ClickHouseFundingRateStore.list_tradable_funding_instruments`: the query used
  `max(updated_at) AS updated_at` while other aggregate expressions referenced
  `updated_at`. The fix aliases the aggregate as `latest_updated_at` and adds a
  regression test.
- GitHub CI and backend deploy are green for the fixed `main` revision.
- Direct proof collection on Mac Studio is blocked at the host-reachability
  layer: `ssh -o ConnectTimeout=10 macstudio true` times out to
  `100.74.213.43`; `ping -c 3 100.74.213.43` has `100%` packet loss;
  `tailscale status` shows the local client cannot synchronize with the
  coordination server and peer reachability may degrade; `tailscale ping
  macstudio-daniil` times out.
- Because SSH is unavailable, this report does not claim checkout parity,
  `/opt/roehub/app` file parity, ClickHouse canonical row counts, scheduler
  `scheduler_funding_catchup_*` samples or absence of fresh scheduler tracebacks
  for `d14050b2`.

Superseded diagnostic: earlier Codex-local probes to `127.0.0.1:8123` and
`127.0.0.1:9202` refused connections, but those probes were against the Codex
host, not the Mac Studio target runtime, and are not acceptance evidence.

Decision: Stage `01` is not accepted. It is implemented, locally tested, pushed
to `origin/main` and deployed by the backend workflow. Acceptance remains
blocked until Mac Studio reachability is restored and
`post_main_production_runtime_proof` proves live
`scheduler_funding_catchup_*` samples plus canonical funding writes from the
deployed `main` revision.

## Cold-Head Review Receipt

Cold-head review: completed
Mode: cold self-review fallback
Review scope: Stage `01` funding storage implementation, provider contract,
scheduler contract, Prometheus/runbook artifacts, stage report, stage ledger and
docs index state.
Review instructions:
`architecture-review/references/cold-head-plan-prompt-pack-review.md`
Verdict: Block. Stage `01` acceptance must wait for
`post_main_production_runtime_proof`: Mac Studio reachability, checkout/runtime
parity for `d14050b235807d60ae1d8cbf951bb651e40f1f45`, then runtime smoke with
live `scheduler_funding_catchup_*` samples and canonical funding rows.
Blockers fixed: report no longer treats Codex-local loopback refusal as target
runtime unavailability; Mac Studio path contract is recorded; report status no
longer says accepted or ready; ledger marks Stage `01` blocked; DTO contract
impact corrected to additive compatible-change; business impact and conditional
service-call coverage added; Bybit non-positive `fundingInterval` runtime
failure fixed in branch commit `f94c8fa4`.
Local follow-up check: `uv run python -m tools.docs.generate_docs_index --check`
passed; required Stage `01` ruff/pyright/pytest gates passed; full post-main
ruff/pyright/pytest/docs-index/diff-check passed; GitHub CI and backend deploy
passed for `d14050b2`.
Residual risks: direct Mac Studio proof is blocked by current SSH/Tailscale peer
reachability; runtime may be deployed by workflow, but this report does not
claim funding catch-up success until ClickHouse, scheduler metrics and logs are
checked through Mac Studio.

## Residual Risks

- Real-boundary acceptance is blocked by missing direct
  `post_main_production_runtime_proof`, not by missing local code, CI or deploy
  workflow evidence.
- The target revision is on `main` at
  `d14050b235807d60ae1d8cbf951bb651e40f1f45`, but Mac Studio SSH is currently
  unreachable from the operator machine; do not call the green deploy workflow
  alone changed-code runtime proof for funding catch-up.
- Scheduler `scheduler_funding_catchup_*` samples and canonical funding rows are
  still required before Stage `01` can move from blocked to accepted.
