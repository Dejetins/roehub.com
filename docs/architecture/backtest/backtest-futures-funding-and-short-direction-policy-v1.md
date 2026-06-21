---
doc_id: backtest-futures-funding-and-short-direction-policy-v1
title: Backtest Futures Funding And Short Direction Policy v1
status: draft-ready-for-implementation
date: 2026-06-21
language: ru
stage_ledger: docs/architecture/backtest/backtest-futures-funding-and-short-direction-policy-v1-stage-reports/backtest-futures-funding-and-short-direction-policy-v1-stage-ledger.md
prompt_pack: .codex/agents/generated/backtest-futures-funding-and-short-direction-policy-v1/
source_plan: external pasted Markdown, not saved in repository
---

# Backtest Futures Funding And Short Direction Policy v1

## Статус и короткий вывод

Документ переводит базовый план в исполнимую архитектуру и stage-gated prompt pack
для Roehub. План принят как отдельная новая линия backtest work. Он не
переоткрывает закрытую линию `backtest-compute-acceleration-v1`: production
baseline остается Stage 05 + Stage 12 composite default
`stage_05_and_12_no_risk`, а funding/short policy реализуется отдельным
пакетом.

`User required before start: nothing`. Для начала Stage `00` не нужны новые
секреты или пользовательские артефакты. Реальные implementation stages должны
использовать только публичные market-data endpoints Binance/Bybit и существующий
prod env contract репозитория.

Главные исправления к базовому плану:

- Bybit external API category для USDT perpetual futures должен быть `linear`,
  не `futures`; внутренний Roehub `market_type=futures` мапится в `linear`.
- Runtime preflight сейчас знает `long_only` и `long_short_reversal`; standalone
  `short` пока не является реальным backtest direction mode и должен быть
  добавлен явно.
- `StrategySpecV1` сейчас не хранит direction; short launch нельзя считать
  полностью проведенным до решения, где direction живет в strategy/run/launch
  metadata.
- Funding не должен заменять `total_return_pct`; для futures jobs добавляется
  net-of-funding family of metrics. Эффективная сортировка futures funding jobs
  по умолчанию становится `total_return_pct_net_of_funding`, при этом gross
  `total_return_pct` сохраняется как отдельный показатель.
- Funding history не должен быть ручным-only процессом: текущий
  `market-data-scheduler` обязан автоматически догонять funding для всех
  exchange-discovered tradable futures pairs Binance и Bybit, а не только для
  whitelist/`ENABLED` инструментов.
- Старые spot jobs с short-enabled semantics остаются читаемыми и неизменными,
  но новые short/long-short backtests и launches разрешены только для futures.

## Проверенный baseline

Код и документы, которые задают текущую границу:

| Область | Текущий факт | Вывод для плана |
| --- | --- | --- |
| Backtest routes | `apps/api/routes/backtests.py` уже имеет jobs, preflight, top, variant, lazy trades, create-strategy endpoints. | Новые поля должны быть additive в текущем job API, без нового parallel API. |
| Runtime defaults/preflight | `src/trading/contexts/backtest/application/services/v2/preflight.py` имеет `direction_modes=("long_only","long_short_reversal")`, без funding config. | Добавить normalized `execution.funding` и standalone `short`. |
| Top rows | `BacktestJobTopVariant` хранит summary-only rows, `trades_json` должен оставаться null. | Funding top adjustment не должен сохранять full trade tape в top rows. |
| Variant hash | `top_result_assembly.py` строит `variant_hash` из canonical params, без funding. | Funding mode/coverage/effective ranking должны входить в identity для новых jobs. |
| Lazy detail | `lazy_trades_detail.py` уже вычисляет exact TP/SL exit and trade detail. | TP/SL funding должен использовать тот же exit path, не дублировать несовместимую логику. |
| Lazy cache key | `BacktestLazyTradesCacheKey` содержит `artifact_manifest_hash`, но не funding hash. | Добавить явный `funding_manifest_hash` или гарантированно включить funding family в root manifest hash; для диагностики нужен отдельный hash. |
| Artifact runtime | `contracts.py` знает `prices`, `signals`, `signal_features`, `mappings`, `hit_times`; artifact-level directions включают `short-only`. | Funding становится новой artifact family; runtime preflight должен догнать artifact contract. |
| Market data | `market_data_ddl.sql` использует `market_id UInt16`: 1 binance spot, 2 binance futures, 3 bybit spot, 4 bybit futures. | Funding DDL должен reuse `market_id UInt16`, не вводить новый exchange id словарь. |
| Reference data | `market-data-reference-data-sync-v2.md` и `SyncWhitelistToRefInstrumentsUseCase` делают `ref_instruments` whitelist-driven; `EnabledInstrumentReader` читает только `status='ENABLED'`. | Funding universe не может зависеть только от текущего whitelist. Stage `01` должен добавить dedicated futures funding universe из exchange metadata или отдельный reader/table, не включая свечной ingestion по всем символам. |
| Market-data scheduler | `apps/scheduler/market_data_scheduler/main/main.py` и `wiring/modules/market_data_scheduler.py` уже запускают `market-data-scheduler` с `/metrics` на `9202`, all-enabled-instruments passes, concurrency и periodic jobs. | Funding должен быть новым scheduler job в этом процессе, не отдельным неуправляемым daemon. |
| Prometheus/runbook | `docs/runbooks/market-data-metrics-reference-ru.md` уже описывает `market-data-scheduler` metrics; `infra/macos/prometheus/prometheus.prod.yml` скрапит `127.0.0.1:9202`. | Funding metrics, alert rules и runbook должны быть добавлены в существующий monitoring baseline. |
| CLI | `apps/cli/main/main.py` dispatches `sync-instruments`, `rest-catchup`, `backfill-1m`, `backtest-artifact-publish`. | Funding catch-up и artifact publish должны быть зарегистрированы в dispatcher. |
| Strategy launch | `apps/api/routes/strategies.py` блокирует только `testnet + spot + short-like`. | Политика должна блокировать spot short-like и в paper, и в testnet, на API уровне. |
| Scenario matrix | `scenario_matrix.py` сейчас считает paper spot short `paper_only`. | CJM меняется: short-like direction требует futures even in paper. |
| Web UI | `backtests.js` default state `market_type="spot"` и `direction="long_short_reversal"`. | Это противоречие нужно исправить: short-like default должен auto-switch/lock futures или default должен стать long-only for spot. |

## Внешние API факты

Проверено по официальной документации 2026-06-21:

- Binance USD-M Futures funding history:
  `GET /fapi/v1/fundingRate`, `limit` default `100`, max `1000`, without
  `startTime/endTime` returns recent `200`, result ascending by
  `fundingTime`; shares `500/5min/IP` limit with funding info.
  Source: <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History>
- Binance funding info:
  `GET /fapi/v1/fundingInfo`, returns adjusted cap/floor and
  `fundingIntervalHours` for symbols with adjusted funding parameters; same
  shared `500/5min/IP` limit.
  Source: <https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-Info>
- Bybit funding history:
  `GET /v5/market/funding/history`, `category` is `linear` or `inverse`,
  `symbol` is required uppercase, `limit` default `200`, max `200`; passing
  only `startTime` returns an error, passing only `endTime` returns records up
  to that end.
  Source: <https://bybit-exchange.github.io/docs/v5/market/history-fund-rate>
- Bybit instruments info:
  `GET /v5/market/instruments-info`, category includes `linear` and `inverse`;
  `linear` universe needs cursor pagination because the default page is 500 and
  the full list can be larger. Response includes `fundingInterval`,
  `upperFundingRate`, `lowerFundingRate`, `settleCoin`.
  Source: <https://bybit-exchange.github.io/docs/v5/market/instrument>
- Bybit mark price kline:
  `GET /v5/market/mark-price-kline`, `symbol` and `interval` required, futures
  max `limit=1000`, list order is reverse by start time.
  Source: <https://bybit-exchange.github.io/docs/v5/market/mark-kline>

## Цели

1. Добавить funding rate history как отдельный market-data type для futures.
2. Добавить funding artifacts в backtest artifact runtime v2.
3. Для futures backtests считать net-of-funding metrics без замены gross
   `total_return_pct`.
4. Учитывать funding после base scoring на final candidate pool/top variants,
   чтобы не раздувать hot path full grid.
5. Для TP/SL variants считать funding по фактическому exit time.
6. Сделать `short` и `long_short_reversal` futures-only для backtest create,
   preflight, strategy launch и UI.
7. Старые persisted spot short-like jobs не мутировать: читать можно, запуск в
   strategy/live запрещен с CTA rerun as futures.

## Не цели

- Не добавлять inverse perpetual funding в v1. Bybit `inverse` и Binance coin-m
  futures остаются отдельным будущим расширением.
- Не использовать funding для spot.
- Не переписывать backtest engine или старую acceleration line.
- Не хранить full trade tape в top-N rows.
- Не вводить private UI-only API.
- Не печатать DSN, API keys, bearer tokens или secret-like values в отчетах.

## Business impact layer

Funding changes the user-facing meaning of futures backtest profitability:
gross `total_return_pct` remains available, but futures ranking and selected
variant detail must expose net-of-funding fields so users do not launch a
strategy from a misleading gross-only result. The short-direction CJM also
changes intentionally: new `short` and `long_short_reversal` work is futures-only
even in paper mode, while historical spot short-like jobs remain readable with a
rerun-as-futures path. Business risk is highest when funding freshness is stale,
so market-data readiness, degraded warnings and Prometheus alerts are part of the
acceptance contract rather than optional operations work.

## Целевое решение

### 1. Funding принадлежит market_data context

Funding ingestion, normalization and coverage lives under
`src/trading/contexts/market_data`. Backtest context consumes funding as
published artifact arrays and readiness summaries. Backtest code не ходит в
exchange REST directly.

Target modules:

- `src/trading/contexts/market_data/application/ports/sources/funding_rate_history_source.py`
- `src/trading/contexts/market_data/application/ports/stores/funding_rate_writer.py`
- `src/trading/contexts/market_data/application/ports/stores/funding_rate_coverage_reader.py`
- `src/trading/contexts/market_data/application/use_cases/backfill_funding_rates.py`
- `src/trading/contexts/market_data/adapters/outbound/clients/funding_rate_history_source.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/funding_rate_store.py`
- `apps/cli/commands/funding_rate_catchup.py`
- `apps/scheduler/market_data_scheduler/wiring/modules/market_data_scheduler.py`
- `src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py`
- `configs/{dev,test,prod}/market_data.yaml`

### 2. ClickHouse schema

Implementation should add an idempotent migration, preferably
`migrations/clickhouse/funding_rates_ddl.sql`, following the existing
`market_data` style and `market_id UInt16` contract.

Logical target schema:

```sql
CREATE TABLE IF NOT EXISTS market_data.ref_futures_funding_instruments
(
    market_id UInt16,
    symbol LowCardinality(String),
    instrument_key String,
    external_category LowCardinality(String), -- binance:futures or bybit:linear semantics
    exchange_status LowCardinality(String),
    is_tradable UInt8,
    settle_coin LowCardinality(Nullable(String)),
    funding_interval_minutes Nullable(UInt16),
    funding_interval_source LowCardinality(String),
    upper_funding_rate Nullable(Float64),
    lower_funding_rate Nullable(Float64),
    updated_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (market_id, symbol);

CREATE TABLE IF NOT EXISTS market_data.raw_binance_funding_rates
(
    market_id UInt16,
    symbol LowCardinality(String),
    instrument_key String,
    funding_time DateTime64(3, 'UTC'),
    funding_time_ms UInt64,
    funding_rate Float64,
    mark_price Nullable(Float64),
    funding_interval_minutes Nullable(UInt16),
    funding_interval_source LowCardinality(String),
    adjusted_cap Nullable(Float64),
    adjusted_floor Nullable(Float64),
    source LowCardinality(String) DEFAULT 'rest',
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3),
    ingest_id Nullable(UUID)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(funding_time)
ORDER BY (market_id, symbol, funding_time);

CREATE TABLE IF NOT EXISTS market_data.raw_bybit_funding_rates
(
    market_id UInt16,
    symbol LowCardinality(String),
    instrument_key String,
    external_category LowCardinality(String), -- v1: linear
    settle_coin LowCardinality(Nullable(String)),
    funding_time DateTime64(3, 'UTC'),
    funding_time_ms UInt64,
    funding_rate Float64,
    mark_price Nullable(Float64),
    funding_interval_minutes Nullable(UInt16),
    funding_interval_source LowCardinality(String),
    upper_funding_rate Nullable(Float64),
    lower_funding_rate Nullable(Float64),
    source LowCardinality(String) DEFAULT 'rest',
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3),
    ingest_id Nullable(UUID)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(funding_time)
ORDER BY (market_id, symbol, funding_time);

CREATE TABLE IF NOT EXISTS market_data.canonical_funding_rates
(
    market_id UInt16,
    symbol LowCardinality(String),
    instrument_key String,
    funding_time DateTime64(3, 'UTC'),
    funding_time_ms UInt64,
    funding_rate Float64,
    mark_price Nullable(Float64),
    funding_interval_minutes Nullable(UInt16),
    funding_interval_source LowCardinality(String),
    data_quality UInt8,
    source LowCardinality(String),
    ingested_at DateTime64(3, 'UTC'),
    ingest_id Nullable(UUID)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMM(funding_time)
ORDER BY (market_id, symbol, funding_time);
```

Materialized views may populate `canonical_funding_rates` from both raw tables,
but they are optional if the repository pattern favors explicit store writes.
The key requirement is one canonical read path for artifact publishing and
coverage checks.

### 3. Funding catch-up и automatic scheduler mode

Manual `funding-rate-catchup` remains necessary for bounded repair and operator
work, but production freshness is owned by the existing `market-data-scheduler`
process on port `9202`.

Runtime topology:

- Reuse `apps/scheduler/market_data_scheduler/main/main.py`.
- Add scheduler job config:

```yaml
market_data:
  scheduler:
    jobs:
      funding_rate_catchup:
        interval_seconds: 1800
        due_mode: funding_interval_aligned
        startup_bootstrap_enabled: true
        settlement_lag_minutes: 10
        binance_standard_interval_hours: 8
        allow_interval_metadata_failure_fallback: false
        tail_lookback_hours: 24
        historical_gap_audit_hours: 24
```

- Add `funding_rate_catchup` to the periodic scheduler job list after
  `sync_whitelist` and `enrich`.
- `interval_seconds` is the scheduler wake-up cadence, not the funding download
  cadence. Default `1800` means "check due work every 30 minutes"; it must not
  download the full futures universe every minute or every wake.
- Actual provider downloads are funding-interval aligned per symbol:
  - funding interval metadata is mandatory for due calculation;
  - Binance must call `/fapi/v1/fundingInfo`; if a symbol has an adjusted row,
    use `fundingIntervalHours`; if a tradable USD-M symbol is absent from that
    adjusted-only response, use the exchange standard `8h` interval and persist
    `funding_interval_source='binance_standard_8h_no_adjustment_row'`;
  - if Binance `fundingInfo` fails globally, do not silently claim Binance
    funding readiness from the `8h` default; mark the run degraded/failed via
    metrics and stage evidence unless an explicit emergency fallback is enabled;
  - Bybit must use instruments-info `fundingInterval` for each `linear`
    instrument; missing interval metadata for a tradable symbol marks that symbol
    degraded/skipped until metadata is available;
  - compute due work as
    `next_funding_time = last_funding_time + funding_interval`; fetch only when
    `now >= next_funding_time + settlement_lag_minutes`;
  - a symbol that is not due is skipped without provider funding-history call.
- The job first refreshes a dedicated funding universe for all tradable futures
  instruments from exchange metadata:
  - Binance USD-M futures from `/fapi/v1/exchangeInfo`;
  - Bybit v1 scope from `/v5/market/instruments-info?category=linear`;
  - persist the snapshot in `ref_futures_funding_instruments` or an equivalent
    dedicated table/reader that does not overload `EnabledInstrumentReader`.
- The catch-up job enumerates all rows in that funding universe where:
  - `market_type=futures` via `market_id in (2, 4)`;
  - `is_tradable=1`;
  - exchange is Binance or Bybit.
- Do not mark every exchange-discovered futures pair as `status='ENABLED'` in
  `ref_instruments`, because that would unintentionally expand candle WS/REST
  ingestion and backtest symbol universe.
- The job must not be BTCUSDT-only and must not depend on current UI selected
  symbol.
- The job uses the same use case as manual CLI, with a `run_mode`:
  `startup_bootstrap | tail | gap_audit | repair_symbol`.
- Bootstrap mode fills from symbol/provider funding start or market configured
  lower bound to the latest settled funding event whose timestamp is at least
  `settlement_lag_minutes` behind `now`.
- Tail mode re-reads a safety lookback window, default `24h`, to tolerate late
  provider corrections and idempotently replace rows.
- Gap-audit mode scans a bounded historical window, default last `24h`, and can
  be widened manually by CLI if an incident requires broader repair.
- Normal steady-state behavior is therefore roughly one funding-history fetch per
  symbol per funding interval, usually once per `8h`, plus bounded tail repair.
  The scheduler wake-up is intentionally more frequent only to avoid missing
  provider-specific interval changes and late publication.

Manual `funding-rate-catchup` must support:

- single instrument: `--exchange`, `--market-type futures`, `--symbol`,
  `--start`, `--end`;
- all exchange-discovered tradable Binance/Bybit futures instruments from the
  dedicated funding universe;
- scheduler-equivalent mode for `--all-futures-from-funding-universe`;
- dry-run/summary mode;
- idempotent re-run over the same window;
- provider rate-limit backoff;
- provider-specific cursor/window rules.

Bybit mapping is explicit:

| Roehub field | Bybit API field |
| --- | --- |
| `market_type=futures` | `category=linear` for v1 |
| `symbol=BTCUSDT` | `symbol=BTCUSDT` |
| `start/end` window | cannot call with only `startTime`; use bounded windows or only `endTime` bootstrap |

### 4. Prometheus metrics, alerts and runbook

Funding freshness is an operational contract. Stage `01` must add scheduler
metrics to `MarketDataSchedulerMetrics` and document them in
`docs/runbooks/market-data-metrics-reference-ru.md`.

Required metrics on `market-data-scheduler`:

| Metric | Type | Labels | Purpose |
| --- | --- | --- | --- |
| `scheduler_funding_catchup_instruments_total` | Counter | `exchange`, `market_type`, `status` | Instruments processed by automatic or manual funding catch-up. |
| `scheduler_funding_catchup_events_written_total` | Counter | `exchange`, `market_type` | Canonical funding events written. |
| `scheduler_funding_catchup_windows_total` | Counter | `exchange`, `market_type`, `status` | Provider request windows completed, failed, skipped or rate-limited. |
| `scheduler_funding_catchup_provider_requests_total` | Counter | `exchange`, `status` | Provider REST requests by final status. |
| `scheduler_funding_catchup_rate_limit_wait_seconds` | Histogram | `exchange` | Provider throttle/backoff wait time. |
| `scheduler_funding_catchup_last_success_unixtime` | Gauge | `exchange`, `market_type` | Last successful automatic funding catch-up. |
| `scheduler_funding_catchup_oldest_lag_seconds` | Gauge | `exchange`, `market_type` | Oldest lag across exchange-discovered tradable futures symbols. |
| `scheduler_funding_catchup_missing_instruments` | Gauge | `exchange`, `market_type` | Count of exchange-discovered tradable futures symbols with no canonical funding rows. |
| `scheduler_funding_catchup_degraded_instruments` | Gauge | `exchange`, `market_type` | Count of symbols with partial or stale coverage. |

Cardinality rule: do not use `symbol` as a Prometheus label for funding
freshness. Per-symbol diagnostics belong in structured logs and ClickHouse
queries; Prometheus carries exchange/market aggregate status.

Required alert/runbook artifacts:

- `infra/macos/prometheus/rules/market-data-funding.rules.yml`;
- include that rule file in `infra/macos/prometheus/prometheus.prod.yml`;
- update `tests/unit/infra/test_monitoring_assets.py`;
- update `docs/runbooks/market-data-metrics-reference-ru.md`;
- update `docs/runbooks/market-data-metrics.md` if it is the operator-facing
  summary.

Alert rules:

| Alert | Severity | Expression intent |
| --- | --- | --- |
| `MarketDataFundingCatchupErrorsGrowing` | warning | `scheduler_job_errors_total{job="funding_rate_catchup"}` or failed funding windows increase over 15m. |
| `MarketDataFundingNoRecentSuccess` | warning | No successful funding catch-up per exchange/market for more than the configured freshness window, default `10h` for 8h funding plus settlement/backoff tolerance. |
| `MarketDataFundingLagHigh` | warning | `scheduler_funding_catchup_oldest_lag_seconds` exceeds the configured lag threshold, default `12h` for Binance/Bybit futures. |
| `MarketDataFundingMissingInstruments` | warning | Missing futures funding instruments stay non-zero for 30m after bootstrap. |
| `MarketDataFundingIntervalMetadataDegraded` | warning | Interval metadata degraded/skipped instruments stay non-zero for 30m; this includes Bybit missing `fundingInterval` or Binance `fundingInfo` endpoint failure without explicit emergency fallback. |

Runtime proof before Stage `01` acceptance:

```bash
curl -fsS http://127.0.0.1:9202/metrics | rg '^scheduler_funding_catchup_'
```

The proof must show metrics names and non-secret label values only.

Service calls, retry and idempotency:

| Caller | Callee | Auth | Timeout/retry | Idempotency / unknown state | Failure behavior |
| --- | --- | --- | --- | --- | --- |
| `market-data-scheduler` `funding_rate_catchup` | Binance `/fapi/v1/exchangeInfo`, required `GET /fapi/v1/fundingInfo`, `GET /fapi/v1/fundingRate` | none, public market-data | Use per-market `rest.timeout_s`, `rest.retries`, exponential backoff and provider rate-limit waits from `market_data.yaml`. | Funding universe dedupes by `(market_id, symbol)`; funding events dedupe by `(market_id, symbol, funding_time_ms, source)`; unknown response/write state is repaired by rerunning the same closed window because writes are idempotent. | Per-symbol failure increments failed metrics, logs summarized provider status, continues other symbols. Global `fundingInfo` failure degrades/fails Binance readiness rather than silently using 8h for all symbols. |
| `market-data-scheduler` `funding_rate_catchup` | Bybit `/v5/market/instruments-info?category=linear` and `GET /v5/market/funding/history` | none, public market-data | Same timeout/backoff; Bybit calls must use `category=linear` for v1 and bounded `startTime+endTime` windows. | Same universe/event dedupe; never call Bybit with only `startTime`; rerun bounded windows after unknown state. | Per-symbol failure increments failed metrics, logs summarized `retCode`, continues other symbols. |
| Funding use case | ClickHouse raw/canonical funding store | ClickHouse env config; do not print secrets | Bounded insert batches; retry only through idempotent rerun path, not blind infinite loops. | `ReplacingMergeTree(ingested_at)` plus stable `(market_id, symbol, funding_time)` ordering makes window replays safe. | Store errors mark instrument failed for this cycle and raise scheduler job error if systemic. |
| Prometheus | `market-data-scheduler` `/metrics` on `9202` | local scrape only | scrape interval from Prometheus config | metrics are observational only; no side effects | Down scrape is handled by existing `up{job="market-data-scheduler"}` plus funding-specific stale alerts. |

### 5. Funding artifact family

Funding is a new artifact family under the same published root as prices,
signals, mappings and hit-times. Suggested logical layout:

```text
funding/
  futures/
    <exchange>/
      <symbol>/
        funding_time.i64.npy
        funding_rate.f64.npy
        mark_price.f64.npy
        interval_minutes.u16.npy
        data_quality.u8.npy
        manifest.yaml
```

Root manifest rules:

- `funding_manifest_hash` is recorded per instrument/timeframe/window.
- Root `artifact_manifest_hash` changes when funding artifacts used by a job
  change.
- Lazy detail cache key stores `funding_manifest_hash` explicitly for
  observability and invalidation.
- Partial funding coverage is allowed only with `coverage_status=degraded` and
  warning codes in preflight/job/result.
- Missing full funding family for futures can be degraded for backtest creation,
  but launch/readiness UI must show that net metrics are degraded.

### 6. Request and preflight contract

Add normalized execution funding config:

```json
{
  "execution": {
    "direction_mode": "long_only | short | long_short_reversal",
    "funding": {
      "mode": "include_when_futures | off",
      "coverage_policy": "degraded_with_warning"
    }
  }
}
```

Defaults:

| Market type | Direction | Funding default |
| --- | --- | --- |
| `spot` | `long_only` | `off` |
| `spot` | `short` / `long_short_reversal` | invalid |
| `futures` | `long_only` | `include_when_futures` |
| `futures` | `short` | `include_when_futures` |
| `futures` | `long_short_reversal` | `include_when_futures` |

`GET /api/backtests/runtime-defaults` should expose
`direction_market_compatibility` and funding default metadata so browser code
does not duplicate hard-coded policy.

`POST /api/backtests/preflight` should return:

- `funding_readiness.status`: `ready | degraded | unavailable | not_applicable`;
- `funding_readiness.warning_codes`;
- `funding_readiness.coverage_ratio`;
- `funding_readiness.window`;
- `funding_readiness.funding_manifest_hash` when artifact exists;
- `direction_market_compatibility` for selected state.

Request hash impact: new jobs include normalized funding config in the hash.
Existing jobs are immutable and not rehashed.

### 6. Funding calculation

For every open position and every funding event where
`entry_time < funding_time <= exit_time`:

```python
side_sign = 1.0 if direction == "long" else -1.0
notional_quote = qty_base * mark_price
funding_pnl_quote = -side_sign * notional_quote * funding_rate
```

Positive `funding_pnl_quote` means strategy receives funding. This matches the
linear perpetual convention where positive funding rate means longs pay shorts.

Required edge cases:

- No funding events inside trade window: funding PnL is `0`.
- Missing `mark_price`: fallback to candle close only if explicitly recorded in
  `data_quality` and warning codes.
- Reversal trades are split by actual long/short segments before applying
  funding.
- TP/SL variants must use the exact exit resolved by the existing detail path,
  including same-bar TP/SL precedence.
- Funding is calculated in quote currency and converted to return using the same
  capital base as existing return metrics.

### 7. Candidate pool and ranking

Funding adjustment is applied after base scoring on a bounded candidate pool:

```python
funding_adjustment_candidate_pool_size = max(top_n * 5, top_n + 100)
```

Persist metadata:

- `funding_adjustment_scope = "candidate_pool"`;
- `funding_adjustment_candidate_pool_size`;
- `funding_adjustment_exact_global_ranking = false`;
- `requested_ranking_metric`;
- `effective_ranking_metric`.

Decision for v1: when a futures job includes funding and the requested ranking
metric is gross `total_return_pct`, the effective ranking metric becomes
`total_return_pct_net_of_funding`. Gross `total_return_pct` is preserved in
summary metrics and UI columns. This avoids a misleading futures top-N where the
visible best strategy is materially worse after funding.

### 8. Result API and UI fields

Add summary metrics:

- `total_return_pct_net_of_funding`;
- `funding_return_pct`;
- `funding_pnl_quote`;
- `funding_events_count`;
- `funding_data_quality`;
- `funding_warning_codes`;
- `funding_included`;
- `funding_adjustment_scope`.

Lazy trade rows add:

- `funding_pnl_quote`;
- `funding_return_pct`;
- `funding_events_count`;
- `funding_data_quality`;
- optional `funding_events` for chart overlay.

Browser CJM:

- Backtest form cannot submit `spot + short`.
- Backtest form cannot submit `spot + long_short_reversal`.
- Selecting short-like direction auto-switches market type to futures or blocks
  spot with server-provided reason.
- Results table shows gross return and net-of-funding return side by side.
- Launch modal for old spot short-like jobs blocks strategy launch and shows
  rerun-as-futures action.
- Funding degraded warning is visible before create, in job summary and in
  selected variant detail.

### 9. Strategy launch policy

API validation must reject any launch where:

```python
direction in {"short", "long_short_reversal"} and market_type != "futures"
```

Reason code:

```text
short_direction_requires_futures_market
```

This replaces the narrower current policy that blocks only
`testnet + spot + short-like`. The same rule applies to paper and testnet.

Direction storage decision for v1:

- do not make a breaking change to `StrategySpecV1`;
- persist backtest-derived direction in launch/run metadata and read models;
- ensure live/profile/run code consumes the same direction metadata that API
  validation checks;
- if a later stage needs library-visible direction in `StrategySpecV1`, introduce
  it as additive metadata with explicit compatibility tests.

## Contract impact classification

| Dimension | Classification | Notes |
| --- | --- | --- |
| Public backtest API response | `compatible-change` | Additive funding fields and readiness metadata. |
| Public backtest API request | `compatible-change` with identity change | New optional `execution.funding`; normalized request hash changes for new funding-aware jobs. |
| Direction enum | `compatible-change` for tolerant clients, possible `breaking-change` for strict clients | Adds standalone `short` and rejects old spot short-like submissions. |
| Strategy launch API | `breaking-change` intentional | `paper + spot + short-like` changes from allowed/paper-only to rejected. |
| Browser CJM | `breaking-change` intentional | Existing spot short-like path becomes rerun-only. |
| ClickHouse schema | `compatible-change` | New funding tables/MVs. |
| Funding instrument universe | `compatible-change` | Adds dedicated all-futures funding universe; must not silently convert all exchange symbols into `ENABLED` candle ingestion instruments. |
| Market-data scheduler config | `compatible-change` | Adds optional `scheduler.jobs.funding_rate_catchup`; production config should enable it explicitly. |
| Prometheus metrics | `compatible-change` | Adds scheduler funding series; avoid symbol labels to control cardinality. |
| Alert/runbook semantics | `compatible-change` | Adds funding freshness alerts and operator actions. |
| Postgres persistence | `compatible-change` if summary JSON only; `unknown` until lazy cache schema checked | Adding cache key fields may require migration. |
| Artifact manifest | `compatible-change` with cache invalidation | New funding family changes manifest hashes for futures jobs. |
| Exchange service calls | `compatible-change` | New public unauthenticated market-data calls with rate limits. |
| Performance | `unknown` until benchmarked | Funding is post-pool only; acceptance requires evidence. |

## Rollout stages

The generated prompt pack implements this order:

| Stage | Name | Outcome |
| --- | --- | --- |
| `00` | Review baseline and freeze contract | Re-read plan, code boundaries, docs, API facts; record exact file manifest and stage ledger entry before code. |
| `01` | Funding storage, automatic catch-up and observability | ClickHouse funding schema, dedicated all-futures funding universe, sources, writer, manual CLI, `market-data-scheduler` automatic all-futures catch-up, Prometheus metrics, alerts, runbook and real-boundary proof. |
| `02` | Funding artifact family and coverage | Publish/load funding arrays from automatically maintained `canonical_funding_rates`, manifest hash, coverage reader, artifact tests. |
| `03` | Preflight/runtime defaults funding readiness | Normalize funding request, direction compatibility, readiness metadata. |
| `04` | No-risk funding adjustment | Compute net-of-funding candidate-pool metrics for no-risk variants. |
| `05` | TP/SL funding adjustment | Reuse exact TP/SL exit logic for funding-aware variant detail and top adjustment. |
| `06` | Result API, lazy detail and persistence | Add summary/lazy fields, cache key semantics, DTO/read model tests. |
| `07` | Futures-only short policy API and browser CJM | Enforce short-like futures-only in API, scenario matrix and web UI. |
| `08` | Final verification and delivery | Run full gates, browser/runtime evidence, docs ledger/index update, publish only if explicitly executing delivery. |

## Required validation ladder

Focused stages should run the narrowest meaningful tests first, then broaden when
touching shared contracts:

- `uv run ruff check <touched targets>`;
- `uv run pyright <touched targets>`;
- `uv run pytest -q <focused tests>`;
- `python -m tools.docs.generate_docs_index --check` when docs changed;
- `curl -fsS http://127.0.0.1:9202/metrics | rg '^scheduler_funding_catchup_'`
  for Stage `01` runtime observability proof;
- benchmark/performance evidence for stages `04` and `05`;
- browser QA evidence for stage `07`;
- pre-ship gate and direct delivery proof for stage `08`.

Funding correctness fixtures must cover:

- long pays positive funding;
- short receives positive funding;
- negative funding flips receiver/payer;
- funding at entry timestamp excluded;
- funding at exit timestamp included;
- missing funding window returns degraded warning, not hard failure;
- Bybit `market_type=futures` maps to `category=linear`;
- old spot short-like job is readable but launch-blocked.

## Risk register

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Net ranking changes top-N semantics for futures | High | Persist requested/effective ranking metrics and show gross/net columns. |
| Candidate-pool adjustment can miss true global net winner | Medium | Persist `exact_global_ranking=false`; benchmark pool size; do not claim full-grid exactness. |
| TP/SL funding duplicates exit logic incorrectly | High | Reuse existing exact detail service or extract shared exit resolver before applying funding. |
| Bybit API category sent as `futures` | High | Adapter-level mapping tests for `futures -> linear`. |
| Funding stays manual-only and silently goes stale | High | Stage `01` must extend `market-data-scheduler` for all exchange-discovered tradable futures instruments and expose freshness metrics/alerts. |
| Funding universe accidentally expands candle ingestion to every exchange futures symbol | High | Use dedicated funding universe table/reader; do not mark all exchange futures as `ENABLED` in existing whitelist-driven `ref_instruments`. |
| Funding interval metadata is silently missing | High | Bybit `fundingInterval` is required per symbol; Binance `fundingInfo` is required globally, with explicit `binance_standard_8h_no_adjustment_row` only for symbols absent from the adjusted-only response. Missing metadata increments degraded metrics and blocks readiness claims. |
| Prometheus per-symbol funding labels explode cardinality | Medium | Metrics aggregate by exchange/market/status; per-symbol details stay in logs/ClickHouse. |
| Strategy direction not stored where live code consumes it | High | Stage `07` must trace direction from backtest snapshot to run/launch metadata and live readiness. |
| Funding degraded warnings hidden in UI | Medium | Preflight, result summary and launch modal all expose warning codes. |
| Provider rate limits slow all-symbol catch-up | Medium | Bounded windows, checkpointing, dry-run, backoff and resumable idempotent writes. |
| Docs/prompt drift from old acceleration line | Medium | Separate stage ledger and prompt pack; do not edit old closed plan except references if needed. |

## Cold-head review result

Cold-head review: completed  
Mode: cold self-review fallback  
Verdict: Release after fixes  
Blockers fixed: Bybit category corrected from raw `futures` to `linear`; standalone `short` runtime gap called out; StrategySpec direction gap called out; browser default `spot + long_short_reversal` contradiction called out; net ranking ambiguity resolved as explicit v1 decision; automatic all-futures funding scheduler mode added; dedicated funding universe added so whitelist does not limit coverage; mandatory interval metadata contract added; Prometheus funding metrics, alerts and runbook coverage added.  
Residual risks: performance impact of funding post-pool adjustment; exact global net ranking remains approximate; live direction storage must be verified in Stage `07`; Stage `01` must prove exchange-discovered all-futures scheduler enumeration and `/metrics` output; provider API behavior may drift and must be rechecked before implementation.
