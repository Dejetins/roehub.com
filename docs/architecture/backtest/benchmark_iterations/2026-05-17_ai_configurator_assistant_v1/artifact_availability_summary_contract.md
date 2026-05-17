# Artifact Availability Summary Contract

Дата: 2026-05-17.

Статус: current contract for Assistant v1 Iteration 02A.

## Назначение

`/opt/roehub/state/backtest_artifacts/v2/availability_summary.yaml` — publisher-owned source of
truth для AI Configurator по реально доступным backtest artifacts.

AI request path не сканирует artifact root и не использует ClickHouse, exchange APIs, market
reference или UI catalog как источник доступных symbols/periods.

## Ownership

| Surface | Owner |
| --- | --- |
| Scanner | `backtest_artifacts` application service |
| Atomic writer | `backtest_artifacts` filesystem outbound adapter |
| Scheduler trigger | `backtest-artifact-publisher` after successful publish cycle |
| Manual trigger | `backtest-artifact-publish --regenerate-summary-only` |
| Consumer | later `BacktestAiContextSnapshotBuilder` |

## Source Inputs

Summary строится только из artifact YAML state:

1. `<root>/<exchange>/<market>/<symbol>/current.yaml`;
2. `<root>/<exchange>/<market>/<symbol>/<active_slot>/manifest.yaml`.

Instrument исключается, если:

- нет `current.yaml`;
- `current.yaml` поврежден или не проходит strict loader;
- active slot отсутствует;
- active `manifest.yaml` отсутствует или поврежден;
- `manifest.identity` не совпадает с `exchange/market/symbol`;
- `manifest.slot_generation` или `manifest.asof_date` не совпадает с `current.yaml`;
- фактический SHA-256 active `manifest.yaml` не совпадает с `current.yaml.manifest_sha256`;
- нет ни одного timeframe с price + mapping + signal manifest coverage.

## YAML Schema

```yaml
schema_version: 1
generated_at_utc: "2026-05-17T00:00:00Z"
artifact_root: "/opt/roehub/state/backtest_artifacts/v2"
artifact_root_schema_version: 2
summary_hash: "<64 hex sha256>"
source: "artifact_publisher_active_slot_scan"
instruments:
  binance/spot/BTCUSDT:
    exchange: "binance"
    market: "spot"
    symbol: "BTCUSDT"
    active_slot: "slot_a"
    slot_generation: 7
    asof_date: "2026-05-02"
    published_at_utc: "2026-05-02T01:36:16Z"
    manifest_sha256: "<64 hex sha256>"
    start_date: "2017-08-17"
    end_date: "2026-05-02"
    backtest_timeframes:
      - "15m"
      - "30m"
      - "1h"
    timeframes:
      15m:
        start_date: "2017-08-17"
        end_date: "2026-05-02"
        bars: 304672
        price_available: true
        signals_available: true
        mappings_available: true
        indicator_ids:
          - "ma.ema"
    hit_times:
      timeframe: "15m"
      available: true
```

## Date Rules

- `timeframes.<tf>.start_date` берется из `prices.<tf>.coverage.open_time_start`.
- `timeframes.<tf>.end_date` берется из `prices.<tf>.coverage.close_time_end`.
- top-level `start_date` по instrument = максимальный `start_date` среди
  `backtest_timeframes`.
- top-level `end_date` по instrument = минимальный `end_date` среди `backtest_timeframes`.
- `1m` может быть price artifact, но не попадает в `backtest_timeframes`, если нет mapping +
  signal coverage.

## Hash Rules

`summary_hash` — deterministic SHA-256 over canonical JSON form of the summary payload without
`summary_hash` and without `generated_at_utc`.

Result:

- repeated generation over identical active artifacts keeps the same `summary_hash`;
- `generated_at_utc` may change without changing `summary_hash`;
- later AI context snapshot can record `summary_hash` as artifact-state identity.

## Write Rules

Writer uses same-root atomic replacement:

```text
availability_summary.yaml.tmp -> availability_summary.yaml
```

The temp file is flushed and fsynced before `os.replace`; the parent directory is fsynced after
rename when the platform allows it.

## Compatibility

This is an additive persisted artifact contract. It does not change current artifact slots,
`current.yaml`, active `manifest.yaml`, `/backtests/jobs`, or request-hash identity.
