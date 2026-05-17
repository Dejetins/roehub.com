# Context Snapshot Contract

Дата: 2026-05-18.

Статус: current contract for Assistant v1 Iteration 02B.

## Назначение

`BacktestAiContextSnapshotBuilder` строит backend-owned compact context для одного запроса AI
Configurator. Snapshot передается дальше в prompt builder и не дает модели доступ к filesystem,
market reference или полному universe symbols.

Source of truth для доступных `exchange/market/symbol`, `timeframe` и периодов:

```text
availability_summary.yaml
```

Market reference, UI catalog и exchange APIs не являются источниками доступности для этого
snapshot.

## Ownership

| Surface | Owner |
| --- | --- |
| Snapshot builder | `trading.contexts.backtest.application.ai_configurator` |
| Availability summary port | `BacktestAiAvailabilitySummaryRepository` |
| Filesystem adapter | `backtest.adapters.outbound.ai_configurator_context` |
| Indicator defaults source | `configs/prod/indicators.yaml` |
| Executable signal support | `supported_indicator_ids_for_signals_v1()` |
| Hard definitions | `trading.contexts.indicators.domain.definitions.all_defs()` |

## Snapshot Schema

Top-level:

- `schema_version: 1`
- `source: backtest_ai_context_snapshot_v1`
- `snapshot_hash`
- `summary_hash`
- `summary_generated_at_utc`
- `resolved_symbol`
- `exchange`
- `market_type`
- `instrument_key`
- `ignored_symbols`
- `warnings`
- `allowed_values`
- `period`
- `timeframe_periods`
- `indicators`
- `indicator_audit`
- `provenance`

Model-facing prompt context exposes:

- `allowed_values.symbol` as exactly one-item list with the resolved symbol;
- summary-derived `allowed_values.timeframe`;
- summary-derived top-level period and per-timeframe periods;
- available indicator entries only;
- `ignored_symbols` and warnings for multi-symbol user requests.

It does not expose:

- full symbol universe;
- local `artifact_root`;
- local summary path;
- raw filesystem paths.

## Symbol Rules

One prompt context resolves exactly one symbol.

If the user asks for several symbols, the first resolved symbol is used for config context and
the remaining symbols are recorded in:

```json
{
  "ignored_symbols": ["ETHUSDT"],
  "warnings": ["multiple_symbol_request: using first symbol and recording ignored_symbols"]
}
```

If the first requested symbol is absent from `availability_summary.yaml`, snapshot building fails
closed with `BacktestAiContextSnapshotUnavailable`.

## Indicator Availability

Each prod indicator from `configs/prod/indicators.yaml` is classified as available or excluded.

Available means:

1. indicator exists in `configs/prod/indicators.yaml`;
2. indicator exists in hard definitions;
3. indicator exists in `supported_indicator_ids_for_signals_v1()`;
4. indicator has compute defaults, or has no hard compute axes;
5. selected symbol summary has artifact coverage for at least one backtest timeframe.

Exclusion reasons:

- `missing_hard_definition`;
- `missing_signal_registry`;
- `missing_compute_defaults`;
- `missing_summary_coverage`.

## Axis Contract

Each indicator axis is represented as one of:

- `range`;
- `explicit`;
- `none`.

`structure.percent_rank.window` remains explicit:

```json
{
  "mode": "explicit",
  "values": [10, 14, 20, 28, 42, 56, 84, 126]
}
```

No-window indicators expose:

```json
{
  "window_axis": {
    "mode": "none"
  }
}
```

## Fail-Closed Rules

Snapshot building fails closed when:

- `availability_summary.yaml` is missing;
- YAML payload is not a mapping;
- `schema_version` is not `1`;
- `source` is not `artifact_publisher_active_slot_scan`;
- `instruments` is empty;
- `summary_hash` is missing, malformed, or does not match content;
- selected symbol or timeframe payload is invalid.

## Hash Rules

`snapshot_hash` is deterministic SHA-256 over canonical JSON form of the snapshot payload with
`snapshot_hash` blanked.

`summary_hash` is copied from `availability_summary.yaml` and remains the artifact-state identity.

## Compatibility

This is an additive internal AI prompt-context contract. It does not change `/backtests/jobs`,
conversation API, UI payloads, persisted job request hashes, or artifact slot contracts.
