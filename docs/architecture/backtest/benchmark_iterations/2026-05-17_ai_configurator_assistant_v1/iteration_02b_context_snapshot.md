# Iteration 02B Context snapshot

Дата: 2026-05-18.

Статус: local implementation complete, delivery pending.

## Цель

Сделать backend-owned context snapshot для AI Configurator, который строится из
`availability_summary.yaml`, indicator catalog, executable signal registry, hard definitions,
runtime defaults и AI config limits.

## Что изменено

- Добавлен `BacktestAiContextSnapshotBuilder`.
- Добавлен port `BacktestAiAvailabilitySummaryRepository`.
- Добавлен filesystem adapter `FilesystemBacktestAiAvailabilitySummaryRepository`.
- Добавлены DTO:
  - `BacktestAiContextSnapshot`;
  - `BacktestAiContextAxis`;
  - `BacktestAiIndicatorAvailability`.
- В `configs/prod/backtest_ai_configurator.yaml` добавлен блок `context_snapshot`.
- Snapshot строится для одного resolved symbol.
- Multiple-symbol request записывает первый symbol как resolved и остальные как
  `ignored_symbols`.
- Snapshot валидирует `availability_summary.yaml` fail-closed, включая `summary_hash`.
- Model-facing context exposes `allowed_values.symbol` как один resolved symbol, а не полный
  symbol universe.
- `structure.percent_rank` сохраняет explicit window values.
- No-window indicators получают `window_axis.mode=none`.
- Indicator audit классифицирует все 40 prod indicators.
- Добавлен contract doc: `context_snapshot_contract.md`.

## Snapshot schema

Top-level snapshot:

- `schema_version`
- `source`
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

Model-facing context:

- `allowed_values.symbol`: exactly one symbol;
- `allowed_values.timeframe`: from `availability_summary.yaml`;
- `period` and `timeframe_periods`: from `availability_summary.yaml`;
- `indicators`: only available indicators;
- `ignored_symbols`: explicit ignored symbols for multi-symbol requests.

Local focused test sample:

```json
{
  "allowed_values.symbol": ["BTCUSDT"],
  "ignored_symbols": ["ETHUSDT"],
  "structure.percent_rank.window_axis": {
    "mode": "explicit",
    "values": [10, 14, 20, 28, 42, 56, 84, 126]
  },
  "volume.obv.window_axis": {
    "mode": "none"
  }
}
```

## Indicator audit

Local focused tests build a BTCUSDT synthetic summary with all prod indicator ids from
`configs/prod/indicators.yaml`.

Result:

- total prod indicators: 40;
- available: 40;
- excluded: 0.

Focused regression also removes `structure.percent_rank` from summary coverage and verifies:

```json
{
  "indicator_id": "structure.percent_rank",
  "reason": "missing_summary_coverage"
}
```

## Контрактное влияние

| Surface | Classification | Notes |
| --- | --- | --- |
| Public API contract | none | No route or browser-visible payload added in 02B. |
| Port contract | compatible-change | Additive `BacktestAiAvailabilitySummaryRepository`. |
| DTO schema | compatible-change | Additive internal snapshot DTOs for later prompt builder. |
| Persisted schema | none | No DB migration or persisted job shape change. |
| Config schema | compatible-change | Additive optional `context_snapshot` block with prod defaults. |
| Request hash/cache identity | none | Backtest request and artifact identity unchanged. |
| Artifact contract | none | Consumes existing `availability_summary.yaml`; no writer change. |

## Проверки

Completed locally:

```text
uv run pytest -q tests/unit/contexts/backtest/application/ai_configurator
```

Result: `40 passed`.

```text
uv run ruff check src/trading/contexts/backtest/application/ai_configurator src/trading/contexts/backtest/adapters/outbound/ai_configurator_context src/trading/contexts/backtest/adapters/outbound/config/backtest_ai_configurator_runtime_config.py tests/unit/contexts/backtest/application/ai_configurator
```

Result: passed.

```text
uv run pyright
```

Result: passed, `0 errors`.

```text
uv run python -m tools.docs.generate_docs_index --check
```

Result: passed.

## Mac Studio

Pending direct-main delivery.

Required post-deploy smoke:

```text
cd /opt/roehub/app
PYTHONPATH=/opt/roehub/app/src:/opt/roehub/app \
  /opt/roehub/app/.venv/bin/python - <<'PY'
from trading.contexts.backtest.adapters.outbound import (
    FilesystemBacktestAiAvailabilitySummaryRepository,
    YamlBacktestGridDefaultsProvider,
)
from trading.contexts.backtest.application.ai_configurator.context_snapshot import (
    BacktestAiContextSnapshotBuilder,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
)

provider = YamlBacktestGridDefaultsProvider.from_yaml(
    config_path="configs/prod/indicators.yaml"
)
builder = BacktestAiContextSnapshotBuilder(
    availability_summary_repository=FilesystemBacktestAiAvailabilitySummaryRepository(
        "/opt/roehub/state/backtest_artifacts/v2/availability_summary.yaml"
    ),
    defaults_provider=provider,
    runtime_defaults_service=BacktestRuntimeDefaultsService(
        defaults_provider=provider,
        runtime_config=BacktestRuntimeConfig(
            hit_times_tp_levels_pct=(1.0,),
            hit_times_sl_levels_pct=(1.0,),
            artifact_config_hash="c" * 64,
        ),
    ),
)
snapshot = builder.build(user_message="Собери RSI для BTCUSDT")
print(snapshot.snapshot_hash)
print(snapshot.summary_hash)
print(snapshot.indicator_audit)
print(snapshot.model_prompt_context()["allowed_values"]["symbol"])
PY
```

## Delivery

Pending.

## Acceptance marker

Accepted: false.

Blocking reason: pending direct-main delivery and Mac Studio smoke.

Next iteration allowed: false.
