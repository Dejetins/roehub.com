# Indicators — Volume

## Scope

Документ фиксирует поддерживаемые `volume.*` идентификаторы, ссылки на реализацию и критичные edge-cases группы Volume.

## Source of truth

- Спецификация формул: `docs/architecture/indicators/indicators_formula.yaml`
- Runtime defaults: `configs/prod/indicators.yaml`
- Hard defs: `src/trading/contexts/indicators/domain/definitions/volume.py`

## Supported indicator IDs

### IDs in docs/architecture/indicators/indicators_formula.yaml

- `volume.ad_line`
- `volume.cmf`
- `volume.mfi`
- `volume.obv`
- `volume.volume_sma`
- `volume.vwap`
- `volume.vwap_deviation`

### IDs with defaults in configs/prod/indicators.yaml

- `volume.ad_line`
- `volume.cmf`
- `volume.mfi`
- `volume.obv`
- `volume.volume_sma`
- `volume.vwap`
- `volume.vwap_deviation`

### Diff between formula and prod defaults

- Нет расхождений.

## Deep-dive docs

- `docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`

## Key code locations

- Definitions: `src/trading/contexts/indicators/domain/definitions/volume.py`
- Numba kernels: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volume.py`
- NumPy oracle: `src/trading/contexts/indicators/adapters/outbound/compute_numpy/volume.py`
- Dispatch: `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- Group tests:
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volume_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_volume_oracle.py`

## Group-specific gotchas

- Кумулятивные цепочки (`ad_line`, `obv`) сбрасывают состояние на NaN-gap.
- `cmf`, `vwap`, `vwap_deviation` зависят от `sum(volume)`; при нуле получают `NaN`.
- `vwap_deviation` по v1 возвращает primary output (`vwap_upper`), хотя формула содержит несколько выходов.
- Все расчёты чувствительны к синхронности OHLCV: рассинхрон массива сразу проявляется NaN propagation.

## How this group is validated

- Сравнение Numba vs NumPy oracle для rolling, cumulative и div-by-zero сценариев.
- Проверка NaN policy на входах с разрывами timeline.

## Change checklist for this group

1. Обновить `docs/architecture/indicators/indicators_formula.yaml` для `volume.*`.
2. Синхронизировать defaults в `configs/{prod,dev,test}/indicators.yaml`.
3. Обновить `src/trading/contexts/indicators/domain/definitions/volume.py`.
4. Обновить `compute_numba` + `compute_numpy` реализации и тесты.
5. Проверить primary output mapping и ссылки в `docs/architecture/indicators/README.md`.
