# Indicators — Structure

## Scope

Документ фиксирует поддерживаемые `structure.*` идентификаторы, различия между formula spec и prod defaults, а также gotchas группы Structure/Normalization.

## Source of truth

- Спецификация формул: `docs/architecture/indicators/indicators_formula.yaml`
- Runtime defaults: `configs/prod/indicators.yaml`
- Hard defs: `src/trading/contexts/indicators/domain/definitions/structure.py`

## Supported indicator IDs

### IDs in docs/architecture/indicators/indicators_formula.yaml

- `structure.candle_body`
- `structure.candle_body_atr`
- `structure.candle_body_pct`
- `structure.candle_lower_wick`
- `structure.candle_lower_wick_atr`
- `structure.candle_lower_wick_pct`
- `structure.candle_range`
- `structure.candle_range_atr`
- `structure.candle_stats`
- `structure.candle_stats_atr_norm`
- `structure.candle_upper_wick`
- `structure.candle_upper_wick_atr`
- `structure.candle_upper_wick_pct`
- `structure.distance_to_ma_norm`
- `structure.percent_rank`
- `structure.pivot_high`
- `structure.pivot_low`
- `structure.pivots`
- `structure.zscore`

### IDs with defaults in configs/prod/indicators.yaml

- `structure.candle_stats`
- `structure.candle_stats_atr_norm`
- `structure.distance_to_ma_norm`
- `structure.percent_rank`
- `structure.pivots`
- `structure.zscore`

### Diff between formula and prod defaults

`configs/prod/indicators.yaml` не содержит defaults для части wrapper-идентификаторов, хотя они описаны в `docs/architecture/indicators/indicators_formula.yaml`:

- `structure.candle_body`
- `structure.candle_body_atr`
- `structure.candle_body_pct`
- `structure.candle_lower_wick`
- `structure.candle_lower_wick_atr`
- `structure.candle_lower_wick_pct`
- `structure.candle_range`
- `structure.candle_range_atr`
- `structure.candle_upper_wick`
- `structure.candle_upper_wick_atr`
- `structure.candle_upper_wick_pct`
- `structure.pivot_high`
- `structure.pivot_low`

## Deep-dive docs

- `docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`

## Key code locations

- Definitions: `src/trading/contexts/indicators/domain/definitions/structure.py`
- Numba kernels: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/structure.py`
- NumPy oracle: `src/trading/contexts/indicators/adapters/outbound/compute_numpy/structure.py`
- Dispatch: `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- Group tests:
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_structure_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_structure_oracle.py`

## Group-specific gotchas

- Multi-output индикаторы (`candle_stats`, `candle_stats_atr_norm`, `pivots`) в v1 экспортируют primary output; wrappers дают доступ к отдельным компонентам.
- Для `*_pct` деление на `range==0` возвращает `NaN`; для `*_atr` деление на `atr==0` тоже `NaN`.
- `pivots` использует confirm-окно (`left/right`): pivot появляется позже, после подтверждения правым окном.
- `distance_to_ma_norm` комбинирует EMA и ATR; NaN-gap в любой из цепочек переносится в итог как `NaN`.

## How this group is validated

- Сравнение Numba vs NumPy oracle по wrappers/multi-output mapping и NaN policy.
- Отдельные проверки div-by-zero (`range==0`, `atr==0`) и confirm semantics для pivots.

## Change checklist for this group

1. Обновить `docs/architecture/indicators/indicators_formula.yaml` для `structure.*`.
2. Синхронизировать defaults в `configs/{prod,dev,test}/indicators.yaml`.
3. Обновить `src/trading/contexts/indicators/domain/definitions/structure.py`.
4. Обновить `compute_numba` + `compute_numpy` реализации и тесты.
5. Проверить согласованность wrappers и ссылки в `docs/architecture/indicators/README.md`.
