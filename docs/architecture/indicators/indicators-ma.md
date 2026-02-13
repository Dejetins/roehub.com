# Indicators — MA

## Scope

Документ фиксирует поддерживаемые `ma.*` идентификаторы, ссылки на реализацию и практические ограничения для сопровождения группы MA.

## Source of truth

- Спецификация формул: `docs/architecture/indicators/indicators_formula.yaml`
- Runtime defaults: `configs/prod/indicators.yaml`
- Hard defs: `src/trading/contexts/indicators/domain/definitions/ma.py`

## Supported indicator IDs

### IDs in docs/architecture/indicators/indicators_formula.yaml

- `ma.dema`
- `ma.ema`
- `ma.hma`
- `ma.lwma`
- `ma.rma`
- `ma.sma`
- `ma.tema`
- `ma.vwma`
- `ma.wma`
- `ma.zlema`

### IDs with defaults in configs/prod/indicators.yaml

- `ma.dema`
- `ma.ema`
- `ma.hma`
- `ma.lwma`
- `ma.rma`
- `ma.sma`
- `ma.tema`
- `ma.vwma`
- `ma.wma`
- `ma.zlema`

### Diff between formula and prod defaults

- Нет расхождений.

## Deep-dive docs

- `docs/architecture/indicators/indicators-ma-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`

## Key code locations

- Definitions: `src/trading/contexts/indicators/domain/definitions/ma.py`
- Numba kernels: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py`
- NumPy oracle: `src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py`
- Dispatch: `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- Group tests:
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_ma_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_ma_oracle.py`

## Group-specific gotchas

- EMA/RMA цепочки работают с reset-on-NaN: состояние не переносится через NaN-gap.
- `ma.vwma` возвращает `NaN`, если `sum(volume)==0` в окне.
- `ma.hma` использует детерминированные округления (`floor(window/2)` и `floor(sqrt(window))`).
- В kernels есть runtime alias `ma.smma`; source of truth для API/конфигов остаётся в `docs/architecture/indicators/indicators_formula.yaml` и `configs/prod/indicators.yaml`.

## How this group is validated

- Сравнение Numba vs NumPy oracle по shape/dtype/NaN semantics.
- Проверка warmup, guard и layout инвариантов через compute engine tests.

## Change checklist for this group

1. Обновить `docs/architecture/indicators/indicators_formula.yaml` для `ma.*`.
2. Синхронизировать defaults в `configs/{prod,dev,test}/indicators.yaml`.
3. Обновить `src/trading/contexts/indicators/domain/definitions/ma.py`.
4. Обновить `compute_numba` + `compute_numpy` реализации и тесты.
5. Проверить ссылки в `docs/architecture/indicators/indicators-overview.md` и `docs/architecture/indicators/README.md`.
