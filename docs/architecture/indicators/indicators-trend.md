# Indicators — Trend

## Scope

Документ фиксирует поддерживаемые `trend.*` идентификаторы, ссылки на реализацию и семантические ограничения группы Trend.

## Source of truth

- Спецификация формул: `docs/architecture/indicators/indicators_formula.yaml`
- Runtime defaults: `configs/prod/indicators.yaml`
- Hard defs: `src/trading/contexts/indicators/domain/definitions/trend.py`

## Supported indicator IDs

### IDs in docs/architecture/indicators/indicators_formula.yaml

- `trend.adx`
- `trend.aroon`
- `trend.chandelier_exit`
- `trend.donchian`
- `trend.ichimoku`
- `trend.keltner`
- `trend.linreg_slope`
- `trend.psar`
- `trend.supertrend`
- `trend.vortex`

### IDs with defaults in configs/prod/indicators.yaml

- `trend.adx`
- `trend.aroon`
- `trend.chandelier_exit`
- `trend.donchian`
- `trend.ichimoku`
- `trend.keltner`
- `trend.linreg_slope`
- `trend.psar`
- `trend.supertrend`
- `trend.vortex`

### Diff between formula and prod defaults

- Нет расхождений.

## Deep-dive docs

- `docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`

## Key code locations

- Definitions: `src/trading/contexts/indicators/domain/definitions/trend.py`
- Numba kernels: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/trend.py`
- NumPy oracle: `src/trading/contexts/indicators/adapters/outbound/compute_numpy/trend.py`
- Dispatch: `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- Group tests:
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_trend_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_trend_oracle.py`

## Group-specific gotchas

- Multi-output индикаторы (`adx`, `ichimoku`, `donchian`, `keltner`, `chandelier_exit`) в v1 публично возвращают только primary output.
- `psar` и `supertrend` stateful: на NaN-gap выполняется reset состояния/направления.
- `ichimoku` использует `shift` с `displacement`; неправильная трактовка знака ломает согласование с formula spec.
- `linreg_slope` чувствителен к warmup и NaN внутри окна: отсутствие полного окна даёт `NaN`, это ожидаемо.

## How this group is validated

- Сравнение Numba vs NumPy oracle по детерминированной variant индексации и NaN policy.
- Отдельная проверка stateful reset semantics для `psar` и `supertrend`.

## Change checklist for this group

1. Обновить `docs/architecture/indicators/indicators_formula.yaml` для `trend.*`.
2. Синхронизировать defaults в `configs/{prod,dev,test}/indicators.yaml`.
3. Обновить `src/trading/contexts/indicators/domain/definitions/trend.py`.
4. Обновить `compute_numba` + `compute_numpy` реализации и тесты.
5. Проверить primary output mapping и ссылки в `docs/architecture/indicators/README.md`.
