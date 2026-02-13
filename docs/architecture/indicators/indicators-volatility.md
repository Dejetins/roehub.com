# Indicators — Volatility

## Scope

Документ фиксирует поддерживаемые `volatility.*` идентификаторы, ссылки на реализацию и эксплуатационные нюансы группы Volatility.

## Source of truth

- Спецификация формул: `docs/architecture/indicators/indicators_formula.yaml`
- Runtime defaults: `configs/prod/indicators.yaml`
- Hard defs: `src/trading/contexts/indicators/domain/definitions/volatility.py`

## Supported indicator IDs

### IDs in docs/architecture/indicators/indicators_formula.yaml

- `volatility.atr`
- `volatility.bbands`
- `volatility.bbands_bandwidth`
- `volatility.bbands_percent_b`
- `volatility.hv`
- `volatility.stddev`
- `volatility.tr`
- `volatility.variance`

### IDs with defaults in configs/prod/indicators.yaml

- `volatility.atr`
- `volatility.bbands`
- `volatility.bbands_bandwidth`
- `volatility.bbands_percent_b`
- `volatility.hv`
- `volatility.stddev`
- `volatility.tr`
- `volatility.variance`

### Diff between formula and prod defaults

- Нет расхождений.

## Deep-dive docs

- `docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`

## Key code locations

- Definitions: `src/trading/contexts/indicators/domain/definitions/volatility.py`
- Numba kernels: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py`
- NumPy oracle: `src/trading/contexts/indicators/adapters/outbound/compute_numpy/volatility.py`
- Dispatch: `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- Group tests:
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volatility_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_volatility_oracle.py`

## Group-specific gotchas

- `volatility.tr` использует fallback на `high-low`, если `prev_close` недоступен (NaN на границе).
- `volatility.atr` сглаживает TR через RMA и reset-on-NaN, что может давать повторный warmup после NaN-hole.
- `volatility.bbands_bandwidth` и `volatility.bbands_percent_b` отдают `NaN` при нулевом знаменателе.
- `volatility.hv` требует положительные значения входа для log-return; `source<=0` приводит к `NaN`.

## How this group is validated

- Сравнение Numba vs NumPy oracle для rolling/stateful/div-by-zero семантики.
- Проверка корректной обработки NaN-holes и детерминированного variant mapping.

## Change checklist for this group

1. Обновить `docs/architecture/indicators/indicators_formula.yaml` для `volatility.*`.
2. Синхронизировать defaults в `configs/{prod,dev,test}/indicators.yaml`.
3. Обновить `src/trading/contexts/indicators/domain/definitions/volatility.py`.
4. Обновить `compute_numba` + `compute_numpy` реализации и тесты.
5. Проверить ссылки в `docs/architecture/indicators/indicators-overview.md` и `docs/architecture/indicators/README.md`.
