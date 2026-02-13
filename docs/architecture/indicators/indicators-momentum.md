# Indicators — Momentum

## Scope

Документ фиксирует поддерживаемые `momentum.*` идентификаторы, ключевые ссылки на реализацию и типовые ошибки сопровождения для группы Momentum.

## Source of truth

- Спецификация формул: `docs/architecture/indicators/indicators_formula.yaml`
- Runtime defaults: `configs/prod/indicators.yaml`
- Hard defs: `src/trading/contexts/indicators/domain/definitions/momentum.py`

## Supported indicator IDs

### IDs in docs/architecture/indicators/indicators_formula.yaml

- `momentum.cci`
- `momentum.fisher`
- `momentum.macd`
- `momentum.ppo`
- `momentum.roc`
- `momentum.rsi`
- `momentum.stoch`
- `momentum.stoch_rsi`
- `momentum.trix`
- `momentum.williams_r`

### IDs with defaults in configs/prod/indicators.yaml

- `momentum.cci`
- `momentum.fisher`
- `momentum.macd`
- `momentum.ppo`
- `momentum.roc`
- `momentum.rsi`
- `momentum.stoch`
- `momentum.stoch_rsi`
- `momentum.trix`
- `momentum.williams_r`

### Diff between formula and prod defaults

- Нет расхождений.

## Deep-dive docs

- `docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`

## Key code locations

- Definitions: `src/trading/contexts/indicators/domain/definitions/momentum.py`
- Numba kernels: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/momentum.py`
- NumPy oracle: `src/trading/contexts/indicators/adapters/outbound/compute_numpy/momentum.py`
- Dispatch: `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- Group tests:
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_momentum_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_momentum_oracle.py`

## Group-specific gotchas

- RSI/TRIX/MACD/PPO используют stateful EMA/RMA цепочки и reset-on-NaN.
- Для ROC/PPO/stoch-производных есть чувствительность к делению на 0: результат `NaN`, не clamp.
- MACD/PPO критичны к согласованности параметров `fast_window < slow_window`; иначе получаются неожиданные формы кривой даже при валидном типе.
- Для `stoch_rsi` warmup складывается из warmup RSI и последующего rolling-окна.

## How this group is validated

- Сравнение Numba vs NumPy oracle по всем поддерживаемым `momentum.*`.
- Проверка NaN-propagation, reset-on-NaN и div-by-zero сценариев.

## Change checklist for this group

1. Обновить `docs/architecture/indicators/indicators_formula.yaml` для `momentum.*`.
2. Синхронизировать defaults в `configs/{prod,dev,test}/indicators.yaml`.
3. Обновить `src/trading/contexts/indicators/domain/definitions/momentum.py`.
4. Обновить `compute_numba` + `compute_numpy` реализации и тесты.
5. Проверить ссылки в `docs/architecture/indicators/indicators-overview.md` и `docs/architecture/indicators/README.md`.
