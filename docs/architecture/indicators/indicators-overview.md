# Indicators Overview (Milestone 2)

`indicators` в Milestone 2 покрывает только вычисление индикаторных тензоров и их контрактные входы/выходы.

## В scope

- Registry hard definitions + YAML defaults.
- Grid materialization и deterministic `variant_key v1`.
- Dense candles ACL (`CandleArrays`).
- CPU/Numba compute engine core (`compute_numba`) с warmup и total memory guard.

## Вне scope

- Backtesting, staged A/B, PnL/MDD/SL/TP.
- Оптимизация стратегий и top-N selection.
- CUDA/GPU реализация.

## Основные контракты

- Port: `trading.contexts.indicators.application.ports.compute.IndicatorCompute`
- DTO output: `trading.contexts.indicators.application.dto.IndicatorTensor`
- Layout: `trading.contexts.indicators.domain.entities.Layout`

## Compute engine core

Реализация расположена в:

- `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py`

См. детально: `docs/architecture/indicators/indicators-compute-engine-core.md`.
