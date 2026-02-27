---
title: План миграции kernels на float32/mixed precision (v1)
version: 1
status: draft
owner: indicators
---

# План миграции kernels на float32/mixed precision (v1)

Документ описывает phase-5 план оптимизации после `docs/architecture/indicators/indicators-grid-compute-perf-optimization-plan-v1.md`.
Цель - повысить throughput и снизить память за счет controlled перехода с full `float64` intermediates на `float32`/mixed precision без регрессий в семантике индикаторов.

## Цель

- Уменьшить memory footprint и время расчета на больших `T x V` (включая сценарии до `450_000` баров).
- Сохранить deterministic behavior, variant ordering и публичные контракты API/DTO.
- Не допустить численных регрессий в критичных индикаторах (signal-sensitive outputs).

## Scope

В scope этого плана входят:

1. Введение precision policy для kernels (`float32`, mixed, `float64`) по группам индикаторов.
2. Поэтапная миграция low-risk kernels на `float32`.
3. Поэтапная миграция medium-risk kernels на mixed precision.
4. Явная фиксация high-risk kernels, которые остаются в `float64` на v1.
5. Добавление тестов численной эквивалентности и performance smoke-гейтов.

## Non-goals

- Изменение формул индикаторов, primary output mapping, или `variant_key_v1` семантики.
- Изменение REST контрактов (`/indicators/compute`, `/indicators/estimate`).
- Переписывание kernels под GPU/CUDA.

## Ключевой принцип

Не делать "all-in float32" сразу. Использовать tiered precision policy:

- `Tier A (float32 end-to-end)` - low-risk kernels.
- `Tier B (mixed)` - вход/выход `float32`, но чувствительные аккумуляторы/промежуточные расчеты в `float64`.
- `Tier C (float64 core)` - high-risk kernels, где точность при `float32` неприемлема для текущих формул.

## AS-IS -> TO-BE (суть изменений)

### 1) Kernel compute path

AS-IS:

```python
out_f64 = _indicator_variants_f64(...)
return np.ascontiguousarray(out_f64.astype(np.float32, copy=False))
```

TO-BE (пример Tier A):

```python
out_f32 = _indicator_variants_f32(...)
return np.ascontiguousarray(out_f32, dtype=np.float32)
```

TO-BE (пример Tier B):

```python
out_f32 = _indicator_variants_mixed_f32(...)
# внутри: критичные аккумуляторы в float64, выходная матрица float32
return np.ascontiguousarray(out_f32, dtype=np.float32)
```

### 2) Общие primitives

AS-IS: `_common.py` ориентирован на `*_f64` rolling/ewma primitives.

TO-BE:

- добавить `*_f32` варианты для low-risk путей;
- добавить mixed primitives (f32 output + f64 accumulator) для windowed/statistical kernels;
- сохранить текущие `*_f64` entrypoints для high-risk paths.

### 3) Engine-level dispatch

AS-IS: implicit full-f64 core (почти для всех kernels).

TO-BE:

- явная precision policy на уровне compute-dispatch (по `indicator_id`);
- deterministic fallback на f64 для неизвестных/непокрытых путей;
- без изменений shape/layout/output contracts (`IndicatorTensor.values: float32`).

## Precision policy по индикаторам (v1)

### Tier A - мигрируем на float32 end-to-end в первую очередь

- `ma.sma`, `ma.ema`, `ma.rma`, `ma.smma`, `ma.wma`, `ma.lwma`, `ma.dema`, `ma.tema`, `ma.zlema`, `ma.hma`
- `structure.candle_*` wrappers, `structure.pivots` wrappers
- `trend.aroon`, `trend.donchian` (после oracle-check)

### Tier B - mixed precision (f32 output + f64 accumulator)

- `ma.vwma`
- `momentum.rsi`, `momentum.roc`, `momentum.cci`, `momentum.williams_r`, `momentum.stoch`
- `momentum.trix`, `momentum.stoch_rsi`, `momentum.macd`, `momentum.ppo`
- `trend.adx`, `trend.vortex`, `trend.supertrend`, `trend.psar`, `trend.keltner`
- `volume.vwap`, `volume.vwap_deviation`
- `structure.zscore`, `structure.percent_rank`, `structure.distance_to_ma_norm`

### Tier C - оставляем float64 core на v1

- `volatility.variance`
- `volatility.stddev`
- `volatility.hv`
- `volatility.bbands`
- `volatility.bbands_bandwidth`
- `volatility.bbands_percent_b`
- `trend.linreg_slope`
- `volume.obv`, `volume.ad_line`, `volume.cmf`, `volume.mfi`

Причина Tier C: высокий риск численной нестабильности при `float32` из-за
`E[x^2] - E[x]^2`, длинных cumulative chains, или diff of large sums near zero.

## Почему не full float32 everywhere

- Для `450k` баров и price-scale BTC (`50k-100k`) `float32` квантование (ULP) уже измеримо.
- Для signal-sensitive outputs (MACD/PPO/linreg near zero) даже малые abs errors могут менять sign/crossover.
- Для variance-family при больших окнах и "плоском" рынке `float32` дает заметный относительный drift.

## Рабочий план внедрения

### Шаг 0 - baseline и harness

1. Зафиксировать benchmark-набор (включая `450k` bars, NaN holes, flat/trending/high-vol regimes).
2. Добавить accuracy harness: сравнение нового пути с baseline f64 на одинаковом grid.
3. Добавить signal-stability метрики: sign flips/crossover flips для чувствительных индикаторов.

### Шаг 1 - primitives и policy wiring

1. Добавить в `_common.py`:
   - `rolling_sum_grid_f32`, `rolling_mean_grid_f32`, `ewma_grid_f32`;
   - mixed helpers (`*_mixed_f32`) с f64 accumulator.
2. В `engine.py` добавить deterministic precision dispatch map по `indicator_id`.

### Шаг 2 - Tier A migration

1. Мигрировать low-risk kernels на `*_f32` core.
2. Обновить unit tests + oracle snapshots для Tier A индикаторов.
3. Прогнать perf smoke и зафиксировать gain.

### Шаг 3 - Tier B migration

1. Мигрировать medium-risk kernels на mixed path.
2. Добавить тесты на stability around zero (`macd/ppo/linreg-like outputs`, где применимо).
3. Зафиксировать допуски по метрикам (см. Acceptance).

### Шаг 4 - Tier C freeze + docs

1. Явно отметить Tier C в коде/доках как "float64 core by design (v1)".
2. Добавить TODO/criteria для потенциального future migration.

## Acceptance criteria (Definition of Done)

- Публичные контракты не изменены (`IndicatorTensor.values` остаётся `float32`).
- Variant ordering/layout invariants неизменны.
- Для Tier A/B индикаторов выполнены accuracy thresholds:
  - `p99_rel_error <= 5e-4` (или stricter для bounded oscillators),
  - `p99_abs_error` в рамках scale-aware порогов,
  - `sign_flip_rate` для signal-critical outputs <= `0.1%` и только в near-zero зоне.
- Performance: у Tier A/B наблюдается measurable speedup или memory reduction на representative grids.
- Все quality gates проходят.

## Quality gates

```bash
uv run ruff check .
uv run pyright
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/compute_numba
uv run pytest -q tests/perf_smoke/contexts/indicators
python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

## Затрагиваемые файлы (ожидаемые)

Код:

- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/momentum.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/trend.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volume.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/structure.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/__init__.py`

Тесты:

- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_ma_kernels.py`
- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_momentum_kernels.py`
- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_trend_kernels.py`
- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volatility_kernels.py`
- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volume_kernels.py`
- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_structure_kernels.py`
- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_engine.py`
- `tests/perf_smoke/contexts/indicators/test_indicators_ma.py`
- `tests/perf_smoke/contexts/indicators/test_indicators_vol_mom.py`
- `tests/perf_smoke/contexts/indicators/test_indicators_trend_volume.py`
- `tests/perf_smoke/contexts/indicators/test_indicators_structure.py`

Документация:

- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/architecture/indicators/indicators-ma-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md`
- `docs/architecture/indicators/README.md`

## Риски

- "Скрытые" sign/crossover flips в near-zero зонах (`macd/ppo/linreg-like`).
- Расхождения на длинных cumulative цепочках (`obv/ad_line`) при full f32.
- Нестабильность variance-family при больших окнах и низкой волатильности.

## Mitigations

- Поэтапная миграция с reversible feature flags (внутренние policy knobs).
- Signal-stability regression tests на fixed datasets.
- Tier C freeze для high-risk kernels до появления более устойчивых формул/алгоритмов.
