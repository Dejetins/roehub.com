# Indicators — Compute Engine Core (Numba) v1

Документ фиксирует фактическую реализацию `IND-EPIC-05` в `indicators`: CPU/Numba skeleton, warmup, runtime config, total memory guard.

## Реализация

### Ключевые файлы

- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py`
- `src/trading/platform/config/indicators_compute_numba.py`
- `apps/api/wiring/modules/indicators.py`
- `apps/api/main/app.py`

### Контракт

Реализация `NumbaIndicatorCompute` реализует
`trading.contexts.indicators.application.ports.compute.IndicatorCompute`:

- `estimate(grid, max_variants_guard)`
- `compute(req)`
- `warmup()`

## Layout

Поддерживаются оба layout:

- `TIME_MAJOR` → `values.shape == (T, V)`
- `VARIANT_MAJOR` → `values.shape == (V, T)`

Где:

- `T` — длина `candles.ts_open`
- `V` — произведение кардинальностей axes

По умолчанию используется `TIME_MAJOR`, если `layout_preference` не задан.

## Total memory guard

Единственный публичный budget-параметр:

- `max_compute_bytes_total`

Оценка:

- `bytes_out = T * V * 4` (выход всегда `float32`)
- `bytes_total_est = bytes_out + ceil(bytes_out * 0.20) + 67108864`

Если `bytes_total_est > max_compute_bytes_total`, выбрасывается `ComputeBudgetExceeded` с deterministic details:

- `T`
- `V`
- `bytes_out`
- `bytes_total_est`
- `max_compute_bytes_total`

## Common kernels

`kernels/_common.py` включает:

- NaN utils (`is_nan`, `nan_to_zero`, `zero_to_nan`, `first_valid_index`)
- rolling primitives (`rolling_sum_grid_f64`, `rolling_mean_grid_f64`)
- EWMA primitive (`ewma_grid_f64`, EMA/RMA alpha switch)
- memory helpers (`estimate_tensor_bytes`, `estimate_total_bytes`, `check_total_budget_or_raise`)
- minimal write kernels для layout (`write_series_grid_time_major`, `write_series_grid_variant_major`)

Декораторы Numba: `parallel=True`, `fastmath=True`, `cache=True`.

## Runtime config

Загрузчик: `trading.platform.config.load_indicators_compute_numba_config`.

Источники (приоритет):

1. env overrides
2. `compute.numba` в `configs/<env>/indicators.yaml` или файле из `ROEHUB_INDICATORS_CONFIG`
3. defaults

Параметры:

- `numba_num_threads`
- `numba_cache_dir`
- `max_compute_bytes_total` (default `5 * 1024**3`)

Поддерживаемые env overrides:

- threads: `ROEHUB_NUMBA_NUM_THREADS`, `NUMBA_NUM_THREADS`
- cache dir: `ROEHUB_NUMBA_CACHE_DIR`, `NUMBA_CACHE_DIR`
- memory budget: `ROEHUB_MAX_COMPUTE_BYTES_TOTAL`, `MAX_COMPUTE_BYTES_TOTAL`

## Warmup и fail-fast

`ComputeNumbaWarmupRunner`:

- применяет `NUMBA_NUM_THREADS` и `NUMBA_CACHE_DIR`
- проверяет writable `NUMBA_CACHE_DIR` (fail-fast)
- выполняет JIT warmup (`rolling_sum`, `rolling_mean`, `ewma`, write kernels)
- логирует warmup (`warmup_seconds`, effective threads, cache dir, kernels)

Startup wiring:

- `apps/api/wiring/modules/indicators.py::build_indicators_compute`
- `apps/api/main/app.py::create_app`

Warmup выполняется при создании API приложения.

## Ошибки

- `ComputeBudgetExceeded` — превышен total memory budget.
- `MissingRequiredSeries` — отсутствуют требуемые series для variant/source mapping.
- `GridValidationError` — ошибки grid/layout/guard инвариантов.
- `UnknownIndicatorError` — неизвестный `indicator_id`.

## Ограничения v1

- Реализация в EPIC-05 — skeleton (indicator-agnostic): выдаёт тензор с детерминированным variant mapping источников.
- Конкретные индикаторные формулы добавляются в последующих EPIC.
- Контракт выхода не меняется: `IndicatorTensor.values` всегда `float32`.

## Тесты

- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_common_kernels.py`
- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_engine.py`
- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_runtime_wiring.py`
- `tests/unit/platform/config/test_indicators_compute_numba_config.py`
- `tests/perf_smoke/contexts/indicators/test_compute_numba_perf_smoke.py`
