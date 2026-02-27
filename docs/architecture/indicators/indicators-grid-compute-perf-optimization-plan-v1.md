---
title: План оптимизации производительности расчета сеток индикаторов (v1)
version: 1
status: draft
owner: indicators
---

# План оптимизации производительности расчета сеток индикаторов (v1)

Документ фиксирует пошаговый план изменений по 4 направлениям оптимизации (пункты 1-2-3-4) для `indicators` compute path без изменения бизнес-семантики индикаторов и порядка вариантов.

## Цель

- Ускорить расчет индикаторных сеток на больших `T x V`.
- Снизить пиковое потребление памяти в hot path.
- Сохранить детерминизм (`axes order`, `variant order`, `nan_policy`, `layout` контракт).

## Scope

В этот план входят только 4 пункта:

1. Быстрые wins в compute/API/config path.
2. Замена `np.ndindex`-экспансий на векторный путь через `np.repeat + np.tile`.
3. Переход на unique-source pipeline (уменьшение дублирования `(V, T)` source matrix).
4. Перепись тяжелых variant kernels на in-place запись без лишних временных массивов.

## Non-goals

- Изменение формул индикаторов, signal rules, или контрактов `IndicatorTensor`.
- Изменение публичного API payload (`/indicators/compute`, `/indicators/estimate`) за рамками эквивалентного preflight поведения.
- Добавление GPU/CUDA или сторонних native-зависимостей.

## Инварианты (обязательно сохранить)

- Идентичная materialization семантика `explicit`/`range`.
- Идентичный deterministic variant ordering от `IndicatorDef.axes`.
- Идентичная NaN policy (`propagate`, reset-on-NaN там, где уже зафиксировано).
- Идентичные формы выхода:
  - `TIME_MAJOR -> (T, V)`
  - `VARIANT_MAJOR -> (V, T)`
- Идентичные domain errors/guards (`GridValidationError`, `ComputeBudgetExceeded`, `Estimate*Exceeded`).

## Baseline: текущие узкие места

1. Лишняя копия полного тензора в `VARIANT_MAJOR` path.
2. Двойной preflight в `POST /indicators/compute`.
3. Python-level `np.ndindex` экспансии на больших `V`.
4. Дублирование source series в полном `(V, T)` matrix для source-parameterized групп.
5. Пер-variant временные аллокации внутри Numba kernels.

Наблюдаемые масштабы сеток по defaults (`configs/prod/indicators.yaml`):

- `ma.sma`: `1176` variants (`source=6 x window=196`)
- `trend.ichimoku`: `376614` variants
- `momentum.stoch_rsi`: `6652368` variants

Микро-бенч для осевых экспансий (порядок величин):

- `V=97,524`: `np.ndindex` ~14 ms vs `repeat/tile` ~0.06 ms
- `V=376,614`: `np.ndindex` ~54 ms vs `repeat/tile` ~0.24 ms
- `V=6,652,368`: `np.ndindex` ~972 ms vs `repeat/tile` ~2.7 ms

## Пункт 1: Быстрые wins (compute/API/config)

### 1.1 Убрать лишнюю копию в `VARIANT_MAJOR`

AS-IS:

```python
variant_series_matrix = <already computed VxT float32>
values = np.empty((variants, t_size), dtype=np.float32, order="C")
write_series_grid_variant_major(values, variant_series_matrix)
```

TO-BE:

```python
variant_series_matrix = <already computed VxT float32 C-contiguous>
values = variant_series_matrix  # zero-copy path
```

Ожидаемый эффект: меньше один full-tensor copy, ниже wall-time и пик памяти.

### 1.2 Убрать дублирующий preflight в `/indicators/compute`

AS-IS: последовательно вызываются `compute.estimate(...)` и `estimator.estimate_batch(...)` для одного и того же grid.

TO-BE: оставить один preflight путь в router (deterministic и эквивалентный по guards), без двойной materialization перед фактическим `compute(...)`.

### 1.3 Вернуть реалистичный `max_compute_bytes_total` в prod config

AS-IS: `configs/prod/indicators.yaml` содержит аномально высокий budget.

TO-BE: выровнять с контрактом и окружениями `dev/test` (5 GiB), чтобы guard снова защищал от OOM на широких сетках.

### Файлы (пункт 1)

- `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- `apps/api/routes/indicators.py`
- `configs/prod/indicators.yaml`
- Тесты:
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_engine.py`
  - `tests/unit/contexts/indicators/api/test_indicators_compute.py`
  - `tests/unit/contexts/indicators/api/test_indicators_estimate.py`

## Пункт 2: `np.ndindex` -> `np.repeat + np.tile`

Требование реализации: использовать именно `np.repeat + np.tile`.

### AS-IS

```python
items = []
for coordinate in np.ndindex(axis_lengths):
    items.append(axis_values[coordinate[axis_index]])
return tuple(items)
```

### TO-BE

```python
repeat = product(axis_lengths[axis_index + 1:]) or 1
tile = product(axis_lengths[:axis_index]) or 1
expanded = np.tile(np.repeat(np.asarray(axis_values), repeat), tile)
return tuple(expanded)
```

### Где заменить

- `_variant_int_values(...)`
- `_variant_float_values(...)`
- `_variant_window_values(...)`
- `_variant_source_labels(...)`

### Файлы (пункт 2)

- `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- Тесты:
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_engine.py`
  - (добавить) unit tests для детерминированного равенства `ndindex` vs `repeat/tile` на representative shapes

## Пункт 3: Unique-source pipeline

### Проблема

Для source-parameterized индикаторов строится полный `source_variants: (V, T)` с повторяющимися строками одного и того же source, что раздувает память и bandwidth.

### AS-IS

```python
variant_source_labels = (..., "close", "close", "hlc3", ...)
source_variants = _build_variant_source_matrix(variant_source_labels, available_series, t_size)
out = compute_*_grid_f32(source_variants=source_variants, ...)
```

### TO-BE

```python
groups = group_variant_indexes_by_source(variant_source_labels)
out = np.empty((V, T), dtype=np.float32)
for source_name, variant_indexes in groups:
    # compute only for subset variants of one source
    out_group = compute_*_grid_f32_for_source(source=available_series[source_name], params_subset=...)
    scatter_rows(out, variant_indexes, out_group)
```

Примечание: для `ma.*` отдельный оптимизированный путь уже есть; фокус пункта 3 - volatility/momentum/trend(structural source-path), где сейчас используется `_build_variant_source_matrix(...)`.

### Файлы (пункт 3)

- `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/momentum.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/trend.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/structure.py`
- Тесты:
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_momentum_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volatility_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_trend_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_structure_kernels.py`
  - `tests/perf_smoke/contexts/indicators/test_indicators_vol_mom.py`
  - `tests/perf_smoke/contexts/indicators/test_indicators_trend_volume.py`
  - `tests/perf_smoke/contexts/indicators/test_indicators_structure.py`

## Пункт 4: In-place kernels (без лишних временных массивов)

### AS-IS

```python
for variant_index in nb.prange(variants):
    out[variant_index, :] = _series_f64(...)
```

Проблема: `_series_f64` часто создает новые `np.empty(...)` для каждого variant.

### TO-BE

```python
for variant_index in nb.prange(variants):
    _series_into_f64(out[variant_index, :], ...)
```

Сначала покрываем самые тяжелые по памяти/времени paths:

- `trend.psar`
- `volatility.bbands*`
- `momentum.stoch_rsi` и `momentum.macd/ppo`
- `volume.vwap_deviation`

### Файлы (пункт 4)

- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/trend.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volatility.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/momentum.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volume.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py` (wiring на новые kernel entrypoints)
- Тесты:
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_momentum_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volatility_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_trend_kernels.py`
  - `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volume_kernels.py`
  - `tests/perf_smoke/contexts/indicators/test_indicators_vol_mom.py`
  - `tests/perf_smoke/contexts/indicators/test_indicators_trend_volume.py`

## Порядок внедрения

1. Пункт 1 (быстрые wins) + регрессионные тесты.
2. Пункт 2 (`repeat/tile`) + детерминизм-тесты осей.
3. Пункт 3 (unique-source) с feature-parity test suite.
4. Пункт 4 (in-place kernels) по группам индикаторов в 2-3 PR.

## Что обновить в документации

Обновляемые документы (после реализации каждого пункта):

- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/architecture/indicators/indicators-ma-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md`
- `docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md`
- `docs/runbooks/indicators-numba-warmup-jit.md` (если меняется warmup набор kernels/signatures)
- `docs/runbooks/indicators-why-nan.md` (если меняются диагностические примеры на больших сетках)
- `docs/architecture/indicators/README.md` (ссылка на этот план)

## Проверки, чтобы не сломать поведение

```bash
uv run ruff check .
uv run pyright
uv run pytest -q tests/unit/contexts/indicators/adapters/outbound/compute_numba
uv run pytest -q tests/unit/contexts/indicators/api
uv run pytest -q tests/perf_smoke/contexts/indicators
uv run python -m tools.docs.generate_docs_index --check
```

## Критерии готовности

- Пункты 1-4 внедрены без изменения функциональных результатов тестов и контрактов API.
- Пиковая память на representative grid снижается (зафиксировано perf-smoke/report).
- Время расчета на больших `V` снижается, особенно на source-heavy и multi-param grids.
- Документация синхронизирована с кодом и индексом `docs/architecture/README.md`.
