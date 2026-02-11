# Indicators — Compute Engine Core (Numba) (v1)

Этот документ фиксирует архитектуру, контракты и реализацию для **IND-EPIC-05 — Compute engine core (Numba) + warmup + threads/cache config** в bounded context `indicators`.

Цель EPIC:

- дать **общий каркас вычислений** (Numba CPU), на который дальше “навешиваются” индикаторы;
- стандартизировать **layout**, сборку результата `IndicatorTensor`, и общие примитивы (`rolling`, `ewma`, NaN policy);
- сделать **warmup** (JIT compile) при старте процесса;
- включить конфиг:
  - `numba_num_threads` (реально влияет),
  - `numba_cache_dir` (работает в Docker),
  - единый **guard по памяти** (один параметр, включая workspace).

Документ **берёт самое полезное** из прототипа движка (ipynb): подход к раскрытию сеток, dtype-политика, njit-паттерны, warmup, простые и быстрые rolling/ewma ядра.

---

## Scope / Non-goals

### In scope (EPIC-05)

1. Общие примитивы и guards:
   - `nan utils`
   - `rolling sum / rolling mean primitives`
   - `ewma primitive`
   - `guards/helpers` (variants, bytes, total memory)

2. Реализация `IndicatorComputePort` (CPU/Numba):
   - выбор layout
   - аллокация выхода
   - сборка `IndicatorTensor`
   - единый “compute plan”: validate → allocate → run kernel → materialize tensor

3. Warmup runner:
   - прогрев ключевых kernels на типовых shapes
   - логирование

4. Конфиг Numba:
   - threads (`NUMBA_NUM_THREADS` / nb.set_num_threads)
   - cache dir (`NUMBA_CACHE_DIR`)

### Out of scope (EPIC-05)

- backtest / staged A/B / топы / оптимизация стратегий / MDD / SL/TP;
- реализация конкретных индикаторов (это IND-EPIC-06/07/08/09);
- jobs/очереди/хранение результатов (позже).

> Важно: в этом EPIC **нет** вычисления MDD и т.п. MDD относится к будущему backtest-слою.

---

## Placement (куда кладём)

**Файл документа:**
- `docs/architecture/indicators/indicators-compute-engine-core.md`

**Код (paths):**
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/`
  - `kernels/_common.py`
  - `engine.py` (или `compute_service.py`)
  - `warmup.py`
- `src/trading/platform/config/` (или текущий конфиг-слой проекта)
  - `indicators_config.py` / `config_models.py` (в зависимости от текущих соглашений)

---

## Source of Truth

- Контрактные типы `IndicatorDef / AxisDef / IndicatorTensor / Layout` — определены в EPIC-01 (domain/application).
- Grid builder/estimator (раскрытие range/explicit, guard `variants <= 600k`) — EPIC-03.
- Здесь (EPIC-05) — **движок compute**, который:
  - принимает уже валидированный `ComputeRequest` (или DTO),
  - оценивает **memory guard**,
  - считает и возвращает `IndicatorTensor`.

---

## Key decisions

### 1) Dtype policy (Milestone 2)

- **Выход всегда `float32`** (фикс Milestone 2).
- Внутри kernels допускается `float64` для накоплений (как в прототипе: sums/ema).
- `fastmath=True`, `parallel=True`, `cache=True` разрешены.

**Почему не float16 сейчас:**
- контракт выхода `float32` зафиксирован,
- float16 часто даёт критичную потерю точности на ценовых рядах (особенно для крупных цен),
- возможная оптимизация хранения/кэша — позже отдельным EPIC.

### 2) One-param memory guard (total)

Вводим один параметр:
- `max_compute_bytes_total` (по умолчанию **5 GiB**, конфигурируемо)

Движок сам резервирует “запас” под workspace и накладные расходы.

**Смысл:** независимо от сочетания `T` (время) и `V` (variants), общий memory budget ограничен.

### 3) Layout поддерживаем оба, default выберем позже

Поддерживаем:
- `TIME_MAJOR`: shape `(T, V)`
- `VARIANT_MAJOR`: shape `(V, T)`

Default фиксируем в EPIC-10 по бенчмарку. В EPIC-05 должны уметь аллоцировать и корректно заполнять оба.

### 4) Никаких staged/selection логик в compute

Compute engine — это “тензорный калькулятор индикаторов”.
Селекция/топ-N/оценка доходности — отдельные контексты/эпики.

---

## Config (platform)

### 1) indicators.compute_numba

Минимальный набор:

- `numba_num_threads: int`
- `numba_cache_dir: str | None`
- `max_compute_bytes_total: int` (bytes)

Рекомендации defaults:
- `numba_num_threads`: `min(physical_cores, 16)` (или явное значение в конфиге)
- `numba_cache_dir`: внутри контейнера writable, например `/var/lib/roehub/numba_cache`
- `max_compute_bytes_total`: `5 * 1024**3`

### 2) Runtime mapping

При старте процесса:

- `NUMBA_NUM_THREADS` выставляем через `numba.set_num_threads(n)` (и/или env)
- `NUMBA_CACHE_DIR` выставляем через env `NUMBA_CACHE_DIR=...`

Принцип:
- env overrides > config defaults
- если dir не writable → fail-fast с понятной ошибкой (иначе cache не будет работать в Docker).

---

## Memory guard (total) — формулы и политика

### Величины

- `T` = количество баров в dense timeline (из CandleFeed ACL, EPIC-04)
- `V` = количество вариантов (из grid builder/estimator, EPIC-03)
- `dtype_out` = float32 (4 bytes)

Выходной тензор:
- `bytes_out = T * V * 4`

Workspace (оценка):
- зависит от индикатора и реализации.
- в EPIC-05 вводим “консервативную оценку”:
  - `bytes_workspace_est = workspace_factor * bytes_out + workspace_fixed_bytes`

Где:
- `workspace_factor` — по умолчанию `0.20` (20%)
- `workspace_fixed_bytes` — по умолчанию `64 MiB` (под временные буферы/оси/служебные)

Итог:
- `bytes_total_est = bytes_out + bytes_workspace_est`

Guard:
- если `bytes_total_est > max_compute_bytes_total` → **422** (validation error)
  - payload включает `T`, `V`, `bytes_out`, `bytes_total_est`, `max_compute_bytes_total`
  - рекомендация: уменьшить период или сетку.

> Важно: chunking помогает уменьшить workspace, но **не уменьшает bytes_out**, если мы должны вернуть тензор целиком. Поэтому основной драйвер — размер выхода.

### Safety margins

Параметры (внутренние константы движка):
- `WORKSPACE_FACTOR_DEFAULT = 0.20`
- `WORKSPACE_FIXED_DEFAULT = 64 MiB`

Для отдельных индикаторов позже можно завести per-indicator overrides (например ATR/EMA требуют чуть больше), но в EPIC-05 достаточно общего механизма.

---

## Numba kernels common (_common.py)

Файл: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py`

### 1) NaN utils

Цели:
- быстрые проверки NaN/valid
- единая политика пропуска NaN для rolling/ewma примитивов

Примитивы:
- `is_nan(x: float) -> bool` (через `math.isnan`)
- `nan_to_zero(x)`, `zero_to_nan(x)` (по необходимости)
- `first_valid_index(x: float32[T]) -> int` (optional, если нужно для warmup/стартов)

NaN policy (для baseline):
- пока нет значения окна → `NaN`
- NaN в исходных свечах (holes) → индикатор на этой позиции обычно `NaN` (или “пропуск” для накопления — решается per-indicator; в EPIC-05 только инструменты).

### 2) Rolling primitives

Нужно ядро, на котором строятся SMA/rolling mean/std/прочее.

Базовые формы (как в прототипе):
- rolling sum по окнам `windows[]` (параллельно по окнам)
- rolling mean = rolling sum / window

Рекомендованная форма API (внутренняя):
- `_rolling_sum_grid_f64(src_f64: (T,), windows_i64: (M,)) -> (T,M) float64`
- `_rolling_mean_grid_f64(...) -> (T,M) float64`

Почему float64 внутри:
- снижает накопление ошибки при суммах.

### 3) EWMA primitive

Для EMA/RMA и многих композиций.

Форма:
- `_ewma_grid_f64(src_f64: (T,), windows_i64: (M,), alpha_mode: enum) -> (T,M) float64`
  - для EMA: `alpha = 2/(w+1)`
  - для RMA/SMMA: `alpha = 1/w`

В EPIC-05 фиксируем только “примитив”, конкретные MA (EMA/RMA) реализуются в EPIC-06, но опираются на это.

### 4) Guards/helpers

- `estimate_tensor_bytes(T:int, V:int, dtype_bytes:int=4) -> int`
- `estimate_total_bytes(bytes_out:int, factor:float, fixed:int) -> int`
- `check_total_budget_or_raise(...)`

Возвращаемые ошибки должны быть понятны API-слою (code/message/details).

---

## Compute engine (IndicatorComputePort) — contract and flow

Файл: `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`

### 1) Responsibilities

`IndicatorComputePort` implementation отвечает за:

- принять input arrays (уже dense, уже contiguous)
- принять `IndicatorDef` + `GridSpec` (уже валидировано/раскрыто в EPIC-03)
- выбрать layout
- применить **memory total guard**
- аллоцировать выходной массив `values`
- вызвать kernel индикатора (пока будет “dispatcher stub” до EPIC-06/07)
- собрать `IndicatorTensor(values, axes, layout, meta)`

### 2) Minimal “compute plan” (унифицированный)

1. `validate_inputs_presence`:
   - есть ли нужные series (close/high/low/volume) в payload
2. `resolve_axes`:
   - осевые значения уже вычислены grid builder’ом (explicit/range)
3. `T = len(ts_open)`; `V = variants`
4. `bytes_out` + `bytes_total_est` → `guard`
5. `allocate(values)`:
   - `np.empty(shape, dtype=np.float32, order='C')`
6. `run_kernel(...)`
7. `build_meta`:
   - warmup flag (если первый запуск)
   - wall time
   - bytes_out / bytes_total_est
   - nan policy summary
8. return `IndicatorTensor`

### 3) Layout mapping

- `TIME_MAJOR`:
  - shape `(T, V)`
  - удобно для “временных” операций и UI-рендера
- `VARIANT_MAJOR`:
  - shape `(V, T)`
  - удобно для последующего прохода по вариантам

Внутри kernels надо быть последовательным:
- либо писать два варианта kernels (под layout),
- либо писать в один layout и при необходимости делать `transpose` (но это дорого по памяти/времени).

Рекомендация для EPIC-05:
- движок поддерживает оба layout’а **на уровне аллокации и метаданных**,
- kernels на старте делаем TIME_MAJOR (как default), а второй layout — через отдельную реализацию позже (EPIC-10/последующие), либо через “variant-major kernels” для нескольких ключевых индикаторов.

(Это ок, потому что в EPIC-05 мы строим каркас; реальные kernels начнут появляться в EPIC-06.)

---

## Warmup runner (JIT compile)

Файл: `src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py`

### 1) What warmup does

- Выполняет один короткий прогон ключевых примитивов из `_common.py`:
  - rolling sum/mean
  - ewma
  - (минимальный stub kernel для записи в tensor, если нужен)

Цель:
- убрать JIT latency из первого пользовательского запроса,
- прогреть cache.

### 2) Where warmup runs

- На старте сервиса (API process) в wiring/composition root:
  - например `apps/api/main.py` или `apps/api/wiring/app_container.py` (по текущим соглашениям проекта)
- Warmup должен быть:
  - **однократным**
  - логируемым (duration + threads + cache_dir + shapes)

### 3) Warmup shapes

Используем типовые размеры, достаточно маленькие:
- `T = 2048`
- `windows = [5, 10, 20, 50, 100]`

Данные:
- `np.linspace` / random float32
- обязательно contiguous arrays

### 4) Logging

Минимум:
- `warmup_done=true`
- `warmup_seconds`
- `numba_num_threads_effective`
- `numba_cache_dir`
- `kernels: ["rolling_mean", "ewma", ...]`

---

## Threads & cache: ensuring they “really work”

### 1) Threads

При инициализации:
- применяем `numba.set_num_threads(config.numba_num_threads)`
- в лог:
  - requested threads
  - `numba.get_num_threads()` (effective)

В DoD тесте:
- прогон одного и того же kernel на достаточно большой задаче,
- сравнить время при `1` vs `N` потоков (smoke assertion “N быстрее хотя бы на X%” не всегда стабилен, но можно проверять что `get_num_threads()` меняется и что нет regression по времени).

### 2) Cache dir

Перед warmup:
- выставить env `NUMBA_CACHE_DIR`
- проверить, что директория существует и writable (создать файл/поддир)
- если нет — fail-fast с ошибкой конфигурации (понятное сообщение для Docker).

В DoD:
- контейнерный запуск не падает,
- warmup пишет cache (как минимум dir существует и writable; прямое доказательство “кеш использован” может быть не детерминированным, но мы фиксируем корректную настройку).

---

## How this reuses the prototype engine (what we take)

Из ipynb прототипа мы переносим только **строительные блоки**, полезные для compute контекста индикаторов:

1) **Numba patterns**:
- `@njit(parallel=True, fastmath=True, cache=True)`
- внутренние накопления в float64
- параллелизация по оси параметров (`for j in prange(M)`)

2) **Rolling/EWMA mechanics**:
- rolling sum/mean по окнам
- EMA/RMA как рекуррентные формулы

3) **Guard philosophy**:
- “комбинаторика” (у нас уже есть публичный guard 600k в EPIC-03)
- “внутренний safety guard” — в EPIC-05 делаем **total bytes guard** (один параметр)

4) **Warmup idea**:
- выделенный прогрев kernels до первого запроса

Из прототипа **не переносим**:
- CUDA, staged A/B, trades-only, MDD, SL/TP, preselect/topN.

---

## Error model (для API слоя)

Compute engine должен уметь возвращать доменные ошибки (которые API мапит в HTTP 422):

1) `ComputeBudgetExceeded`:
- details: `T`, `V`, `bytes_out`, `bytes_total_est`, `max_total`

2) `MissingRequiredSeries`:
- какие series требуются по `IndicatorDef`, какие фактически есть

3) `InvalidLayout` (если клиент запросил неизвестное)

---

## DoD (EPIC-05)

1) **Warmup при старте**:
- выполняется один раз и логируется (duration, shapes, threads, cache_dir).

2) **NUMBA_NUM_THREADS реально влияет**:
- `numba.get_num_threads()` отражает конфиг,
- perf-smoke подтверждает отсутствие деградации и корректную смену потоков.

3) **NUMBA_CACHE_DIR задействован**:
- при старте директория writable,
- warmup/compute не падают в контейнере из-за cache.

4) **Memory total guard работает**:
- при завышенных `T*V` compute отказывает с понятной 422 ошибкой,
- в ошибке есть расчёты bytes.

---

## Implementation checklist (what must appear in repo)

### Files to add

- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py`

### Config wiring

- `src/trading/platform/config/...`:
  - модель `IndicatorsComputeNumbaConfig`
  - загрузка из `configs/*` + env overrides
  - установка env/numba threads before warmup

### Tests

- `tests/unit/contexts/indicators/test_compute_budget_guard.py`
- `tests/perf_smoke/contexts/indicators/test_numba_threads_and_warmup.py`

---

## Notes for follow-ups (not in EPIC-05)

- EPIC-06/07 начнут добавлять реальные kernels индикаторов, опираясь на `_common.py`.
- EPIC-10 зафиксирует default layout по бенчмарку.
- Возможная оптимизация хранения (float16/int16+scale) — отдельный EPIC, если будет нужна экономия диска/кэша.

---
