# indicators-structure-normalization-compute-numba-v1.md

# Indicators — Structure / Normalization features compute (Numba/Numpy) + outputs v1

Этот документ является **source of truth** для **IND-EPIC-09 — Structure/Normalization features (“признаки режима”)** в bounded context `indicators`.

Цель EPIC: добавить “режимные” признаки (нормализации и структурные характеристики свечи) **без ухода в microstructure/market breadth**.  
Все индикаторы считаются **тензором по сетке параметров** на CPU через Numba, имеют **оракул Numpy** для unit tests, подключены к `POST /indicators/compute`, и защищены guards от взрыва комбинаторики/памяти.

Связанные документы (читать вместе):

- `docs/architecture/indicators/indicators-overview.md`
- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`
- `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`
- `docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md`
- `docs/architecture/indicators/indicators_formula.yaml` (формулы structure/normalization — спецификация)
- `configs/prod/indicators.yaml` (+ `configs/dev|test/indicators.yaml`) — defaults/bounds
- (референс-структура доков) `docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md`
- (референс-структура доков) `docs/architecture/indicators/indicators-volatility-momentum-compute-numba-v1.md`

Связанные ключевые файлы/папки:

- Numba runtime: `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- Numba kernels common: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py`
- Numba kernels Structure (добавляем): `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/structure.py`
- Numpy oracle Structure (добавляем): `src/trading/contexts/indicators/adapters/outbound/compute_numpy/structure.py`
- Warmup: `src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py`
- Grid materialization: `src/trading/contexts/indicators/application/services/grid_builder.py`
- Candle arrays DTO: `src/trading/contexts/indicators/application/dto/candle_arrays.py`
- CandleFeed ACL: `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py`
- API route: `apps/api/routes/indicators.py`
- API DTO: `apps/api/dto/indicators.py`
- Domain defs: `src/trading/contexts/indicators/domain/definitions/structure.py`

---

## Scope / Non-goals

### In scope (EPIC-09)

1) Реализация всех индикаторов из группы `structure.*`, которые присутствуют в `configs/*/indicators.yaml` и/или являются явной частью EPIC-09:

**Structure / Normalization**
- `structure.zscore(window, source)`
- `structure.percent_rank(window, source)`
- `structure.candle_stats()`
- `structure.candle_stats_atr_norm(atr_window)`
- `structure.pivots(left, right)`
- `structure.distance_to_ma_norm(window, source)` *(добавляем в YAML defaults и формулы)*

2) Полный compute-путь:
- Numba kernels + Numpy oracle с **идентичной семантикой NaN policy**
- Подключение dispatch в compute engine
- Warmup-policy обновлена под новые индикаторы

3) Тесты:
- unit: `compute_numba` vs `compute_numpy` (включая NaN/holes и деления на ноль)
- perf-smoke: обязательно на `structure.percent_rank` и `structure.zscore`

### Out of scope (EPIC-09)

- `structure.heikin_ashi` — удалён из `docs/architecture/indicators/indicators_formula.yaml` и `configs/*/indicators.yaml` (в EPIC-09 не реализуется).
- Изменение публичного API/DTO `/indicators/compute` под multi-output или int-output.
- Runtime DSL parsing `indicators_formula.yaml` (формулы — спецификация, реализация — вручную в kernels).
- microstructure / market breadth / order-book признаки.

---

## Source of Truth

1) Идентичности индикаторов и параметры:
- `src/trading/contexts/indicators/domain/definitions/structure.py`
- YAML defaults:
  - `configs/prod/indicators.yaml` (+ `configs/dev|test/indicators.yaml`)

2) Формулы (спецификация):
- `docs/architecture/indicators/indicators_formula.yaml`
  Для каждого `structure.*`, присутствующего в YAML defaults, должна существовать запись формулы.
  **Важно:** формулы реализуются вручную в kernels, но должны соответствовать смыслу `ops`.

3) Реальный input формат:
- `CandleFeed` возвращает dense timeline + NaN holes (см. `indicators-candlefeed-acl-dense-timeline-v1.md`).

---

## Ключевые решения

### 1) Один `/compute` = один индикатор (v1)

`POST /indicators/compute` в v1 выполняет расчёт **одного** `indicator_id` по его grid.

### 2) Multi-output индикаторы: v1 возвращает primary output (фикс)

Некоторые `structure.*` по спецификации имеют несколько outputs (`candle_stats`, `candle_stats_atr_norm`, `pivots`).
В v1 API/DTO остаётся прежним: **возвращается один float32 тензор**.

**Primary output mapping (фикс v1):**
- `structure.candle_stats` → `body_pct`
- `structure.candle_stats_atr_norm` → `body_atr`
- `structure.pivots` → `pivot_high`

**Примечание:** остальные outputs допускается считать внутри kernels (для будущего расширения), но наружу в v1 не экспортируются.

### 3) Все “режимные” фичи доступны как single-output wrappers (рекомендация EPIC-09)

Чтобы использовать все компоненты multi-output индикаторов в v1 без изменения API, добавляем **одно-выходные wrapper-индикаторы** (через `call + alias` в `indicators_formula.yaml`, и явные kernels/oracle функции).

Wrappers должны быть доступны через registry + compute так же, как обычные индикаторы.

**Wrappers для `structure.candle_stats`:**
- `structure.candle_body` → `body`
- `structure.candle_range` → `range`
- `structure.candle_upper_wick` → `upper_wick`
- `structure.candle_lower_wick` → `lower_wick`
- `structure.candle_body_pct` → `body_pct`
- `structure.candle_upper_wick_pct` → `upper_wick_pct`
- `structure.candle_lower_wick_pct` → `lower_wick_pct`

**Wrappers для `structure.candle_stats_atr_norm`:**
- `structure.candle_body_atr` → `body_atr`
- `structure.candle_range_atr` → `range_atr`
- `structure.candle_upper_wick_atr` → `upper_wick_atr`
- `structure.candle_lower_wick_atr` → `lower_wick_atr`

**Wrappers для `structure.pivots`:**
- `structure.pivot_high` → `pivot_high`
- `structure.pivot_low` → `pivot_low`

### 4) NaN policy (фикс v1) + div-by-zero policy

Candles содержат NaN дырки. Compute **не делает импутацию**.

**Rolling/windowed ops** (rolling_mean/sum/min/max/std/var/percent_rank/zscore/…):
- `t < window-1` → `NaN` (warmup зона)
- если в окне есть NaN → output NaN

**Stateful ops** (ema/rma/cumsum и т.п., если применяются внутри реализации):
- до первого валидного `x[t]` → NaN
- при первом валидном `x[t]`: seed `y[t] = x[t]`
- если `x[t]` NaN → `y[t]=NaN` и **state reset**
- после reset, при следующем валидном `x[t]` стартуем заново с seed `y[t]=x[t]`

**Деление на ноль (фикс v1):**
- `range = high - low`:
  - если `range == 0` → любые `*_pct` → `NaN`
- `atr == 0`:
  - любые `*_atr` → `NaN`
  - `distance_to_ma_norm` → `NaN`

### 5) Float32 output (фикс)

`IndicatorTensor.values` — `float32`.
Внутри kernels допустимы `float64` accumulator’ы (implementation detail), но выход фиксирован как `float32`.

### 6) Guards применяются до аллокаций тензора

До расчёта:
- `variants <= max_variants_per_compute`
- `estimated_memory_bytes <= max_compute_bytes_total`
  где `estimated_memory_bytes` считается так же, как в других EPIC: `bytes_out + reserve`,
  reserve = `max(64MiB, 20%)`.

---

## Реализуемые индикаторы Structure/Normalization (v1)

### A) `structure.zscore(window, source)`
**Формула:** `structure.zscore` (см. `indicators_formula.yaml`).
- `mu = rolling_mean(source, window)`
- `sd = rolling_std(source, window, ddof=0)`
- `z = (source - mu) / sd`

NaN policy:
- rolling warmup и NaN-propagation
- div-by-zero: если `sd == 0` → `NaN`

Primary output: `zscore`.

### B) `structure.percent_rank(window, source)`
**Формула:** `structure.percent_rank`.
- `rolling_percent_rank(source, window, scale=100.0, tie_mode="le")`

NaN policy:
- rolling warmup и NaN-propagation

Primary output: `percent_rank`.

### C) `structure.candle_stats()`
**Формула:** `structure.candle_stats`.
Outputs:
- `range = high - low`
- `body = abs(close - open)`
- `upper_wick = high - max(open, close)`
- `lower_wick = min(open, close) - low`
- `body_pct = body / range`
- `upper_wick_pct = upper_wick / range`
- `lower_wick_pct = lower_wick / range`

NaN policy:
- если любой из входов candle (open/high/low/close) NaN → все derived NaN на этом t
- div-by-zero: если `range == 0` → `*_pct` NaN

Primary output (фикс v1): `body_pct`.

Wrappers: см. раздел “single-output wrappers”.

### D) `structure.candle_stats_atr_norm(atr_window)`
**Формула:** `structure.candle_stats_atr_norm`.
- вызывает `structure.candle_stats`
- вызывает `volatility.atr(high, low, close, window=atr_window)`
- делит `body/range/upper_wick/lower_wick` на `atr`

NaN policy:
- ATR rolling warmup и NaN-propagation
- div-by-zero: если `atr == 0` → `*_atr` NaN

Primary output (фикс v1): `body_atr`.

Wrappers: см. раздел “single-output wrappers”.

### E) `structure.pivots(left, right)`
**Формула:** `structure.pivots`.
- `pivothigh(high, left, right, strict=true, shift_confirm=true)`
- `pivotlow(low, left, right, strict=true, shift_confirm=true)`
- outputs: `pivot_high`, `pivot_low` (values series<float>)

NaN policy:
- при NaN в окнах подтверждения pivot → NaN (pivot не подтверждается)
- “shift_confirm=true” означает, что pivot появляется только после правого окна подтверждения (consistent v1).

Primary output (фикс v1): `pivot_high`.

Wrappers: `structure.pivot_high`, `structure.pivot_low`.

### F) `structure.distance_to_ma_norm(window, source)` *(добавляем)*
Назначение: “distance-to-MA normalized” в ATR-единицах.

**Формула (фикс v1):**
- `ma = ema(source, window, init="sma")`
- `atr = volatility.atr(high, low, close, window=window)`
- `dist_to_ma_norm = (source - ma) / atr`

Выбор сделан осознанно:
- один `window` контролирует и сглаживание MA, и нормализацию ATR → меньше комбинаторики,
- EMA даёт режимную метрику, менее “ступенчатую”, чем SMA, и соответствует общей библиотеке ops.

NaN policy:
- EMA: reset-on-NaN (см. общий stateful policy)
- ATR: rolling warmup/NaN-propagation
- div-by-zero: если `atr == 0` → NaN

Primary output: `distance_to_ma_norm`.

---

## Sources (input series)

### Доступные источники (InputSeries)

Base candles:
- `open`, `high`, `low`, `close`, `volume`

Derived series:
- `hl2 = (high + low) / 2`
- `hlc3 = (high + low + close) / 3`
- `ohlc4 = (open + high + low + close) / 4`

Правила:
- derived вычисляются в float32;
- NaN пропускается естественно (если любой компонент NaN → derived NaN);
- derived создаются как contiguous arrays.

---

## Warmup policy (v1)

Warmup используется:
- для preflight оценок,
- для unit тестов корректной NaN зоны,
- для объяснимости поведения output.

Правила warmup (минимальный индекс, после которого допускаются валидные значения при отсутствии NaN дырок):

- `structure.zscore(window, ...)` → warmup = `window`
- `structure.percent_rank(window, ...)` → warmup = `window`
- `structure.candle_stats()` → warmup = `0`
- `structure.candle_stats_atr_norm(atr_window)` → warmup = `atr_window`
- `structure.pivots(left, right)` → warmup ≈ `left + right` *(плюс подтверждение через `shift_confirm=true`)*
- `structure.distance_to_ma_norm(window, ...)` → warmup = `window`

Реализация:
- обновить `src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py` под новые `structure.*` IDs и wrappers.

---

## Execution pipeline (compute v1)

### 1) Request → Grid → Axes
Вход: `ComputeRequest` (`apps/api/dto/indicators.py`) содержит:
- `indicator_id`
- `grid` (explicit/range specs)
- `layout` (optional hint)
- `max_variants_guard` (optional override, иначе runtime default)

Grid материализуется через:
- `src/trading/contexts/indicators/application/services/grid_builder.py`

Результат:
- `AxisDef` по каждой оси (например `window`, `source`, `atr_window`, `left/right`)
- `variants = product(axis_lengths)`

### 2) CandleFeed → CandleArrays
Используется `CandleFeed` ACL:
- `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py`

Получаем dense timeline `[start, end)` на 1m, с NaN дырками.

### 3) Guards
Применяются до больших аллокаций:
- variants guard (default)
- memory guard (`max_compute_bytes_total`)

Ошибки: детерминированные payload’ы (422) как в compute engine core.

### 4) Compute (Numba) + Output tensor
- Numba kernels: `compute_numba/kernels/structure.py`
- output: `IndicatorTensor(values=float32, axes, layout, meta)`

### 5) Oracle (Numpy)
- Numpy path: `compute_numpy/structure.py`
Используется в тестах как эталон на одинаковых входах и той же NaN policy.

---

## API: POST /indicators/compute (v1)

Назначение: вычислить **один** индикатор по сетке параметров.

- Endpoint: `apps/api/routes/indicators.py`
- DTO: `apps/api/dto/indicators.py`

Семантика:
- request задаёт `indicator_id` + grid
- система читает свечи через CandleFeed (market_data ACL)
- возвращает `IndicatorTensor` (в пределах guards)

---

## Tests

### Unit tests (обязательные)

Покрываем:

1) корректность axes/materialization (включая порядок осей)
2) совпадение numba vs numpy на фиксированном seed
3) корректность NaN policy:
   - warmup зоны для rolling окон (zscore/percent_rank/atr)
   - div-by-zero policy (range==0, atr==0, sd==0)
4) dtype/output layout invariants (float32, contiguous)
5) pivots: подтверждение (`shift_confirm=true`) и корректные NaN на краях

Предлагаемые пути:
- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_structure_kernels.py` *(добавить)*
- `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_structure_oracle.py` *(добавить)*

Минимальный “эталонный набор” для полного numba-vs-numpy сравнения:
- `structure.zscore` (rolling_mean/std + div)
- `structure.percent_rank` (ranking + ties)
- `structure.candle_stats_atr_norm` (call + atr + div-by-zero)
- `structure.pivots` (confirm/shift semantics)
- `structure.distance_to_ma_norm` (ema stateful + atr + div)

Wrappers:
- покрыть как “delegation correctness” (wrapper output == соответствующий output базового multi-output) на небольших входах.

### Perf-smoke (обязательный)

Требование EPIC-09: perf-smoke на `percent_rank` и `zscore`.

Предлагаемый тест:
- `tests/perf_smoke/contexts/indicators/test_indicators_structure.py` *(добавить)*

Сценарий:
- длина T реалистичная для CPU (например 100k–300k), без жёсткого SLA
- индикаторы: `structure.percent_rank`, `structure.zscore`
- несколько окон (например 20 и 200) + один source (например `close`)
- assert:
  - не падает,
  - укладывается в guards,
  - output float32,
  - формы детерминированы.

---

## Determinism & ordering rules (фикс)

1) Оси grid:
- `explicit.values` сохраняют порядок request (без сортировки)
- `range(start, stop_incl, step)` материализуется строго инклюзивно

2) Variants ordering:
- порядок осей в `IndicatorDef` определяет порядок перемножения и variant indexing
- enumeration детерминирована при одинаковом request

3) Candles duplicates/holes:
- политика `last-wins` на дубли
- “out of range” игнорируется
- NaN holes сохраняются (см. CandleFeed ACL документ)

---

## Целевое размещение в репозитории

### Numba kernels
- добавить: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/structure.py`
- обновить: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/__init__.py`

### Numpy oracle
- добавить: `src/trading/contexts/indicators/adapters/outbound/compute_numpy/structure.py`
- обновить: `src/trading/contexts/indicators/adapters/outbound/compute_numpy/__init__.py`

### Engine wiring
- обновить маршрутизацию в:
  - `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- при необходимости: `warmup.py`

### Domain defs
- обновить / расширить:
  - `src/trading/contexts/indicators/domain/definitions/structure.py`
  - убедиться, что wrappers присутствуют как отдельные `IndicatorDef` (single-output).

### YAML defaults
- добавить defaults для `structure.distance_to_ma_norm` в:
  - `configs/prod/indicators.yaml`
  - `configs/dev/indicators.yaml`
  - `configs/test/indicators.yaml`
- если wrappers должны быть доступны через grid-builder/registry так же, как остальные, им тоже нужны defaults (либо explicit-only policy, если так принято в registry — решение фиксируется в `indicators-registry-yaml-defaults-v1.md`).

### Formula spec
- добавить формулы для:
  - `structure.distance_to_ma_norm`
  - wrapper-индикаторов (call + alias на базовые multi-output), если их нет

Файл:
- `docs/architecture/indicators/indicators_formula.yaml`

### Docs index
- добавить ссылку в `docs/architecture/indicators/README.md`
  на этот документ: `docs/architecture/indicators/indicators-structure-normalization-compute-numba-v1.md`

---

## DoD (EPIC-09)

EPIC-09 считается выполненным, если:

1) `POST /indicators/compute` считает все `structure.*` индикаторы из YAML defaults и применяет guards (variants + memory).
2) Реализованы Numba kernels для:
   - `structure.zscore`
   - `structure.percent_rank`
   - `structure.candle_stats` + wrappers
   - `structure.candle_stats_atr_norm` + wrappers
   - `structure.pivots` + wrappers
   - `structure.distance_to_ma_norm`
3) Есть Numpy oracle с идентичной семантикой + unit tests “numba vs numpy” (минимум на репрезентативном наборе, включая div-by-zero и NaN holes).
4) Есть perf-smoke тест для `structure.percent_rank` и `structure.zscore`.
5) Документация обновлена:
   - добавлен этот документ,
   - `docs/architecture/indicators/README.md` содержит ссылку,
   - публичные экспорты через `__init__.py` стабильны,
   - docstrings в новых/изменённых классах/протоколах ссылаются на этот документ и связанные файлы.

