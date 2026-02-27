# Indicators — Trend + Volume compute (Numba/Numpy) + outputs v1

Этот документ является **source of truth** для **IND-EPIC-08 — Trend + Volume (каналы/пробои/денежный поток)** в bounded context `indicators`.

Цель EPIC: дать следующий “вертикальный” результат — **trend.* и volume.* индикаторы считаются тензором по сетке параметров** на CPU через Numba, с **оракулом Numpy** для unit tests, подключены к `POST /indicators/compute`, и защищены guards от взрыва комбинаторики/памяти.

Связанные документы (читать вместе):

- `docs/architecture/indicators/indicators-overview.md`
- `docs/architecture/indicators/indicators-compute-engine-core.md`
- `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`
- `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`
- `docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md`
- `docs/architecture/indicators/indicators_formula.yaml` (формулы trend/volume — спецификация)
- `configs/prod/indicators.yaml` (+ `configs/dev|test/indicators.yaml`) — defaults/bounds

Связанные ключевые файлы/папки:

- Numba runtime: `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- Numba kernels common: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py`
- Numba kernels Trend (добавляем): `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/trend.py`
- Numba kernels Volume (добавляем): `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volume.py`
- Numpy oracle Trend (добавляем): `src/trading/contexts/indicators/adapters/outbound/compute_numpy/trend.py`
- Numpy oracle Volume (добавляем): `src/trading/contexts/indicators/adapters/outbound/compute_numpy/volume.py`
- Warmup: `src/trading/contexts/indicators/adapters/outbound/compute_numba/warmup.py`
- Grid materialization: `src/trading/contexts/indicators/application/services/grid_builder.py`
- Candle arrays DTO: `src/trading/contexts/indicators/application/dto/candle_arrays.py`
- CandleFeed ACL: `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py`
- API route: `apps/api/routes/indicators.py`
- API DTO: `apps/api/dto/indicators.py`
- Domain defs: `src/trading/contexts/indicators/domain/definitions/trend.py`, `.../volume.py`

---

## Scope / Non-goals

### In scope (EPIC-08)

1) Полная реализация **всех** индикаторов из групп `trend.*` и `volume.*`,
которые присутствуют в `configs/prod/indicators.yaml` (и **аналогично** в `dev/test`):

**Trend**
- `trend.adx(window, smoothing)`
- `trend.aroon(window)`
- `trend.chandelier_exit(window, mult)`
- `trend.donchian(window)`
- `trend.ichimoku(conversion_window, base_window, span_b_window, displacement)`
- `trend.keltner(window, mult)`
- `trend.linreg_slope(window, source)`
- `trend.psar(accel_start, accel_step, accel_max)`
- `trend.supertrend(window, mult)`
- `trend.vortex(window)`

**Volume**
- `volume.ad_line()`
- `volume.cmf(window)`
- `volume.mfi(window)`
- `volume.obv()`
- `volume.volume_sma(window)`
- `volume.vwap(window)`
- `volume.vwap_deviation(window, mult)`

2) Numba kernels + Numpy oracle с **идентичной семантикой NaN policy** и детерминированным варианто-индексингом.

3) Подключение dispatch в compute engine + обновление warmup (если нужно).

4) Тесты:
- unit: `compute_numba` vs `compute_numpy` (на репрезентативном подмножестве, но с покрытием ключевых NaN/guard семантик)
- perf-smoke: “trend+volume grid” на реалистичной длине T (без жёсткого SLA, но обязательно smoke + guard coverage)

### Out of scope (EPIC-08)

- Изменение публичного API/DTO `/indicators/compute` под multi-output или int-output.
- Runtime DSL parsing `indicators_formula.yaml` (формулы — спецификация, реализация — вручную в kernels).

---

## Source of Truth

1) Идентичности индикаторов и параметры:
- `src/trading/contexts/indicators/domain/definitions/trend.py`
- `src/trading/contexts/indicators/domain/definitions/volume.py`
- YAML defaults:
  - `configs/prod/indicators.yaml` (+ dev/test)

2) Формулы (спецификация):
- `docs/architecture/indicators/indicators_formula.yaml`
  Для каждого `trend.*` и `volume.*`, присутствующего в YAML defaults, должна существовать запись формулы.
  **Важно:** в EPIC-08 формулы реализуются вручную в kernels, но должны соответствовать смыслу `ops`.

3) Реальный input формат:
- `CandleFeed` возвращает dense timeline + NaN holes (см. `indicators-candlefeed-acl-dense-timeline-v1.md`).

---

## Ключевые решения

### 1) Один `/compute` = один индикатор (v1)

`POST /indicators/compute` в v1 выполняет расчёт **одного** `indicator_id` по его grid.

Batch — отдельная orchestration задача (не в EPIC-08).

### 2) Multi-output индикаторы: v1 возвращает primary output (фикс)

Некоторые индикаторы по спецификации имеют несколько outputs (например ADX/Ichimoku/Donchian/Keltner/Chandelier/VWAP deviation).
В v1 API/DTO остаётся прежним: **возвращается один float32 тензор**.

**Primary output mapping (фикс v1):**
- `trend.adx` → `adx`
- `trend.aroon` → `aroon_osc`
- `trend.chandelier_exit` → `chandelier_long`
- `trend.donchian` → `donchian_mid`
- `trend.ichimoku` → `span_a` *(см. также раздел про shift)*
- `trend.keltner` → `middle`
- `trend.psar` → `psar`
- `trend.supertrend` → `supertrend`
- `trend.vortex` → `vi_plus`
- `volume.vwap_deviation` → `vwap_upper` *(как “верхняя граница”, зависит от `mult`)*

**Примечание:** остальные outputs допускается считать внутри kernels (для будущего расширения), но наружу в v1 не экспортируются.

Для source-parameterized trend-path (`trend.linreg_slope`) compute engine использует grouped-source pipeline:

- глобальные `variant_source_labels` группируются по unique source;
- kernel вызывается отдельно по каждой source-группе и subset параметров;
- group-результаты scatter’ятся в исходные global variant indexes без изменения порядка.

### 3) In-place heavy kernels (phase 4)

Для наиболее тяжёлых trend/volume kernels в variant-циклах включён in-place write path:

- запись primary output выполняется сразу в `out[variant_index, :]`;
- для hot-кейсов используются `*_into_*` helper-ы, чтобы убрать лишние per-variant
  временные массивы;
- публичные entrypoints `compute_trend_grid_f32(...)` и `compute_volume_grid_f32(...)`
  сохраняют прежние сигнатуры и контракт.

Phase 4 покрывает следующие индикаторы:

- `trend.psar`
- `volume.vwap_deviation`

### 4) Int outputs (direction flags) не возвращаются в v1

`trend.psar` и `trend.supertrend` в формуле имеют `*_dir: series<int>`.
В v1 `IndicatorTensor.values` — `float32`. Поэтому direction:
- либо вообще не считается,
- либо считается, но не возвращается (implementation detail).

### 5) NaN policy (фикс v1, включая сброс направления)

Candles содержат NaN дырки. Compute **не делает импутацию**.

**Rolling/windowed ops** (rolling_mean/sum/min/max/std/var/…):
- `t < window-1` → `NaN` (warmup зона)
- если в окне есть NaN → output NaN

**Stateful ops** (ema/rma и любые цепочки, которые на них опираются):
- до первого валидного `x[t]` → NaN
- при первом валидном `x[t]`: seed `y[t] = x[t]`
- если `x[t]` NaN → `y[t]=NaN` и **state reset**
- после reset, при следующем валидном `x[t]` стартуем заново с seed `y[t]=x[t]`

**Directional state (psar/supertrend):**
- при NaN на входах, где невозможно обновить шаг → output NaN и **direction/state reset**
- после reset — state и direction переинициализируются “с нуля” на первом валидном фрагменте

### 6) Shift semantics (фикс v1, как в DSL)

`shift(x, periods)` должен быть семантически **единым** во всём контексте (используется в TR/ADX/Ichimoku и т.д.).
Конкретная реализация должна соответствовать уже принятой в `_common.py` / существующих kernels.

**Требование EPIC-08:** kernels Trend/Volume обязаны повторить поведение `shift` из общих primitives (не “своё”).

**Ichimoku note:** по спецификации `span_a`/`span_b` используют `shift(..., periods: -displacement)`.
Это фиксируется как “истина” для v1 и реализуется строго по `indicators_formula.yaml`.

### 7) Float32 output (фикс)

`IndicatorTensor.values` — `float32`.
Внутри kernels допустимы `float64` accumulator’ы (implementation detail), но выход фиксирован как `float32`.

### 7.1) Phase-5 precision policy (f32/mixed)

Для trend/volume phase-5 policy фиксируется так:

- `Tier A` (`float32`): `trend.aroon`, `trend.donchian`.
- `Tier B` (`mixed precision`): `trend.adx`, `trend.vortex`, `trend.supertrend`,
  `trend.psar`, `trend.keltner`, `volume.vwap`, `volume.vwap_deviation`.
- `Tier C` (`float64` core): `trend.linreg_slope`, `volume.obv`, `volume.ad_line`,
  `volume.cmf`, `volume.mfi`.

Policy не меняет внешние API/DTO: `IndicatorTensor.values` по-прежнему `float32`.

### 8) Guards применяются до аллокаций тензора

До расчёта:
- `variants <= max_variants_per_compute`
- `estimated_memory_bytes <= max_compute_bytes_total`
  где `estimated_memory_bytes` считается так же, как в EPIC-03/05: `bytes_out + reserve`,
  reserve = `max(64MiB, 20%)`.

---

## Реализуемые индикаторы Trend (v1)

### A) `trend.adx(window, smoothing)`
**Формула:** `indicators_formula.yaml` → `trend.adx`.
Ключевая семантика:
- DM/TR → Wilder smoothing (RMA)
- DI и DX → ADX = RMA(DX, smoothing)
- NaN дырки → reset цепочек rma и любых stateful частей.

**Primary output:** `adx`.

### B) `trend.aroon(window)`
**Формула:** `trend.aroon`.
- rolling_time_since_max/min → aroon_up/down → aroon_osc
- NaN дырки → NaN propagation через rolling ops.

**Primary output:** `aroon_osc`.

### C) `trend.chandelier_exit(window, mult)`
**Формула:** `trend.chandelier_exit`.
- ATR(window) (Wilder RMA) + rolling max/min high/low
- long/short уровни

**Primary output:** `chandelier_long`.

### D) `trend.donchian(window)`
**Формула:** `trend.donchian`.
- upper = rolling_max(high), lower = rolling_min(low), mid = (upper+lower)/2

**Primary output:** `donchian_mid`.

### E) `trend.ichimoku(conversion_window, base_window, span_b_window, displacement)`
**Формула:** `trend.ichimoku`.
Outputs: conversion, base, span_a, span_b, lagging (см. shift note).
NaN дырки:
- любые rolling на окнах → NaN при пропусках
- shift/out-of-range → NaN.

**Primary output:** `span_a`.

### F) `trend.keltner(window, mult)`
**Формула:** `trend.keltner`.
- tp = (h+l+c)/3
- middle = EMA(tp, window) c init="sma" (и reset-on-NaN)
- band = ATR(window)*mult

**Primary output:** `middle`.

### G) `trend.linreg_slope(window, source)`
**Формула:** `trend.linreg_slope`.
- rolling_linreg_slope(y=source, x_mode="0..N-1")
- NaN дырки → NaN в окнах.

**Primary output:** `slope`.

### H) `trend.psar(accel_start, accel_step, accel_max)`
**Формула:** `trend.psar` через `op: psar_wilder`.
- stateful, direction/state reset на NaN дырках
- direction — int output (не возвращаем).

**Primary output:** `psar`.

### I) `trend.supertrend(window, mult)`
**Формула:** `trend.supertrend` через `op: supertrend`, atr_mode="wilder".
- stateful, direction/state reset на NaN дырках
- direction — int output (не возвращаем).

**Primary output:** `supertrend`.

### J) `trend.vortex(window)`
**Формула:** `trend.vortex`.
- vm+/vm- + rolling_sum + TR rolling_sum

**Primary output:** `vi_plus`.

---

## Реализуемые индикаторы Volume (v1)

### A) `volume.ad_line()`
**Формула:** `volume.ad_line`.
- MFM = ((close-low) - (high-close)) / (high-low)
- MFV = MFM * volume
- AD = cumsum(MFV)
NaN дырки:
- если нет high/low/close/volume → NaN и reset (cumsum).

**Primary output:** `ad_line`.

### B) `volume.cmf(window)`
**Формула:** `volume.cmf`.
- `cmf = rolling_sum(mfv) / rolling_sum(volume)`
- mfv берётся как `diff(ad_line, 1)`.

**Primary output:** `cmf`.
Особое правило:
- если `sum(volume)==0` → NaN (div-by-zero policy)

### C) `volume.obv()`
**Формула:** `volume.obv`.
- dir = sign(diff(close))
- signed_v = dir*volume
- obv = cumsum(signed_v)
NaN дырки → NaN и reset cumsum.

**Primary output:** `obv`.

### D) `volume.mfi(window)`
**Формула:** `volume.mfi`.
- tp = (h+l+c)/3
- rmf = tp*volume
- pos/neg потоки по сравнению tp vs prev_tp
- mfi = 100 - 100/(1+mr)
NaN дырки → NaN propagation (rolling_sum) + reset на diff/shift.

**Primary output:** `mfi`.

### E) `volume.volume_sma(window)`
**Формула:** `volume.volume_sma`.
- rolling_mean(volume, window)

**Primary output:** `volume_sma`.

### F) `volume.vwap(window)`
**Формула:** `volume.vwap`.
- tp = (h+l+c)/3
- vwap = rolling_sum(tp*volume)/rolling_sum(volume)
Если `sum(volume)==0` → NaN.

**Primary output:** `vwap`.

### G) `volume.vwap_deviation(window, mult)`
**Формула:** `volume.vwap_deviation`.
- vwap(window) + rolling_std(tp - vwap, window) → band = stdev*mult → upper/lower

**Primary output (фикс v1):** `vwap_upper`.
(Потому что это единственный output, который зависит и от `window`, и от `mult`, и отражает “deviation band”.)

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
- `AxisDef` по каждой оси (например `window`, `mult`, `smoothing`, `source`, и т.д.)
- `variants = product(axis_lengths)`

### 2) CandleFeed → CandleArrays
Используется `CandleFeed` ACL:
- `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py`

Получаем dense timeline `[start, end)` на 1m, с NaN дырками.

### 3) Guards
Применяются до больших аллокаций:
- variants guard (default: 600k, если не переопределено)
- memory guard (default: 5 GiB)

Ошибки: детерминированные payload’ы (422) как в EPIC-03.

### 4) Compute (Numba) + Output tensor
- Numba kernels:
  - `compute_numba/kernels/trend.py`
  - `compute_numba/kernels/volume.py`
- output: `IndicatorTensor(values=float32, axes, layout, meta)`

### 5) Oracle (Numpy)
- Numpy path:
  - `compute_numpy/trend.py`
  - `compute_numpy/volume.py`
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
   - warmup зоны для rolling окон
   - reset-on-NaN для stateful цепочек (ADX smoothing, psar/supertrend)
4) dtype/output layout invariants (float32, contiguous)

Предлагаемые пути:
- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_trend_kernels.py`
- `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_volume_kernels.py`
- `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_trend_oracle.py`
- `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_volume_oracle.py`

Минимальный “эталонный набор” для полного numba-vs-numpy сравнения:
- Trend: `trend.adx`, `trend.supertrend`, `trend.ichimoku` (покрывает rolling+stateful+shift)
- Volume: `volume.vwap_deviation`, `volume.mfi`, `volume.obv` (покрывает rolling_std + cumsum state)

Остальные допускается покрыть sanity-check’ами (shape/dtype/warmup/guards), но без пропусков ключевых семантик.

### Perf-smoke (обязательный)

- `tests/perf_smoke/contexts/indicators/test_indicators_trend_volume.py`

Сценарий:
- реальный `CandleFeed` (или реалистичная фикстура dense arrays)
- grid: например `trend.adx: window 5..120 × smoothing 5..50` (guarded subset),
  и `volume.vwap_deviation: window 5..200 × mult 0.5..4.0` (guarded subset)
- assert: не падает, укладывается в guards, логирует метрики времени/размера.

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
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/trend.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/volume.py`
- обновить: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/__init__.py`

### Numpy oracle
- `src/trading/contexts/indicators/adapters/outbound/compute_numpy/trend.py`
- `src/trading/contexts/indicators/adapters/outbound/compute_numpy/volume.py`
- обновить: `src/trading/contexts/indicators/adapters/outbound/compute_numpy/__init__.py`

### Engine wiring
- обновить маршрутизацию в:
  - `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
- при необходимости: `warmup.py`

### Docs
- добавить ссылку в `docs/architecture/indicators/README.md`
  на этот документ: `docs/architecture/indicators/indicators-trend-volume-compute-numba-v1.md`

---

## DoD (EPIC-08)

EPIC-08 считается выполненным, если:

1) `POST /indicators/compute` считает все `trend.*` и `volume.*` индикаторы из YAML defaults и применяет guards (variants + memory).
2) Реализованы Numba kernels для всех:
   - Trend: `adx, aroon, chandelier_exit, donchian, ichimoku, keltner, linreg_slope, psar, supertrend, vortex`
   - Volume: `ad_line, cmf, mfi, obv, volume_sma, vwap, vwap_deviation`
3) Есть Numpy oracle с идентичной семантикой + unit tests “numba vs numpy” (минимум на репрезентативном подмножестве + sanity checks на остальное).
4) Есть perf-smoke тест для trend+volume grid на реалистичной длине T.
5) Документация обновлена:
   - добавлен этот документ,
   - `docs/architecture/indicators/README.md` содержит ссылку,
   - публичные экспорты через `__init__.py` стабильны,
   - docstrings в новых/изменённых классах/протоколах ссылаются на этот документ и связанные файлы.
