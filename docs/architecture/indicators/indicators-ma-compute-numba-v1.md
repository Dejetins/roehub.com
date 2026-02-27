Название документа: **`indicators-ma-compute-numba-v1.md`**
Куда кладём: **`docs/architecture/indicators/indicators-ma-compute-numba-v1.md`**

---

# Indicators — MA compute (Numba/Numpy) + sources v1

Этот документ является source of truth для **IND-EPIC-06 — Реализация группы MA + базовые “строительные блоки”** в bounded context `indicators`.

Цель EPIC: дать первый “вертикальный” результат — **MA индикаторы считаются тензором по сетке параметров** на CPU через Numba, с **оракулом Numpy** для тестов, и подключены к API `POST /indicators/compute`.

Связанные документы (читать вместе):

* `docs/architecture/indicators/indicators-overview.md`
* `docs/architecture/indicators/indicators-compute-engine-core.md`
* `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`
* `docs/architecture/indicators/indicators-candlefeed-acl-dense-timeline-v1.md`
* `docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md`
* `docs/architecture/indicators/indicators_formula.yaml` (базовые формулы MA)

Связанные ключевые файлы/папки:

* Numba runtime: `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
* Numba kernels common: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/_common.py`
* Numba kernels MA (добавляем): `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py`
* Numpy oracle (добавляем): `src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py`
* Grid materialization: `src/trading/contexts/indicators/application/services/grid_builder.py`
* Candle arrays DTO: `src/trading/contexts/indicators/application/dto/candle_arrays.py`
* CandleFeed ACL: `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py`
* API route: `apps/api/routes/indicators.py`
* API DTO: `apps/api/dto/indicators.py`
* Config defaults: `configs/prod/indicators.yaml` (аналогично dev/test)

---

## Scope / Non-goals

### In scope (EPIC-06)

1. Реализация MA kernels на Numba:

* `ma.sma`, `ma.ema`, `ma.wma`/`ma.lwma`, `ma.rma`/`ma.smma`, `ma.vwma`
* multi-param / compositional:

  * `ma.dema`, `ma.tema`, `ma.zlema`, `ma.hma`

2. Sources для MA:

* базовые: `open`, `high`, `low`, `close`
* derived: `hl2`, `hlc3`, `ohlc4`

3. Numpy oracle для MA с **точно такой же семантикой** (для unit tests).

4. API `POST /indicators/compute`:

* считает **один индикатор за запрос** (один `indicator_id` + его `GridSpec`),
* использует `CandleFeed` (dense arrays + NaN holes),
* применяет guards (variants ≤ 600k, memory ≤ max_compute_bytes_total).

5. Тесты:

* unit: `compute_numba` vs `compute_numpy` на фиксированном seed (включая NaN-дырки),
* perf-smoke: “MA grid на реальных свечах” (без жёсткого SLA по времени, но обязателен smoke + guard coverage).

### Out of scope (EPIC-06)

* KAMA/ALMA/FRAMA/McGinley, envelopes (можно добавить позже отдельным follow-up).
* runtime DSL parsing `indicators_formula.yaml` (документ — источник спецификации, но реализация кодом).
* batch compute (несколько индикаторов в одном `/compute`) — это следующий уровень orchestration/jobs.

---

## Source of Truth

1. Идентичности индикаторов и их параметры:

* `src/trading/contexts/indicators/domain/definitions/ma.py`
* YAML defaults (UI ranges/steps):

  * `configs/prod/indicators.yaml` (и `dev/test`)

2. Формулы (спецификация):

* `docs/architecture/indicators/indicators_formula.yaml`
  Для каждого `ma.*`, который присутствует в YAML defaults, должна существовать запись формулы в `indicators_formula.yaml`.
  **Важно:** в EPIC-06 формулы реализуются вручную в kernels, но должны соответствовать смыслу `ops`.

3. Реальный input формат:

* `CandleFeed` возвращает dense timeline + NaN holes (см. `indicators-candlefeed-acl-dense-timeline-v1.md`).

---

## Ключевые решения

### 1) Один `/compute` = один индикатор (v1)

`POST /indicators/compute` в v1 выполняет расчёт **одного** `indicator_id` по его grid.
Batch оценка и ограничение размера батча делаются через `POST /indicators/estimate` (EPIC-03).

### 2) Sources — часть grid (обязательная ось, если параметризована)

Если индикатор параметризует `source`, то `source` является **обязательной** осью grid.
Порядок значений `source` в `explicit.values` сохраняется как в request (без сортировки).

### 3) NaN policy для MA (фикс v1)

Candles содержат NaN дырки. Compute **не делает импутацию**.

**SMA/WMA/VWMA (окна):**

* `t < window-1` → `NaN` (warmup зона)
* если в окне есть NaN → output NaN

**EMA/RMA (stateful):**

* до первого валидного `x[t]` → NaN
* при первом валидном `x[t]`: seed `y[t] = x[t]`
* если `x[t]` NaN → `y[t]=NaN` и **state reset**
* после reset, при следующем валидном `x[t]` стартуем заново с seed `y[t]=x[t]`

Эта политика гарантирует, что EMA не “перетягивает” состояние через пропуски.

### 4) HMA округления (детерминированно)

В HMA:

* `w2 = floor(window / 2)`
* `sqrt_w = floor(sqrt(window))`

### 5) Float32 output (фикс)

`IndicatorTensor.values` — `float32`.
Внутри kernels допустимы `float64` accumulator’ы (implementation detail), но выход фиксирован как `float32`.

### 5.1) Phase-5 precision policy (f32/mixed)

В MA kernels введён явный аргумент `precision` в `compute_ma_grid_f32(...)` c режимами:

- `float32`
- `mixed precision`
- `float64`

Policy для MA в v1:

- `Tier A` (`float32`): `ma.sma`, `ma.ema`, `ma.rma`/`ma.smma`, `ma.wma`/`ma.lwma`,
  `ma.dema`, `ma.tema`, `ma.zlema`, `ma.hma`.
- `Tier B` (`mixed precision`): `ma.vwma` (float32 output + float64 rolling accumulators).
- `Tier C`: для MA группы не используется как отдельная целевая миграция в phase-5.

### 6) Guards применяются до аллокаций тензора

До расчёта:

* `variants <= max_variants_per_compute`
* `estimated_memory_bytes <= max_compute_bytes_total`
  Где `estimated_memory_bytes` считается так же, как в EPIC-03/05: `bytes_out + reserve`, reserve = `max(64MiB, 20%)`.

---

## Реализуемые индикаторы MA (v1)

### A) Single-param MA

1. `ma.sma(window, source)`
   rolling mean

2. `ma.ema(window, source)`
   EMA с alpha = `2 / (window + 1)` и NaN reset policy (см. выше)

3. `ma.rma(window, source)` / `ma.smma(window, source)`
   RMA/SMMA как EMA с alpha = `1 / window` и той же NaN reset policy

4. `ma.wma(window, source)` / `ma.lwma(window, source)`
   линейно-взвешенное скользящее среднее (weights 1..window)

5. `ma.vwma(window, source)`
   VWMA = sum(price*volume) / sum(volume)

* если volume NaN или сумма volume = 0 → NaN
* NaN в окне → NaN

### B) Compositional / multi-step MA

1. `ma.dema(window, source)`
   DEMA = `2*EMA(x) − EMA(EMA(x))`

2. `ma.tema(window, source)`
   TEMA = `3*EMA − 3*EMA(EMA) + EMA(EMA(EMA))`

3. `ma.zlema(window, source)`
   ZLEMA (zero-lag EMA): входная серия компенсируется лагом (v1: лаг `floor((window-1)/2)`), затем EMA на скорректированной серии.
   (В документации формула должна быть отражена через steps/ops аналогично.)

4. `ma.hma(window, source)`
   HMA через WMA-композицию:

* `w2 = floor(window/2)`
* `sqrt_w = floor(sqrt(window))`
* `hma = WMA( 2*WMA(x,w2) − WMA(x,window), sqrt_w )`

---

## Sources (input series)

### Доступные источники (InputSeries)

* base: `open`, `high`, `low`, `close`
* derived:

  * `hl2 = (high + low) / 2`
  * `hlc3 = (high + low + close) / 3`
  * `ohlc4 = (open + high + low + close) / 4`

Правила:

* derived вычисляются в float32;
* NaN пропускается естественно (если любой компонент NaN → derived NaN);
* derived создаются как contiguous arrays.

---

## Execution pipeline (compute v1)

### 1) Request → Grid → Axes

* Вход: `ComputeRequest` (`apps/api/dto/indicators.py`) содержит:

  * `indicator_id`
  * `grid` (explicit/range specs)
  * `layout` (optional hint)
  * `max_variants_guard` (optional override, иначе runtime default)
* Grid материализуется через:

  * `src/trading/contexts/indicators/application/services/grid_builder.py`
* Результат:

  * `AxisDef` по каждой оси (`window`, `source`)
  * `variants = product(axis_lengths)`

### 2) CandleFeed → CandleArrays

* Использовать `CandleFeed` ACL:

  * `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/market_data_candle_feed.py`
* Получаем dense timeline `[start, end)` на 1m, с NaN дырками.

### 3) Guards

* применяются до больших аллокаций:

  * variants guard (600k default)
  * memory guard (5 GiB default)
* ошибки: детерминированные payload’ы (422) как в EPIC-03.

### 4) Compute (Numba) + Output tensor

* Numba path:

  * `compute_numba/kernels/ma.py` вызывает primitives из `_common.py`
  * output: `IndicatorTensor(values=float32, axes, layout, meta)`

### 5) Oracle (Numpy)

* Numpy path:

  * `compute_numpy/ma.py`
* Используется в тестах как эталон (на одинаковых входах и той же NaN policy).

---

## API: POST /indicators/compute (v1)

**Назначение:** вычислить один MA индикатор по сетке параметров.

* Endpoint: `apps/api/routes/indicators.py`
* DTO: `apps/api/dto/indicators.py`

**Семантика:**

* request задаёт `indicator_id` + grid (params/inputs).
* система читает свечи через CandleFeed (market_data ACL).
* возвращает `IndicatorTensor` (или компактный ответ v1 — в зависимости от уже принятого формата `/compute`).

**Важно:** если в v1 решено отдавать не весь тензор (например, только metadata), это должно быть отдельным контрактом. Для EPIC-06 предполагается, что тензор возвращается в “серверном” формате (внутренний вызов/use-case), а API может отдавать либо ссылку, либо малый результат — это зависит от текущего API контракта. В Milestone 2 допустимо вернуть тензор в памяти (ограничено guards).

---

## Tests

### Unit tests (обязательные)

Покрываем:

* корректность axes/materialization (window×source),
* совпадение numba vs numpy на фиксированном seed,
* корректность NaN policy:

  * warmup зона,
  * NaN дырки (state reset для EMA/RMA),
* dtype/output layout invariants (float32, contiguous).

Предлагаемые пути:

* `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_ma_kernels.py`
* `tests/unit/contexts/indicators/adapters/outbound/compute_numpy/test_ma_oracle.py`
* возможно расширение `tests/unit/contexts/indicators/adapters/outbound/compute_numba/test_engine.py`

### Perf-smoke (обязательный)

* `tests/perf_smoke/contexts/indicators/test_indicators_ma.py`
* сценарий:

  * реальный `CandleFeed` (или фикстура dense arrays),
  * grid: windows 1..200 × sources 4–6 (≤ 600k),
  * assert: не падает, укладывается в guards, логирует метрику времени/размера.

---

## Determinism & ordering rules (фикс)

1. Оси grid:

* `explicit.values` сохраняют порядок request (без сортировки).
* `range(start, stop_incl, step)` материализуется строго инклюзивно.

2. Variants ordering:

* порядок осей в `IndicatorDef` определяет порядок перемножения и variant indexing.
* variant enumeration детерминирована при одинаковом request.

3. Duplicates/holes в candles:

* политика `last-wins` на дубли,
* “out of range” игнорируется,
* NaN holes сохраняются (см. EPIC-04 документ).

---

## Целевое размещение в репозитории

### Numba kernels

* `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/ma.py`
* обновить: `src/trading/contexts/indicators/adapters/outbound/compute_numba/kernels/__init__.py`

### Numpy oracle

* `src/trading/contexts/indicators/adapters/outbound/compute_numpy/ma.py`
* обновить: `src/trading/contexts/indicators/adapters/outbound/compute_numpy/__init__.py`

### Engine wiring

* обновить маршрутизацию в:

  * `src/trading/contexts/indicators/adapters/outbound/compute_numba/engine.py`
* если нужен use-case:

  * `src/trading/contexts/indicators/application/use_cases/` (новый use-case compute)

### API

* `apps/api/routes/indicators.py` (добавить/расширить `/indicators/compute`)
* `apps/api/dto/indicators.py` (request/response)

### Docs

* добавить ссылку в `docs/architecture/indicators/README.md`

---

## DoD (EPIC-06)

EPIC-06 считается выполненным, если:

1. `POST /indicators/compute` считает MA по сетке (пример: windows 1..200 × sources 4) и применяет guards (≤ 600k + memory ≤ 5GiB).
2. Реализованы Numba kernels для:

   * SMA/EMA/WMA/LWMA/RMA/VWMA
   * DEMA/TEMA/ZLEMA/HMA
3. Есть Numpy oracle с идентичной семантикой и unit tests “numba vs numpy”.
4. Есть perf-smoke тест для MA grid на реалистичной длине T.
5. Документация обновлена:

   * добавлен этот документ,
   * `docs/architecture/indicators/README.md` содержит ссылку,
   * публичные экспорты через `__init__.py` стабильны.

---
