# Indicators — Grid Builder + Batch Estimator + Guards (v1)

**Документ:** `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`
**Контекст:** `src/trading/contexts/indicators`
**EPIC:** **IND-EPIC-03 — Grid builder + estimator + guards (600k)**

Этот документ фиксирует архитектуру, контракты и детерминированную семантику для:

* раскрытия сеток параметров (`explicit` / `range`) **для каждого индикатора**,
* подсчёта итогового количества комбинаций **для всего батча** (несколько индикаторов + SL/TP),
* оценки памяти батча,
* публичных guards:

  * `max_variants_per_compute = 600_000` (лимит на *total_variants* батча),
  * `max_compute_bytes_total` (лимит на *estimated_memory_bytes* батча; дефолт 5 GiB, конфигурируемо),
* API endpoint: `POST /indicators/estimate` (возвращает только totals, без preview осей).

> Важно: Milestone 2 **не** включает backtest. `estimate` — это **preflight** (budget check) до запуска будущих расчётов.

---

## Связанные документы и исходники

### Архитектура indicators (контекст)

* `docs/architecture/indicators/indicators-overview.md`
* `docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md`
* `docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md`
* `docs/architecture/indicators/indicators-compute-engine-core.md`

### Доменные/DTO контракты (уже в репозитории)

* Grid/DTO:

  * `src/trading/contexts/indicators/application/dto/grid.py`
  * `src/trading/contexts/indicators/application/dto/estimate_result.py`
* Domain defs:

  * `src/trading/contexts/indicators/domain/entities/indicator_def.py`
  * `src/trading/contexts/indicators/domain/entities/param_def.py`
  * `src/trading/contexts/indicators/domain/entities/axis_def.py`
  * `src/trading/contexts/indicators/domain/specifications/grid_spec.py`
  * `src/trading/contexts/indicators/domain/specifications/grid_param_spec.py`
* Ошибки:

  * `src/trading/contexts/indicators/domain/errors/grid_validation_error.py`
  * `src/trading/contexts/indicators/domain/errors/unknown_indicator_error.py`

### API слой (существует и расширяется)

* Route:

  * `apps/api/routes/indicators.py`
* DTO API (если расширяем/добавляем):

  * `apps/api/dto/indicators.py`

---

## Scope / Non-goals

### In scope (EPIC-03)

1. **Grid builder**

* раскрытие `explicit` и `range(start/stop_incl/step)` в детерминированные оси,
* валидация против `IndicatorDef` (тип/границы/step),
* правило: если `source` параметризуемый — это **обязательная ось** и участвует в комбинаторике.

2. **Batch estimator**

* расчёт `total_variants` для **всего батча** (несколько индикаторов + risk SL/TP),
* оценка `estimated_memory_bytes` (float32) + internal reserve под workspace,
* расчёт `T` (длина плотного таймлайна) **без чтения свечей**: только из `time_range` и `timeframe`.

3. **Guards**

* `total_variants <= max_variants_per_compute (600k)`,
* `estimated_memory_bytes <= max_compute_bytes_total` (дефолт 5 GiB).

4. **API**

* `POST /indicators/estimate` возвращает только totals.

### Out of scope (EPIC-03)

* любые preview осей (`first/last`, примеры значений, списки значений),
* compute формулы/тензоры (EPIC-06+),
* загрузка свечей из market_data для estimate,
* float16/mixed precision (фиксируем float32 в Milestone 2),
* job/queue/batch execution.

---

## Ключевые решения

### 1) Estimate — всегда по **всему батчу**

Пользователь выбирает **несколько индикаторов**, задаёт диапазоны параметров, задаёт SL/TP и нажимает “рассчитать комбинации”. Сервер обязан вернуть две итоговые величины:

* `total_variants`
* `estimated_memory_bytes`

### 2) SL/TP участвуют в комбинаторике

`total_variants` включает:

* произведение вариантов по всем индикаторам,
* умножение на варианты по SL и TP.

### 3) `source` — обязательная ось, если параметризуется

Если индикатор допускает выбор `source` (например `close/hlc3/...`) и он объявлен параметризуемым в `IndicatorDef`, то `source` участвует в вариантах как обычная ось.

### 4) Порядок `explicit values` сохраняем как в request

Это важно для UX и повторяемости “как пользователь задал”. Валидация не меняет порядок.

### 5) Один memory-параметр, reserve рассчитываем внутри

Оперируем одним конфигом `max_compute_bytes_total`, а reserve на workspace вычисляется автоматически внутри estimator’а (policy фиксируем ниже).

---

## Модель батча и формулы

### Термины

* **Indicator block** — один индикатор с его осями (params + inputs axes) → `indicator_variants`.
* **Risk block** — оси SL и TP → `risk_variants`.
* **Batch** — декартово произведение всех блоков → `total_variants`.

### Итоговая комбинаторика

Для каждого индикатора `i`:

```
indicator_variants_i = Π len(axis_values_ik)
```

Risk:

```
sl_variants = len(sl_axis_values)
tp_variants = len(tp_axis_values)
risk_variants = sl_variants * tp_variants
```

Batch:

```
total_variants = (Π indicator_variants_i) * risk_variants
```

---

## Grid builder (application service)

**Файл:** `src/trading/contexts/indicators/application/services/grid_builder.py`

### Supported specs v1

#### Explicit values

* `mode: explicit`
* `values: [...]`
* длина оси: `len(values)` (порядок сохраняется как в request)

#### Range (inclusive)

* `mode: range`
* `start`, `stop_incl`, `step`
* значения: `start, start+step, ... <= stop_incl`

### Валидация (Must)

Для каждого axis spec:

1. **Тип**

* соответствует `ParamDef.kind` (`int/float/enum`).

2. **Bounds**

* все значения должны попадать в `[hard_min, hard_max]` (или `enum_values` для enum).

3. **Step**

* `step > 0`.
* для `range` обязательно задан.
* для `explicit` step не нужен, но значения всё равно валидируются на bounds и тип.

4. **Range non-empty**

* range обязан материализоваться хотя бы в одно значение.

5. **Детерминизм float-range**

* materialization через `n = floor((stop_incl - start) / step) + 1`,
* `value_i = start + i*step`,
* затем детерминированная нормализация (если есть `precision`/квантование в домене — использовать его; иначе строгая математика + минимальный eps только для проверок).

> Примечание: строгую логику квантования/precision лучше централизовать рядом с `ParamDef`/`GridParamSpec`, чтобы одинаково работало и в estimate, и в compute.

---

## Estimator: total_variants + memory estimate

### Входы оценщика

Estimator использует только:

* batch request (списки индикаторов + их grid specs + risk spec),
* `IndicatorRegistryPort` (`src/trading/contexts/indicators/application/ports/registry/indicator_registry.py`) для получения `IndicatorDef`,
* `time_range` и `timeframe` (для расчёта `T`).

Estimator **не** читает свечи.

### Расчёт T (длина плотного таймлайна)

`T` — ожидаемая длина плотного массива свечей:

* для `timeframe=1m`:

  ```
  T = dateDiff('minute', start, end)  // для полуинтервала [start, end)
  ```
* в терминах shared-kernel предпочтительно использовать `TimeRange`/`Timeframe`:

  * `src/trading/shared_kernel/primitives/time_range.py`
  * `src/trading/shared_kernel/primitives/timeframe.py`

> Полуинтервал `[start, end)` обязателен для детерминизма и совместимости с CandleFeed контракта.

---

## Оценка памяти (policy v1)

Milestone 2 фиксирует `float32` output для тензоров индикаторов.

### Компоненты памяти

1. **Candles arrays (dense OHLCV + ts_open)**

* OHLCV: 5 массивов `float32` → `5 * 4` байт на элемент,
* ts_open: `int64` → `8` байт на элемент:

```
bytes_candles = T * (5*4 + 8)
```

2. **Indicator tensors**
   В v1 считаем: **1 float32 tensor на индикатор** с shape `(T, indicator_variants_i)` (layout не влияет на объём):

```
bytes_indicators = Σ (T * indicator_variants_i * 4)
```

> Если позже индикатор имеет несколько output-компонент (например bands), расширяем на `output_components_i`:
> `T * indicator_variants_i * output_components_i * 4`.

3. **Reserve (workspace)**
   Один параметр для пользователя — `max_compute_bytes_total`, но reserve рассчитываем внутри:

```
reserve = max(64MiB, 0.20 * (bytes_candles + bytes_indicators))
```

4. **Итог**

```
estimated_memory_bytes = bytes_candles + bytes_indicators + reserve
```

### Почему risk (SL/TP) не увеличивает память индикаторов

Risk параметры влияют на количество вариантов будущего backtest, но **не увеличивают** размер indicator tensors (тензоры зависят от параметров индикаторов). Поэтому в memory estimate v1 risk не участвует.

---

## Guards (public)

### Variants guard

* `total_variants <= max_variants_per_compute`
* default: `600_000`
* если превышение → `422` с понятной ошибкой и значениями:

  * `total_variants`
  * `max_variants_per_compute`

### Memory guard

* `estimated_memory_bytes <= max_compute_bytes_total`
* default: `5 GiB` (через конфиг runtime indicators)
* если превышение → `422` с:

  * `estimated_memory_bytes`
  * `max_compute_bytes_total`
  * * breakdown (опционально для лога; в API v1 не обязателен)

---

## API: POST /indicators/estimate (v1)

**Файл:** `apps/api/routes/indicators.py`

### Назначение

UI получает **только**:

* `total_variants`
* `estimated_memory_bytes`

Никаких preview значений осей.

### Request (draft contract)

В v1 предполагаем, что request содержит:

* `timeframe`, `time_range`,
* `indicators[]` с `indicator_id` и grid specs по inputs/params,
* `risk` с `sl` и `tp` grid specs,
* (опционально) overrides guards, но дефолты берём из runtime config.

### Response (строгий минимальный)

```json
{
  "schema_version": 1,
  "total_variants": 123456,
  "estimated_memory_bytes": 2147483648
}
```

### Ошибки

`422 Unprocessable Entity`:

* grid invalid (bounds/type/step/empty range/unknown ids),
* variants exceeded,
* memory exceeded.

---

## Детерминизм & ordering

### Explicit ordering (фикс)

* порядок `values` в `mode: explicit` сохраняется как в request (включая `source`).

### Независимость от порядка индикаторов в запросе

* `total_variants` и `estimated_memory_bytes` не зависят от порядка перечисления индикаторов в request.

### Float-range детерминизм

* materialization строго по индексу `i` и `n`, без накопления ошибок через “while x <= stop”.

---

## Интеграции и зависимости по EPIC’ам

* **EPIC-01/02**: `IndicatorDef`/`ParamDef` — источник hard bounds и структуры осей.
* **EPIC-05**: reserve policy и memory-guard согласован с `max_compute_bytes_total`.
* **EPIC-04**: estimate не читает market_data, но использует ту же семантику `T` как у CandleFeed dense timeline.

---

## Тесты (минимальный DoD)

Рекомендуемые файлы:

* `tests/unit/contexts/indicators/application/services/test_grid_builder.py`
* `tests/unit/contexts/indicators/api/test_indicators_estimate.py` (или рядом с текущими тестовыми соглашениями)

Покрыть:

1. **Range int**: `(5..7 step 1)` → 3 значения
2. **Range float**: детерминизм `n` и значений
3. **Validation**:

* `step <= 0`
* `start > stop_incl`
* out of bounds
* unknown `indicator_id` / unknown param/input

4. **Batch total_variants**:

* 2 индикатора + SL/TP → корректное перемножение

5. **Guards**:

* превышение 600k
* превышение memory (например большой `T`)

---

## Paths / где лежит реализация

* Grid builder:

  * `src/trading/contexts/indicators/application/services/grid_builder.py`
* API endpoint:

  * `apps/api/routes/indicators.py`
* DTO request/response (если нужно расширить/добавить):

  * `apps/api/dto/indicators.py`
  * `src/trading/contexts/indicators/application/dto/grid.py`
  * `src/trading/contexts/indicators/application/dto/estimate_result.py`

---

## Definition of Done (EPIC-03)

1. `POST /indicators/estimate` возвращает **только**:

   * `total_variants`
   * `estimated_memory_bytes`

2. `total_variants` учитывает:

   * все индикаторы в батче,
   * `source` как ось (если параметризуемый input),
   * `sl`/`tp` как оси.

3. Guards работают:

   * `total_variants > 600k` → `422`
   * `estimated_memory_bytes > max_compute_bytes_total` → `422`

4. Детерминизм:

   * `explicit` сохраняет порядок request,
   * float-range materialization детерминированный.

5. Добавлены unit-тесты на grid builder + estimate endpoint.

---

## Что обновить в docs/README индикаторов

Добавить ссылку на этот документ в:

* `docs/architecture/indicators/README.md`

Рекомендуемая строка:

* `Indicators — Grid Builder + Batch Estimate + Guards (v1)` → `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md`
