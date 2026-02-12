
---

# Indicators — CandleFeed ACL: dense timeline + NaN holes (v1)

Этот документ фиксирует архитектуру и контракты для **IND-EPIC-04 — CandleFeed ACL: dense timeline + NaN holes** в bounded context `indicators`.

Цель EPIC:

* индикаторы/compute получают **быстрые contiguous numpy arrays** и **не думают про пропуски**;
* построение временной сетки выполняется **детерминированно** по `TimeRange` и `Timeframe`;
* пропуски свечей представлены как **NaN** (без импутации/интерполяции).

---

## Scope / Non-goals

### In scope (EPIC-04)

1. Application port `CandleFeed` (уже существует) и его реализация как ACL-адаптера к `market_data`.
2. Функция/алгоритм `load_ohlcv_dense(...)`:

   * читает канонические свечи,
   * строит плотный таймлайн,
   * вставляет NaN в missing.
3. Детерминированные правила:

   * строгая валидация `time_range` на кратность таймфрейму,
   * политика обработки дублей: **last-wins**,
   * свечи вне диапазона: **игнорировать**.
4. Unit-тесты на корректность плотной сетки, NaN-дырки, порядок и политику дублей.

### Out of scope (EPIC-04)

* лечение пропусков (интерполяция/forward-fill) — запрещено в M2;
* чтение не-1m таймфреймов (в v1 целимся в `canonical_candles_1m`);
* оптимизации хранения результатов, кэширование, prefetch — отдельные EPIC’и;
* бектест/SLTP симуляция — не входит в Milestone 2.

---

## Связанные документы и файлы

### Документация (source-of-truth по соседним частям)

* `docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md` — application ports + DTO v1.
* `docs/architecture/indicators/indicators-compute-engine-core.md` — compute engine core (EPIC-05).
* `docs/architecture/indicators/indicators-grid-builder-estimate-guards-v1.md` — grid builder/estimate/guards (EPIC-03).
* `docs/architecture/market_data/market-data-application-ports.md` — порты и сторы `market_data`.
* `docs/architecture/shared-kernel-primitives.md` — `TimeRange`, `Timeframe`, `MarketId`, `Symbol`.

### Ключевые файлы в репозитории (реализация/контракты)

**Indicators (port + DTO):**

* `src/trading/contexts/indicators/application/ports/feeds/candle_feed.py`
* `src/trading/contexts/indicators/application/dto/candle_arrays.py`

**Indicators (ACL adapter target path):**

* `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/` *(создаётся/используется в EPIC-04)*

**Market Data (store port style reference):**

* `src/trading/contexts/market_data/application/ports/stores/` *(референс по стилю портов)*

---

## Source of Truth

* **Контракт данных на вход compute** фиксируется DTO `CandleArrays`:
  `ts_open[int64]` + `open/high/low/close/volume[float32]`, все массивы одной длины и contiguous.
* **Плотная временная сетка** строится только из `TimeRange` и `Timeframe`.
* **Пропуски** выражаются `NaN` и сохраняются дальше по пайплайну (`NaN propagation`).
* **Не читаем напрямую инфраструктуру** из `indicators`: доступ к свечам только через ACL-адаптер к `market_data`.

---

## Ключевые решения

### 1) Строгая кратность `TimeRange` таймфрейму

`TimeRange` должен быть строго кратен `timeframe` (в v1: `1m`).

* Если `(end - start) % timeframe != 0` → ошибка валидации (422 на уровне API / доменная ошибка на уровне портов).
* Семантика диапазона: **полуинтервал** `[start, end)`.

*(Это согласовано с текущей семантикой в estimate/guards: диапазон не “округляется”.)*

### 2) Плотный таймлайн строится независимо от данных

`ts_open` генерируется как арифметическая прогрессия:

* `step_ms = timeframe.to_millis()` (для 1m: 60_000)
* `T = (end - start) / step_ms`
* `ts_open[i] = start + i * step_ms`, `i = 0..T-1`

Это гарантирует:

* детерминизм,
* корректную длину тензора,
* единый “time grid” для всех индикаторов/вариантов.

### 3) Пропуски — только NaN, без импутации

Все OHLCV серии инициализируются `NaN` и заполняются только там, где есть свеча.

* “лечения” пропусков нет (не делаем forward-fill / interpolation).
* NaN считается “истиной” для compute.

### 4) Политика дублей: last-wins (детерминированно)

Если в источнике встречаются дубли с одинаковым `ts_open`:

* после детерминированной сортировки по `ts_open` применяется политика **last-wins**.

Это даёт:

* стабильное поведение,
* resilience к редким дубликатам при ingest/merge.

### 5) Свечи вне диапазона — игнорируем (детерминированно)

Если store/источник вернул свечу с `ts_open < start` или `ts_open >= end`:

* такая свеча **игнорируется**.

Это защищает контур `indicators` от “грязных” источников и сохраняет детерминизм.

---

## Контракты: CandleFeedPort и CandleArrays

### CandleFeedPort

Контрактный порт остаётся в application слое:

* `src/trading/contexts/indicators/application/ports/feeds/candle_feed.py`

v1 метод:

* `load_1m_dense(market_id: MarketId, symbol: Symbol, time_range: TimeRange) -> CandleArrays`

Семантика:

* возвращает свечи на полуинтервале `[start, end)`,
* строит плотную сетку времени для 1m,
* вставляет NaN на пропусках.

> Примечание: для будущего расширения на другие ТФ допускается внутренний helper `load_ohlcv_dense(..., timeframe)` в адаптере, но публичный порт v1 остаётся `1m`-специфичным.

### CandleArrays (DTO)

* `src/trading/contexts/indicators/application/dto/candle_arrays.py`

Инварианты v1:

* все массивы 1D и одинаковой длины `T`,
* `ts_open.dtype == int64`, строго возрастающий,
* `open/high/low/close/volume.dtype == float32`,
* массивы contiguous (C-order).

---

## Алгоритм `load_ohlcv_dense(...)`

### Входные данные

* `market_id`, `symbol`, `time_range`
* `timeframe` (v1: фикс `1m`)

### Выход

* `CandleArrays(ts_open, open, high, low, close, volume)`

### Псевдо-алгоритм (детерминированный)

1. **Validate alignment**

   * `step_ms = timeframe.to_millis()`
   * `delta = end - start`
   * `delta % step_ms == 0` MUST hold, иначе ошибка.

2. **Allocate arrays**

   * `T = delta // step_ms`
   * `ts_open = arange(T, dtype=int64) * step_ms + start`
   * `open/high/low/close/volume = empty(T, float32); fill(NaN)`

3. **Load sparse candles from market_data**

   * вызвать `market_data` store port (ACL) для чтения свечей на `[start, end)`.

4. **Deterministic normalization**

   * свечи сортируются по `ts_open` ascending (стабильная сортировка),
   * свечи вне `[start, end)` игнорируются.

5. **Materialize into dense arrays**

   * `idx = (ts_open_candle - start) // step_ms`
   * заполнение OHLCV в позиции `idx`
   * при дублях `ts_open` — last-wins достигается естественно за счёт порядка после сортировки.

6. **Return CandleArrays**

   * гарантируется contiguous + инварианты DTO.

---

## Размещение в репозитории

### Adapter path (ACL)

Создаём/используем:

* `src/trading/contexts/indicators/adapters/outbound/feeds/market_data_acl/`

  * `__init__.py` (стабильные экспорты)
  * `market_data_candle_feed.py` *(рекомендуемое имя)*

Адаптер зависит от `market_data` портов чтения свечей (store port).

### Wiring

Подключение адаптера в composition root:

* `apps/api/wiring/modules/indicators.py`
  CandleFeedPort биндим на `MarketDataCandleFeed` (ACL adapter).

---

## Ошибки и валидация

В v1 достаточно:

* ошибка “time_range not aligned to timeframe” (validation error)
* (опционально) ошибка “empty time_range” если `T == 0` и это запрещено политикой (обычно запрещено)

Дубли и out-of-range свечи **не являются ошибкой** в v1:

* дубли → last-wins
* out-of-range → ignore

---

## Тесты (DoD)

Цель тестов — подтвердить:

1. **Длина и таймлайн**

* `len(ts_open) == T`,
* `ts_open` строго возрастающий и соответствует `[start, end)`.

2. **NaN holes**

* пропущенный timestamp приводит к `NaN` во всех OHLCV на позиции.

3. **Порядок и детерминизм**

* входные свечи могут быть неотсортированы → выход корректен.

4. **Duplicates policy**

* при двух свечах с одинаковым `ts_open` побеждает **последняя** (после сортировки).

5. **Out-of-range handling**

* свечи с `ts_open < start` или `>= end` игнорируются и не ломают результат.

6. **Alignment**

* некратный `time_range` → ошибка валидации.

Рекомендуемые пути тестов:

* `tests/unit/contexts/indicators/adapters/outbound/feeds/test_market_data_acl_candle_feed.py`
* (опционально) `tests/perf_smoke/...` без жёсткого SLA, только sanity на больших `T` + contiguous.

---

## Perf notes (чтобы не стать бутылочным горлом)

Рекомендации к реализации:

* не использовать per-candle dict lookup,
* заполнение делать через индексы `idx` и vectorized assignment,
* аллокации: `np.empty + fill(np.nan)` для float32 массивов,
* обеспечить contiguous arrays (C-order) на выходе DTO.

---

## DoD (EPIC-04)

EPIC-04 считается выполненным, если:

1. Реализован `CandleFeedPort` адаптер `market_data_acl`, возвращающий `CandleArrays` с плотным `ts_open` и NaN holes.
2. Соблюдены решения:

   * strict alignment,
   * last-wins по дублям,
   * out-of-range игнорируется.
3. Добавлены unit-тесты, подтверждающие длину, NaN-дырки, порядок, дубликаты, alignment.
4. Построение dense arrays не является очевидным bottleneck (perf-smoke/sanity по необходимости).

---
