# Shared Kernel — Primitives (Trading)

Этот документ фиксирует минимальные доменные примитивы Shared Kernel и правила, которые должны быть одинаковыми во всех bounded contexts.

Цель:
- единые идентичности (`MarketId`, `Symbol`, `InstrumentId`);
- единая идентичность пользователя (`UserId`) для межконтекстной связности;
- единые правила времени (`UtcTimestamp`, `TimeRange`);
- единая каноническая свеча (`Candle`) и отдельно её метаданные (`CandleMeta`);
- единый параметр таймфрейма (`Timeframe`) и правила rollup из 1m.

Источник правды по хранению данных рынка: ClickHouse DDL (см. таблицы `market_data.*`), в частности `market_data.canonical_candles_1m`.


## Ключевые решения

### 1) Идентичность рынка
- Рынок идентифицируется `MarketId` (UInt16 в ClickHouse), соответствующий `market_data.ref_market.market_id`.
- `market_type` (spot/futures) и биржа (binance/bybit) — детали справочника `ref_market`, в доменных примитивах отдельно не хранятся.

### 2) Идентичность инструмента
- Доменный идентификатор инструмента: `InstrumentId = (market_id, symbol)`.
- Поле `instrument_key` в свечных таблицах — trace/debug строка (для логов/отладки), **не** доменная идентичность.

### 3) Время
- Канонический формат времени в домене: timezone-aware `datetime` в UTC.
- Точность: миллисекунды (совместимо с `DateTime64(3, 'UTC')`).

### 4) Диапазоны времени
- `TimeRange` трактуется как полуинтервал: `[start, end)`.

### 5) Числа (OHLCV)
- Цены и объёмы в `Candle` представлены как `float`, совместимо с `Float64` в ClickHouse.
- Точный денежный учёт (Decimal) — отдельное решение для будущих контекстов (execution/pnl/risk), не для канонических свечей.

### 6) Таймфреймы
- Хранение: только 1m canonical (`canonical_candles_1m`) — источник правды.
- Любые другие TF — derived, строятся rollup’ом из 1m.
- Rollup возвращает только **закрытые и полные бакеты** (строгая полнота).


## Примитивы

### MarketId

**Purpose**  
`MarketId` — стабильный идентификатор рынка (биржа + тип рынка) из `market_data.ref_market.market_id`. Используется как ключ рынка во всех свечных таблицах.

**Representation**  
- целое число (концептуально `UInt16` в ClickHouse; в домене — value object вокруг `int`).

**Invariants**  
- `market_id > 0`.

**Source of truth (DDL)**  
- `market_data.ref_market.market_id UInt16`
- `market_id UInt16` в `raw_*` и `canonical_*`.

**Serialization**  
- как целое число (например, JSON number).


---

### Symbol

**Purpose**  
`Symbol` — обозначение инструмента внутри конкретного `MarketId` (например `BTCUSDT`).

**Representation**  
- строка.

**Normalization**  
- `strip()`
- `upper()`

**Invariants**  
- не пустая строка после нормализации.

**Source of truth (DDL)**  
- `symbol LowCardinality(String)` в свечных таблицах и в `ref_instruments`.

**Serialization**  
- как строка.


---

### InstrumentId

**Purpose**  
`InstrumentId` — доменная идентичность инструмента. Определяется парой: `(market_id, symbol)`.

> Примечание: поле `instrument_key` (`"{exchange}:{market_type}:{symbol}"`) — trace/debug поле.  
> Оно помогает отладке и трассировке ingestion, но не заменяет доменный ключ.

**Fields**  
- `market_id: MarketId`
- `symbol: Symbol`

**Invariants**  
- оба поля валидны по своим инвариантам.

**Source of truth (DDL)**  
- свечные таблицы используют сортировку/ключ, включающий `(market_id, symbol, ts_open/open_time)`.

**Serialization**  
- канонично: `{ "market_id": <int>, "symbol": "<str>" }`.


---

### UserId

**Purpose**  
`UserId` — сквозной идентификатор пользователя для всех bounded contexts, где нужен
стабильный субъект (strategy/backtest/optimize/risk/identity).

**Representation**  
- value object вокруг `UUID` (см. `src/trading/shared_kernel/primitives/user_id.py`).

**Invariants**  
- значение обязано быть валидным `UUID`;
- строковый формат должен быть канонически парсируем через UUID-представление.

**Source of truth (DDL)**  
- `identity_users.user_id UUID` в `migrations/postgres/0001_identity_v1.sql`.

**Serialization**  
- как UUID-строка (например `00000000-0000-0000-0000-000000000001`).

> Примечание: `UserId` — не Telegram-specific идентификатор.  
> `telegram_user_id` остаётся отдельным identity-атрибутом/ключом входа.


---

### UtcTimestamp

**Purpose**  
`UtcTimestamp` — единый тип времени в системе с жёстким требованием UTC (запрещаем смешение локального времени и naive datetime).

**Representation**  
- timezone-aware `datetime` в UTC.

**Precision**  
- миллисекунды (совместимо с `DateTime64(3, 'UTC')`).

**Invariants**  
- timezone-aware, интерпретируется как UTC (naive datetime запрещён).

**Source of truth (DDL)**  
`DateTime64(3, 'UTC')` используется в:
- `open_time`, `close_time`
- `ts_open`, `ts_close`
- `ingested_at`
- `updated_at` в справочниках

**Serialization**  
- ISO-строка в UTC (например `2026-02-04T12:34:56.789Z`) или согласованный формат проекта, при условии UTC + ms.


---

### TimeRange

**Purpose**  
`TimeRange` — стандартный диапазон времени для запросов данных, backfill/ingestion и построения derived таймфреймов.

**Fields**  
- `start: UtcTimestamp`
- `end: UtcTimestamp`

**Boundary rule**  
- диапазон — полуинтервал `[start, end)`.

**Invariants**  
- `start < end`.

**Serialization**  
- `{ "start": <UtcTimestamp>, "end": <UtcTimestamp> }`.


---

### Timeframe (derived from 1m source)

**Purpose**  
`Timeframe` — параметр запроса/расчёта (для стратегий/бектеста/фидов), а не обязательное отражение “как хранится в БД”.  
Источник правды по хранению свечей: `market_data.canonical_candles_1m` (1m).  
Любые `5m/15m/1h/...` — derived timeframes, строятся rollup’ом из 1m.

**Representation**
- `code`: строковый код (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`).
- `duration`: длительность таймфрейма как концепт (минуты/часы/дни).

**Supported set**
- `1m` (base / source of truth)
- `5m`, `15m`, `1h`, `4h`, `1d` (derived)

Набор может расширяться, но любой derived timeframe должен быть кратен 1 минуте.

**Base timeframe rule**
- `BASE_TIMEFRAME = 1m`
- все прочие таймфреймы строятся исключительно из 1m-свечей.

#### Rollup rules (from 1m)

1) **Bucket alignment (UTC, epoch-aligned)**  
- `bucket_open = floor(ts_open, timeframe)`  
- `bucket_close = bucket_open + timeframe.duration`

Примеры:
- `15m`: `:00, :15, :30, :45`
- `1h`: `HH:00`
- `1d`: `00:00` UTC

2) **Inclusion rule (полуинтервал)**  
В бакет входят 1m-свечи с:
- `bucket_open <= ts_open < bucket_close`

3) **Canonical OHLCV aggregation**
- `open`  = open первой 1m-свечи в бакете (по `ts_open`)
- `close` = close последней 1m-свечи в бакете
- `high`  = max(high) по бакету
- `low`   = min(low) по бакету
- `volume_base`  = sum(volume_base)
- `volume_quote` = sum(volume_quote) если доступно, иначе `NULL`
- дополнительные поля (если есть в canonical): суммируются (`trades_count`, `taker_buy_*`) или `NULL`.

4) **Completeness rule (фиксируем)**
Rollup возвращает только **закрытые и полные бакеты**:
- бакет закрыт, если его `bucket_close` целиком лежит внутри запрошенного `TimeRange` (учитывая `[start, end)`),
- бакет полный, если присутствуют **все** 1m-свечи, которые должны входить в бакет по правилу включения.

Если бакет неполный или не закрыт — derived-свеча для него не возвращается.

**Storage rule**
- хранение: только `canonical_candles_1m` (1m)
- derived вычисляются из 1m по правилам rollup.

**Serialization**
- в запросах/DTO таймфрейм сериализуется как строка `code`.


---

### Candle

**Purpose**  
`Candle` — каноническое рыночное представление свечи для стратегий/бектеста/ML.  
Источник правды по 1m: `market_data.canonical_candles_1m`.

**Fields (market data)**
- `instrument_id: InstrumentId`  (эквивалент `(market_id, symbol)`)
- `ts_open: UtcTimestamp`
- `ts_close: UtcTimestamp`
- `open: float`
- `high: float`
- `low: float`
- `close: float`
- `volume_base: float`
- `volume_quote: Optional[float]`

**Invariants**
- `ts_open < ts_close`
- `high >= max(open, close)`
- `low <= min(open, close)`
- `volume_base >= 0`
- если `volume_quote` задан, то `volume_quote >= 0`

**Numeric type decision**
- float (совместимо с `Float64` в ClickHouse).

**Source of truth (DDL)**
`market_data.canonical_candles_1m`:
- `ts_open`, `ts_close`
- `open/high/low/close`
- `volume_base`, `volume_quote`
и ключи `(market_id, symbol)`.

**Serialization**
- объект с полями свечи + вложенный `instrument_id`.


---

### CandleMeta

**Purpose**  
`CandleMeta` — метаданные происхождения и записи свечи (ingestion metadata).  
Хранится отдельно от рыночных полей (`Candle`) и не смешивается с рыночной логикой.

**Fields (ingestion metadata)**
- `source: str` — источник данных, допустимые значения (минимум):
  - `ws`   (websocket)
  - `rest` (REST API)
  - `file` (ручная/пакетная загрузка исторических данных из файлов)
- `ingested_at: UtcTimestamp` — версия записи (для `ReplacingMergeTree(ingested_at)`)
- `ingest_id: Optional[UUID]` — идентификатор запуска ingestion (если используется)
- `instrument_key: str` — trace/debug строка для логов/отладки (например `"{exchange}:{market_type}:{symbol}"`)
- `trades_count: Optional[int]`
- `taker_buy_volume_base: Optional[float]`
- `taker_buy_volume_quote: Optional[float]`

**Invariants**
- `source` принадлежит допустимому набору.
- `ingested_at` — валидный `UtcTimestamp`.

**Source of truth (DDL)**
`market_data.canonical_candles_1m`:
- `source`, `ingested_at`, `ingest_id`, `instrument_key`
- `trades_count`, `taker_buy_volume_base`, `taker_buy_volume_quote`

**Serialization**
- как объект метаданных (например вложенное поле `meta` рядом с `candle` в DTO),
  либо как отдельная структура, сопровождающая свечу в результатах feed’ов/репозиториев.
