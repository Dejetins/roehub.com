# Market Data — Application Ports (Walking Skeleton v1)

Этот документ фиксирует минимальные контракты (ports) application-слоя для bounded context `market_data`
в рамках первого вертикального среза (walking skeleton).

Цель:
- определить порты для ingestion/backfill 1m свечей из источников `ws/rest/file`;
- записывать данные в ClickHouse через `raw_*` таблицы;
- получать канонические 1m свечи из `market_data.canonical_candles_1m` (источник правды) с контролируемым дедупом на хвосте.

Источник правды по хранению данных рынка: ClickHouse DDL, в частности:
- `market_data.raw_binance_klines_1m`
- `market_data.raw_bybit_klines_1m`
- `market_data.canonical_candles_1m`
- MV `mv_raw_*_to_canonical_1m` (raw -> canonical)


## Ключевые решения

### 1) Canonical формируется автоматически (через MV), прямых записей в canonical нет
Application-слой ingestion пишет только в `raw_*` (через порт `RawKlineWriter`).
Таблица `canonical_candles_1m` заполняется Materialized View автоматически.

Следствие:
- порта `CanonicalCandleWriter` **не существует**;
- “источник правды” для чтения — `canonical_candles_1m`.


### 2) Источники данных — `ws/rest/file`, но порты не зависят от конкретной биржи
`CandleMeta.source` фиксирует происхождение: `ws | rest | file`.

Порты не содержат “Binance/Bybit/Vision” в именах и не встраивают знания о протоколах/форматах.
Конкретная биржа и формат данных — детали адаптеров и wiring (composition root).

### 3) Дедуп при чтении canonical делаем только на хвосте последних 24 часов
Таблица `canonical_candles_1m` использует `ReplacingMergeTree(ingested_at)`, что означает eventual dedup
(дубликаты возможны до фоновых merge).

Чтобы не использовать `FINAL/argMax` на всей истории, контракт чтения фиксирует:
- дедуп обязателен **только** для данных, пересекающих последние 24 часа;
- для более старых данных допускается “как есть”, так как они считаются уже смердженными
  (или обеспечиваются операционной политикой `OPTIMIZE` на последних партициях).

Дедуп реализуется внутри адаптера ClickHouse (implementation detail), например через:
- `ORDER BY ingested_at DESC LIMIT 1 BY (market_id, symbol, ts_open)` для хвоста,
или другие эквивалентные техники без раскрытия в application-слое.

### 4) Clock — обязательная зависимость application-слоя
Clock нужен для:
- стабильного определения границы “последних 24 часов”;
- детерминированных тестов;
- согласованной трактовки “сейчас” внутри use-cases.

Clock не заменяет `UtcTimestamp` и не влияет на доменные примитивы. Это port application-слоя.


## DTO (application)

DTO в application-слое — не Pydantic, не зависит от transport/API.

### CandleWithMeta

**Purpose**  
`CandleWithMeta` — упаковка канонической свечи `Candle` и метаданных ingestion `CandleMeta` как единого результата
(для ingestion pipeline и для чтения canonical при необходимости).

**Fields**
- `candle: Candle`
- `meta: CandleMeta`

**Invariants**
- `candle` валиден по инвариантам `Candle`
- `meta` валиден по инвариантам `CandleMeta`


## Ports

### Clock

**Purpose**  
`Clock` — источник “текущего времени” для application-слоя в UTC.

**Contract**
- `now() -> UtcTimestamp`

**Invariants**
- `now()` всегда возвращает timezone-aware UTC (`UtcTimestamp`).


---

### CandleIngestSource

**Purpose**  
`CandleIngestSource` — порт получения 1m свечей из внешних источников (`ws/rest/file`) для ingestion/backfill.
Порт source-agnostic: не привязан к конкретной бирже, протоколу, формату или способу доставки.

**Contract**
- `stream_1m(instrument_id: InstrumentId, time_range: TimeRange) -> Iterator[CandleWithMeta]`

**Semantics**
- возвращает свечи в пределах полуинтервала `[time_range.start, time_range.end)`
- SHOULD: выдача отсортирована по `candle.ts_open` по возрастанию
- `meta.source` принадлежит допустимому набору: `ws | rest | file`
- `meta.instrument_key` — trace/debug поле (например `"{exchange}:{market_type}:{symbol}"`), формируется в адаптере


---

### RawKlineWriter

**Purpose**  
`RawKlineWriter` — порт записи 1m свечей в ClickHouse raw-таблицы (`raw_*_klines_1m`).
Canonical формируется автоматически через MV.

**Contract**
- `write_1m(rows: Iterable[CandleWithMeta]) -> None`

**Semantics**
- адаптер записывает данные в соответствующую raw-таблицу (маршрутизация — implementation detail)
- запись должна быть устойчивой к повторным вставкам (re-ingestion/backfill):
  дубликаты возможны на raw/каноническом уровне до merge, но не должны “ломать” downstream чтения/пайплайны


---

### CanonicalCandleReader

**Purpose**  
`CanonicalCandleReader` — порт чтения канонических 1m свечей из `market_data.canonical_candles_1m`.

**Contract**
- `read_1m(instrument_id: InstrumentId, time_range: TimeRange, clock: Clock) -> Iterator[CandleWithMeta]`

*(Примечание: clock передаётся как зависимость через конструктор реализации; в контракте показан смысл.)*

**Semantics**
- возвращает свечи в пределах полуинтервала `[time_range.start, time_range.end)`
- SHOULD: выдача отсортирована по `candle.ts_open` по возрастанию

**Dedup rule (важно)**
- для части диапазона, пересекающей “последние 24 часа” относительно `clock.now()`, порт гарантирует:
  - не более одной записи на ключ `(instrument_id, candle.ts_open)`
  - выбирается “последняя версия” по `meta.ingested_at`
- для данных старше 24 часов допускается чтение “как есть” (без дополнительного дедупа в запросе)

Дедуп реализуется внутри адаптера ClickHouse и не раскрывается потребителям.


## Notes for Strategy / Live Tail (future)

Этот документ описывает ports `market_data` для walking skeleton ingestion и базового чтения canonical.

Стратегии в дальнейшем будут:
- делать bootstrap (snapshot) из `canonical_candles_1m`,
- затем получать “tail” из общего live-потока (`ws`) без постоянного чтения из БД,
- выполнять дедуп “на лету” в памяти (например numpy/структуры индексов) как часть Strategy Feed.

Чтобы это было совместимо:
- canonical reader и live источник должны отдавать одинаковую модель данных (`Candle` + `CandleMeta`)
  и согласованный ключ дедупа: `(InstrumentId, ts_open)`.
