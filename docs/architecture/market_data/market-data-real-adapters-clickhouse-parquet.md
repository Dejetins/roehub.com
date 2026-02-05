# Market Data — Real Adapters: ClickHouse + Parquet Source (Walking Skeleton v1)

Этот документ фиксирует “компромиссный” шаг walking skeleton v1:
- реальные адаптеры ClickHouse (write raw + read canonical),
- реальный file-источник из `.parquet` для backfill,
- без сетевых биржевых клиентов (`rest/ws`) на данном этапе.

Цель:
- запустить end-to-end сценарий:
  `Parquet (.parquet) -> CandleIngestSource -> RawKlineWriter (ClickHouse raw_*) -> MV -> canonical`
- сохранить архитектурные границы (use-case зависит только от ports).


## Связанные документы
- `docs/architecture/shared-kernel-primitives.md`
- `docs/architecture/market_data/market-data-application-ports.md`
- `docs/architecture/market_data/market-data-use-case-backfill-1m.md`


## Ключевые решения

### 1) Источник `.parquet` обязан содержать `market_id` и `symbol`
Поскольку доменная идентичность инструмента фиксирована как:
`InstrumentId = (market_id, symbol)`,

Parquet-источник должен предоставлять:
- `market_id` — целое число (эквивалент UInt16 в ClickHouse);
- `symbol` — строка (будет нормализована правилами `Symbol`).

Эти поля считаются обязательными независимо от того, один инструмент в файле или несколько.

### 2) `instrument_key` не требуется от parquet и генерируется адаптером
Поле `instrument_key` — trace/debug (не доменная идентичность).  
Parquet-источник не обязан его содержать.

Parquet-адаптер обязан сгенерировать `instrument_key` самостоятельно (канонично и стабильно).
В walking skeleton v1 фиксируем минимальный формат:

- `instrument_key = f"{market_id.value}:{symbol}"`

(Позже можно заменить на более человекочитаемый формат `{exchange}:{market_type}:{symbol}`
после подключения `ref_market` и правил mapping.)

### 3) Источник правды по свечам — `canonical_candles_1m`, но запись идёт в `raw_*`
Запись выполняется только в `raw_*` таблицы через port `RawKlineWriter`.

Таблица `canonical_candles_1m` заполняется автоматически Materialized View:
- `mv_raw_binance_to_canonical_1m`
- `mv_raw_bybit_to_canonical_1m`

Следствие:
- прямых записей в canonical нет;
- корректность mapping raw->canonical контролируется DDL/MV.

### 4) Dedup в canonical reader — только на хвосте последних 24 часов (без FINAL)
`canonical_candles_1m` использует `ReplacingMergeTree(ingested_at)` → возможны дубликаты до merge.

Чтобы избегать `FINAL/argMax` на всей истории, `CanonicalCandleReader` гарантирует дедуп
только для части диапазона, пересекающей последние 24 часа относительно `Clock.now()`.

Реализация в ClickHouse адаптере выполняет selective dedup на хвосте, например:
- `ORDER BY ingested_at DESC LIMIT 1 BY (market_id, symbol, ts_open)`.

Для данных старше 24 часов допускается чтение “как есть” (без доп. дедупа в запросе).


## Реализуемые ports

В рамках этого шага реализуются следующие application ports:

### CandleIngestSource
Реализация: Parquet-источник.
- `stream_1m(instrument_id: InstrumentId, time_range: TimeRange) -> Iterator[CandleWithMeta]`

### RawKlineWriter
Реализация: ClickHouse writer.
- `write_1m(rows: Iterable[CandleWithMeta]) -> None`

### CanonicalCandleReader
Реализация: ClickHouse reader.
- `read_1m(instrument_id: InstrumentId, time_range: TimeRange) -> Iterator[CandleWithMeta]`

### Clock
Реализация: SystemClock (platform).
- `now() -> UtcTimestamp`


## Adapter placement (folder structure)

### Parquet source
- `src/trading/contexts/market_data/adapters/outbound/clients/files/parquet_candle_ingest_source.py`

### ClickHouse persistence
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/raw_kline_writer.py`
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py`

Дополнительно (implementation detail адаптеров ClickHouse):
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py`
  - тонкий слой над конкретным драйвером ClickHouse, чтобы unit-тестировать SQL и payload без реальной БД.

### Platform time
- `src/trading/platform/time/system_clock.py`


## Parquet ingestion — requirements & mapping

### Required columns (обязательные)
Parquet обязан содержать:
- `market_id` (int)
- `symbol` (str)
- `ts_open` (datetime-like, timezone-aware или приводимый к UTC)
- `open`, `high`, `low`, `close` (float-like)
- `volume_base` (float-like)

`ts_close` может отсутствовать: в v1 допускается вычисление как `ts_open + 1 minute`
(так как use-case — 1m backfill).

### Optional columns (опциональные)
Если присутствуют — используем, если нет — заполняем `None`:
- `ts_close` (иначе вычисляем)
- `volume_quote`
- `trades_count`
- `taker_buy_volume_base`
- `taker_buy_volume_quote`

### Normalization rules
- `symbol` нормализуется правилами `Symbol` (strip + upper).
- `market_id` валидируется правилами `MarketId` (`> 0`).
- `ts_open`/`ts_close` приводятся к `UtcTimestamp` (UTC + ms), naive datetime запрещён.
- `instrument_id` формируется как `InstrumentId(MarketId(market_id), Symbol(symbol))`.
- `instrument_key` генерируется по правилу (см. Key decision #2).

### Meta.source
Для parquet ingestion фиксируем:
- `meta.source = "file"`

### Meta.ingested_at
В v1 (walking skeleton) фиксируем правило:
- `meta.ingested_at = clock.now()` (один timestamp на весь прогон или на батч — решение use-case/адаптера;
  минимально допустимо: один на прогон).

(Технически raw таблицы имеют default `now64(3)`, но мы сохраняем явный `ingested_at` в доменной модели,
чтобы стабильно воспроизводить прогоны и тесты.)

### Meta.ingest_id
В v1:
- `meta.ingest_id = None` (может быть добавлен позже на уровне orchestration/CLI/worker).


## ClickHouse RawKlineWriter — mapping and normalization

Writer принимает `CandleWithMeta` и пишет в одну из raw таблиц:
- `raw_binance_klines_1m` если `market_id` ∈ {1, 2}
- `raw_bybit_klines_1m` если `market_id` ∈ {3, 4}

Маршрутизация по `market_id` — implementation detail адаптера.

### Normalization for non-nullable raw fields
В ClickHouse DDL raw-таблиц ряд полей не Nullable.
Если в доменной модели они отсутствуют (`None`), writer обязан нормализовать:
- отсутствующие numeric значения → `0` / `0.0`
- отсутствующие optional для bybit trade/taker полей → вставляются как `NULL` в canonical через MV (как в DDL),
  но raw writer заполняет “как требует raw-таблица”.

Это правило относится только к уровню адаптера raw writer и не влияет на доменные примитивы.


## ClickHouse CanonicalCandleReader — last-24h dedup

Reader читает `market_data.canonical_candles_1m` и реализует правило:
- “старую” часть диапазона читает обычным SELECT (без дедупа);
- “хвост” последних 24 часов читает с дедупом без использования FINAL.

Cutoff вычисляется как:
- `cutoff = clock.now() - 24 hours`
- далее используется пересечение `TimeRange` с `[cutoff, +inf)`.

Ключ дедупа:
- `(market_id, symbol, ts_open)`

Победитель выбирается по:
- максимальному `ingested_at` (последняя версия).


## Testing strategy

### Unit tests (без реального ClickHouse)
- Parquet-источник: тестируем обязательные поля, фильтрацию по `InstrumentId` и `[start, end)`,
  корректное построение `InstrumentId` и генерацию `instrument_key`.
- ClickHouse writer/reader: тестируем построение SQL и payload через `gateway` (FakeGateway),
  включая маршрутизацию по `market_id` и last-24h dedup разрез диапазона.

### Integration tests (позже)
- тесты с реальным ClickHouse (docker) для проверки MV и end-to-end вставки/чтения.
