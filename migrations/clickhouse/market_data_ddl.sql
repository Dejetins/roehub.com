```md
/* ============================================================
   ref_market — справочник “площадок” (биржа + тип рынка) с фиксированным market_id
   Пример market_id:
     1 = binance spot
     2 = binance futures
     3 = bybit  spot
     4 = bybit  futures

   Поля:
   - market_id      UInt16  : фиксированный ID
   - exchange_name  String  : имя биржи ("binance", "bybit")
   - market_type    String  : тип рынка ("spot", "futures")
   - market_code    String  : код ("binance:spot", "bybit:futures")
   - is_enabled     UInt8   : 1/0
   - count_symbols  UInt32  : сколько символов сейчас в справочнике (можно обновлять джобой)
   - updated_at     DateTime64(3, UTC): версия записи для ReplacingMergeTree
   ============================================================ */

```
```sql
CREATE DATABASE IF NOT EXISTS market_data;
USE market_data;

CREATE TABLE IF NOT EXISTS market_data.ref_market
(
    market_id      UInt16,                                   -- фиксированный ID (1..N)
    exchange_name  LowCardinality(String),                    -- "binance" | "bybit"
    market_type    LowCardinality(String),                    -- "spot" | "futures"
    market_code    LowCardinality(String),                    -- "binance:spot", "bybit:futures", ...
    is_enabled     UInt8 DEFAULT 1,                           -- 1 = активен, 0 = выключен
    count_symbols  UInt32 DEFAULT 0,                          -- актуализируется отдельно (не триггерится автоматически)
    updated_at     DateTime64(3, 'UTC') DEFAULT now64(3)      -- версия записи
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (market_id);
```

```md
/* ============================================================
   ClickHouse DDL (RU) — "2 RAW + 1 CANONICAL" + 2 СПРАВОЧНИКА
   + Materialized View (MV) для raw -> canonical

   Таблицы со свечами (ровно 3):
   1) raw_binance_klines_1m
   2) raw_bybit_klines_1m
   3) canonical_candles_1m

   Справочники (ровно 2):
   4) ref_market        — справочник "биржа + тип рынка" с фиксированным market_id
   5) ref_instruments   — справочник символов по market_id (торгуется/не торгуется + метаданные)

   MV (НЕ таблицы со свечами, но пишут в canonical автоматически):
   - mv_raw_binance_to_canonical_1m
   - mv_raw_bybit_to_canonical_1m

   Важное:
   - market_id фиксирован:
       1 = binance spot
       2 = binance futures
       3 = bybit  spot
       4 = bybit  futures
   - symbol хранится строкой (LowCardinality) — это нормально и эффективно.
   - “торгуется/не торгуется” хранится в ref_instruments, а не дублируется в каждой минуте.
   ============================================================ */
```

```sql
CREATE TABLE IF NOT EXISTS market_data.ref_instruments
(
    market_id     UInt16,
    symbol        LowCardinality(String),

    status        LowCardinality(String) DEFAULT 'UNKNOWN',
    is_tradable   UInt8 DEFAULT 1,

    base_asset    LowCardinality(Nullable(String)),
    quote_asset   LowCardinality(Nullable(String)),

    price_step    Nullable(Float64),
    qty_step      Nullable(Float64),
    min_notional  Nullable(Float64),

    updated_at    DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (market_id, symbol);

```
```md
/* ------------------------------------------------------------
   (1) raw_binance_klines_1m — сырые свечи Binance 1m (как пришло)
   ------------------------------------------------------------ */
```
```sql
CREATE TABLE IF NOT EXISTS market_data.raw_binance_klines_1m
(
    market_id      UInt16,                                  -- 1 (binance spot) или 2 (binance futures)
    symbol         LowCardinality(String),
    instrument_key String,                                  -- "binance:spot:BTCUSDT" (формируешь в ingestion)

    -- поля Binance kline (твой список)
    open_time      DateTime64(3, 'UTC'),
    open           Float64,
    high           Float64,
    low            Float64,
    close          Float64,
    volume         Float64,                                  -- base volume
    close_time     DateTime64(3, 'UTC'),

    quote_asset_volume            Float64,                   -- quote volume
    number_of_trades              UInt32,
    taker_buy_base_asset_volume   Float64,
    taker_buy_quote_asset_volume  Float64,

    -- метаданные ingestion
    source         LowCardinality(String) DEFAULT 'unknown',  -- "ws" | "rest"
    ingested_at    DateTime64(3, 'UTC') DEFAULT now64(3),
    ingest_id      Nullable(UUID)
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(open_time)
ORDER BY (market_id, symbol, open_time)
SETTINGS index_granularity = 8192;
```
```md
/* ------------------------------------------------------------
   (2) raw_bybit_klines_1m — сырые свечи Bybit 1m (как пришло)
   ------------------------------------------------------------ */
```

```sql
CREATE TABLE IF NOT EXISTS market_data.raw_bybit_klines_1m
(
    market_id      UInt16,                                  -- 3 (bybit spot) или 4 (bybit futures)
    symbol         LowCardinality(String),
    instrument_key String,                                  -- "bybit:spot:BTCUSDT"

    interval_min   UInt16,                                  -- обычно 1
    start_time_ms  UInt64,
    start_time_utc DateTime64(3, 'UTC'),

    open           Float64,
    high           Float64,
    low            Float64,
    close          Float64,
    volume         Float64,                                  -- base volume
    turnover       Float64,                                  -- quote volume (bybit naming)

    -- метаданные ingestion
    source         LowCardinality(String) DEFAULT 'unknown',  -- "ws" | "rest" | "file"
    ingested_at    DateTime64(3, 'UTC') DEFAULT now64(3),
    ingest_id      Nullable(UUID)
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(start_time_utc)
ORDER BY (market_id, symbol, start_time_utc)
SETTINGS index_granularity = 8192;
```
```md
/* ------------------------------------------------------------
   (3) canonical_candles_1m — канонические свечи 1m (единый формат)
   - Это единственная таблица, которую читают стратегии/бектест/ML.
   - ReplacingMergeTree(ingested_at): “последняя версия победит” после фоновых merge.
   ------------------------------------------------------------ */
```
```sql
CREATE TABLE IF NOT EXISTS market_data.canonical_candles_1m
(
    market_id      UInt16,                                  -- 1..4
    symbol         LowCardinality(String),
    instrument_key String,                                  -- "{exchange}:{market_type}:{symbol}" (для логов/отладки)

    ts_open        DateTime64(3, 'UTC'),
    ts_close       DateTime64(3, 'UTC'),

    open           Float64,
    high           Float64,
    low            Float64,
    close          Float64,

    volume_base    Float64,
    volume_quote   Nullable(Float64),

    trades_count            Nullable(UInt32),
    taker_buy_volume_base   Nullable(Float64),
    taker_buy_volume_quote  Nullable(Float64),

    source         LowCardinality(String),                   -- "ws" | "rest"
    ingested_at    DateTime64(3, 'UTC'),                     -- версия записи (для ReplacingMergeTree)
    ingest_id      Nullable(UUID)
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY toYYYYMMDD(ts_open)
ORDER BY (market_id, symbol, ts_open)
SETTINGS index_granularity = 8192;
```

```md
/* ============================================================
   Materialized Views (MV) raw -> canonical
   Важно:
   - MV НЕ является “дополнительной таблицей со свечами”.
   - MV автоматически пишет строки в canonical_candles_1m при INSERT в raw_*.
   - Поэтому canonical появляется “сразу”, без отдельной джобы.
   ============================================================ */

/* ------------------------------------------------------------
   MV: Binance raw -> canonical
   Маппинг:
   - open_time  -> ts_open
   - close_time -> ts_close
   - volume     -> volume_base
   - quote_asset_volume -> volume_quote
   - taker_* и trades_count заполняются (у Binance есть)
   ------------------------------------------------------------ */
```

```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data.mv_raw_binance_to_canonical_1m
TO market_data.canonical_candles_1m
AS
SELECT
    market_id,
    symbol,
    instrument_key,

    open_time  AS ts_open,
    close_time AS ts_close,

    open, high, low, close,

    volume             AS volume_base,
    quote_asset_volume AS volume_quote,

    number_of_trades             AS trades_count,
    taker_buy_base_asset_volume  AS taker_buy_volume_base,
    taker_buy_quote_asset_volume AS taker_buy_volume_quote,

    source,
    ingested_at,
    ingest_id
FROM market_data.raw_binance_klines_1m;
```
```md
/* ------------------------------------------------------------
   MV: Bybit raw -> canonical
   Маппинг:
   - start_time_utc -> ts_open
   - ts_close = ts_open + 1 minute (для 1m)
   - volume -> volume_base
   - turnover -> volume_quote
   - trades_count/taker_* = NULL (у Bybit в твоём источнике их нет)
   ------------------------------------------------------------ */
```
```sql
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data.mv_raw_bybit_to_canonical_1m
TO market_data.canonical_candles_1m
AS
SELECT
    market_id,
    symbol,
    instrument_key,

    start_time_utc                       AS ts_open,
    (start_time_utc + INTERVAL 1 MINUTE) AS ts_close,

    open, high, low, close,

    volume   AS volume_base,
    turnover AS volume_quote,

    CAST(NULL, 'Nullable(UInt32)')  AS trades_count,
    CAST(NULL, 'Nullable(Float64)') AS taker_buy_volume_base,
    CAST(NULL, 'Nullable(Float64)') AS taker_buy_volume_quote,

    source,
    ingested_at,
    ingest_id
FROM market_data.raw_bybit_klines_1m;
```