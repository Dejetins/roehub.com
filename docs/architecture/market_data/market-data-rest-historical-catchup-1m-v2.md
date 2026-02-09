# Market Data — REST Historical Catch-up + Gap Fill 1m (v2)

Этот документ фиксирует правила и минимальные механизмы **REST-догонки** и **автоматического gap fill**
для bounded context `market_data` на этапе v2.

Цель:
- уметь **догонять хвост** 1m свечей до “последней закрытой минуты” (per instrument);
- уметь **находить и чинить гэпы** 1m свечей по всей доступной истории инструмента;
- писать только в `raw_*`, canonical формируется MV;
- быть совместимыми с автоматическим пайплайном: `REST catch-up → WS live → Redis Streams для стратегий`.

Источник правды по хранению свечей: ClickHouse DDL:
- `market_data.raw_binance_klines_1m`
- `market_data.raw_bybit_klines_1m`
- `market_data.canonical_candles_1m`
- MV `mv_raw_*_to_canonical_1m` (raw → canonical)

Связанные документы:
- `Market Data — Runtime Config & Invariants (v2)` (market_data.yaml + whitelist.csv + time slicing ≤ 7d)
- `Market Data — Application Ports (Walking Skeleton v1)` (ports ingestion/writer/reader)
- `Market Data — Reference Data Sync (Whitelist -> ClickHouse) (v2)` (ref_market/ref_instruments)


## Ключевые решения

### 1) REST-адаптеры используют `requests` и нормализуют данные в `CandleWithMeta`
- HTTP-клиент проекта: **requests**.
- Адаптеры REST обязаны отдавать `CandleWithMeta`:
  - `Candle` — рыночные поля
  - `CandleMeta` — ingestion metadata (`source="rest"`, `ingested_at`, `ingest_id`, `instrument_key`, optional trades/volumes).

### 2) Пишем только **закрытые** 1m свечи (“closed 1m only”)
REST ingestion не пишет “текущую незакрытую минуту”.
Граница end для догонки:
- `end = floor_to_minute(clock.now())` (полуинтервал `[start, end)`, end не включён).

### 3) Canonical остаётся “источником правды” для определения хвоста и анализа покрытия
- “Хвост” определяется по `canonical_candles_1m` (как по источнику правды).
- Gap fill тоже основан на canonical, потому что raw может содержать дубликаты до merge.

### 4) Защита от Code 252: time slicing по UTC-дням (≤ 7 дней на insert)
Любой исторический диапазон режется на чанки по `max_days_per_insert` из runtime config (≤ 7).
Это гарантирует, что один insert не пересекает > 7 дневных партиций.

### 5) Up-to-date rate limits: autodetect по response headers (без “хардкода” лимитов)
Мы **не храним** “точные rate limits” в `market_data.yaml`, потому что:
- Binance отдаёт “использованный вес” в response headers `X-MBX-USED-WEIGHT-(interval)` (по IP).
- Bybit отдаёт лимиты/остаток/время сброса в headers `X-Bapi-Limit`, `X-Bapi-Limit-Status`, `X-Bapi-Limit-Reset-Timestamp`.

Следствие:
- runtime config содержит параметры **limiter/autodetect** (safety_factor, max_concurrency, backoff/retry),
а не сами лимиты.

### 6) Reuse: ingestion pipeline использует существующий `Backfill1mCandlesUseCase`
REST catch-up и gap fill не изобретают новую “батч-логику записи”.
Они переиспользуют:
- `CandleIngestSource.stream_1m(...)` (REST реализация)
- `RawKlineWriter.write_1m(...)` (ClickHouse raw writer)
- `Backfill1mCandlesUseCase(batch_size=10k/20k)` как “выполнятор” ingestion в ClickHouse.

### 7) Отчётность: stdout/logs + метрики Prometheus
В v2 “отчёт” не пишется в отдельную БД-таблицу.
Канал отчётности зависит от entrypoint:
- CLI / notebook: stdout (text/json) — контракт для операторов.
- Automation (scheduler/worker): structured logs + Prometheus metrics (для Grafana/alerts).


## REST Endpoints (факты, важные для адаптеров)

### Binance Spot (market_id=1)
- Endpoint: `GET /api/v3/klines`
- `limit`: default 500, max 1000.

### Binance Futures USD-M (market_id=2)
- Endpoint: `GET /fapi/v1/klines`
- `limit`: default 500, max 1500.
- Вес запроса зависит от LIMIT (важно для autodetect).

### Bybit V5 (market_id=3 spot, market_id=4 futures)
- Endpoint: `GET /v5/market/kline`
- `limit`: [1..1000], default 200.
- Ответ `list` сортирован **в обратном порядке** по `startTime` (descending).


## Нормализация данных в `CandleWithMeta`

### Candle.ts_open / ts_close
- `ts_open` = open/start time, UTC, ms точность (через `UtcTimestamp`).
- `ts_close` = `ts_open + 1 minute` (каноничная минутная сетка).

### Candle volumes
- `volume_base` = volume (объём в базовом активе)
- `volume_quote`:
  - Binance: quoteAssetVolume
  - Bybit: turnover

### CandleMeta
- `source = "rest"`
- `ingested_at = clock.now()`
- `ingest_id`: UUID “на страницу/кусок” REST (для трассировки)
- `instrument_key = f"{exchange}:{market_type}:{symbol}"` (exchange/market_type из runtime config)

Тот же формат `instrument_key` обязателен и для file/parquet backfill, чтобы все ingestion paths
писали единый каноничный ключ.


## Ports (добавления для v2)

Для EPIC 2 текущего `CanonicalCandleReader.read_1m()` недостаточно, потому что:
- нужно быстро получать “последнюю свечу” без угадывания окна;
- нужно агрегатно оценивать покрытие по дням и находить дни с пропусками.

Поэтому добавляем минимальный порт “индекса canonical”:

### CanonicalCandleIndexReader

**Purpose**  
Быстрые индексовые операции по `canonical_candles_1m` для tail и gap detection.

**Contract**
- `last_ts_open(instrument_id: InstrumentId) -> UtcTimestamp | None`
- `first_ts_open(instrument_id: InstrumentId) -> UtcTimestamp | None`
- `count_distinct_ts_open_by_utc_day(instrument_id: InstrumentId, time_range: TimeRange) -> Mapping[str, int]`
  - key: `YYYY-MM-DD` (UTC day), value: countDistinct(ts_open)
- `list_ts_open_in_range(instrument_id: InstrumentId, time_range: TimeRange) -> Sequence[UtcTimestamp]`
  - используется точечно для “проблемных дней” (максимум 1440 ts_open на день)

**Semantics**
- все операции трактуют время как UTC;
- `last_ts_open`/`first_ts_open` относятся к canonical (источник правды), без FINAL;
- “tail” 24h может применять дедуп-аналогично `ClickHouseCanonicalCandleReader`.


## Use-case: RestCatchUp1mUseCase

### Purpose
Догнать свечи инструмента до “последней закрытой минуты”, а также найти и восстановить гэпы по истории.

### Inputs
- `instrument_id: InstrumentId`
- `runtime_config: MarketDataRuntimeConfig`
- `clock: Clock`
- `index_reader: CanonicalCandleIndexReader`
- `source: CandleIngestSource` (REST реализация)
- `writer: RawKlineWriter`
- `batch_size: int` (10k/20k)
- `max_days_per_insert: int` (из runtime config, ≤ 7)

### Semantics (tail catch-up)
1) `end = floor_to_minute(clock.now())`
2) `last = index_reader.last_ts_open(instrument_id)`
3) Если `last is None`:
   - use-case не угадывает “начало истории” сам.
   - bootstrap диапазон задаёт orchestration/ops (например “догнать от даты X”).
4) Иначе:
   - `start = last + 1 minute`
   - если `start >= end` → хвост уже догнан
5) Делается time slicing по UTC-дням:
   - `slice_time_range_by_utc_days(TimeRange(start, end), max_days=max_days_per_insert)`
6) Каждый slice прогоняется через ingestion pipeline (батчами по `batch_size`).

### Semantics (gap scan + fill)
Gap fill выполняется автоматически, без ручных команд:

1) Определяем “доступный диапазон” по canonical:
   - `min_ts_open = index_reader.first_ts_open(...)`
   - `max_ts_open = index_reader.last_ts_open(...)`
   - если None → gap scan пропускаем

2) Оцениваем покрытие по дням:
   - строим `TimeRange(min_ts_open, max_ts_open + 1m)`
   - `daily_counts = index_reader.count_distinct_ts_open_by_utc_day(...)`

3) Для каждого UTC-дня считаем `expected`:
   - обычно 1440 минут
   - для первого/последнего дня — expected считается по пересечению с общим диапазоном

4) Для дней, где `count < expected`:
   - получаем список `ts_open` за день: `list_ts_open_in_range(day_range)`
   - в памяти строим missing intervals (склеиваем подряд идущие пропуски)
   - каждый missing interval прогоняем тем же REST ingestion механизмом (time slicing + batching)

### Invariants
- Пишем только закрытые 1m (end всегда “вниз до минуты”).
- Любой insert пересекает ≤ 7 UTC-дней (runtime config guard).
- Source=rest всегда проставляется в `CandleMeta.source`.

### Dedup safety (gap fill)
Для gap-fill применяется дополнительная защитная проверка перед записью:
- для каждого chunk `[start,end)` use-case читает `distinct_ts_opens` из canonical index;
- сравнение делается по минутным ключам (`epoch_minutes`), а не по “сырым” datetime;
- если минута уже есть в canonical, запись этой минуты пропускается.

Это защищает от ложных gap-диапазонов при timezone/ms-шумах и не даёт наращивать дубликаты
в canonical при повторном восстановлении истории.


## Verification SQL (post-run)
Проверка глобальной динамики missing/duplicates:

```sql
-- global duplicates/missing
SELECT
  first_m,
  last_m,
  dateDiff('minute', first_m, last_m) + 1 AS candles_expected,
  candles_actual,
  (dateDiff('minute', first_m, last_m) + 1) - candles_actual AS candles_missing,
  rows_total - candles_actual AS duplicate_rows
FROM
(
  SELECT
    min(m) AS first_m,
    max(m) AS last_m,
    count() AS rows_total,
    uniqExact(m) AS candles_actual
  FROM
  (
    SELECT toStartOfMinute(ts_open) AS m
    FROM market_data.canonical_candles_1m
  )
);
```

Диагностика дней с остаточными gap-минутами:

```sql
SELECT
  day,
  expected,
  actual,
  expected - actual AS missing
FROM
(
  SELECT
    toDate(ts_open) AS day,
    uniqExact(toStartOfMinute(ts_open)) AS actual,
    1440 AS expected
  FROM market_data.canonical_candles_1m
  GROUP BY day
)
WHERE missing > 0
ORDER BY missing DESC
LIMIT 50;
```

Проверка дубликатов по конкретному инструменту:

```sql
SELECT
  count() - uniqExact(toStartOfMinute(ts_open)) AS dup
FROM market_data.canonical_candles_1m
WHERE instrument_key = 'binance:spot:BTCUSDT';
```


## Адаптер: REST CandleIngestSource (routing по market_id)

REST ingest source использует `runtime_config.market_by_id(instrument_id.market_id)` и выбирает:
- base_url (per market)
- параметры limiter/backoff
- формат и параметры под конкретную биржу

Маппинг market_id → exchange/market_type:
- 1: binance/spot
- 2: binance/futures
- 3: bybit/spot
- 4: bybit/futures

Bybit `category`:
- spot → `category="spot"`
- futures → `category="linear"`


## Entrypoints: CLI / Notebook / Automation

### CLI: rest-catchup
CLI нужен для разработки/ops и отладки. Отчёт выводится в stdout (text/json).

Формы запуска:
- `rest-catchup --market-id 1 --symbol BTCUSDT [--config ...] [--batch-size ...]`
- `rest-catchup --all-from-ref-instruments [--only-enabled] [--config ...] [--batch-size ...]`

### Notebook: scripts/data/market_data/02_rest_catchup_1m.ipynb
Ноутбук повторяет тот же сценарий, в ручном режиме:
- загрузить runtime config
- загрузить env (/etc/roehub/roehub.env)
- запустить catch-up для одного инструмента
- запустить для всех enabled инструментов

### Automation: Scheduler/Worker
REST catch-up является частью автоматического пайплайна.

Минимальный сценарий v2:
- Scheduler периодически ставит job:
  - `rest_catchup_all_enabled` (по enabled инструментам из `ref_instruments`)
- Worker выполняет:
  - по каждому инструменту запускает `RestCatchUp1mUseCase`
  - соблюдает `rest.limiter.max_concurrency` (верхняя граница параллелизма)
  - применяет retries/backoff для временных ошибок

Отчётность в automation:
- structured logs (json) на каждый инструмент и на общий запуск
- Prometheus metrics (см. ниже) для Grafana/alerts


## Observability / Metrics (минимум для v2)

### Tail metrics
- `rows_written_total`
- `batches_written_total`
- `slices_total`
- `duration_s`
- `rows_per_s`
- `lag_to_now_s` (после догонки должен быть близок к 0)

### Gap metrics
- `days_scanned_total`
- `days_with_gaps_total`
- `gap_intervals_total`
- `gap_rows_written_total`
- `gap_rows_skipped_existing_total` (защитный дедуп: минуты, уже присутствующие в canonical)

### Error metrics
- `rest_errors_total` (by exchange/market_type/status_code)
- `rest_retries_total`
- `rest_rate_limit_sleeps_total`


## Out of scope (в этом документе)
- WS live ingestion (stream → raw, SLO ≤ 1s)
- Redis Streams feed для стратегий (онлайн-доставка)
- Enrich инструментов из REST бирж (base/quote/steps/min_notional)
