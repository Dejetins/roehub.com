# Market Data — Use-Case: Backfill 1m Candles (Walking Skeleton v1)

Этот документ фиксирует первый application use-case bounded context `market_data` для walking skeleton v1.

Цель:
- выполнить ingestion/backfill канонических 1m свечей из источников `ws/rest/file`;
- записать данные в ClickHouse через `raw_*` таблицы (через port `RawKlineWriter`);
- обеспечить потоковую обработку и ограниченное потребление памяти (batching).

Источник правды по хранению 1m свечей:
- `market_data.canonical_candles_1m` (заполняется автоматически через MV из `raw_*`).


## Контекст и границы

### Что use-case делает
- читает 1m свечи из `CandleIngestSource` для `(InstrumentId, TimeRange)`;
- пишет их в `RawKlineWriter` батчами;
- собирает простой отчёт выполнения (для CLI/логов/наблюдаемости).
- сохраняет `meta.instrument_key` в едином каноничном формате
  `"{exchange}:{market_type}:{symbol}"` (едино с REST catch-up).

### Что use-case НЕ делает (осознанно)
- не читает `canonical_candles_1m` для проверки записи (read-back) — это отдельный сценарий/проверка качества;
- не пытается определить “какие свечи уже есть” и пропускать их (skip-existing);
- не занимается дедупликацией на уровне canonical (ReplacingMergeTree — eventual dedup);
- не реализует стратегический “live tail” и дедуп “на лету” в памяти — это будет частью Strategy Feed / ACL.


## Связанные документы
- `docs/architecture/shared-kernel-primitives.md`
- `docs/architecture/market_data/market-data-application-ports.md`


## Ports (dependencies)

Use-case зависит только от application ports:

- `CandleIngestSource`
  - `stream_1m(instrument_id: InstrumentId, time_range: TimeRange) -> Iterator[CandleWithMeta]`
- `RawKlineWriter`
  - `write_1m(rows: Iterable[CandleWithMeta]) -> None`
- `Clock`
  - `now() -> UtcTimestamp`

ClickHouse, Materialized Views, биржи, протоколы и форматы данных — детали адаптеров и wiring.


## DTO (application)

DTO в application-слое — не Pydantic, не зависит от transport/API.

### Backfill1mCommand

**Purpose**  
Команда на backfill 1m свечей по инструменту и диапазону.

**Fields**
- `instrument_id: InstrumentId`
- `time_range: TimeRange`  (семантика полуинтервала `[start, end)`)

**Invariants**
- поля валидны по инвариантам `InstrumentId` и `TimeRange`.


---

### Backfill1mReport

**Purpose**  
Отчёт выполнения backfill для наблюдаемости (CLI/логи/метрики).

**Fields**
- `instrument_id: InstrumentId`
- `time_range: TimeRange`
- `started_at: UtcTimestamp`
- `finished_at: UtcTimestamp`
- `candles_read: int`       — сколько свечей пришло из источника
- `rows_written: int`       — сколько строк отправлено в writer (обычно == candles_read)
- `batches_written: int`    — сколько батчей записано

**Notes**
- отчёт не гарантирует количество уникальных свечей в `canonical_candles_1m`:
  canonical использует `ReplacingMergeTree(ingested_at)`, дедуп является eventual.


## Алгоритм (поведение use-case)

### 1) Диапазон времени
Use-case работает строго по семантике `TimeRange = [start, end)`.

### 2) Потоковая обработка
Use-case не держит все данные в памяти:
- читает свечи из `CandleIngestSource` итератором;
- буферизует до размера батча;
- пишет батч через `RawKlineWriter`;
- очищает буфер и продолжает.

### 3) Batching policy
Batching — обязательная часть walking skeleton v1 для эксплуатационной пригодности.

- use-case принимает параметр `batch_size: int` при создании (через wiring).
- алгоритм: буферизуем до `batch_size` строк и делаем `write_1m(...)`, затем очищаем буфер.

Рекомендованный размер батча для CLI в v1:
- `batch_size = 10_000`

Пояснение:
- batching ограничивает память процесса;
- batching снижает риск ошибок ClickHouse на больших диапазонах (например, лимиты по числу партиций в одном INSERT-блоке);
- throughput остаётся стабильным (несколько крупных вставок вместо одной гигантской).

Примечание:
- в v1 use-case допускает передачу `batch_size=None` (если wiring так решит),
  но CLI по умолчанию использует batching (см. CLI-док).

### 4) Повторные прогоны (идемпотентность на уровне сценария)
Use-case допускает повторный запуск на том же диапазоне:
- он не вычисляет “что уже есть”;
- он повторно пишет то, что выдаёт источник.

Дубликаты возможны на raw/canonical уровне до merge, но downstream чтение и пайплайны не должны “ломаться”.
Оптимизация “skip-existing” — отдельный следующий use-case/улучшение.

### 5) Ошибки
В walking skeleton v1:
- ошибки источника/записи пробрасываются наружу как исключения;
- единый слой application errors/кодов будет добавлен позже, когда стабилизируем эксплуатационные сценарии.

### 6) Временные метки
- `started_at = clock.now()` до начала чтения;
- `finished_at = clock.now()` после завершения последней записи.


## Notes for Strategy / Live Tail (future)

Стратегии будут делать:
1) bootstrap (snapshot) из `canonical_candles_1m` через `CanonicalCandleReader`,
2) затем получать live tail из общего потока (`ws`) без чтения БД “по одной свече”,
3) выполнять дедуп “на лету” в памяти (например numpy/структуры индексов) по ключу:
   `(InstrumentId, ts_open)`.

Backfill use-case обеспечивает совместимость, потому что пишет данные,
которые затем становятся доступны через canonical и имеют ту же модель (`Candle` + `CandleMeta`).
