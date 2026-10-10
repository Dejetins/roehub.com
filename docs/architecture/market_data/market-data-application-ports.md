# Market Data — Application Ports (Walking Skeleton v1)

Этот документ фиксирует минимальные контракты (ports) application-слоя для bounded context `market_data`
в рамках первого вертикального среза (walking skeleton).

Цель:
- определить порты для ingestion/backfill 1m свечей из источников `ws/rest/file`;
- записывать данные в ClickHouse через `raw_*` таблицы;
- получать канонические 1m свечи из `market_data.canonical_candles_1m` (источник правды) детерминированно для offline/precompute чтения.

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

### 3) Canonical reader возвращает детерминированное duplicate-free чтение на всём диапазоне
Таблица `canonical_candles_1m` использует `ReplacingMergeTree(ingested_at)`, что означает eventual dedup
(дубликаты возможны до фоновых merge).

Контракт чтения фиксирует не механизм, а результат:
- consumer получает duplicate-free canonical rows на всём запрошенном диапазоне;
- порядок строк детерминирован (`ts_open ASC`);
- реализация дедупа остаётся implementation detail ClickHouse-адаптера.

Текущая реализация использует один `SELECT ... FINAL` по всему диапазону. Это дороже, чем selective tail dedup,
поэтому reader остаётся ориентированным на offline/precompute workloads, а не на hot path runtime.

### 4) Clock — обязательная зависимость application-слоя
Clock нужен для:
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
- `read_1m(instrument_id: InstrumentId, time_range: TimeRange) -> Iterator[CandleWithMeta]`

**Semantics**
- возвращает свечи в пределах полуинтервала `[time_range.start, time_range.end)`
- SHOULD: выдача отсортирована по `candle.ts_open` по возрастанию

**Reader contract**
- порт гарантирует duplicate-free canonical rows на всём диапазоне
- конкретная техника дедупа (`FINAL` или эквивалент) скрыта внутри адаптера
- reader ориентирован на deterministic offline/precompute reads, а не на hot-path live ingestion


## Notes for Strategy / Live Tail (future)

Этот документ описывает ports `market_data` для walking skeleton ingestion и базового чтения canonical.

Стратегии в дальнейшем будут:
- делать bootstrap (snapshot) из `canonical_candles_1m`,
- затем получать “tail” из общего live-потока (`ws`) без постоянного чтения из БД,
- выполнять дедуп “на лету” в памяти (например numpy/структуры индексов) как часть Strategy Feed.

Чтобы это было совместимо:
- canonical reader и live источник должны отдавать одинаковую модель данных (`Candle` + `CandleMeta`)
  и согласованный ключ дедупа: `(InstrumentId, ts_open)`.


## Artifact consumer observations (2026-10-10)

Backtest Artifacts owns `StoredCoordinates` in its application reconciliation
service. Its `StoredSourceObservation` contains an instrument and nullable first/last
actual minute; `observation_page` is a bounded ordered keyset page at one exclusive
cutoff. Empty observations retain coordinate traversal. `observation` serves the
existing optional post-write hint. The ClickHouse composition adapter reads the
Market Data exact bitmap; the application imports no concrete persistence adapter.
The previous coordinate-only `page` and reconciler constructor remain supported.

This does not replace the Market Data canonical index port used by ingestion:
its scalar bounds still read canonical data, with null rather than epoch defaults
for empty input. Existing bounded history verification owns index repair. Exchange
absence markers are never candle observations. See the reference API's “Artifact
source observations” and “Verified source availability and index repair” sections.

## Shared provider admission (S03, 2026-10-10)

Исторический заголовок Walking Skeleton не означает ограничение текущих портов.
Добавлен `application/ports/provider_admission.py`: `ProviderRequest` содержит
provider, явный egress scope, общий предел concurrency и набор `QuotaWindow`
(dimension, seconds, limit, cost). `ProviderAdmission.reserve` атомарно резервирует
все окна или возвращает классифицированный временный отказ. `ProviderPermit.check`
проверяет владение; `complete` вызывается только после подтверждённой остановки I/O,
учитывает заголовки/cooldown и освобождает точный permit. Потраченная квота не
возвращается. Истечение TTL не доказывает остановку запроса.

PostgreSQL adapter владеет транзакциями, временем квот и durable metadata; HTTP
adapter — endpoint policy, монотонным deadline и ограниченным пулом отдельных
HTTP-процессов с переиспользуемыми session. Каждый запрос эксклюзивно занимает
transport; число transport не превышает `ingestion.rest_concurrency_instruments`,
независимо от числа вызывающих потоков. Ожидание пула входит в общий deadline.
`provider_io_guard` передаёт проверку текущего
application-owner в синхронный HTTP вызов, не связывая application с PG/HTTP types.
Порт не вводит service/daemon и не держит SQL-транзакцию во время сетевого I/O.

`HistoryBoundsStore.defer` сохраняет token-checked queued work с `next_retry_at`
и sanitized `error_code`. Временный отказ admission проходит через probe/source
без превращения в отсутствие истории. HTTP/body validation и реальные свечи
остаются источником confirmed bounds. Существующие API DTO и права не расширены.

Совместимость: persistence — additive `compatible-change`; новые port implementers
должны реализовать admission/defer, а HTTP composition требует common PG и explicit
scope (`breaking-change` для старой конфигурации без scope). Координированный
upgrade/rollback и cleanup только после подтверждённого drain описаны в local
Market Data recovery runbook, раздел S03.

## Async recovery admission (S04, 2026-10-11)

`AsyncStreamRecoveryStore` дополняет синхронный порт: `remember_many` возвращает
receipts в порядке входных ключей только после commit; `complete_many` удаляет
только совпадающие `(task_key, token)`. `BatchedRecoveryAdmission` объединяет
не более 256 записей за 2 ms, ограничивает очередь вместе с in-flight batch
20 000 записями и длины symbol/reason 128 символами. Failed receipt требует
явного rearm; автоматическая смена token запрещена. Синхронные потребители
сохраняют прежний порт и persisted schema.

Collector использует одну async PG session для advisory lease, admission,
pending/retry/fail/advance/completion и socket health. Операции сериализованы;
транзакция не охватывает CH/REST. Deadline 3 s закрывает session перед отменой
ожидания, включая потерянный сетевой ответ. Неизвестный commit не считается
откатом: lifecycle прекращается, restart получает lease и сверяет прежние ключи.
Неизвестная receipt сохраняется; exactly-once между PG и CH не обещается.

Buffer резервирует capacity до persistence. В общий row budget входят buffer,
DB admissions и ожидающие producers: максимум `2 * max_buffer_rows`, из них
не более `max_buffer_rows` уже допущены к persistence/write. Дополнительно
применяется 64 MiB budget учтённых retained bytes (консервативная оценка объекта
и служебных структур, не RSS). При переполнении intake получает backpressure/
QueueFull, не durable acceptance; reconnect/tail repair покрывает такие arrivals.
Отмена до persistence свободна; начатая owned admission завершается независимо
от повторных отмен caller. После commit row принадлежит buffer либо остаётся
в journal для restart. Только реальная canonical свеча разрешает закрыть
`ws_recovery`; absence marker этого права не даёт.

Совместимость additive async port — `compatible-change`; persisted shape,
recovery keys/tokens и API DTO — `none`. Upgrade/rollback требует drain collector
и сохранения journal; прежняя версия читает оставленные pending receipts.
