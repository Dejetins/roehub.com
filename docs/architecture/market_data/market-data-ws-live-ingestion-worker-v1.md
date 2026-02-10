# Market Data — WS Live Ingestion Worker & Maintenance Scheduler (v1)

Этот документ описывает **EPIC 3 — WS Live Ingestion Worker: close-only, <1s to raw** и сопутствующий **maintenance scheduler**.
Формат и уровень детализации ориентированы на то, чтобы по документу можно было **однозначно реализовать код** (через Codex агент)
и развернуть сервисы (docker-compose / k8s позже).

---

## Цели эпика

### A) Live WS worker: `market-data-ws-worker`

Постоянный процесс, который:

1) Подключается к WebSocket 4 рынков:
- `market_id=1` Binance Spot
- `market_id=2` Binance Futures (USD-M)
- `market_id=3` Bybit Spot (V5)
- `market_id=4` Bybit Futures (V5 linear)

2) Обрабатывает **только closed 1m candles** и записывает их в **ClickHouse raw** (`raw_*_klines_1m`)
с SLO: **p95 “receive closed → inserted raw” ≤ 1s (локально)**.

3) Детектирует разрывы (gap) **по последовательности минут** на инструмент:
- если видим пропуск минут в WS потоке → инициируем REST fill
- при reconnect / рестарте → сравниваем хвост и делаем REST догонку

---

### B) Maintenance scheduler: `market-data-scheduler`

Отдельный процесс, который поддерживает “систему в порядке” и не обязан быть внутри live worker:

1) **sync whitelist → ref_instruments** (если whitelist остаётся источником)
2) **enrich ref_instruments** (base/quote/steps/min_notional)
3) **страховочная REST догонка** по расписанию (например каждый час / сутки)
4) наблюдаемость: прогресс, текущее действие, ошибки, лимиты REST

---

## Где лежит документация

- Архитектура / контракты и поведение:
  - `docs/architecture/market_data/market-data-ws-live-ingestion-worker-v1.md`  *(этот документ)*

- (Опционально позже) отдельный runbook:
  - `docs/runbooks/market-data-ws-worker.md`
  - `docs/runbooks/market-data-scheduler.md`

---

## Термины и семантика времени

### Closed 1m candle
Свеча “закрыта”, если источник явно помечает 1m kline как финальную за минуту.
- Binance kline streams: поле `k.x == true` означает “kline is closed”. citeturn9view0
- Binance USD-M futures kline streams: `k.x == true`. citeturn13view0
- Bybit V5 public kline: поле `confirm == true` означает “kline is closed”. citeturn14view0

> Live worker **игнорирует** незакрытые/промежуточные апдейты.

### “Последовательность минут” (простыми словами)
У каждого инструмента есть ожидаемая цепочка `ts_open`:
- … 12:00, 12:01, 12:02, 12:03, …
Если после 12:02 пришла свеча 12:05 — значит **пропущены** 12:03 и 12:04 (gap = 2 минуты).
Это и есть “gap detection по последовательности ts_open”.

### Minute key
Везде, где сравниваем минуты, используем **minute bucket**:
- `toStartOfMinute(ts_open)` в ClickHouse
- в Python: `ts_open` нормализован до “точной минуты” (без секунд/мс)

---

## Хранилище и текущие таблицы (источник правды)

### Raw tables
- `market_data.raw_binance_klines_1m`
- `market_data.raw_bybit_klines_1m`

### Canonical (через MV)
- `market_data.canonical_candles_1m` формируется автоматически через:
  - `mv_raw_binance_to_canonical_1m`
  - `mv_raw_bybit_to_canonical_1m`

**Важно**: Live worker пишет **только** в raw. Прямых записей в canonical нет.

---

## Инструменты (whitelist) и загрузка из ClickHouse

Источник списка активных инструментов для live worker и scheduler:
- `market_data.ref_instruments`
- критерий отбора (минимальный):
  - `status == 'ENABLED'`
  - `is_tradable == 1`

---

## Архитектура: компоненты и потоки

Ниже описан **целевой runtime** (2 контейнера, 2 процесса):

### 1) `market-data-ws-worker` (высокий приоритет: live SLO)

**Pipeline A: WS → Buffer → Raw insert**
1) WS клиент получает сообщение.
2) Парсит и нормализует в `Candle` + `CandleMeta` (DTO `CandleWithMeta`).
3) Фильтрует: **только closed 1m**.
4) Передаёт свечу в буфер вставок.
5) Буфер flush’ится:
   - по таймеру `flush_interval_ms` (≤ 500ms; дефолт 250ms)
   - или по размеру `max_buffer_rows` (пример: 2000)
6) Flush вызывает `RawKlineWriter.write_1m(rows)` — это момент **insert begin/end**.
7) После submit в insert buffer worker делает best-effort publish в Redis Streams:
   - stream: `md.candles.1m.<instrument_key>`
   - id: `<ts_open_epoch_ms>-0`
   - ошибки Redis логируются/метрятся и не останавливают ingestion.

**Pipeline B: Gap detection → REST fill (фон)**
- Срабатывает по двум причинам:
  1) “внутри WS”: пропуск минут в последовательности
  2) “на reconnect/рестарт”: догонка от последней известной минуты (canonical) до текущего момента

REST fill вызывает существующий use-case `RestCatchUp1mUseCase` (или режим/подмножество его логики),
чтобы дозаписать raw (и, через MV, canonical).

> REST fill **не должен блокировать** WS ingestion. Это отдельный пул задач/очередь.

### 2) `market-data-scheduler` (фоновый maintenance)

Содержит три независимых периодических джоба:

- **Job S1**: `sync_whitelist_to_ref_instruments`
- **Job S2**: `enrich_ref_instruments`
- **Job S3**: `rest_insurance_catchup`

Стартовая последовательность процесса (выполняется один раз на запуск):
- `S1` (seed `ref_market` + sync whitelist)
- `S2` (enrich `ref_instruments` из биржевых instrument-info endpoint’ов)
- startup scan:
  - для каждого enabled/tradable инструмента читаем canonical bounds до `now_floor`
  - если bounds пустые: bootstrap `[earliest_available_ts_utc, now_floor)`
  - если `canonical_min > earliest_available_ts_utc + 1m`: historical backfill
    `[earliest_available_ts_utc, canonical_min)`
  - tail insurance: `[max(canonical_max + 1m, now_floor - tail_lookback), now_floor)`
  - все задачи enqueue в фоновой REST queue (не блокируют основной loop scheduler).

Периодический `S3` выполняется в две фазы:
- phase-1 (planner queue): enqueue только `scheduler_bootstrap` и `historical_backfill`
  для инструментов, у которых canonical пустой или начинается позже `earliest_available_ts_utc`.
- phase-2 (full catchup): для всех enabled/tradable инструментов запускается
  `RestCatchUp1mUseCase` (tail + full gap scan по историческому диапазону canonical),
  чтобы закрывать внутренние multi-day дыры, включая дни, где в canonical пока 0 строк.

---

## WS worker: точные правила поведения

### A. Загрузка инструментов и разбиение на соединения

1) Читаем из ClickHouse список активных инструментов:
   `SELECT market_id, symbol FROM market_data.ref_instruments WHERE status='ENABLED' AND is_tradable=1`

2) Группируем по `market_id`.

3) Для каждого `market_id` создаём N WS соединений, где:
- `max_symbols_per_connection` берём из runtime config `market_data.markets[*].ws.max_symbols_per_connection`.
- Binance spot/futures: комбинированные стримы `/stream?streams=...`
- Bybit: подписки пачками (ограничение на args уже в адаптере).

4) На reconnect: resubscribe на тот же набор символов.

### B. Только closed 1m

Для каждого источника:
- Binance Spot/Futures:
  - сообщение kline содержит `k` и флаг `k.x` (closed). citeturn9view0turn13view0
  - 1m фильтруем по `k.i == "1m"` (или по подписке только 1m)
- Bybit V5 (spot/linear):
  - сообщение kline содержит `confirm` (closed). citeturn14view0
  - фильтруем по topic/interval: `kline.1` (1 minute)

**Иначе** — игнорируем (но считаем метрику “ignored_non_closed”).

### C. Буферизация и SLO

Буфер гарантирует:
- flush по таймеру ≤ `flush_interval_ms` (конфиг; дефолт 250ms)
- flush по размеру ≤ `max_buffer_rows` (конфиг; дефолт 2000)

**SLO метрика**:
- p95 `closed_received_to_raw_insert_seconds ≤ 1.0`

Точки измерения:
- `t_received_utc`: сразу после JSON parse + нормализации в `CandleWithMeta`
- `t_insert_start_utc`: прямо перед `gateway.insert_rows(...)` внутри raw writer
- `t_insert_done_utc`: сразу после успешного `insert_rows`

---

## Gap detection: правила и действия

### 1) Gap на лету (внутри WS потока)

Для каждого `InstrumentId` в памяти храним `last_seen_minute`.

При получении новой closed свечи с `m = floor_to_minute(ts_open)`:
- если `last_seen_minute is None`: установить `last_seen_minute = m`
- иначе:
  - `expected = last_seen_minute + 1 minute`
  - если `m == expected`: ok
  - если `m <= last_seen_minute`: late/duplicate/out-of-order → не считаем gap (считаем метрику)
  - если `m > expected`: gap `[expected, m)` → enqueue REST fill

### 2) На reconnect (WS переподключение)

Для каждого инструмента, который был на соединении:
1) `canonical_last = CanonicalCandleIndexReader.max_ts_open_lt(before=now_floor)`
2) если `canonical_last is None`: enqueue bootstrap backfill
3) иначе, если `canonical_last < now_floor - 1 minute`: enqueue tail fill `[canonical_last + 1 minute, now_floor)`

---

## Bootstrap/Historical boundary

Если в canonical данных нет (`bounds_1m(..., before=now_floor) == (None, None)`) — грузим **всю доступную историю**:
- `start = market_data.markets[*].rest.earliest_available_ts_utc`
- `end = now_floor`

Concurrency ограничиваем: максимум 4 инструмента одновременно.

Если canonical уже не пустой, но начинается позже earliest boundary:
- `canonical_min > earliest_available_ts_utc + 1m` → историческая догрузка:
  `[earliest_available_ts_utc, canonical_min)`.

Это закрывает production-case, когда worker успел записать только свежий хвост, и
`canonical` перестал быть “пустым” до запуска scheduler.

---

## Maintenance scheduler: джобы и расписание

- S1) Sync whitelist → ref_instruments (on start + периодически)
- S2) Enrich ref_instruments (on start + раз в 6–24 часа)
- S3) REST insurance catchup (каждый час, lookback 2–6 часов) + startup/periodic scan
  исторических “дыр” относительно `earliest_available_ts_utc`.
  В runtime это реализовано как:
  - planner enqueue (`scheduler_bootstrap`/`historical_backfill`) и
  - per-instrument `RestCatchUp1mUseCase` для tail+gap заполнения.

---

## Конфигурация (runtime config)

Используем существующий YAML (`configs/*/market_data.yaml`), расширяем при необходимости:

- `market_data.ingestion.flush_interval_ms: 250`
- `market_data.ingestion.max_buffer_rows: 2000`
- `market_data.ingestion.rest_concurrency_instruments: 4`
- `market_data.ingestion.tail_lookback_minutes: 180`
- `market_data.markets[*].rest.earliest_available_ts_utc: "YYYY-MM-DDTHH:MM:SSZ"`
- `market_data.scheduler.jobs.*` (интервалы)

---

## Запуск процессов (текущие entrypoint’ы)

- Worker:
  - `python -m apps.worker.market_data_ws.main.main --config configs/dev/market_data.yaml --metrics-port 9201`
- Scheduler:
  - `python -m apps.scheduler.market_data_scheduler.main.main --config configs/dev/market_data.yaml --whitelist configs/dev/whitelist.csv --metrics-port 9202`

---

## Метрики (Prometheus)

Worker поднимает `/metrics` через `prometheus_client`.
Job name: **`market-data-ws-worker`**.

Минимум:
- WS reconnects/messages/errors
- CH insert batches/rows/duration/errors
- SLO histograms:
  - `ws_closed_to_insert_start_seconds`
  - `ws_closed_to_insert_done_seconds`
- REST fill tasks/active/errors/duration
- Redis live feed:
  - `redis_publish_total`
  - `redis_publish_errors_total`
  - `redis_publish_duplicates_total`
  - `redis_publish_duration_seconds`
- Scheduler job runs/duration/errors + progress:
  - `scheduler_job_runs_total{job=...}`
  - `scheduler_job_errors_total{job=...}`
  - `scheduler_job_duration_seconds{job=...}`
  - `scheduler_tasks_planned_total{reason=...}`
  - `scheduler_tasks_enqueued_total{reason=...}`
  - `scheduler_startup_scan_instruments_total`
  - `scheduler_rest_catchup_instruments_total{status=ok|failed|skipped_no_seed}`
  - `scheduler_rest_catchup_tail_minutes_total`
  - `scheduler_rest_catchup_tail_rows_written_total`
  - `scheduler_rest_catchup_gap_days_scanned_total`
  - `scheduler_rest_catchup_gap_days_with_gaps_total`
  - `scheduler_rest_catchup_gap_ranges_filled_total`
  - `scheduler_rest_catchup_gap_rows_written_total`

---

## Graceful shutdown

SIGTERM/SIGINT:
1) stop intake
2) flush buffers
3) close ws
4) exit

---

## Примечания

- Одна реплика worker (как указано).
- Дубликаты в raw допустимы; canonical dedup — ReplacingMergeTree(ingested_at).
- REST не может “сгенерировать” свечи, которых у биржи нет — это нормально.
- Для worker/scheduler runtime используется thread-local ClickHouse gateway (один CH client на поток), чтобы при `rest_concurrency_instruments > 1` не ловить session-конфликт `Attempt to execute concurrent queries within the same session`.
- Временный workaround для старых инстансов без фикса: `market_data.ingestion.rest_concurrency_instruments: 1`.
