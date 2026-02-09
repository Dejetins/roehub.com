# Market Data — Runtime Config & Invariants (v2)

Этот документ фиксирует runtime-конфигурацию и инварианты исполнения для bounded context `market_data`
в рамках этапа v2 (REST + WS + автоматическая догонка хвоста).

Цель:
- один YAML описывает 4 рынка (binance/bybit × spot/futures), endpoints REST/WS, параметры reconnect/ping, flush policy;
- whitelist тикеров задаётся CSV файлом;
- REST backfill режется на чанки так, чтобы один insert не пересекал >7 дневных партиций ClickHouse;
- rate limits берутся **онлайн** у биржи (autodetect), поэтому численные лимиты не хранятся в YAML.

Scope:
- конфигурация и её валидация относятся к wiring/infra (composition roots), не к domain/application.


## Ключевые решения

### 1) Единственный источник runtime-настроек — `configs/.../market_data.yaml`
CLI/worker/notebooks читают YAML на старте процесса.
Настройки меняются без правок кода: достаточно изменить YAML/CSV и перезапустить процесс.

Конфиг читается на старте процесса (без hot reload). Это упрощает эксплуатацию и делает запуск детерминированным.

### 2) Whitelist тикеров — CSV файл (источник правды)
Whitelist определяет, какие инструменты:
- попадают в `ref_instruments` (через отдельный use-case в EPIC 1),
- подписываются в WS,
- догоняются через REST.

CSV — операторский формат, легко редактируется и версионируется.

### 3) Rate limits — autodetect (онлайн), численные лимиты не хранятся в YAML
Численные лимиты могут изменяться биржей, поэтому значения берём **из ответов/метаданных API**:

- **Binance**:
  - `exchangeInfo.rateLimits[]` даёт текущие caps/лимиты.
  - REST-ответы возвращают заголовки `X-MBX-USED-WEIGHT-*` (usage), что позволяет адаптивно ограничивать скорость запросов.

- **Bybit**:
  - REST-ответы возвращают заголовки `X-Bapi-Limit`, `X-Bapi-Limit-Status`, `X-Bapi-Limit-Reset-Timestamp`,
    что достаточно для динамической подстройки лимитера на уровне endpoint’ов.

В YAML остаются только *управляющие параметры лимитера* (safety factor, max concurrency), чтобы:
- ограничить параллелизм,
- иметь консервативный режим до получения первых заголовков/metadata.

### 4) REST backfill использует time-slicing по UTC дням: максимум 7 дней на insert
ClickHouse raw таблицы партиционированы по дню (`toYYYYMMDD` по времени открытия свечи).
Чтобы избегать ошибок и лимитов на количество партиций/parts в одной вставке,
любой диапазон backfill режется на чанки так, чтобы один insert затрагивал не более 7 дневных партиций.


## YAML Schema (market_data.yaml)

### Top-level
- `version: int` — версия схемы конфига (для будущей миграции)
- `market_data: {...}` — настройки market_data

### market_data.markets[]
Каждый элемент описывает один рынок.

Required fields:
- `market_id: int` — соответствует `market_data.ref_market.market_id`
- `exchange: "binance" | "bybit"`
- `market_type: "spot" | "futures"`
- `market_code: str` — например `binance:spot`

REST:
- `rest.base_url: str`
- `rest.earliest_available_ts_utc: str (ISO-8601, UTC, not in future)`
- `rest.timeout_s: float`
- `rest.retries: int`
- `rest.backoff.base_s: float`
- `rest.backoff.max_s: float`
- `rest.backoff.jitter_s: float`

REST limiter (управляющие параметры):
- `rest.limiter.mode: "autodetect"` — фиксировано для v2
- `rest.limiter.safety_factor: float` — доля от лимита, которую используем (например 0.8)
- `rest.limiter.max_concurrency: int` — максимальная параллельность запросов

WS:
- `ws.url: str`
- `ws.ping_interval_s: float`
- `ws.pong_timeout_s: float`
- `ws.reconnect.min_delay_s: float`
- `ws.reconnect.max_delay_s: float`
- `ws.reconnect.factor: float`
- `ws.reconnect.jitter_s: float`
- `ws.max_symbols_per_connection: int`

### market_data.ingestion
Flush/maintenance policy для WS worker и scheduler:
- `flush_interval_ms: int` — максимальная задержка удержания данных в буфере перед вставкой в ClickHouse.
  Даже если буфер не достиг `max_buffer_rows`, по таймеру выполняется flush.
  Меньше `flush_interval_ms` → ниже latency до raw, но больше insert-операций.
- `max_buffer_rows: int` — предохранитель на всплески: flush при достижении размера буфера.
- `rest_concurrency_instruments: int` — ограничение параллельных REST fill/insurance задач по инструментам.
- `tail_lookback_minutes: int` — lookback для scheduler insurance catchup (S3).

SLO guidance:
- чтобы уложиться в “≤1s до raw” (лучше 0.5s), рекомендуется `flush_interval_ms` порядка 250–500 ms
  (при разумном `max_buffer_rows`, например 1k–5k).

### market_data.scheduler.jobs
Периодические maintenance интервалы:
- `sync_whitelist.interval_seconds`
- `enrich.interval_seconds`
- `rest_insurance_catchup.interval_seconds`

Operational semantics:
- scheduler на старте делает startup scan по enabled/tradable инструментам:
  - bootstrap `[earliest_available_ts_utc, now_floor)` для пустого canonical
  - historical backfill `[earliest_available_ts_utc, canonical_min)` если canonical начинается позже earliest boundary
  - tail insurance `[max(canonical_max + 1m, now_floor - tail_lookback), now_floor)`.
- observability startup scan:
  - `scheduler_startup_scan_instruments_total`
  - `scheduler_tasks_planned_total{reason=...}`
  - `scheduler_tasks_enqueued_total{reason=...}`

### market_data.backfill
Политика REST backfill:
- `max_days_per_insert: 7` — константа этапа v2 (не конфигурируется выше 7)
- `chunk_align: "utc_day"` — режем по границам UTC дней


## Whitelist CSV schema

File: `whitelist.csv`

Required columns:
- `market_id` — int
- `symbol` — str (trim, upper)
- `is_enabled` — 0/1

Semantics:
- `is_enabled=1` → инструмент активен: участвует в подписках и backfill
- `is_enabled=0` → инструмент игнорируется

Duplicates:
- если встречается повтор ключа `(market_id, symbol)`, применяется правило:
  - последняя строка побеждает (last-win), с warning в лог


## Time-slicing (≤7 дней на insert)

Input:
- `TimeRange [start, end)` в UTC

Rule:
- диапазон режется по границам UTC-дней так, чтобы каждый чанк затрагивал не более 7 дневных партиций CH.

Algorithm (conceptual):
- cursor = start
- while cursor < end:
  - anchor = floor_to_utc_day(cursor)              # 00:00 UTC
  - boundary = anchor + 7 days                     # day boundary
  - slice_end = min(end, boundary)
  - yield [cursor, slice_end)
  - cursor = slice_end

Guarantee:
- один чанк не пересекает >7 партиций `toYYYYMMDD(ts_open)`.


## Notes
- Конфиг читается на старте процесса (без hot reload).
- Валидируем YAML/CSV строго и падаем быстро, чтобы не получать “тихие” частичные запуски.
- `rest.earliest_available_ts_utc` — обязательный ключ, используется worker/scheduler для bootstrap/historical границ.
- Для конкурентных `asyncio.to_thread(...)` путей worker/scheduler используют thread-local CH gateway
  (один ClickHouse client на поток), чтобы избежать session-конфликта concurrent queries.
- Adapters могут логировать “observed rate limits” (из заголовков/metadata) для диагностики,
  но не используют захардкоженные численные лимиты из конфига.
