````md
# Strategy realtime output via Redis Streams v1

Архитектурный контракт v1 для публикации realtime метрик и событий стратегии в Redis Streams (per-user), чтобы UI мог подписаться и видеть состояние run’ов.

## Цель

Дать UI простой способ **подписаться на пользователя** и получать:
- **метрики** по запущенным run’ам (warmup, checkpoint, lag, gap/repair и т.д.);
- **события** уровня “полезно пользователю” (изменение состояния run, остановка, ошибка).

## Контекст

У нас уже есть:
- Live Runner (STR-EPIC-03), который читает `md.candles.1m.<instrument_key>` из Redis Streams и детерминированно обновляет `StrategyRun` (checkpoint, warmup, state).  
  См. `src/trading/contexts/strategy/application/services/live_runner.py` и док: `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md`.
- Redis уже используется как transport (market_data live feed), и есть принцип **best-effort semantics**: ошибки Redis не должны ломать воркер.  
  См. `docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md`.

В текущей архитектуре отсутствует стабильный realtime output для UI, который можно “tail’ить” по пользователю без доступа к БД.

## Scope

В рамках v1 добавляем **публикацию в Redis Streams**:

### Streams (per-user)
- `strategy.metrics.v1.user.<user_id>`
- `strategy.events.v1.user.<user_id>`

### Payload schema v1 (все значения — строки)

#### Metrics payload fields
`schema_version, ts, strategy_id, run_id, metric_type, value, instrument_key, timeframe`

#### Events payload fields
`schema_version, ts, strategy_id, run_id, event_type, payload_json, instrument_key, timeframe`

Правила:
- `schema_version` фиксировано `"1"`.
- `ts` — ISO-8601 UTC строка с `Z` (например `"2026-02-17T12:34:56.000Z"`).
- `payload_json` — строка JSON-объекта (с детерминированной сериализацией: `sort_keys=true`, `separators=(",", ":")`, `ensure_ascii=true`).

### Fixed enumerations (v1)

#### metric_type (фиксированный список v1)
Базовый набор (из требований):
- `warmup_processed_bars` — сколько “закрытых” баров засчитано в warmup (value: `"0".."N"`).
- `checkpoint_ts_open` — текущий checkpoint `ts_open` (value: ISO UTC строка или `""` если None).
- `lag_seconds` — отставание в секундах относительно `now` (value: `"0".."N"`).
- `candles_processed_total` — сколько 1m свечей было принято как contiguous (value: `"0".."N"`).
- `rollup_bucket_closed` — закрыт ли bucket rollup на этом шаге (value: `"0"`/`"1"`).
- `gap_detected` — обнаружен ли gap на входе (value: `"0"`/`"1"`).
- `repair_missing_bars` — сколько 1m баров не хватает до непрерывности на момент gap (value: `"0".."N"`).

Дополнительные метрики (рекомендуемые, чтобы UI/дебаг закрывали “все вопросы” текущей реализации live-runner):
- `warmup_required_bars` — требуемое число баров warmup (value: `"1".."N"`), из `numeric_max_param_v1`.
- `warmup_satisfied` — удовлетворён ли warmup (value: `"0"`/`"1"`).
- `run_state` — текущее состояние run (value: `"starting"|"warming_up"|"running"|"stopping"|"stopped"|"failed"`).
- `rollup_bucket_count_1m` — сколько 1m свечей набрано в текущем rollup bucket (value: `"0".."N"`).
- `rollup_bucket_open_ts` — `bucket_open_ts` (value: ISO UTC строка или `""` если None).
- `repair_attempt` — номер попытки repair, когда есть gap (value: `"0".."N"`), `0` если repair не выполнялся.
- `repair_continuous` — удалось ли восстановить непрерывность до `target_ts_open` (value: `"0"`/`"1"`).
- `dropped_non_contiguous_total` — сколько раз candle была отброшена как не-contiguous (value: `"0".."N"`).

> Важно: список `metric_type` **явно фиксируется** в этом документе и **не расширяется молча**. Расширение = новый документ `-v2` или явный ADR/изменение контракта.

#### event_type (фиксированный список v1, только “полезно пользователю”)
- `run_state_changed` — изменение состояния run (payload_json: `{"from":"...","to":"..."}`).
- `run_stopped` — run завершён в `stopped` (payload_json: `{}` или `{"reason":"user_request"}` если появится).
- `run_failed` — run упал (payload_json: `{"error":"<last_error>"}`).

> Non-goal: gap-repair и rollup события (шум для пользователя). Они остаются как метрики/внутренний дебаг при необходимости.

## Non-goals

- WebSocket/SSE gateway для UI (только запись в Redis Streams).
- Exactly-once delivery. Семантика best-effort как у market_data publisher.
- Горизонтальный scale live-runner / шардирование по пользователям/инструментам.
- Управление retention/trim (MAXLEN) в v1 (обсуждается отдельно).
- Consumer groups/ACK для UI (UI читает как tail через `XREAD`).
- Публикация “каждой свечи” как event (слишком шумно).
- Публикация gap-repair событий и rollup событий в events stream.

## Ключевые решения

### 1) Per-user Redis Streams для realtime output (metrics + events)

UI подписывается на два стабильных стрима по `user_id`, без доступа к БД и без знания внутренней структуры воркера.

Связано:
- Live runner: `src/trading/contexts/strategy/application/services/live_runner.py`
- Redis Streams стиль: `docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md`

Последствия:
- ✅ Простое потребление в UI: `XREAD ... STREAMS strategy.metrics... $`.
- ⚠️ Кардинальность ключей (по пользователям) и рост объёма данных → нужен retention план (вне v1).

### 2) Строгая schema v1 и “все значения строками”

Делаем payload максимально простым для UI и совместимым с Redis Streams (аналогично market_data): все поля — строки, `payload_json` — строка JSON.

Последствия:
- ✅ UI не ломается от типов, сериализация стабильна.
- ⚠️ Нужно явно фиксировать формат строк (ISO UTC, `"0"/"1"`, счётчики как `"123"`).

### 3) Детерминированные message IDs: `<ts_epoch_ms>-<seq>`

Чтобы:
- обеспечивать стабильный порядок в рамках одного `ts`;
- иметь возможность считать `duplicates_total` (как signal повтора/коллизии).

Правило:
- `ts_epoch_ms` берётся из поля `ts` (ISO → epoch_ms).
- `seq` присваивается **детерминированно** в рамках `(user_id, stream_kind, ts_epoch_ms)`:
  - формируется список публикаций за этот `ts_epoch_ms`,
  - сортируется стабильным ключом (пример):  
    `(kind, strategy_id, run_id, instrument_key, timeframe, metric_type|event_type)`,
  - `seq = index` (0..N-1).

Последствия:
- ✅ Повторная публикация тем же воркером “на тот же `ts`” детектится как duplicate.
- ⚠️ При будущей multi-instance архитектуре нужна координация/шардирование, иначе коллизии вероятны (вне v1).

### 4) Best-effort semantics для Redis publish

Ошибки Redis (подключение/таймаут/ResponseError) **не ломают** live-runner воркер:
- publish ошибки логируются,
- увеличиваются Prometheus счётчики,
- runner продолжает работу (как в market_data publisher контракте).

Последствия:
- ✅ Стабильность воркера.
- ⚠️ UI может пропустить часть обновлений при проблемах Redis (принимаем в v1).

### 5) Точка публикации: внутри Strategy Live Runner, после успешной персистенции

Realtime output публикуется из live-runner, потому что только он детерминированно знает:
- checkpoint,
- warmup progress,
- transitions `starting/warming_up/running/stopping/stopped/failed`,
- факт gap/repair результата.

Инвариант: publish делается **после** успешного `run_repository.update(...)` и **до/после** `ack` свечи — но так, чтобы “ack без сохранения” не происходил.

## Контракты и инварианты

- Stream names фиксированы:
  - `strategy.metrics.v1.user.<user_id>`
  - `strategy.events.v1.user.<user_id>`
- `schema_version == "1"` для всех записей.
- Все значения payload — строки.
- `ts` всегда ISO-8601 UTC с `Z`.
- `payload_json` — валидный JSON object string с детерминированной сериализацией.
- `metric_type` и `event_type` только из фиксированных списков v1.
- Message ID для `XADD` имеет формат `<ts_epoch_ms>-<seq>`, где `seq` детерминированен внутри `(user_id, stream_kind, ts_epoch_ms)`.
- Best-effort: любые ошибки Redis publish **не меняют** run state и **не падают** наружу как фатальная ошибка итерации.
- Приватность: streams адресуются по `user_id` и предназначены для UI слоя с корректной авторизацией (авторизация вне scope).

## Связанные файлы

Существующие (точки интеграции/референсы):
- `src/trading/contexts/strategy/application/services/live_runner.py` — основная логика обработки, warmup/rollup/checkpoint/state; точка эмита метрик/событий.
- `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py` — wiring воркера, Prometheus метрики, Redis-конфиг.
- `configs/dev/strategy_live_runner.yaml` — текущая конфигурация Redis (host/port/db/password_env/timeouts).
- `src/trading/contexts/strategy/domain/entities/strategy_run.py` — run state machine и поля (`user_id`, `checkpoint_ts_open`, `last_error`).
- `src/trading/contexts/strategy/adapters/outbound/messaging/redis/redis_streams_live_candle_stream.py` — референс реализации Redis client + паттерны обработки `ResponseError`.
- `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md` — базовый документ live-runner.
- `docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md` — best-effort semantics референс.
- `docs/runbooks/market-data-redis-streams.md` — стиль runbook команд Redis Streams.

Ожидаемые новые/изменяемые (в рамках STR-EPIC-04 имплементации):
- `src/trading/contexts/strategy/application/ports/realtime_output_publisher.py` — порт публикации (metrics/events).
- `src/trading/contexts/strategy/adapters/outbound/messaging/redis/redis_streams_realtime_output_publisher.py` — Redis Streams publisher adapter.
- `src/trading/contexts/strategy/adapters/outbound/messaging/redis/__init__.py` — стабильные экспорты.
- `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py` — подключение publisher + Prometheus publish_* метрики.
- `configs/{dev,prod,test}/strategy_live_runner.yaml` — секция конфигурации realtime output (enabled + префиксы/таймауты при необходимости).
- `tests/unit/contexts/strategy/**` — unit-тесты publisher’а и интеграции в runner.

## Как проверить

```bash
# запускать из корня репозитория
ruff check .
uv run pyright
uv run pytest -q

# (опционально) локальный запуск воркера
uv run python -m apps.worker.strategy_live_runner.main.main --config configs/dev/strategy_live_runner.yaml
````

### Runbook: redis-cli команды для проверки realtime streams

Подставь реальный `USER_ID` (строкой) и адрес redis (если не локально).

```bash
# 1) Посмотреть последние 10 метрик
redis-cli XREVRANGE "strategy.metrics.v1.user.<USER_ID>" + - COUNT 10

# 2) Посмотреть последние 10 событий
redis-cli XREVRANGE "strategy.events.v1.user.<USER_ID>" + - COUNT 10

# 3) Подписаться (tail) на новые метрики (аналог "follow")
redis-cli XREAD BLOCK 0 STREAMS "strategy.metrics.v1.user.<USER_ID>" $

# 4) Подписаться (tail) на новые события
redis-cli XREAD BLOCK 0 STREAMS "strategy.events.v1.user.<USER_ID>" $

# 5) Проверить, что стрим существует и увидеть meta (длина, first/last entry)
redis-cli XINFO STREAM "strategy.metrics.v1.user.<USER_ID>"
redis-cli XINFO STREAM "strategy.events.v1.user.<USER_ID>"

# 6) Быстро узнать длину
redis-cli XLEN "strategy.metrics.v1.user.<USER_ID>"
redis-cli XLEN "strategy.events.v1.user.<USER_ID>"
```

## Риски и открытые вопросы

* Риск: рост объёма данных в per-user streams без retention → Redis память/CPU.
  Влияние: деградация Redis, нагрузка на воркеры и UI.
  Что дальше: определить retention/trim стратегию (MAXLEN ~) и частоту публикаций.

* Риск: при будущем multi-instance live-runner возможны коллизии message_id (дубликаты) на одном user stream.
  Влияние: потеря части сообщений или рост duplicates_total.
  Что дальше: шардирование по user_id/стратегиям или отказ от детерминированного ID в multi-instance режиме (в отдельном дизайне).

* Вопрос: публикуем ли `checkpoint_ts_open` как `""` при `None` или не публикуем метрику вообще?
  Рекомендация v1: публиковать всегда, `""` означает отсутствие checkpoint.

* Вопрос: частота публикации метрик — “на каждую принятую свечу” или “на итерацию runner” (poll loop)?
  Рекомендация v1: ключевые метрики публиковать на принятую свечу (checkpoint/warmup/lag), остальные — по необходимости/на transition.

```
```
