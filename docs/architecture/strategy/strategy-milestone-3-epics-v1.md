

---

# Strategy — User Goal & Scope (Milestone 3 / EPIC 0)

Документ фиксирует целевой пользовательский сценарий и границы `Strategy v1` (Milestone 3) перед реализацией EPIC’ов 1–13.

## Цель (с точки зрения пользователя)

Пользователь умеет:

1. Собрать конфигурацию стратегии (**spec**) из данных:

* инструмент (`instrument_id` + `instrument_key`)
* таймфрейм (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`; `1m` в live v1 поддерживается)
* набор индикаторов/параметров
* `signal template` (template-based, без DSL)

2. Сохранить стратегию (**обязательная привязка к user_id**).

3. Запускать/останавливать стратегию.

4. В личном кабинете видеть:

* **realtime** метрики/ивенты (через stream)
* **history/truth** данные (из Postgres)

5. Если появились события типа “trade/open/close/…” — получать уведомления в Telegram (если бот подключён).

## Scope / Non-goals (Milestone 3)

### In scope

* Новый bounded context: `src/trading/contexts/strategy/*` (domain + application ports + basic services).
* PG persistence (DDL + repository adapters):

  * стратегии пользователя
  * запуски (runs)
  * события/метрики (truth/history)
* Live runner worker:

  * читает Redis Streams 1m: `md.candles.1m.<instrument_key>`
  * делает rollup в TF стратегии (для TF > `1m`); для TF=`1m` использует базовые 1m (passthrough)
  * гарантирует правило “только закрытые и полные бакеты”
  * умеет repair missing 1m из ClickHouse canonical (`market_data.canonical_candles_1m`)
* Realtime output:

  * Redis Streams для UI: метрики/ивенты (**schema_version=1**)
* Telegram hooks:

  * порт + best-effort адаптер (v1 допускает заглушку)

### Out of scope

* execution: ордера/филы/позиции/комиссии/PnL;
* полноценный backtest engine;
* DSL сигналов (делаем template-based);
* distributed scaling >1 worker instance (заложить в дизайн, но реализация v1 — один воркер).

## Источники данных и контракты (на уровне интеграции)

### Вход (live хвост)

Redis Streams (source of truth для live хвоста):

* Stream: `md.candles.1m.<instrument_key>`
* Message ID: `<epoch_ms>-0` (где epoch_ms = `ts_open` в UTC ms)
* Payload schema_version: `"1"`
* Важное: **все поля строками**, `ts_open/ts_close` — ISO UTC

Документация источника:

* `docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md`
* `docs/runbooks/market-data-redis-streams.md`

Доменные примитивы, на которые маппим payload:

* `src/trading/shared_kernel/primitives/candle.py`
* `src/trading/shared_kernel/primitives/candle_meta.py`

### Repair источник (дырки)

ClickHouse canonical 1m (source of truth для дыр):

* `market_data.canonical_candles_1m`

Текущая реализация reader’а canonical (market_data context):

* `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py`

## Таймфреймы и rollup (Strategy v1)

### Разрешённые TF (live v1)

* `1m`, `5m`, `15m`, `1h`, `4h`, `1d`
* `1m` в live v1 поддерживается как базовый TF (без rollup)

### Rollup правила (строго)

* bucket alignment: `Timeframe.bucket_open(ts)` (epoch-aligned UTC)
* бакет считается закрытым, когда получена 1m свеча с `ts_close == bucket_close`
* derived свеча выпускается **только если бакет полный**:

  * все 1m свечи для `[bucket_open, bucket_close)` присутствуют
* для TF=`1m` rollup-шаг не выполняется (используются входные закрытые 1m свечи)
* агрегация OHLCV:

  * open = first
  * close = last
  * high = max
  * low = min
  * volumes = sum

## Warmup (Strategy v1)

Без warmup стратегия начнёт генерировать сигналы “на голой истории”.

Правило v1:

* `warmup_bars` **не задаётся в spec/API вручную**; runner вычисляет его детерминированно из `spec.indicators`
* алгоритм v1: `numeric_max_param_v1` (`max` по положительным numeric-параметрам индикаторов, fallback=`1`)
* live runner **не выполняет evaluation**, пока не накоплено вычисленное `warmup_bars` свечей TF стратегии
* вычисленный `warmup_bars` фиксируется в `run.metadata_json.warmup` для трассировки
* warmup seed берём не из Redis, а из ClickHouse canonical:

  * при старте run: загрузить диапазон `warmup_bars * timeframe.duration` до `now_floor`
  * построить derived свечи и заполнить буфер

## Telegram (semantics v1)

* Telegram — best-effort: ошибки **не ломают** live runner.
* Telegram notify формируется из StrategyEvent типов `trade_open|trade_close|order_*|...`
* В Milestone 3 допустима заглушка адаптера (логирует), если нет надёжного хранения связки user↔chat.

## Open Questions (для следующих EPIC’ов)

* OQ-001: Как в API определяется `user_id` (аутентификация/авторизация).
* OQ-002: Какой механизм миграций Postgres используем и как он запускается в CI/CD.
* OQ-003: Где хранится binding Telegram (token/chat-id/user).

---
