Все вопросы закрыты. Документ фиксируем как новый архитектурный контракт для STR-EPIC-03.

**Название файла:** `strategy-live-runner-redis-streams-v1.md`
**Куда кладём:** `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md`

Ниже — готовое содержимое по вашему шаблону.

````md
# Strategy — Live Runner via Redis Streams (v1)

Фиксирует контракт STR-EPIC-03: как один live-runner обслуживает множество пользователей/стратегий, читая 1m свечи из Redis Streams, делая rollup, warmup, checkpointing и repair(read).

## Цель

Определить детерминированную и воспроизводимую модель исполнения Strategy v1 в live-режиме:
- один процесс live-runner обслуживает множество активных run’ов (разных пользователей/стратегий);
- вход — WS closed 1m candles через Redis Streams;
- первичная истина прогресса — checkpoint в Postgres (`strategy_runs.checkpoint_ts_open`);
- gaps закрываются чтением из ClickHouse canonical (read-only);
- warmup вычисляется из `spec.indicators` и фиксируется в `run.metadata_json`.

## Контекст

Market Data WS worker публикует закрытые 1m свечи в Redis Streams по схеме `md.candles.1m.<instrument_key>` (best-effort), при этом ClickHouse canonical остаётся историческим источником истины.

В Strategy v1 уже зафиксированы:
- неизменяемое хранение run’ов/событий в Postgres;
- инвариант: не более 1 активного run на одну стратегию;
- checkpoint по базовой 1m (`checkpoint_ts_open`).

Нужно зафиксировать поведение live-runner так, чтобы:
- не требовалась отдельная инфраструктура очередей для run’ов (в v1 — polling PG);
- обеспечивалась идемпотентность и устойчивость к дублям/out-of-order;
- gaps восстанавливались через canonical CH, без запуска ingestion.

## Scope

Входит:
- модель исполнения: **один live-runner** обслуживает много пользователей и много стратегий;
- обнаружение активных run’ов: **polling Postgres** по состояниям `starting|warming_up|running|stopping`;
- чтение live-потока: Redis Streams `md.candles.1m.<instrument_key>` (consumer group);
- вычисление/фиксация warmup из `spec.indicators` (numeric_max_param_v1) в `run.metadata_json.warmup`;
- rollup из 1m в TF стратегии (включая TF=`1m`);
- strict монотонность checkpoint + repair(read) из ClickHouse canonical;
- переходы run state: `starting → warming_up → running → stopped` (и обработка stop).

## Non-goals

- Горизонтальное масштабирование live-runner (в v1 **только 1 инстанс**).
- Гарантия “в Redis значит уже в canonical”: публикация best-effort и может опережать durable-хранилище.
- UI-стримы/пейлоады для фронта (это STR-EPIC-04).
- Запуск ingestion из Strategy (repair(read) только читает canonical CH, не инициирует ingestion).

## Ключевые решения

### 1) Один live-runner обслуживает множество пользователей и стратегий (multi-tenant runner)

Live-runner агрегирует активные run’ы из Postgres и группирует их по `instrument_key`.
Для каждого `instrument_key` читается один Redis stream, и одна 1m свеча fan-out’ится на все run’ы, которым она нужна.

Причины выбора:
- минимальный ops (один процесс);
- нет дублирования чтения одной и той же свечи “на пользователя”;
- предсказуемая нагрузка и простые метрики.

Последствия:
- v1 не масштабируется горизонтально без шардинга по инструментам;
- нужен локальный кеш активных run’ов (обновляемый polling’ом).

### 2) Активные run’ы обнаруживаются polling’ом Postgres (v1)

Live-runner периодически запрашивает список run’ов в состояниях:
`starting|warming_up|running|stopping`.

Рекомендуемый параметр:
- `poll_interval_seconds`: конфигурируемо; дефолт **5s**.

Причины выбора:
- минимальные зависимости (без отдельной очереди);
- достаточная реактивность для v1.

Последствия:
- возможна задержка до `poll_interval_seconds` при старте/остановке run.

### 3) В v1 допускается ровно один инстанс live-runner

Redis Streams consumer group делит сообщения между consumers, поэтому запуск нескольких инстансов без шардинга приведёт к “размазке” свечей.
В v1 фиксируем **один** инстанс.

Последствия:
- упрощение семантики consumer-group;
- лимит по throughput закрывается только оптимизациями внутри процесса.

### 4) Timeframe live v1 поддерживает включая `1m`

TF стратегии может быть `1m` и любые разрешённые shared-kernel `Timeframe`.
Rollup в случае `1m` — pass-through (derived bucket == базовая свеча).

Последствия:
- live-runner обязан корректно работать на `1m` без “запрета” в контракте.

### 5) Warmup вычисляется runner’ом из `spec.indicators` и сохраняется в run.metadata_json.warmup

`warmup_bars` не задаётся в spec как число вручную.
Runner вычисляет warmup по алгоритму **numeric_max_param_v1** (максимальный “числовой параметр окна/периода” среди индикаторов v1) и сохраняет в:
- `run.metadata_json.warmup` (и, соответственно, `strategy_runs.metadata_json.warmup`).

Последствия:
- warmup становится детерминированной функцией `spec.indicators`;
- изменился spec → меняется warmup (что ожидаемо и воспроизводимо).

### 6) Primary checkpoint — Postgres `strategy_runs.checkpoint_ts_open`, strict монотонность

Primary truth о прогрессе обработки 1m:
- `strategy_runs.checkpoint_ts_open` (последняя обработанная базовая 1m).

Правило обработки входной 1m свечи:
- если `ts_open <= checkpoint_ts_open` → **ignore** (идемпотентность);
- если `ts_open == checkpoint_ts_open + 1m` → нормальная обработка, продвижение checkpoint;
- если `ts_open > checkpoint_ts_open + 1m` → **gap**, запускаем repair(read) из ClickHouse canonical, и **не продвигаем checkpoint**, пока не восстановим непрерывность.

Последствия:
- устойчивость к дублям и повторным доставкам;
- gaps закрываются “истиной” из canonical, а не эвристиками из Redis.

### 7) Repair(read): только read из ClickHouse canonical, derived bucket закрываем строго

При gap по 1m live-runner дочитывает недостающие 1m из ClickHouse canonical (read-only).
Derived bucket (TF стратегии) считается закрытым, только когда **все 1m внутри бакета присутствуют**.

Последствия:
- предсказуемая и детерминированная сборка rollup-баров;
- “late minute” не приводит к пересборке уже закрытого derived bucket (в v1 закрытие строгое).

### 8) Переходы состояний run управляются live-runner (v1)

API создаёт run в состоянии `starting`.
Live-runner выполняет:
- после seed из canonical → перевод `starting → warming_up`;
- после достижения warmup → перевод `warming_up → running`;
- при остановке → доводит до `stopped` (или уважает stop, если stop инициирован use-case API).

Последствия:
- состояние run отражает реальную стадию готовности исполнения;
- UI/оператору проще диагностировать “где мы”.

### 9) Публикация Market Data в Redis best-effort и может опережать durable insert

В WS worker обработка закрытой свечи устроена так:
1) `insert_buffer.submit(row)` — enqueue в асинхронный буфер вставки,
2) `live_candle_publisher.publish_1m_closed(row)` — best-effort XADD в Redis,
3) gap tracking / rest-fill enqueue.

Это означает: **Redis publish не является доказательством, что свеча уже попала в canonical** на момент получения стратегии.
Live-runner:
- использует payload из Redis для live-обработки;
- использует ClickHouse canonical только для repair(read) при gap;
- при repair(read) допускает, что canonical может “догонять” (риск описан ниже).

## Контракты и инварианты

- В v1 запущен ровно **один** live-runner инстанс.
- Live-runner обслуживает множество пользователей/стратегий (multi-tenant).
- Активные run’ы обнаруживаются **polling**’ом Postgres по состояниям: `starting|warming_up|running|stopping`.
- Инвариант хранения: **не более 1 активного run на одну стратегию** (enforced storage-side).
- Входной live stream: `md.candles.1m.<instrument_key>`.
- `instrument_key` стратегии обязан совпадать с canonical/ingestion ключом.
- Primary checkpoint: `strategy_runs.checkpoint_ts_open` по базовой 1m.
- Идемпотентность: `ts_open <= checkpoint_ts_open` → ignore.
- Gap: `ts_open > checkpoint+1m` → repair(read) из ClickHouse canonical; checkpoint не продвигается до восстановления непрерывности.
- Rollup: derived bucket закрывается только когда есть все 1m внутри бакета.
- Warmup: вычисляется runner’ом из `spec.indicators` (numeric_max_param_v1) и сохраняется в `run.metadata_json.warmup`.
- Market Data Redis publish best-effort и может опережать durable вставки; стратегия не должна полагаться на “Redis => canonical already”.

## Связанные файлы

- `apps/worker/market_data_ws/wiring/modules/market_data_ws.py` — WS worker: enqueue raw insert, best-effort Redis publish, gap tracking.
- `docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md` — контракт live feed (stream name, schema, best-effort, ID).
- `src/trading/contexts/market_data/adapters/outbound/messaging/redis/redis_streams_live_candle_publisher.py` — publisher (детерминированный XADD id, дубли/out-of-order как no-op).
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/canonical_candle_reader.py` — чтение canonical 1m (read-only, дедуп хвоста 24h).
- `src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py` — `instrument_id`/`instrument_key`/`timeframe`, валидации spec.
- `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/strategy_run_repository.py` — хранение run’ов, инвариант “один активный run”.
- `docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md` — доменная спецификация runs/events/checkpoint.
- `docs/architecture/strategy/strategy-milestone-3-epics-v1.md` — scope по strategy milestone (обновлённые пункты warmup/timeframe).
- `docs/architecture/roadmap/milestone-3-epics-v1.md` — roadmap STR-EPIC-03/04 границы.
- `docs/architecture/roadmap/base_milestone_plan.md` — базовый план (warmup/runner/контракты).
- `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md` — API-контроль run и описание warmup как вычисляемого.

## Как проверить

```bash
# запускать из корня репозитория
ruff check .
pyright
pytest -q
````

## Риски и открытые вопросы

* Риск: canonical CH может “догонять” публикацию в Redis (publish идёт после enqueue в insert buffer, но до гарантированной flush/durability).
  Влияние: repair(read) может временно не найти “только что опубликованную” свечу в canonical.
  Митигатор: при repair(read) предусмотреть retry/backoff по хвосту, либо ограничить repair(read) только по реально подтверждённым gaps относительно checkpoint.
* Риск: один инстанс live-runner может стать CPU/IO bottleneck при росте числа инструментов/run’ов.
  Митигатор: оптимизация fan-out, батчинг, профилирование; v2 — шардинг по `instrument_key` и несколько инстансов.
* Открытых вопросов по контракту v1: **нет** (все решения зафиксированы в секции “Ключевые решения”).

```

