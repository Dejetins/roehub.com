# Market Data Live Tail Repair v1

План фиксирует восстановление коротких разрывов live-хвоста закрытых 1m свечей для стратегий после blocker `Stage 12.4`: Redis hot cache, bounded ClickHouse repair, REST tail fallback через `Market Data`, repair audit, метрики и доказательства на Mac Studio.

## Статус Документа

| Поле | Значение |
|---|---|
| `status` | `planned` |
| `created_at` | `2026-06-29` |
| `owner` | `Market Data`, `Strategy`, `Live Execution` |
| `trigger` | Blocker Stage `12.4` in `strategy-producer-paper-testnet-trading-v1` |
| `stage_ledger` | `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md` |
| `prompt_pack` | `.codex/agents/generated/market-data-live-tail-repair-v1/` |

## Коротко

`Stage 12.4` доказал, что стратегия может стабильно писать `StrategySignal` и `ExecutionSourceEvent`, пока поток закрытых минутных свечей непрерывен. Но при пропуске одной минуты в Redis stream текущий repair path зависит от ClickHouse. Если ClickHouse или его HTTP-интерфейс временно недоступен, `StrategyLiveRunner` не может восстановить непрерывный хвост и не должен двигать `strategy_runs.checkpoint_ts_open`.

Цель этого плана: сделать короткое восстановление live-хвоста свечей надежным и измеримым, не превращая стратегию в прямого клиента Binance/Bybit.

Целевая цепочка:

```text
Redis stream -> Redis hot cache -> ClickHouse with short timeout/circuit breaker -> exchange REST tail -> repair audit/outbox -> blocked/failed only after policy exhaustion
```

## Business Impact

Этот repair-cycle нужен, чтобы paper/testnet strategy production не останавливался из-за одного короткого пропуска закрытой минутной свечи, если ClickHouse в этот момент временно недоступен. Для бизнеса это означает более надежный непрерывный выпуск сигналов и событий исполнения в тестовом контуре, меньше ручных перезапусков и понятный audit trail, почему gap был восстановлен или почему система безопасно остановилась.

План не включает mainnet trading и не обещает production runtime proof до Stage `06`. До этого stages дают только bounded implementation/proof по своим boundary: Redis, provider chain, ACK policy, metrics и docs.

## Почему Сейчас

| Факт | Значение |
|---|---|
| До сбоя `12.4` | `111` обработанных свечей, `111` `StrategySignal`, `111` `ExecutionSourceEvent`, дубли `0/0/0`, p99 `StrategySignal -> ExecutionSourceEvent` около `57ms`. |
| Точка сбоя | Redis stream `md.candles.1m.binance:spot:BTCUSDT` пропустил свечу `2026-06-27T01:38:00Z` и перешел с `01:37` к `01:39`. |
| Текущий repair | `_repair_gap` в `StrategyLiveRunner` читает только `market_data.canonical_candles_1m FINAL` через ClickHouse. |
| Runtime blocker | Выбранный run `d87917a1-1d72-49a8-b5c5-e40290bd3096` позже стал `failed` после `ConnectionResetError(54, 'Connection reset by peer')` к `http://127.0.0.1:8123`. |
| Вывод | ClickHouse остается исторической истиной, но не может быть единственным быстрым источником восстановления live gap. |

## Scope

Входит:

| Область | Что делаем |
|---|---|
| Market Data | Новый short-tail repair contract, Redis hot cache, REST tail recovery, repair audit/outbox, metrics. |
| Strategy | Новый порт `ClosedCandleTailProvider` вместо прямой зависимости `_repair_gap -> CanonicalCandleReader`. |
| Redis | Отдельный горячий cache для диапазонного чтения по `instrument_key` и `ts_open`; Redis stream остается транспортом доставки. |
| ClickHouse | Исторический источник и reconciliation source; в live repair path только bounded call с коротким timeout/circuit breaker. |
| REST adapters | Используются только через Market Data boundary, только для короткого хвоста строго закрытых 1m свечей. |
| Ops | Метрики, alerts, runbook, runtime-доказательство на Mac Studio. |
| Stage `12.4` | Разблокируем повторный запуск, но не смешиваем repair implementation и 6h soak в один stage. |

Не входит:

| Область | Почему не входит |
|---|---|
| Mainnet trading | Текущий цикл остается `paper`/`testnet`; mainnet submit запрещен. |
| Новый exchange-execution дизайн | Execution path уже принят foundation stages; repair касается входных свечей. |
| Горизонтальное масштабирование live-runner | v1 сохраняет один `strategy-live-runner`; шардирование по `instrument_key` отдельно. |
| Полная замена ClickHouse | ClickHouse остается historical truth и reconciliation source. |
| Прямой REST из Strategy | Нарушает bounded context и secret/provider boundary. |

## Текущее Состояние

| Компонент | Наблюдаемый факт | Источник |
|---|---|---|
| Redis stream publisher | `market-data-ws-worker` публикует закрытые 1m свечи в `md.candles.1m.<instrument_key>` с deterministic stream id `<ts_open_epoch_ms>-0`; publish best-effort. | `redis_streams_live_candle_publisher.py`, `market-data-live-feed-redis-streams-v1.md` |
| Strategy stream consumer | `RedisStrategyLiveCandleStream` читает consumer group `strategy.live_runner.v1`; invalid payloads ack/drop; valid messages ACKаются отдельным вызовом. | `redis_streams_live_candle_stream.py` |
| Current runner gap repair | `_repair_gap` читает только `CanonicalCandleReader.read_1m(...)`; при failure exception может перевести run в `failed`. | `live_runner.py` |
| Existing REST fill | `RestFillRange1mUseCase` читает через `CandleIngestSource` и пишет raw storage; он не является синхронным short-tail provider для Strategy. | `rest_fill_range_1m.py` |
| Existing docs drift | `strategy-live-runner-redis-streams-v1.md` фиксирует gaps only via ClickHouse canonical. | `strategy-live-runner-redis-streams-v1.md` |

## Целевая Архитектура

### Владение

| Контекст | Ответственность |
|---|---|
| `Market Data` | Получает, кеширует, восстанавливает и аудирует short-tail closed candles. |
| `Strategy` | Требует непрерывный диапазон через порт, но не знает REST SDK, API keys, raw provider payloads или ClickHouse failure details. |
| `Live Execution` | Получает `ExecutionSourceEvent` только после того, как стратегия смогла принять свечу и записать `StrategySignal`. |

### Новый Порт

Плановый порт:

```text
src/trading/contexts/strategy/application/ports/closed_candle_tail_provider.py
```

Контракт:

```text
get_closed_1m_tail(instrument_id, instrument_key, start_ts_open, end_ts_open, correlation_id)
  -> ClosedCandleTailResult
```

Правила:

| Правило | Требование |
|---|---|
| Диапазон | Полуинтервал `[start_ts_open, end_ts_open)`, где обе границы выровнены по минуте. |
| Закрытость | Provider возвращает только строго закрытые свечи; текущая незакрытая минута запрещена. |
| Непрерывность | `continuous=true` только если есть каждая 1m свеча от `start` до `end - 1m`. |
| Происхождение | Каждая свеча имеет `source=redis_hot_cache|clickhouse|rest`, `ingest_id`, `ingested_at`, `market_id`, `symbol`, `instrument_key`, `ts_open`, `ts_close`. |
| Дедуп | При нескольких источниках один `ts_open` дает один deterministic row; приоритет источников фиксирован. |
| Ошибки | Частичный результат не продвигает checkpoint; failure записывает repair audit. |

### Redis Hot Cache

Redis stream не является удобным range-store: consumer group pending/ack семантика не дает безопасно восстановить старую пропущенную минуту как диапазон. Поэтому нужен отдельный hot cache:

| Ключ | Назначение |
|---|---|
| `md:hot:1m:<instrument_key>:z` | Sorted set: score = `ts_open_epoch_ms`, member = `ts_open_epoch_ms`. |
| `md:hot:1m:<instrument_key>:h` | Hash: field = `ts_open_epoch_ms`, value = normalized candle JSON. |
| `md:hot:1m:<instrument_key>:meta` | Optional hash for retention/debug counters if needed. |

Retention v1: `6-24h`, default `24h` in production config unless runtime memory proof forces lower value.

Write paths:

| Source | Write to hot cache |
|---|---|
| WS closed candle publish | Required, same normalized payload as stream plus source/provenance. |
| REST tail repair | Required before returning repaired candle to Strategy. |
| ClickHouse repair read | Optional cache backfill for the repaired range; recommended to reduce repeated CH reads. |

### Provider Source Order

| Порядок | Источник | Timeout / policy | Роль |
|---:|---|---|---|
| `1` | Redis hot cache | Redis socket timeout from config | Fast range recovery for short live gaps. |
| `2` | ClickHouse canonical | Short timeout + circuit breaker | Historical truth and normal repair source when healthy. |
| `3` | Exchange REST tail through Market Data adapters | Tail window limit, rate limit, backoff | Last-resort short-tail repair for closed minutes missing from Redis/CH. |
| `4` | Repair audit/outbox miss | Durable record | Block/degrade only after all sources fail or cannot prove continuity. |

REST tail v1 limit: configurable `15-60m`, default `15m`. Anything older is not live-tail repair and must fall back to canonical/backfill workflows.

Stage `03` implementation uses `ClosedCandleTailRepairPolicy.rest_tail_limit_minutes`
for the REST short-tail boundary and `MarketDataClosedCandleTailProvider(...,
clickhouse_circuit_open_seconds=...)` for the in-process ClickHouse failure circuit.
The provider catches ClickHouse reader failures, opens the circuit for the configured
window, and continues to REST tail before returning a miss/failure.

### Checkpoint И ACK Семантика

`strategy_runs.checkpoint_ts_open` remains source of truth for processing progress.

| Ситуация | Правило |
|---|---|
| `ts_open <= checkpoint` | Ignore as stale/idempotent; ACK allowed. |
| `ts_open == checkpoint + 1m` | Process candle; ACK only after checkpoint/result persistence succeeds. |
| `ts_open > checkpoint + 1m` and repair succeeds | Process repaired candles in order, then current candle; ACK after checkpoint reaches current candle. |
| `ts_open > checkpoint + 1m` and repair fails temporarily | Do not mark candle processed. Runner must either leave message pending and reclaim/retry, or record a durable repair backlog proving the candle can be reconstructed from hot cache. The chosen implementation must be tested with a failed repair followed by successful retry. |
| All sources exhausted | Run becomes controlled `blocked` or `failed` according to configured policy; audit/outbox records exact missing range and sources tried. |

Stage `04` implementation selects pending reclaim: `StrategyLiveRunner` does not ACK the triggering Redis stream message until every relevant active run either treats it as stale/idempotent or advances `strategy_runs.checkpoint_ts_open` through that candle. `RedisStrategyLiveCandleStream` reclaims/replays pending entries before reading new `>` entries, using bounded `XAUTOCLAIM` plus current-consumer pending replay.

Stage implementation must choose one concrete policy before code changes:

| Option | Допустимость |
|---|---|
| Pending reclaim with no ACK until checkpoint accepts current candle | Selected in Stage `04`; implemented in runner ACK gating and Redis pending reclaim. |
| Durable repair backlog plus ACK after hot-cache materialization | Allowed only if runtime proof shows no candle loss and retry can reconstruct all pending minutes without Redis stream message. |
| Current behavior: ACK after `_process_candle` returns without checkpoint advance | Not allowed unless paired with durable backlog/hot-cache proof. |

## Service Calls

| Caller | Callee | Style | Contract | Timeout / retry | Failure behavior |
|---|---|---|---|---|---|
| `market-data-ws-worker` | Redis hot cache | sync Redis write | Store one closed 1m candle by `instrument_key`/`ts_open`. | Redis socket timeout; best-effort for WS ingestion, but metric on failure. | WS ingestion continues; repair hit-rate may degrade; alert if cache write errors grow. |
| `StrategyLiveRunner` | `ClosedCandleTailProvider` port | in-process application port | Request continuous `[start,end)` closed 1m tail. | Bounded provider timeout/backoff from config. | No checkpoint advance without continuity. |
| Provider adapter | Redis hot cache | sync Redis range read | Return sorted normalized candles by `ts_open`. | Redis socket timeout, no unbounded retry. | Continue to ClickHouse on miss/error; metric source=`redis_hot_cache`. |
| Provider adapter | ClickHouse canonical | HTTP/native read | Read canonical closed 1m range. | Short timeout + circuit breaker. | Continue to REST tail if circuit open/error; audit source failure. |
| Provider adapter | Market Data REST source | external HTTP via existing adapters | Fetch short closed 1m tail only. | Rate limit/backoff; no current open minute. | Write restored rows to hot cache/audit; miss blocks after policy. |
| Provider adapter | Postgres repair audit/outbox | DB write | Append repair attempt/result. | DB transaction timeout, no raw provider payload. | If audit write fails, repair must fail closed unless explicitly documented as metrics-only non-critical path. |

## Persistence And Audit

New durable audit should live outside ClickHouse so it remains writable when ClickHouse is unstable.

Planned Postgres table:

```text
market_data_candle_repair_events
```

Minimal fields:

| Field | Purpose |
|---|---|
| `event_id` | UUID primary key. |
| `correlation_id` | Runner/provider attempt correlation. |
| `instrument_id`, `instrument_key`, `market_id`, `symbol` | Bounded market identity. |
| `range_start_ts_open`, `range_end_ts_open` | Half-open repaired/missing range. |
| `status` | `attempted|succeeded|miss|failed|circuit_open|rate_limited`. |
| `sources_attempted_json` | Ordered source names and summary statuses; no raw payloads. |
| `restored_ts_opens_json` | Restored minute list. |
| `missing_ts_opens_json` | Remaining missing minute list. |
| `error_code`, `error_summary` | Stable redacted error category. |
| `created_at` | Event time. |

The table is audit/outbox-equivalent for v1. A separate delivery outbox is not required until external notification delivery exists.

## Metrics And Alerts

| Metric / evidence | Labels | Meaning |
|---|---|---|
| `market_data_live_tail_gap_total` | `instrument_key`, `source_stage` bounded/cardinality-reviewed | Gap detected by Strategy repair path. |
| `market_data_live_tail_repair_total` | `source`, `status` | Repair attempts and outcomes. |
| `market_data_live_tail_repair_latency_seconds` | `source`, `status` | Provider latency. |
| `market_data_hot_cache_hit_total` / `market_data_hot_cache_miss_total` | `instrument_key` bounded to configured instruments | Redis cache effectiveness. |
| `market_data_clickhouse_repair_circuit_state` | none or bounded host label | Circuit breaker open/closed. |
| `strategy_live_runner_checkpoint_stall_total` | `reason` | Checkpoint could not advance due gap/repair. |
| `strategy_live_runner_deferred_ack_total` | `reason` | Messages not ACKed or deferred by backlog policy. |

Alerts:

| Alert | Severity | Owner | Runbook action |
|---|---|---|---|
| Gap not repaired within policy | warning/critical | operator | Check Redis hot cache, REST tail, ClickHouse circuit, selected run checkpoint. |
| ClickHouse repair circuit open too long | warning | operator | Verify ClickHouse health; live path should continue via REST for short tail. |
| REST tail repair errors/rate-limit | warning | operator | Verify adapter endpoint/rate limits; reduce repair rate if needed. |
| Hot cache miss on short tail | warning | operator | Verify WS publisher cache writes and retention. |
| Active run has no new `StrategySignal` | critical after threshold | operator | Check checkpoint, gap repair audit, producer health. |

## Redaction

Allowed in logs/reports:

| Allowed | Forbidden |
|---|---|
| `instrument_key`, `market_id`, bounded source name, `ts_open`, source status, stable error code, duration, counts | API keys, secrets, tokens, cookies, DSNs, raw provider payloads, signed URLs, Authorization headers, raw exchange responses, Redis password values |

## План Внедрения

| Stage | Название | Смысл | Acceptance |
|---|---|---|---|
| `01` | Contract foundation and audit schema | Ввести порт `ClosedCandleTailProvider`, DTO результата, config model, Postgres audit migration/repository, docs sync. | Local DB migration/test proves audit events insert/query; fake provider contract proves continuous/missing results; no runtime repair behavior changed. |
| `02` | Redis hot cache | Добавить Redis hot cache writer/reader в Market Data and WS publish path. | Real Redis call writes duplicate candles, range reads sorted `[start,end)`, retention works, cache hit/miss metrics visible. |
| `03` | Tail provider source chain | Реализовать provider chain: Redis hot cache -> ClickHouse short timeout/circuit -> REST tail -> audit. | Integration test/call with ClickHouse failure and fake/rest tail returns continuous range, writes hot cache and audit, never exposes raw provider payload. |
| `04` | Strategy runner integration and ACK policy | Заменить `_repair_gap` на `ClosedCandleTailProvider`, зафиксировать pending/backlog ACK policy. | Direct runner call proves gap repaired and checkpoint advances; failed repair does not lose current/future candle; retry later processes range with no duplicate signals. |
| `05` | Metrics, alerts, runbook | Добавить Prometheus metrics, alert rules, runbook, docs updates. | Runtime metrics endpoint exposes repair/cache/circuit/checkpoint-stall signals after synthetic calls; alert rules parse; runbook has operator steps. |
| `06` | Mac Studio repair proof | Доставить изменения в `main`, дождаться green GitHub Actions/CI, выполнить deploy/sync Mac Studio и только затем доказать controlled missing-minute + ClickHouse unavailable + REST tail recovery. | `post_main_production_runtime_proof`: target revision is on `main`, CI/GitHub Actions are green, deploy/sync into `/opt/roehub/app` is complete, then runner restores missing candle, checkpoint advances, `StrategySignal` and `ExecutionSourceEvent` continue, audit rows/metrics recorded, no mainnet/secret leak. |
| `07` | Stage 12.4 rerun handoff | После accepted repair proof заново выполнить `12.4` или явно открыть его rerun по текущему prompt. | `12.4` reaches accepted 6h evidence or remains blocked with new unrelated blocker; `12.5` opens only after `12.4 accepted`. |

## Планируемые Файлы

| Stage | Основные зоны |
|---|---|
| `01` | `src/trading/contexts/strategy/application/ports/closed_candle_tail_provider.py`, market-data repair DTO/ports, Alembic migration, Postgres repository, docs, tests. |
| `02` | `src/trading/contexts/market_data/adapters/outbound/messaging/redis/*hot_cache*`, `redis_streams_live_candle_publisher.py`, `apps/worker/market_data_ws/wiring/modules/market_data_ws.py`, `configs/prod/market_data.yaml`, tests. |
| `03` | Market Data provider chain/adapters, REST tail adapter reuse, ClickHouse circuit wrapper, repair audit repository integration, tests. |
| `04` | `src/trading/contexts/strategy/application/services/live_runner.py`, `StrategyLiveCandleStream` port/Redis adapter if pending reclaim is chosen, wiring in `apps/worker/strategy_live_runner`, config, tests. |
| `05` | Prometheus metrics wiring in worker modules, alert/runbook docs under `docs/architecture/market_data/` or ops docs, tests/check scripts. |
| `06` | Stage report and ledger updates; code changes only if runtime proof finds a narrow blocker. |
| `07` | Strategy producer Stage `12.4` report/ledger updates; no new market-data code unless a new blocker is found and documented. |

## Затрагиваемая Документация

| Документ | Что обновить |
|---|---|
| `docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md` | Добавить hot cache как отдельный live-tail range store; Redis stream остается fan-out transport. |
| `docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md` | Заменить ClickHouse-only repair на `ClosedCandleTailProvider` chain и обновить ACK/checkpoint contract. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | Зафиксировать, что `12.4` blocked до accepted repair-cycle и Stage `06` `post_main_production_runtime_proof`. |
| `docs/architecture/README.md` | Проверить/обновить через docs index generator. |

## Журнал Итераций

Единый ledger:

```text
docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md
```

Правила:

| Правило | Требование |
|---|---|
| Предыдущий stage | Каждый prompt проверяет, что предыдущий required stage `accepted`, кроме repair/supersede prompt. |
| Tests are gates | Tests/lint/type checks не являются acceptance для runtime stages. |
| Real calls | Каждый stage требует конкретный вызов к затронутой boundary: Redis, DB, provider chain, runner, metrics или Mac Studio boundary. Pre-main checks may be only `target_host_readiness_pre_main` / `read_only_existing_runtime_smoke`; changed-code proof on Mac Studio is only `post_main_production_runtime_proof`. |
| Direct main | После successful validation изменения доставляются в `origin/main` через `publish-ci-deploy`; ветки/worktrees/stashes не создавать без прямой просьбы пользователя. |
| Secrets | Ни один prompt/report/ledger не пишет raw secrets, credentials, cookies, tokens, raw provider payloads. |

## Риски

| Риск | Митигация |
|---|---|
| Redis hot cache увеличит память | Retention `6-24h`, metrics, isolated Redis proof in Stage `02`, and production runtime observation only inside Stage `06` `post_main_production_runtime_proof`. |
| REST tail может упереться в rate limits | Short tail only, rate limiter/backoff, metrics, audit miss rather than blind retry. |
| ACK policy может стать сложной для multi-run stream fan-out | Stage `04` обязан доказать failed repair -> later retry без потери свечи и без duplicate signals. |
| ClickHouse circuit может скрыть настоящую деградацию historical storage | Alert on circuit-open duration; ClickHouse remains reconciliation source. |
| Provider chain может вернуть свечу, еще не попавшую в ClickHouse | Repair audit/outbox records source/provenance; later reconciliation can compare with canonical. |

## Критерии Закрытия Проблемы

Проблема считается закрытой только когда:

1. Redis hot cache range read работает на реальном Redis.
2. Provider chain восстанавливает короткий gap при недоступном ClickHouse через REST tail.
3. `StrategyLiveRunner` не теряет future candle при failed repair and later retry.
4. Repair audit/outbox и Prometheus metrics фиксируют source, latency, status, missing/restored minutes.
5. Stage `06` `post_main_production_runtime_proof` после `main`, green CI/GitHub Actions и deploy/sync показывает controlled missing minute + ClickHouse unavailable + REST tail recovery.
6. `Stage 12.4` rerun снова проходит 6h или блокируется уже другой, явно зафиксированной причиной.
