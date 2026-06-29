# Stage 02: Redis Hot Cache

Статус: `accepted`.

Дата: `2026-06-30`.

## Pre-Start

Ledger gate: `market-data-live-tail-repair-v1-stage-ledger.md` указывал `current_stage=02`; Stage `01` принят и доставлен через direct-main commit `861ec9ebd4d0499d578e84db415a58505afe8172`; CI и downstream deploy workflows для этого SHA прошли успешно. Stage `03` до Stage `02 accepted` был закрыт.

Первый запуск Stage `02` был заблокирован, потому что обязательный real Redis proof нельзя было безопасно выполнить локально: `127.0.0.1:6379` возвращал `Connection refused`, а локальные `redis-cli`, `redis-server`, `docker`, `podman` отсутствовали. Пользователь затем явно разрешил isolated synthetic proof против Redis на `macstudio`:

```text
Разрешаю использовать Redis на macstudio для isolated synthetic proof Stage 02: записать тестовые md:hot:1m:* keys, проверить чтение и удалить их после проверки.
```

Это разрешение сняло blocker только для изолированных test keys и не является `post_main_production_runtime_proof`.

## Что Реализовано

| Область | Итог |
|---|---|
| Redis hot cache adapter | Добавлен `RedisCandleHotCache` для записи закрытых 1m свечей в hash/zset pair и чтения полуинтервала `[start,end)`. |
| Cache key contract | Зафиксированы ключи `md:hot:1m:<instrument_key>:z` и `md:hot:1m:<instrument_key>:h`; sorted set score/member и hash field используют `ts_open_epoch_ms`. |
| Duplicate semantics | Повторная запись той же минуты перезаписывает тот же hash field и zset member, не создавая ambiguous row. |
| Retention | Добавлен config `redis_hot_cache.retention_hours`, production default `24h`, prune удаляет zset member и hash field старше окна. |
| Worker wiring | `market-data-ws-worker` теперь собирает hot-cache publisher и Redis Streams publisher через `FanoutLiveCandlePublisher`; hot cache вызывается перед stream publish. |
| Metrics | Добавлены hooks и worker metrics для write success/error/duration и range read hit/miss/error/duration. |
| Runtime config | В `configs/prod/market_data.yaml` включен `redis_hot_cache` с `key_prefix: "md:hot:1m"` и `retention_hours: 24`. |
| Tests | Добавлены focused unit tests для serialization/duplicate/sorted range/retention/metrics hooks/fan-out и parser tests для `redis_hot_cache`. |

## Redis Key Contract

| Key | Назначение |
|---|---|
| `md:hot:1m:<instrument_key>:z` | Sorted set: score = `ts_open_epoch_ms`, member = `ts_open_epoch_ms`. |
| `md:hot:1m:<instrument_key>:h` | Hash: field = `ts_open_epoch_ms`, value = normalized candle JSON. |

Real proof использовал только test-only instrument key:

```text
stage02-proof:binance:spot:BTCUSDT:20260630
```

Соответствующие временные ключи:

```text
md:hot:1m:stage02-proof:binance:spot:BTCUSDT:20260630:z
md:hot:1m:stage02-proof:binance:spot:BTCUSDT:20260630:h
```

Оба ключа удалены в `finally`; после cleanup `cleanup_remaining_keys=0`.

## Business Impact

Stage `02` добавляет быстрый live-tail range-store для свежих закрытых 1m свечей. Это снижает риск, обнаруженный в strategy-producer Stage `12.4`: если Redis stream пропустил минуту, а ClickHouse временно недоступен, следующий Stage `03` сможет сначала попробовать восстановить короткий хвост из Redis hot cache, не превращая Strategy в прямого клиента exchange REST.

Stage `02` сам по себе еще не меняет `StrategyLiveRunner` и не восстанавливает runtime gap path. Это делает Stage `03`.

## Service Calls / Ops Coverage

| Surface | Stage `02` decision |
|---|---|
| Runtime service calls | `market-data-ws-worker` выполняет best-effort Redis hash/zset write рядом с Redis Streams publish. |
| Auth/secrets/provider payloads | Используется существующий Redis connection config и env var name `ROEHUB_REDIS_PASSWORD`; значения секретов не читаются и не записываются в docs/report. |
| Timeout/retry/fallback behavior | Redis socket/connect timeout переиспользуется из `redis_streams`; write failure логируется, метрикуется и не останавливает WS ingestion. |
| Idempotency/duplicate behavior | Duplicate write по тому же `ts_open_epoch_ms` дает один hash field и один zset member. |
| Unknown-state behavior | Если hash/zset частично повреждены, `read_range` возвращает только строки с payload; continuity проверяет следующий provider layer. |
| Alerts/runbooks | Metrics добавлены; alert rules и runbook остаются Stage `05`. |

## Logging / Redaction Coverage

Hot-cache logs содержат только `instrument_key` и `ts_open_epoch_ms`. Report фиксирует bounded proof facts: target label, test key names, counts, timestamps, metric callback counts и cleanup result. Secrets, DSNs, raw provider payloads, tokens, cookies, Authorization headers и Redis auth values не записывались.

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `src/trading/contexts/market_data/adapters/outbound/messaging/redis/redis_candle_hot_cache.py` | none | none | Redis hot cache writer/reader, hooks and live publisher adapter. | `compatible-change`; new adapter surface and new cache keys. |
| `src/trading/contexts/market_data/adapters/outbound/messaging/redis/fanout_live_candle_publisher.py` | none | none | Fan-out one WS candle to hot cache and stream publisher. | `compatible-change`; preserves `LiveCandlePublisher` call contract. |
| `tests/unit/contexts/market_data/adapters/test_redis_candle_hot_cache.py` | none | none | Unit tests for duplicate write, sorted range read, retention, metrics hooks and fan-out. | `none`; tests only. |
| none | `src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py` | none | Add `RedisHotCacheConfig` and parser defaults. | `compatible-change`; additive config schema. |
| none | `src/trading/contexts/market_data/adapters/outbound/config/__init__.py` | none | Export `RedisHotCacheConfig`. | `compatible-change`; additive import surface. |
| none | `src/trading/contexts/market_data/adapters/outbound/messaging/__init__.py` | none | Export new live-feed adapters. | `compatible-change`; additive import surface. |
| none | `src/trading/contexts/market_data/adapters/outbound/messaging/redis/__init__.py` | none | Export new Redis adapters. | `compatible-change`; additive import surface. |
| none | `apps/worker/market_data_ws/wiring/modules/market_data_ws.py` | none | Wire hot-cache publisher and Prometheus metrics into WS worker. | `compatible-change`; runtime now writes extra Redis keys when enabled. |
| none | `configs/prod/market_data.yaml` | none | Enable production hot cache with `24h` retention. | `compatible-change`; new operational feature flag/config. |
| none | `tests/unit/contexts/market_data/adapters/test_market_data_runtime_config.py` | none | Cover `redis_hot_cache` defaults and explicit config parsing. | `none`; tests only. |
| none | `docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md` | none | Distinguish Redis Streams transport from Redis hot-cache range-store. | `none`; documentation sync. |
| `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/02-redis-hot-cache.md` | none | none | Stage report and validation evidence. | `none`; documentation/evidence only. |
| none | `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/market-data-live-tail-repair-v1-stage-ledger.md` | none | Mark Stage `02 accepted` and open Stage `03`. | `none`; staged workflow state only. |
| none | `docs/architecture/README.md` | none | Docs index refreshed after Stage `02` docs update. | `none`; generated documentation index only. |

Files outside prompt expected paths: none.

## Validation Evidence

| Gate | Result | Evidence |
|---|---:|---|
| Focused hot-cache/config/worker tests | passed | `uv run pytest -q tests/unit/contexts/market_data/adapters/test_redis_candle_hot_cache.py tests/unit/contexts/market_data/adapters/test_market_data_runtime_config.py tests/unit/contexts/market_data/application/services/test_ws_worker_publishes_redis.py` -> `16 passed in 0.24s`. |
| Focused ruff | passed | `uv run ruff check src/trading/contexts/market_data/adapters/outbound/messaging/redis src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py apps/worker/market_data_ws/wiring/modules/market_data_ws.py tests/unit/contexts/market_data/adapters/test_redis_candle_hot_cache.py tests/unit/contexts/market_data/adapters/test_market_data_runtime_config.py` -> `All checks passed!`. |
| Focused pyright | passed | `uv run pyright src/trading/contexts/market_data/adapters/outbound/messaging/redis src/trading/contexts/market_data/adapters/outbound/config/runtime_config.py apps/worker/market_data_ws/wiring/modules/market_data_ws.py tests/unit/contexts/market_data/adapters/test_redis_candle_hot_cache.py tests/unit/contexts/market_data/adapters/test_market_data_runtime_config.py tests/unit/contexts/market_data/application/services/test_ws_worker_publishes_redis.py` -> `0 errors, 0 warnings, 0 informations`. |
| Prompt `ruff` gate | passed | `uv run ruff check src/trading/contexts/market_data apps/worker/market_data_ws tests` -> `All checks passed!`. |
| Prompt `pyright` gate | passed | `uv run pyright src/trading/contexts/market_data apps/worker/market_data_ws tests` -> `0 errors, 0 warnings, 0 informations`. |
| Prompt pytest gate adaptation | passed | `tests/integration` is absent; `uv run pytest -q tests/unit/contexts/market_data` -> `137 passed in 1.74s`. |
| Docs index | passed | `uv run python -m tools.docs.generate_docs_index --check` -> `OK: /Users/daniildegtyarev/Projects/roehub.com/docs/architecture/README.md is up-to-date.` |
| Real Redis duplicate/range proof | passed | `macstudio` Redis through SSH tunnel `127.0.0.1:16379`: `zcard=3`, `hlen=3`, range timestamps `2026-06-30T12:00:00.000Z`, `2026-06-30T12:01:00.000Z`, `2026-06-30T12:02:00.000Z`; duplicate write count `4` produced only `3` rows. |
| Real Redis cleanup | passed | `cleanup_deleted=2`, `cleanup_remaining_keys=0`. |
| Metrics hook proof | passed | Real Redis proof observed `write_success:4`, `write_error:0`, `read_hit:1`, `read_miss:0`, `read_error:0`, `write_duration:4`, `read_duration:1`. |
| Publish-route repo-wide ruff | passed | `uv run ruff check .` -> `All checks passed!`. |
| Publish-route repo-wide pyright | passed | `uv run pyright` -> `0 errors, 0 warnings, 0 informations`. |
| Publish-route repo-wide pytest | passed | `uv run pytest -q -ra` -> `1459 passed, 3 warnings in 65.88s`. |
| Whitespace diff | passed | `git diff --check` -> no output. |

Prompt gate `uv run pytest -q tests/unit/contexts/market_data tests/integration` is adapted because `tests/integration` does not exist in this repository snapshot. The equivalent Stage `02` test surface is `tests/unit/contexts/market_data` plus the real Redis proof above.

## Validation Boundary Note

Stage `02` collected real Redis evidence against `macstudio` via SSH tunnel using isolated synthetic keys and current local code. This is a real Redis adapter proof, not proof that changed code is running in production. `post_main_production_runtime_proof` remains deferred to Stage `06`, after the relevant revision is on `main`, CI/deploy are green, and `/opt/roehub/app` is updated.

## Contract Impact

| Dimension | Classification | Notes |
|---|---|---|
| Public API contract | `none` | No API routes or external HTTP payloads changed. |
| Port contract | `none` | `LiveCandlePublisher` contract is unchanged. |
| DTO schema | `none` | Existing DTOs are unchanged; hot-cache JSON is an internal Redis value format. |
| Persisted schema | `none` | No DB migration or durable table changed. |
| Config schema | `compatible-change` | Adds optional `market_data.live_feed.redis_hot_cache`; missing section defaults to disabled with `24h` retention. |
| Cache key / persistence identity | `compatible-change` | Adds Redis keys `md:hot:1m:<instrument_key>:z` and `md:hot:1m:<instrument_key>:h`; existing stream keys unchanged. |
| Runtime / ops behavior | `compatible-change` | When enabled, WS worker writes additional Redis hot-cache keys best-effort; ingestion continues on Redis write failure. |
| Service-call auth/timeout/retry/error semantics | `compatible-change` | Hot cache reuses Redis connection/auth env var name and socket/connect timeout from `redis_streams`; no retry is added; write errors are logged/metrics-only for WS ingestion. |
| External side-effect idempotency / unknown-state semantics | `compatible-change` | Duplicate identity is `ts_open_epoch_ms`; partial/missing hash payloads are treated as cache miss/partial read for later provider continuity checks. |
| Logs / metrics / redaction semantics | `compatible-change` | Adds bounded hot-cache metrics and logs only `instrument_key`/`ts_open_epoch_ms`; no secrets or raw provider payloads are logged. |
| Alert / runbook semantics | `none` | Stage `02` adds metrics only; alert rules and runbook actions remain Stage `05`. |
| Browser-visible behavior | `none` | No browser routes or UI defaults changed. |
| Performance benchmark / latency claim | `N/A` | Stage `02` makes no speed or latency improvement claim. No baseline/before measurement was collected, so there is no comparable benchmark result; runtime observation is via `redis_hot_cache_write_duration_seconds` and `redis_hot_cache_read_duration_seconds`. |

## Delivery Status

Accepted delivery evidence for this stage is reviewed scoped staging, direct-main commit/push, local publish gates and GitHub Actions/CI after push. The exact commit hash is fixed in the final executor report because the hash cannot be written into the commit that creates it.

## Next Stage Handoff

Stage `03` can start after Stage `02` scoped direct-main delivery and CI are green. The next executor can rely on:

- `RedisCandleHotCache.read_range(...)` returns `ClosedCandleTailRow` rows sorted by `ts_open` for `[start,end)`;
- duplicate hot-cache writes are deterministic by `ts_open_epoch_ms`;
- empty cache range returns `()`, and continuity remains the provider-chain responsibility;
- Redis write failures are best-effort for WS ingestion and are observable through `redis_hot_cache_write_errors_total`;
- Redis Streams remains fan-out transport, while Redis hot cache is the range-store for short live-tail repair.
