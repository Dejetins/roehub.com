# Market Data — Live Feed to Strategies via Redis Streams (v1)

## Purpose
This document defines EPIC 4 live feed delivery for Market Data v2:
- source events: only WebSocket **closed 1m** candles from `market-data-ws-worker`;
- transport: Redis Streams;
- semantics: **best-effort** publishing (Redis failures must not break raw ingestion into ClickHouse and must not stop worker runtime).

ClickHouse remains the historical source of truth (`canonical_candles_1m`).
Redis Streams is an online fan-out channel for strategies.

## Redis Streams vs Hot Cache

Redis Streams and Redis hot cache are separate live-feed surfaces:

| Surface | Keys | Responsibility |
|---|---|---|
| Redis Streams | `md.candles.1m.<instrument_key>` | Online fan-out transport for strategy consumers with consumer-group pending/ack semantics. |
| Redis hot cache | `md:hot:1m:<instrument_key>:z`, `md:hot:1m:<instrument_key>:h` | Short live-tail range-store for deterministic `[start,end)` repair reads. |

Consumer group state is not used as a range-store. A strategy gap repair path must read the hot cache through the Market Data repair provider chain, not scan or reinterpret stream pending entries.

## Business Impact

Бизнес-смысл: Redis Streams сохраняет онлайн-доставку закрытых свечей стратегиям, а Redis hot cache добавляет быстрый ограниченный источник восстановления короткого live-хвоста. Это снижает риск из Stage `12.4`, где один пропуск минуты в stream вместе с временной недоступностью ClickHouse остановил выпуск live signals.

Этот документ не утверждает, что repair path для стратегии уже завершен. Интеграция `StrategyLiveRunner`, ACK policy, alert rules, runbook и production proof остаются следующими stages плана `market-data-live-tail-repair-v1`.

## Service Calls / Ops Coverage

| Surface | Decision |
|---|---|
| WS worker -> Redis Streams | Best-effort `XADD` в `md.candles.1m.<instrument_key>`; failure метрикуется и не должен останавливать raw ingestion. |
| WS worker -> Redis hot cache | Best-effort hash/zset write в `md:hot:1m:<instrument_key>:h` и `md:hot:1m:<instrument_key>:z`; failure метрикуется и не должен останавливать raw ingestion. |
| Strategy -> Redis hot cache | `N/A` в этом документе; Strategy должна использовать будущую цепочку `ClosedCandleTailProvider`, а не прямой Redis access. |
| Exchange REST / ClickHouse repair | `N/A`; документ описывает только Redis live-feed surfaces. |
| Timeout / auth source | Redis socket/connect timeout и имя optional auth env var берутся из `market_data.live_feed.redis_streams`; значения секретов не документируются. |

## Logging / Redaction Coverage

Разрешено в logs/docs: `instrument_key`, bounded Redis key pattern, `ts_open_epoch_ms`, source name, status/count и metric name.

Запрещено в logs/docs: Redis auth values, DSNs, raw exchange payloads, API keys, tokens, cookies, Authorization headers или signed URLs.

## Alerts / Monitoring / Runbook Coverage

Stage `02` задает worker metrics для нового hot cache. Alert rules и operator runbook actions здесь `N/A`; они остаются Stage `05`, когда уже существуют repair provider chain и strategy integration.

## Why Redis Streams
- Native queue semantics for fan-out consumers.
- Consumer groups with pending/ack model for resilient strategy processing.
- Low operational overhead for current stack (single additional service in compose).

## Publisher Scope
- Publisher lives only in WS worker wiring.
- Scheduler and REST catch-up paths do not publish to Redis.
- Only normalized WS closed 1m candles are published.

## Stream Naming
Mode: `per_instrument`.

Pattern:
- `md.candles.1m.<instrument_key>`

Example:
- `md.candles.1m.binance:spot:BTCUSDT`

`instrument_key` must match canonical key used in ingestion metadata and ClickHouse canonical tables.

## Message ID and Ordering
Publisher uses deterministic stream IDs derived from candle open time:
- `id = "<epoch_ms>-0"`
- where `epoch_ms` is `ts_open` in UTC milliseconds.

Behavior:
- duplicate/out-of-order IDs are treated as no-op;
- worker continues processing;
- `redis_publish_duplicates_total` is incremented.

## Message Schema v1 (all fields are strings)
Required fields:
- `schema_version`: `"1"`
- `market_id`: int as string
- `symbol`: string
- `instrument_key`: string
- `ts_open`: ISO8601 UTC with milliseconds
- `ts_close`: ISO8601 UTC with milliseconds
- `open`: float as string
- `high`: float as string
- `low`: float as string
- `close`: float as string
- `volume_base`: float as string
- `volume_quote`: float as string, or empty string when null
- `source`: `"ws"`
- `ingested_at`: ISO8601 UTC with milliseconds
- `ingest_id`: UUID string (`CandleMeta.ingest_id`; fallback to worker process ingest id)

Example payload:

```text
schema_version=1
market_id=1
symbol=BTCUSDT
instrument_key=binance:spot:BTCUSDT
ts_open=2026-02-10T12:34:00.000Z
ts_close=2026-02-10T12:35:00.000Z
open=100.1
high=101.2
low=99.9
close=100.8
volume_base=12.34
volume_quote=1234.5
source=ws
ingested_at=2026-02-10T12:35:00.120Z
ingest_id=00000000-0000-0000-0000-000000000001
```

## Runtime Config
Config section in `market_data.yaml`:

```yaml
market_data:
  live_feed:
    redis_streams:
      enabled: true
      host: "redis"
      port: 6379
      db: 0
      password_env: "ROEHUB_REDIS_PASSWORD"
      socket_timeout_s: 2.0
      connect_timeout_s: 2.0
      stream_mode: "per_instrument"
      stream_prefix: "md.candles.1m"
      retention_days: 7
      maxlen_approx: 10080 # optional; default retention_days * 1440
    redis_hot_cache:
      enabled: true
      key_prefix: "md:hot:1m"
      retention_hours: 24
```

Backward compatibility rule:
- if `market_data.live_feed.redis_streams` is missing, feed is disabled.
- if `market_data.live_feed.redis_hot_cache` is missing, hot cache is disabled with defaults `key_prefix="md:hot:1m"` and `retention_hours=24`.

The hot cache reuses the same Redis connection settings as `redis_streams` in v1.

## Best-Effort Failure Semantics
On Redis failure (connection timeout/unreachable/other):
- worker logs error;
- increments `redis_publish_errors_total`;
- continues normal WS ingestion (`insert_buffer -> raw ClickHouse`) without restart.

On duplicate/out-of-order XADD ID:
- increments `redis_publish_duplicates_total`;
- does not raise;
- continues runtime.

## Metrics (WS worker)
- `redis_publish_total` (Counter): successful publishes.
- `redis_publish_errors_total` (Counter): failed publishes.
- `redis_publish_duplicates_total` (Counter): duplicate/out-of-order ID drops.
- `redis_publish_duration_seconds` (Histogram): publish call duration observation.
- `redis_hot_cache_writes_total` (Counter): successful hot-cache writes.
- `redis_hot_cache_write_errors_total` (Counter): failed hot-cache writes.
- `redis_hot_cache_write_duration_seconds` (Histogram): hot-cache write call duration observation.
- `redis_hot_cache_read_hits_total` (Counter): hot-cache range reads that returned at least one candle.
- `redis_hot_cache_read_misses_total` (Counter): hot-cache range reads that returned no candles.
- `redis_hot_cache_read_errors_total` (Counter): failed hot-cache range reads.
- `redis_hot_cache_read_duration_seconds` (Histogram): hot-cache range read call duration observation.

Benchmark evidence: `N/A` for this document. These metrics define observability fields and do not claim a baseline, candidate measurement, latency target, or throughput improvement.

Hot-cache write failures are best-effort: they must be logged and counted, but must not stop WS ingestion into the raw/canonical pipeline. Range-read errors are surfaced to the later repair provider chain so it can continue to ClickHouse/REST according to policy.

## Consumer Group Conventions
Recommended names:
- group: `strategy.<name>`
- consumer: per instance identifier (`<hostname>-<pid>`, `<pod-name>`, etc.)

## Minimal Python Consumer Example (`redis-py`)

```python
import os
from redis import Redis
from redis.exceptions import ResponseError

redis_client = Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True,
)

instrument_key = "binance:spot:BTCUSDT"
stream = f"md.candles.1m.{instrument_key}"
group = "strategy.mean_reversion"
consumer = os.getenv("HOSTNAME", "local") + "-1"

try:
    redis_client.xgroup_create(name=stream, groupname=group, id="$", mkstream=True)
except ResponseError as exc:
    if "BUSYGROUP" not in str(exc):
        raise

while True:
    events = redis_client.xreadgroup(
        groupname=group,
        consumername=consumer,
        streams={stream: ">"},
        count=200,
        block=5000,
    )
    for _, entries in events:
        for message_id, fields in entries:
            ts_open = fields["ts_open"]
            close_price = float(fields["close"])
            # strategy logic here
            redis_client.xack(stream, group, message_id)
```

## Versioning Approach
- Compatible additions keep `schema_version="1"`.
- Breaking schema changes require new version (`"2"`) and documented migration.
