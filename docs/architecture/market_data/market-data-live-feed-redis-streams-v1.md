# Market Data — Live Feed to Strategies via Redis Streams (v1)

## Purpose
This document defines EPIC 4 live feed delivery for Market Data v2:
- source events: only WebSocket **closed 1m** candles from `market-data-ws-worker`;
- transport: Redis Streams;
- semantics: **best-effort** publishing (Redis failures must not break raw ingestion into ClickHouse and must not stop worker runtime).

ClickHouse remains the historical source of truth (`canonical_candles_1m`).
Redis Streams is an online fan-out channel for strategies.

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
```

Backward compatibility rule:
- if `market_data.live_feed.redis_streams` is missing, feed is disabled.

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
- `redis_publish_duration_seconds` (Histogram): publish latency.

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
