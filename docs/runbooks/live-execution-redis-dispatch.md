# Runbook — Live Execution Redis Dispatch

## Purpose

Stage 12 publishes only durable accepted execution intents to Redis Streams.
Postgres remains the source of truth. Redis is a transport for the future
`exchange-execution` process.

## Streams And Group

| Purpose | Name |
|---|---|
| Request dispatch stream | `execution.requests.v1` |
| Retry marker stream | `execution.requests.retry.v1` |
| Quarantine/DLQ stream | `execution.requests.dlq.v1` |
| Consumer group | `exchange-execution.v1` |

Runtime environment overrides:

| Variable | Default |
|---|---|
| `ROEHUB_EXECUTION_DISPATCH_REDIS_ENABLED` | `true` in `prod`, otherwise `false` |
| `ROEHUB_EXECUTION_DISPATCH_REDIS_HOST` | `127.0.0.1` in `prod`, otherwise `redis` |
| `ROEHUB_EXECUTION_DISPATCH_REDIS_PORT` | `6379` |
| `ROEHUB_EXECUTION_DISPATCH_REDIS_DB` | `0` |
| `ROEHUB_EXECUTION_DISPATCH_REDIS_PASSWORD_ENV` | `ROEHUB_REDIS_PASSWORD` |
| `ROEHUB_EXECUTION_DISPATCH_REQUEST_STREAM` | `execution.requests.v1` |
| `ROEHUB_EXECUTION_DISPATCH_RETRY_STREAM` | `execution.requests.retry.v1` |
| `ROEHUB_EXECUTION_DISPATCH_DLQ_STREAM` | `execution.requests.dlq.v1` |
| `ROEHUB_EXECUTION_DISPATCH_CONSUMER_GROUP` | `exchange-execution.v1` |
| `ROEHUB_EXECUTION_DISPATCH_RETRY_BUDGET` | `3` |
| `ROEHUB_EXECUTION_DISPATCH_BACKPRESSURE_LENGTH` | `10000` |

## Dispatch Semantics

Dispatch is allowed only when:

- `execution_intents.status` is `accepted` or `retry`;
- `execution_intents.risk_status` is `accepted`;
- `dispatch_attempt_count < ROEHUB_EXECUTION_DISPATCH_RETRY_BUDGET`.

The API records the intent and risk result first. After that durable commit, it
claims the row as `dispatching`, publishes one Redis message, and marks the row
`dispatched` with `dispatch_stream_name` and `dispatch_redis_message_id`.

Rejected, `recorded`, `not_evaluated`, `quarantined`, and already dispatched
rows are not republished. Duplicate API replay can retry a `retry` row, but a
`dispatched` row is treated as already complete.

## Redis Checks

Basic stream info:

```bash
redis-cli -h 127.0.0.1 -p 6379 XINFO STREAM execution.requests.v1
redis-cli -h 127.0.0.1 -p 6379 XINFO GROUPS execution.requests.v1
```

Read one message through the consumer group:

```bash
redis-cli -h 127.0.0.1 -p 6379 XREADGROUP GROUP exchange-execution.v1 stage12-debug COUNT 1 STREAMS execution.requests.v1 '>'
```

Inspect pending entries:

```bash
redis-cli -h 127.0.0.1 -p 6379 XPENDING execution.requests.v1 exchange-execution.v1
```

Ack only after the consumer has made the durable state change in Postgres:

```bash
redis-cli -h 127.0.0.1 -p 6379 XACK execution.requests.v1 exchange-execution.v1 1710000000000-0
```

Retry and quarantine streams:

```bash
redis-cli -h 127.0.0.1 -p 6379 XINFO STREAM execution.requests.retry.v1
redis-cli -h 127.0.0.1 -p 6379 XINFO STREAM execution.requests.dlq.v1
```

## Metrics

API `/metrics` exposes bounded Stage 12 counters:

```bash
curl -fsS http://127.0.0.1:8000/metrics | rg 'execution_dispatch_(total|retry_total|dlq_total|backpressure_total|redis_errors_total)'
```

Expected labels are bounded `result` and `reason` values only. Do not add user,
strategy, connection, idempotency key, stream id, or exchange labels.

## Failure Handling

Redis outage:

- status becomes `retry`;
- `dispatch_last_error` records a bounded reason such as `ConnectionError`;
- no exchange submit occurs;
- duplicate API replay may retry after Redis recovers.

Backpressure:

- if the request stream length is at or above the configured threshold, status
  becomes `retry`;
- a retry marker is written to `execution.requests.retry.v1` when Redis is
  reachable;
- no primary dispatch message is written.

Poison message:

- status becomes `quarantined`;
- a marker is written to `execution.requests.dlq.v1` when Redis is reachable;
- repair must inspect the durable intent row and quarantine reason before any
  manual replay.

Ack-after-durable:

- Stage 12 exposes the transport method and Redis group contract;
- future consumers must update durable order/execution state first and call
  `XACK` only after that state change succeeds.
