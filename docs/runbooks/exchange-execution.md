# Runbook — Exchange Execution

## Purpose

`exchange-execution` is the supervised runtime boundary that will own future
exchange order adapters. Stage 13 keeps adapters disabled. The process can read
Redis dispatch messages and the durable intent ledger, publish poison messages
to DLQ, expose health/metrics, and record heartbeat/observation rows, but it
must not submit, cancel, amend or status-check real orders.

## Runtime Contract

| Surface | Value |
|---|---|
| Process module | `python -m apps.exchange_execution.main.main` |
| Config | `configs/prod/exchange_execution.yaml` |
| launchd label | `com.roehub.exchange-execution` |
| Monit service | `roehub_exchange_execution` |
| Health | `http://127.0.0.1:9206/health/ready` |
| Metrics | `http://127.0.0.1:9206/metrics` |
| Request stream | `execution.requests.v1` |
| Retry stream | `execution.requests.retry.v1` |
| DLQ stream | `execution.requests.dlq.v1` |
| Consumer group | `exchange-execution.v1` |
| Adapter mode | `disabled` |

## Fail-Closed Defaults

- `adapter_mode` must be `disabled` in Stage 13. Any other value fails config
  validation.
- Valid dispatched intents are observed and recorded, but not acknowledged while
  adapters are disabled.
- Poison or non-dispatchable messages are recorded to
  `exchange_execution_request_observations`, published to
  `execution.requests.dlq.v1`, then acknowledged after that durable observation.
- Redis remains transport only. Postgres `execution_intents` and later
  order/reconciliation ledgers are the durable source of truth.
- Unknown side effects must be reconciled from durable state or provider state
  before retry. Blind retry is forbidden.

## Health Checks

```bash
curl -fsS http://127.0.0.1:9206/health/ready
curl -fsS http://127.0.0.1:9206/metrics | rg 'exchange_execution_(ready|dependency_ready|adapter_disabled|clock_drift_ms)'
```

Expected Stage 13 readiness may be `degraded` with
`adapter_disabled_stage13` while dependencies are healthy. `not_ready` is the
fail-closed state and should alert.

Readiness covers:

- config loaded and validated;
- Postgres heartbeat write;
- Redis stream and consumer group visibility;
- request stream backpressure threshold;
- DLQ stream visibility;
- Redis server time vs local process clock drift;
- rate-limit guard configuration;
- adapter disabled state.

## Redis Diagnostics

```bash
redis-cli -h 127.0.0.1 -p 6379 XINFO STREAM execution.requests.v1
redis-cli -h 127.0.0.1 -p 6379 XINFO GROUPS execution.requests.v1
redis-cli -h 127.0.0.1 -p 6379 XPENDING execution.requests.v1 exchange-execution.v1
redis-cli -h 127.0.0.1 -p 6379 XINFO STREAM execution.requests.dlq.v1
```

Run one controlled observation through the local internal endpoint:

```bash
curl -fsS -X POST http://127.0.0.1:9206/internal/v1/run-once
```

If the message maps to a durable dispatched/accepted intent, the process records
`adapter_disabled` and leaves the Redis message pending. If the message is
malformed or not dispatchable, the process records `quarantined`, publishes a
DLQ marker, and acks the original message only after the durable observation.

## Postgres Diagnostics

```sql
SELECT service_id, status, status_reason, adapter_mode, heartbeat_at
FROM exchange_execution_process_heartbeats
ORDER BY heartbeat_at DESC
LIMIT 5;

SELECT intent_id, stream_name, redis_message_id, status, status_reason, observed_at
FROM exchange_execution_request_observations
ORDER BY observed_at DESC
LIMIT 20;
```

## Operations

```bash
launchctl print gui/$(id -u)/com.roehub.exchange-execution
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary | rg exchange_execution
curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up{job="exchange-execution"}'
```

Do not enable order adapters from this runbook. Stage 14 must add explicit
testnet-only adapters, config guards, exchange server-time checks, limiter
integration, and secret-clean evidence before any submit path exists.
