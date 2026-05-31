# Runbook — Exchange Execution

## Purpose

`exchange-execution` is the supervised runtime boundary for exchange order
adapters. In disabled mode it only observes Redis dispatch messages and the
durable intent ledger. In Stage 14 testnet mode it may submit, status-check and
cancel native Binance/Bybit testnet orders after durable guards pass. Mainnet
submit remains forbidden.

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
| Adapter mode | `disabled` or `testnet` |

## Fail-Closed Defaults

- `adapter_mode=disabled` keeps Stage 13 no-submit behavior.
- `adapter_mode=testnet` enables Stage 14 native testnet adapters only.
- Any connection whose environment is not `testnet` is hard-blocked before
  submit as `mainnet_hard_block`.
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

Expected disabled-mode readiness may be `degraded` with
`adapter_disabled_stage13` while dependencies are healthy.

Expected Stage 14 testnet readiness reports adapter dependency
`testnet_adapters_ready` when Postgres, Redis, OpenBao Transit and configured
adapters are available. Missing credential resolution degrades or fails
readiness before any submit. `not_ready` is the fail-closed state and should
alert.

Readiness covers:

- config loaded and validated;
- Postgres heartbeat write;
- Redis stream and consumer group visibility;
- request stream backpressure threshold;
- DLQ stream visibility;
- Redis server time vs local process clock drift;
- rate-limit guard configuration;
- adapter disabled or testnet adapter dependency state.

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

In disabled mode, if the message maps to a durable dispatched/accepted intent,
the process records `adapter_disabled` and leaves the Redis message pending. In
testnet mode, a valid message is acknowledged only after a durable
`execution_orders` guard, submit, status/cancel or adapter-error decision. If
the message is malformed or not dispatchable, the process records
`quarantined`, publishes a DLQ marker, and acks the original message only after
the durable observation.

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

SELECT intent_id, exchange_name, environment, status, status_reason, exchange_order_id
FROM execution_orders
ORDER BY updated_at DESC
LIMIT 20;

SELECT exchange_connection_id, exchange_name, environment, status, status_reason
FROM exchange_private_stream_sessions
ORDER BY updated_at DESC
LIMIT 20;
```

## Operations

```bash
launchctl print gui/$(id -u)/com.roehub.exchange-execution
/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary | rg exchange_execution
curl -fsS 'http://127.0.0.1:9090/api/v1/query?query=up{job="exchange-execution"}'
```

Do not enable mainnet order adapters from this runbook. Testnet evidence must
record only redacted provider identifiers and must never include API keys,
secrets, passphrases, signatures, cookies, Authorization headers or raw signed
payloads.
