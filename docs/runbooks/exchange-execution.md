# Runbook — Exchange Execution

## Purpose

`exchange-execution` is the supervised runtime boundary for exchange order
adapters. In disabled mode it only observes Redis dispatch messages and the
durable intent ledger. In testnet mode it may submit, status-check and cancel
native Binance/Bybit testnet orders after durable guards pass, then write
order events, fill facts, funding facts and reconciliation runs. Mainnet submit
remains forbidden.

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
| PITR readiness marker | `ROEHUB_EXECUTION_PITR_VERIFIED=true` after a restore drill |

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
- In prod and test config, `ledger.pitr_required=true`; readiness is degraded
  or not ready as `pitr_restore_not_verified` until the configured PITR marker
  is set after a restore drill.

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

SELECT event_type, status, reason, provider_order_id, observed_at
FROM execution_order_events
ORDER BY observed_at DESC
LIMIT 20;

SELECT provider_trade_id, price, quantity, fee_amount, fee_asset, filled_at
FROM execution_fills
ORDER BY filled_at DESC
LIMIT 20;

SELECT status, reason, local_status, provider_status, fill_count, funding_event_count
FROM execution_reconciliation_runs
ORDER BY completed_at DESC
LIMIT 20;

SELECT exchange_connection_id, exchange_name, environment, status, status_reason
FROM exchange_private_stream_sessions
ORDER BY updated_at DESC
LIMIT 20;

SELECT policy_name, table_name, partition_key, retention_days, archive_before_purge, pitr_required
FROM execution_ledger_retention_policies
ORDER BY policy_name;

SELECT target_time, status, reason, verified_at, row_counts_json
FROM execution_ledger_pitr_drills
ORDER BY verified_at DESC
LIMIT 5;
```

## Reconciliation And PITR

Order status checks append `execution_order_events` and create an
`execution_reconciliation_runs` row. Provider fill facts from Binance
`myTrades` / futures `userTrades` and Bybit `execution/list` are normalized
into `execution_fills` and deduped by `(order_id, provider_trade_id)`. Funding
facts are optional and explicit: spot rows are marked
`spot_funding_not_applicable`; futures rows with no funding facts remain
`funding_reconciliation_pending` and must not be treated as complete PnL.

Retention policy metadata is recorded in
`execution_ledger_retention_policies`. Money-ledger tables use long retention,
archive-before-purge semantics and PITR requirement flags. Before canary or
production-readiness claims, run a target-host restore drill, insert a
redacted `execution_ledger_pitr_drills` row with row counts and status, then
set `ROEHUB_EXECUTION_PITR_VERIFIED=true` for the supervised process and reload
the service.

Metrics:

```bash
curl -fsS http://127.0.0.1:9206/metrics \
  | rg 'execution_reconciliation_total|execution_ledger_backup_restore_total'
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

## Stage 17 Production Readiness

Stage 17 keeps production in `adapter_mode=testnet`. Mainnet submit remains
blocked until a separate explicit canary approval changes the policy outside
this runbook.

Safe canary protocol:

1. Confirm Stage 16 and Stage 17 ledger entries are accepted and deployed.
2. Confirm `ROEHUB_EXECUTION_PITR_VERIFIED=true`, `/health/ready` is `ready`,
   Monit shows `roehub_exchange_execution OK`, and Prometheus target
   `up{job="exchange-execution"}` is `1`.
3. Run only a funded testnet connection with a tiny bounded notional and
   `cancel_after_submit=true` unless the explicit canary approval says
   otherwise.
4. Record source-event, intent, risk, Redis dispatch, order, fill/funding,
   reconciliation and notification rows before declaring the run complete.
5. Compute latency from durable timestamps only: source received, intent
   created, risk decision, Redis dispatch, submit, ack/status, fill,
   reconciliation and notification.
6. Compute slippage only when a bounded expected price is available in the
   source reference or limit price. Otherwise report slippage as unavailable,
   not zero.
7. Verify Redis has no unexplained `XPENDING` or DLQ growth after the canary.
8. Keep all reports secret-safe: no cookies, tokens, ciphertext, raw signed
   payloads, passphrases or provider responses with sensitive fields.

Rollback and kill switch:

```bash
# Producer-side kill switch proof: accepted context must set kill_switch_open=false
# and return risk_reason=kill_switch_closed before Redis dispatch.

# Runtime stop:
launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.exchange-execution.plist

# Runtime restore:
launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.exchange-execution.plist
curl -fsS http://127.0.0.1:9206/health/ready
```

Do not delete ledger rows during rollback. Postgres remains the source of truth
for source events, intents, orders, fills, reconciliation and notification
outbox rows. Redis messages may be acknowledged only after the durable state
change they represent has been written.

## Stage 17 Alert Actions

The repo-managed production rules live in
`infra/macos/prometheus/rules/live-execution-stage17.rules.yml` and are
installed by `scripts/macos/bootstrap_native_prod.sh` into
`/opt/roehub/config/prometheus.rules/`.

| Alert | Severity | Owner | Escalation | Runbook action |
|---|---|---|---|---|
| `LiveExecutionDlqGrowing` | critical | live-execution | Stop canary; keep mainnet disabled. | Inspect `execution.requests.v1`, `execution.requests.dlq.v1`, `exchange_execution_request_observations`, `execution_intents`, then reconcile before replay. |
| `LiveExecutionClockDriftUnsafe` | critical | live-execution | Stop canary and block submit. | Fix host time/NTP, reload `exchange-execution`, rerun `/health/ready` and drift metric checks. |
| `LiveExecutionPrivateStreamMissingForSubmit` | critical | live-execution | Stop canary; reconcile provider state. | Check `exchange_private_stream_sessions`, `execution_order_events` and `execution_reconciliation_runs` for stream/session proof. |
| `LiveExecutionDispatchBackpressure` | warning | live-execution | Pause new dispatch until drained. | Compare `XINFO STREAM`, `XPENDING`, API dispatch retry counters and observations before resuming. |
| `LiveExecutionReconciliationPending` | critical | live-execution | Stop canary and reconcile provider facts. | Query orders, order events, fills and reconciliation rows; do not retry blindly. |
| `LiveExecutionPitrNotVerified` | critical | live-execution | Keep mainnet disabled; block canary approval. | Run PITR restore drill, insert redacted `execution_ledger_pitr_drills`, set `ROEHUB_EXECUTION_PITR_VERIFIED=true`, reload. |
| `LiveExecutionUnknownState` | critical | live-execution | Stop canary; preserve evidence. | Use notification outbox, order events, reconciliation rows and provider status as source of truth before retry. |

Monit is the process-level guard for `roehub_exchange_execution`. Its
Stage 17 contract is critical severity, owner `live-execution`, and the same
escalation: stop canary/mainnet enablement, inspect durable Postgres/Redis
state, then recover through this runbook.
