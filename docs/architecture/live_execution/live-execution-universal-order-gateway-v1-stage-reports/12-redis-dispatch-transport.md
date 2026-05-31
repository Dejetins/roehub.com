# Stage 12: Redis Dispatch Transport

Stage 12 adds the Redis transport after the durable risk gate. Only intents
with `status=accepted` and `risk_status=accepted` can be claimed for dispatch;
rejected and legacy `recorded/not_evaluated` rows are never published.

Date: 2026-05-31.

Status: accepted. Direct-main delivery, CI/deploy and Mac Studio post-deploy
runtime evidence are complete.

## Scope

Included:

- deterministic Redis Streams contract:
  `execution.requests.v1`, `execution.requests.retry.v1`,
  `execution.requests.dlq.v1`, consumer group `exchange-execution.v1`;
- optional production-enabled API dispatch wiring controlled by bounded
  `ROEHUB_EXECUTION_DISPATCH_*` environment settings;
- durable dispatch state on `execution_intents`:
  `accepted`, `dispatching`, `dispatched`, `retry`, `quarantined`;
- dispatch attempt count, Redis stream name/message id, bounded last error and
  updated timestamp columns;
- dispatch service with retry budget, Redis outage handling, backpressure
  retry marker, poison-message quarantine and duplicate replay safety;
- Redis adapter that creates the request consumer group, publishes request,
  retry and DLQ markers, and exposes ack-after-durable-state-change;
- authenticated API response fields for dispatch state;
- bounded API metrics:
  `execution_dispatch_total{result,reason}`,
  `execution_dispatch_retry_total{reason}`,
  `execution_dispatch_dlq_total{reason}`,
  `execution_dispatch_backpressure_total{reason}` and
  `execution_dispatch_redis_errors_total{reason}`;
- runbook `docs/runbooks/live-execution-redis-dispatch.md`;
- focused dispatch service, Redis adapter, API route and migration tests.

Out of scope:

- no `exchange-execution` supervised process;
- no exchange SDK/API call, credential decrypt, signed payload or order submit;
- no mainnet order submission;
- no browser-visible surface change;
- no consumer-side durable order state yet beyond the transport ack method and
  documented `XACK` rule.

## Local Evidence

| Gate | Command | Result |
|---|---|---|
| Focused Stage 12 tests | `uv run pytest -q tests/unit/contexts/live_execution/test_execution_dispatch_service.py tests/unit/contexts/live_execution/adapters/test_redis_execution_dispatch_transport.py tests/unit/apps/api/test_ui_execution_routes.py tests/unit/apps/migrations/test_execution_redis_dispatch_transport_sql.py` | `15 passed`. |
| Focused ruff | `uv run ruff check src/trading/contexts/live_execution apps/api tests/unit/contexts/live_execution tests/unit/apps/api/test_ui_execution_routes.py tests/unit/apps/migrations/test_execution_redis_dispatch_transport_sql.py` | Passed. |
| Focused pyright | `uv run pyright src/trading/contexts/live_execution apps tests/unit/contexts/live_execution tests/unit/apps/api/test_ui_execution_routes.py` | `0 errors`. |
| Required ruff | `uv run ruff check src/trading/contexts/live_execution apps configs tests` | Passed. |
| Required pyright | `uv run pyright src/trading/contexts/live_execution apps tests` | `0 errors`. |
| Required unit/apps scope | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps` | `323 passed, 3 warnings`. |
| Required integration path | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps tests/integration` | Blocked locally because `tests/integration` does not exist in this checkout; no tests ran for that path. |
| Full local suite | `uv run pytest -q -ra` | `1074 passed, 3 warnings`. |
| Docs index | `uv run python -m tools.docs.generate_docs_index --check` | Passed. |
| Whitespace | `git diff --check` | Passed. |

## Delivery Evidence

Direct-main delivery:

- implementation commit: `565f0ace Add execution Redis dispatch transport`;
- pushed to `origin/main`;
- CI `26720381203`: success;
- Publish App Image `26720422220`: success;
- Deploy Backend `26720422250`: success;
- Deploy Web `26720422190` and follow-up `26720426619`: success.

## Runtime Evidence

Production proof used a disposable authenticated smoke session against the
deployed Mac Studio API at `http://127.0.0.1:8000`, direct Postgres reads,
Redis Streams calls through redis-py, and API `/metrics`. The temporary smoke
session was revoked after the probe.

Run id: `stage12-d18b0618`.

| Surface | Evidence | Result |
|---|---|---|
| Mac Studio smoke | `bash scripts/macos/smoke_prod.sh` succeeded after deploy: launchd services, API unauthorized boundary, Redis `PONG`, Postgres service and Tailscale state were healthy. | Pass. |
| API accepted path | Real authenticated `POST /ui/execution/source-events` returned HTTP `201`; `POST /ui/execution/intents` with all risk checks true returned HTTP `201`, `status=dispatched`, `status_reason=redis_xadd_ok`. | Pass. |
| DB accepted dispatch | Postgres row for the accepted intent had `status=dispatched`, `risk_status=accepted`, `risk_reason=risk_gate_accepted`, `dispatch_attempt_count=1`, `dispatch_stream_name=execution.requests.v1`, and a Redis message id. | Pass. |
| Duplicate replay | Exact duplicate intent replay returned HTTP `200`, `duplicate=true`, same `intent_id`; Redis primary stream contained exactly `1` message for that intent. | Pass. |
| Rejected no-dispatch | Missing-risk intent returned HTTP `201`, `status=rejected`, `risk_reason=risk_state_unavailable`; Postgres dispatch fields stayed empty and Redis primary stream contained `0` messages for that rejected intent. | Pass. |
| Redis primary stream | `XINFO STREAM execution.requests.v1` reported length `1`; `XINFO GROUPS` reported one group; `XREADGROUP GROUP exchange-execution.v1 <probe-consumer>` read one message; `XPENDING` was `1`, then `XACK` after durable DB state returned `1`, and `XPENDING` returned `0`. | Pass. |
| Backpressure retry | A direct application-service probe with real Postgres/Redis and `backpressure_max_stream_length=1` returned `retry/dispatch_backpressure`, DB status `retry`, and one marker in `execution.requests.retry.v1`. | Pass. |
| Redis outage/recovery | A direct dispatch probe pointed at unavailable Redis port `6390` returned `retry/ConnectionError`; replay with real Redis returned `dispatched/redis_xadd_ok`, DB status `dispatched`, and one primary stream message. | Pass. |
| Poison/DLQ | A direct dispatch probe with a poison transport over real Redis returned `dlq/stage12_poison_probe`, DB status `quarantined`, and one marker in `execution.requests.dlq.v1`. | Pass. |
| Retry/DLQ stream info | `XINFO STREAM execution.requests.retry.v1` length `1`; `XINFO STREAM execution.requests.dlq.v1` length `1`. | Pass. |
| Metrics | API `/metrics` exposed `execution_dispatch_total`, `execution_dispatch_retry_total`, `execution_dispatch_dlq_total`, `execution_dispatch_backpressure_total`, and `execution_dispatch_redis_errors_total`; `execution_dispatch_total{result="dispatched",reason="redis_xadd_ok"} 1.0` was present from the API accepted path. | Pass. |
| Redaction | Latest 10 entries checked in request, retry and DLQ streams had no `authorization`, `api_key`, `apikey`, `secret`, `token`, `cookie`, `passphrase`, `signature` or `ciphertext` terms. | Pass. |
| Cleanup | Temporary smoke session revoke updated `1` session. Durable smoke ledger rows and Redis messages were retained as audit evidence. | Pass. |

Boundary command summary:

```text
accepted_api 201 201 dispatched redis_xadd_ok
accepted_db dispatched accepted risk_gate_accepted dispatch_attempt_count=1 stream=execution.requests.v1
duplicate 200 true same_intent true primary_messages_for_intent 1
rejected_api 201 201 rejected risk_state_unavailable rejected_stream_count 0
redis_primary length=1 groups=1 xreadgroup=true pending_before=1 acked=1 pending_after=0
backpressure retry dispatch_backpressure db_status=retry retry_stream_messages=1
outage_recovery retry ConnectionError dispatched redis_xadd_ok db_status=dispatched primary_messages=1
poison_dlq dlq stage12_poison_probe db_status=quarantined dlq_messages=1
retry_stream_length 1 dlq_stream_length 1 metrics_present true
stream_secret_scan request=0 retry=0 dlq=0
```

Local workstation runtime limitation:

- `redis-cli`, `pg_isready`, Docker and local Redis/Postgres services are absent;
- `uv run python` Redis ping to `127.0.0.1:6379` returned connection refused;
- required runtime acceptance must therefore be collected from Mac Studio after
  direct-main deploy.

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | compatible-change | `POST /ui/execution/intents` may now return dispatch state fields and `status=dispatched|retry|quarantined` for accepted intents when dispatch wiring is enabled. Existing rejected responses remain rejected. |
| Ports/DTO | compatible-change | Adds `ExecutionDispatchTransport` and dispatch service; extends `ExecutionIntentRepository` with dispatch transition methods. |
| Persistence | compatible-change | Additive Alembic columns and status enum values on `execution_intents`; existing `recorded`, `accepted` and `rejected` rows remain valid. |
| Redis | compatible-change | Adds execution request, retry and DLQ stream names plus consumer group contract. Redis remains transport only. |
| Config | compatible-change | Adds bounded `ROEHUB_EXECUTION_DISPATCH_*` env settings; dispatch defaults enabled only in `prod`. |
| Runtime/Ops | compatible-change | API attempts Redis dispatch for accepted intents in production after durable DB commit; Redis outage fails closed to `retry`. |
| UI/browser | none | No visible page or browser workflow changed. |
| Metrics/logs/redaction | compatible-change | Adds bounded counters without user, strategy, connection, token, cookie, signed payload or secret-bearing labels. |
| External side effects | none | Redis publish only; no exchange API/SDK call or order submission. |

## Rollback

Disable `ROEHUB_EXECUTION_DISPATCH_REDIS_ENABLED` or revert the Stage 12 code
path. Existing dispatch state columns are additive. Rows in `retry` or
`quarantined` should be inspected before manual replay; Redis messages are not
money truth.

## Handoff To Stage 13

Stage `13` can rely on the stream contract, consumer group, retry and DLQ
stream names, and ack-after-durable rule. It must introduce the supervised
`exchange-execution` process without changing the producer contract: process a
request only after reading a durable `dispatched` intent and acknowledge Redis
only after the next durable money-boundary state change succeeds.
