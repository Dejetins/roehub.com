# Stage 12: Redis Dispatch Transport

Stage 12 adds the Redis transport after the durable risk gate. Only intents
with `status=accepted` and `risk_status=accepted` can be claimed for dispatch;
rejected and legacy `recorded/not_evaluated` rows are never published.

Date: 2026-05-31.

Status: pending delivery. Local code gates are complete; direct-main delivery,
CI/deploy and Mac Studio post-deploy runtime evidence must be added before the
stage is accepted.

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

## Runtime Evidence

Pending post-deploy Mac Studio evidence:

- accepted intent creates `dispatched` DB state and one Redis message;
- rejected and missing-risk intents remain non-dispatchable;
- `XINFO`, `XREADGROUP`, `XPENDING`, retry stream and DLQ stream evidence;
- duplicate replay proves no second primary dispatch after `dispatched`;
- Redis outage/recovery proves `retry` then successful dispatch;
- metrics expose dispatch, retry, DLQ, backpressure and Redis error counters;
- no exchange submit, credential decrypt or signed exchange payload exists.

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
