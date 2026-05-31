# Stage 10: Execution Source Intent Order Model

Stage 10 adds the producer-neutral execution ingress contract:
`ExecutionSourceEvent`, `ExecutionRequest`, explicit v1 order model validation,
source-type registry and durable `ExecutionIntent` persistence. The
implementation is additive and records intents as `recorded` only; it does not
run the Stage 11 risk gate, publish Redis dispatch messages, decrypt exchange
credentials, call exchange SDK/API clients, or submit orders.

Date: 2026-05-31.

Status: local validated; runtime acceptance and direct-main delivery pending.

## Scope

Included:

- source registry for `strategy_signal`, `manual_request`, `ml_agent_decision`
  and `ops_test`;
- `ExecutionSourceEvent`, `ExecutionRequest`, `ExecutionOrderModelV1` and
  `ExecutionIntent` domain models;
- v1 order allowlist `market` and `limit`;
- fail-closed rejection for OCO, trailing, TP/SL, amend/replace, multi-leg and
  unsupported order types before Redis/order submit;
- source event idempotency by owner, source type and SHA-256 idempotency hash;
- intent idempotency by owner and SHA-256 idempotency hash;
- durable source-event to intent linking, producer source refs,
  `strategy_signal_id`, order model fields, status and risk-not-evaluated
  markers;
- additive Alembic tables `execution_source_events` and `execution_intents`;
- authenticated API routes `POST /ui/execution/source-events` and
  `POST /ui/execution/intents`;
- bounded API metrics:
  `execution_source_event_total{source_type,result}`,
  `execution_intent_total{source_type,result,reason}` and
  `execution_order_model_rejected_total{source_type,reason}`;
- focused domain, API-route and migration tests.

Out of scope:

- no risk gate decision beyond order-model allowlist;
- no Redis `execution.requests.v1` dispatch;
- no exchange-execution process, adapter call or order submit;
- no mainnet order submission;
- no UI/browser surface change;
- no raw producer payload, credentials, Authorization header, cookies, API keys,
  signed payloads, passphrases or ciphertext in the new source-event refs.

## Local Evidence

| Gate | Command | Result |
|---|---|---|
| Focused Stage 10 tests | `uv run pytest -q tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/apps/api/test_ui_execution_routes.py tests/unit/apps/migrations/test_execution_source_intent_order_model_sql.py` | `11 passed`. |
| Focused ruff | `uv run ruff check src/trading/contexts/live_execution apps/api tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/apps/api/test_ui_execution_routes.py tests/unit/apps/migrations/test_execution_source_intent_order_model_sql.py` | Passed. |
| Focused pyright | `uv run pyright src/trading/contexts/live_execution apps/api tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/apps/api/test_ui_execution_routes.py tests/unit/apps/migrations/test_execution_source_intent_order_model_sql.py` | `0 errors`. |
| Required ruff | `uv run ruff check src/trading/contexts/live_execution apps tests` | Passed. |
| Required pyright | `uv run pyright src/trading/contexts/live_execution apps tests` | `0 errors`. |
| Required unit scope | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps` | `294 passed, 3 warnings`. |

## Runtime Evidence

Pending. Acceptance still requires deployed or otherwise real boundary proof:

- real `POST /api/ui/execution/source-events` and
  `POST /api/ui/execution/intents` calls for all four source types;
- duplicate idempotency calls for source events and intents;
- invalid source policy and unsupported order model calls;
- SQL rows in `execution_source_events` and `execution_intents`;
- Redis evidence proving no execution dispatch stream/messages;
- `/metrics` evidence for source event, intent and order-model rejection
  counters.

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | compatible-change | Adds authenticated `/api/ui/execution/source-events` and `/api/ui/execution/intents`; new execution errors are mapped to `400`, `404` and `422`. |
| Ports/DTO | compatible-change | Adds `ExecutionIntentRepository`, API DTOs and `ExecutionIngressService`; no existing port is changed. |
| Persistence | compatible-change | Adds `execution_source_events` and `execution_intents`; no destructive migration. |
| Redis | none | Stage 10 intentionally does not publish dispatch messages. |
| Config | none | Reuses existing `STRATEGY_PG_DSN`/fail-fast wiring pattern; no new env schema. |
| Runtime/Ops | compatible-change | Adds bounded metrics only; no supervised process or service restart scope beyond API. |
| UI/browser | none | No visible page or browser workflow changed. |
| External side effects | none | No exchange SDK/API call, credential decrypt or order submit. |
| Logs/redaction | compatible-change | Source refs reject sensitive key names and bounded values; API responses include ids/statuses only. |

## Rollback

Disable the new API router wiring or revert the additive commit. Existing
`execution_source_events` and `execution_intents` rows are inert audit data
because Stage 10 never dispatches Redis messages or submits orders.

## Handoff To Stage 11

Stage `11` must consume `recorded` intents, evaluate source-aware risk policy,
write durable accepted/rejected risk state and keep rejected intents out of
Redis dispatch. Stage `10` records `risk_status=not_evaluated` and
`risk_reason=stage11_not_implemented` by design.
