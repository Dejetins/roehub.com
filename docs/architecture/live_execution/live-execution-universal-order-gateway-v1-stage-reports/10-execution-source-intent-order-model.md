# Stage 10: Execution Source Intent Order Model

Stage 10 adds the producer-neutral execution ingress contract:
`ExecutionSourceEvent`, `ExecutionRequest`, explicit v1 order model validation,
source-type registry and durable `ExecutionIntent` persistence. The
implementation is additive and records intents as `recorded` only; it does not
run the Stage 11 risk gate, publish Redis dispatch messages, decrypt exchange
credentials, call exchange SDK/API clients, or submit orders.

Date: 2026-05-31.

Status: accepted. Local gates, direct-main delivery, CI/deploy, Mac Studio
runtime API/DB/Redis/metrics evidence, stage report and ledger update are
complete.

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
| Full local test suite | `uv run pytest -q -ra` | `1045 passed, 3 warnings`. |
| Docs index | `uv run python -m tools.docs.generate_docs_index --check` | Passed. |
| Whitespace | `git diff --check` | Passed. |

## Delivery Evidence

Direct-main delivery:

- implementation commit: `650e4dfb Add execution source intent model`;
- pushed to `origin/main`;
- CI `26718954140`: success;
- Publish App Image `26718993829`: success;
- Deploy Backend `26718993846`: success;
- Deploy Web `26718993847` and follow-up `26718997002`: success.

## Runtime Evidence

Production proof used disposable authenticated smoke sessions against the
deployed Mac Studio API at `http://127.0.0.1:8000`, direct Postgres reads,
Redis `XINFO`/key scan and API `/metrics`. The public edge path is unchanged
because Stage 10 has no browser-visible surface.

Accepted runtime proof:

| Surface | Evidence | Result |
|---|---|---|
| Migration | Mac Studio Postgres `to_regclass` found `execution_source_events` and `execution_intents`. | Pass. |
| Source ingress | Real `POST /ui/execution/source-events` returned HTTP `201` for `strategy_signal`, `manual_request`, `ml_agent_decision` and `ops_test`. The `strategy_signal` call persisted `strategy_signal_id`; all source refs were bounded non-secret refs. | Pass. |
| Intent ingress | Real `POST /ui/execution/intents` returned HTTP `201` for all four source types. Each intent persisted `order_type=market`, `status=recorded`, `status_reason=stage10_recorded_no_dispatch`, `risk_status=not_evaluated`, `risk_reason=stage11_not_implemented`. | Pass. |
| Invalid source | `POST /ui/execution/source-events` for `strategy_signal` without `strategy_signal_id` returned HTTP `400`, code `execution.invalid_source_event`, reason `strategy_signal_id_required`. | Pass. |
| Unsupported order model | `POST /ui/execution/intents` with `take_profit` returned HTTP `422`, code `execution.unsupported_order_model`, reason `tp_sl_not_supported`; the source event outcome became `order_model_rejected` and no intent was created for that request. | Pass. |
| Idempotency | Duplicate source call with the exact same idempotency key returned HTTP `200`, `duplicate=true`, same `source_event_id`; duplicate intent call returned HTTP `200`, `duplicate=true`, same `intent_id`. DB counts for the duplicate smoke user were `source_count=1`, `source_idempotency_hashes=1`, `intent_count=1`, `intent_idempotency_hashes=1`. | Pass. |
| DB source rows | SQL grouped rows for the main smoke user showed source outcomes: `intent_created/stage10_recorded_no_dispatch` for all four source types, plus `manual_request/order_model_rejected/tp_sl_not_supported`; source summary `source_count=6`, linked sources `4`, `strategy_signal_source_count=1`. | Pass. |
| DB intent rows | SQL grouped rows showed one `market/recorded/not_evaluated` intent for `strategy_signal`, `manual_request`, `ml_agent_decision`, and two for `ops_test` because the first replay probe intentionally used a different key before the corrected duplicate-only probe. Corrected duplicate proof showed no duplicate row. | Pass. |
| Redis dispatch absence | Redis scan for `*execution*` returned `0` keys; `XINFO STREAM execution.requests.v1` returned `no such key`. | Pass. |
| Metrics | `/metrics` exposed `execution_source_event_total` for all four source types, `execution_intent_total{reason="stage10_recorded_no_dispatch",result="recorded"}` for all four source types, and `execution_order_model_rejected_total{source_type="manual_request",reason="tp_sl_not_supported"} 1.0`. | Pass. |
| Cleanup | Temporary smoke sessions were revoked; duplicate-proof active sessions after revoke `0`, main probe active sessions after revoke `0`. | Pass. |

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | compatible-change | Adds authenticated `/api/ui/execution/source-events` and `/api/ui/execution/intents`; new execution errors are mapped to `400`, `404` and `422`. |
| Ports/DTO | compatible-change | Adds `ExecutionIntentRepository`, API DTOs and `ExecutionIngressService`; no existing port is changed. |
| Persistence | compatible-change | Adds `execution_source_events` and `execution_intents`; no destructive migration. |
| Redis | none | Stage 10 intentionally does not publish dispatch messages. |
| Config | none | Reuses existing `STRATEGY_PG_DSN`/fail-fast wiring pattern; no new env schema. |
| Runtime/Ops | compatible-change | Adds bounded metrics only; backend API deploy/reload applies the additive migration; no new supervised process. |
| UI/browser | none | No visible page or browser workflow changed. |
| External side effects | none | No exchange SDK/API call, credential decrypt or order submit. |
| Logs/redaction | compatible-change | Source refs reject sensitive key names and bounded values; API responses include ids/statuses only. |

## Rollback

Disable the new API router wiring or revert the additive commit. Existing
`execution_source_events` and `execution_intents` rows are inert audit data
because Stage 10 never dispatches Redis messages or submits orders.

## Handoff To Stage 11

Stage `11` can start from accepted `execution_source_events` and
`execution_intents` contracts. It must consume `recorded` intents, evaluate
source-aware risk policy, write durable accepted/rejected risk state, and keep
rejected intents out of Redis dispatch. Stage `10` records
`risk_status=not_evaluated` and `risk_reason=stage11_not_implemented` by design.
