---
doc: live-execution-stage-16-producer-integrations-notifications
stage: "16"
status: in_progress
canonical_plan: docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md
ledger: docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md
---

# Stage 16: Producer Integrations And Notifications

Stage 16 adds producer outcome linking and a redacted execution notification
outbox on top of the Stage 10-15 source-event, risk, Redis dispatch,
exchange-execution, order, fill, and reconciliation ledgers.

Status: `in_progress`.

Previous stage: Stage `15` is accepted in the iteration ledger.

## Files Changed

Code:

- `src/trading/contexts/live_execution/domain/notification.py`
- `src/trading/contexts/live_execution/domain/execution_source.py`
- `src/trading/contexts/live_execution/application/ports/execution_intent_repository.py`
- `src/trading/contexts/live_execution/application/use_cases/execution_ingress.py`
- `src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py`
- `src/trading/contexts/live_execution/adapters/outbound/persistence/in_memory/execution_intent_repository.py`
- `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/execution_intent_repository.py`
- `src/trading/contexts/strategy/application/ports/execution_producer.py`
- `src/trading/contexts/strategy/adapters/outbound/acl/live_execution_producer.py`
- `src/trading/contexts/strategy/application/services/live_runner.py`
- `apps/api/dto/ui_execution.py`
- `apps/api/routes/ui_execution.py`
- `apps/api/wiring/modules/live_execution.py`
- `apps/api/dto/ui_strategies_dashboard.py`
- `apps/api/wiring/modules/ui_strategies_dashboard.py`
- `apps/api/monitoring.py`
- `apps/exchange_execution/main/app.py`
- `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`
- `apps/web/templates/pages/strategies.html`
- `apps/web/dist/js/pages/strategies.js`
- `apps/web/locales/en.json`
- `apps/web/locales/ru.json`

Schema:

- `alembic/versions/20260603_0030_execution_notifications_producers_v1.py`

Tests:

- `tests/unit/contexts/live_execution/test_execution_ingress_service.py`
- `tests/unit/apps/api/test_ui_execution_routes.py`
- `tests/unit/apps/migrations/test_execution_notifications_producers_sql.py`

## Implementation Summary

- Added `execution_notification_outbox` as an additive Postgres ledger with
  bounded `event_type`, `severity`, `status`, JSON-object labels, owner/time
  index, source/intent indexes, and idempotent dedupe across owner, event type,
  source event, intent, order, and reason.
- Widened `execution_source_events.outcome` for Stage 16 terminal outcomes:
  `risk_rejected`, `submitted`, `filled`, `cancelled`, `failed`,
  `reconciliation_required`, and `handoff_failed`.
- Kept all producer ingress on the existing `ExecutionIngressService` path.
  Rejected risk decisions now update the source event to `risk_rejected` and
  create `producer_rejected` or `producer_kill_switch` outbox rows.
- Added `/api/ui/execution/notifications` POST/GET for redacted outbox probe
  and operator-visible API access.
- Added a Strategy application port, `StrategyExecutionProducer`, and a
  Strategy outbound ACL adapter that records persisted `StrategySignal` rows as
  `ExecutionSourceEvent` rows through live-execution ingress. The worker wires
  it only when `ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED=1`; default is
  fail-closed/off.
- Exchange-execution now creates outbox rows for guard rejections, fills,
  unknown adapter states, and terminal matched/cancelled outcomes. It also
  updates linked source-event outcomes from order/reconciliation state.
- `/strategies` dashboard now exposes a compact
  `signal -> source event -> intent -> execution outcome -> notification` panel
  through additive DTO fields and browser rendering.

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public/API | compatible-change | Adds `/ui/execution/notifications` and additive dashboard fields. Existing execution source/intent endpoints remain compatible. |
| Persistence | compatible-change | Adds `execution_notification_outbox` and widens source-event outcome enum. No existing ledger tables are dropped. |
| Redis | none | Dispatch rules stay unchanged: only accepted/risk-accepted intents are dispatchable. |
| Config | compatible-change | Adds fail-closed `ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED`, default `0`. |
| Runtime/ops | compatible-change | Adds bounded notification metrics in API and exchange-execution. |
| UI/browser | compatible-change | Adds `/strategies` execution outcome panel. Existing panels remain. |
| Logs/redaction | compatible-change | Notification labels reject sensitive key names and bound label sizes. |

## Quality Gates

Passed locally:

- `uv run pytest -q tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/apps/api/test_ui_execution_routes.py tests/unit/apps/migrations/test_execution_notifications_producers_sql.py`
- `uv run ruff check src/trading/contexts/live_execution src/trading/contexts/strategy apps tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/apps/api/test_ui_execution_routes.py tests/unit/apps/migrations/test_execution_notifications_producers_sql.py`
- `node --check apps/web/dist/js/pages/strategies.js`
- `uv run pyright src/trading/contexts/live_execution src/trading/contexts/strategy apps tests/unit/contexts/live_execution/test_execution_ingress_service.py tests/unit/apps/api/test_ui_execution_routes.py tests/unit/apps/migrations/test_execution_notifications_producers_sql.py`
- `uv run pytest -q tests/unit/apps/api/test_ui_strategy_dashboard_routes.py tests/unit/apps/migrations`
- `uv run pytest -q tests/unit/contexts/strategy/application/test_strategy_live_runner.py tests/unit/contexts/live_execution/test_exchange_execution_process.py`

Passed after documentation update:

- `uv run ruff check src/trading/contexts/strategy src/trading/contexts/live_execution apps tests`
- `uv run pyright src/trading/contexts/strategy src/trading/contexts/live_execution apps tests`
- `uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/live_execution tests/unit/apps`
- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`

Pending:

- Runtime API/DB/Redis/browser/Mac Studio acceptance evidence.

## Runtime Evidence

Pending. The stage is not accepted until real boundary calls prove:

- Strategy, manual, ML, and ops producers create source events through one
  ingress.
- Eligible accepted intents dispatch through `execution.requests.v1`.
- Rejected/no-intent paths do not dispatch.
- Postgres rows link `source_event_id`, intents, orders/fills/reconciliation,
  and `execution_notification_outbox`.
- `/strategies` browser UI renders the signal/source/intent/outcome/notification
  link.

## Rollback

- Leave `ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED=0` to disable Strategy
  producer source-event writes.
- Disable caller use of `/ui/execution/notifications` without affecting source
  event or intent ingress.
- If needed before acceptance, revert the Stage 16 commit; the migration is
  additive and downgrade drops only `execution_notification_outbox` and restores
  the previous source-event outcome check.

## Next Handoff

Before accepting Stage 16, run the required full local gates, apply the
migration in the target runtime, perform real API/SQL/Redis/browser probes, and
then update this report plus the ledger from `in_progress` to `accepted` or
`blocked`.
