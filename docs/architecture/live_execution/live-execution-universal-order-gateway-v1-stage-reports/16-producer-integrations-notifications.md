---
doc: live-execution-stage-16-producer-integrations-notifications
stage: "16"
status: accepted
canonical_plan: docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md
ledger: docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md
---

# Stage 16: Producer Integrations And Notifications

Stage 16 adds producer outcome linking and a redacted execution notification
outbox on top of the Stage 10-15 source-event, risk, Redis dispatch,
exchange-execution, order, fill, and reconciliation ledgers.

Status: `accepted`.

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
  (`421 passed, 3 warnings`)
- `uv run python -m tools.docs.generate_docs_index --check`
- `git diff --check`

CI/deploy:

- Commit `89e40026` pushed to `main`.
- CI `26850850053`: successful.
- Publish App Image `26850942477`: successful; Docker cache reservation
  annotation was non-fatal.
- Deploy Backend `26850942515`: successful.
- Deploy Web `26850942466` and follow-up `26850986731`: successful.

## Runtime Evidence

Accepted on Mac Studio after the `main` deployment.

Mac Studio deployment and health:

- `/opt/roehub/app` is the launchd runtime copy and contains the Stage 16
  migration, notification domain module, and `/strategies` UI hooks.
- `bash scripts/macos/smoke_prod.sh` passed: launchd services were loaded,
  API unauthorized boundary returned `401`, Postgres returned `1`, Redis
  returned `PONG`, and Tailscale backend state was `Running`.
- Production database Alembic version was `20260603_0030`; SQL confirmed
  `execution_notification_outbox`, `execution_source_events`, and
  `execution_intents` tables exist.

Runtime producer/API probe:

- Run marker: `stage16-20260602T222018-d29a99c2`.
- Synthetic strategy:
  `d6a4633a-1c14-4cc9-8a53-4c8b1693485d`.
- API/runtime producer calls covered all four source types:
  `strategy_signal`, `manual_request`, `ml_agent_decision`, and `ops_test`.
- Strategy ACL adapter call recorded a monitor-only `StrategySignal` as
  `strategy_signal/no_intent` with reason
  `stage16_monitor_only_no_intent`.
- Strategy kill-switch intent was rejected with
  `risk_reason=kill_switch_closed` and created a
  `producer_kill_switch` notification.
- Manual intent was rejected with
  `risk_reason=manual_recent_auth_required` and created a
  `producer_rejected` notification.
- ML intent was rejected with `risk_reason=ml_agent_policy_missing`; this
  proves ML source decisions do not bypass the risk gate. The same source link
  also carried a redacted `producer_unknown` notification.
- Ops accepted intent returned `status=dispatched`,
  `dispatch_stream_name=execution.requests.v1`, and a Redis message id;
  Redis stream length moved from `12` to `13`.
- Additional ops source-event notification proved terminal visibility with
  `producer_terminal/stage16_cancelled_terminal`.

SQL evidence by strategy id:

| Source type | Outcome | Reason | Count |
|---|---|---|---:|
| `strategy_signal` | `no_intent` | `stage16_monitor_only_no_intent` | 1 |
| `strategy_signal` | `risk_rejected` | `kill_switch_closed` | 1 |
| `manual_request` | `risk_rejected` | `manual_recent_auth_required` | 1 |
| `ml_agent_decision` | `risk_rejected` | `ml_agent_policy_missing` | 1 |
| `ops_test` | `intent_created` | `risk_gate_accepted` | 1 |
| `ops_test` | `recorded` | `source_event_recorded` | 1 |

Notification outbox evidence by strategy id:

| Event type | Severity | Reason | Linked source outcome |
|---|---|---|---|
| `producer_fill` | `info` | `stage16_fill_observed` | `intent_created` |
| `producer_kill_switch` | `critical` | `kill_switch_closed` | `risk_rejected` |
| `producer_rejected` | `warning` | `manual_recent_auth_required` | `risk_rejected` |
| `producer_rejected` | `warning` | `ml_agent_policy_missing` | `risk_rejected` |
| `producer_terminal` | `info` | `stage16_cancelled_terminal` | `recorded` |
| `producer_unknown` | `critical` | `stage16_unknown_state` | `risk_rejected` |

Metrics evidence:

- API metrics exposed bounded counters for:
  `execution_source_event_total`,
  `execution_intent_total`,
  `execution_dispatch_total{result="dispatched",reason="redis_xadd_ok"}`,
  and `execution_notification_outbox_total` with bounded `event_type`,
  `source_type`, and `severity` labels.

Browser evidence:

- Playwright against
  `https://roehub.com/strategies?strategy_id=d6a4633a-1c14-4cc9-8a53-4c8b1693485d`
  used the temporary smoke session.
- Dashboard requests returned `200`.
- Accessibility snapshot and DOM artifact showed the `Execution outcomes`
  panel in `ready` state with `6` rows and coverage for
  `producer_fill`, `producer_unknown`, `producer_kill_switch`,
  `producer_terminal`, `producer_rejected`, and `no_intent`.
- DOM secret scan returned `false`.
- Artifacts:
  - `output/playwright/stage16-producer-notifications-dom-stage16-20260602T222018-d29a99c2.json`
  - `output/playwright/stage16-execution-outcomes-panel-ultrawide-stage16-20260602T222018-d29a99c2.png`

Cleanup:

- Temporary smoke session was revoked; SQL cleanup proof returned
  `stage16_smoke_session_active_after_cleanup=0`.
- Temporary auth state files and remote probe scripts were deleted.

## Rollback

- Leave `ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED=0` to disable Strategy
  producer source-event writes.
- Disable caller use of `/ui/execution/notifications` without affecting source
  event or intent ingress.
- If needed before acceptance, revert the Stage 16 commit; the migration is
  additive and downgrade drops only `execution_notification_outbox` and restores
  the previous source-event outcome check.

## Next Handoff

Stage `17` can start. It must use the accepted producer-neutral ingress,
notification outbox, and `/strategies` outcome link as the producer handoff
surface. Keep `ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED=0` unless a later
stage explicitly enables live Strategy producer writes in production. Redis
remains transport only; durable Postgres source/intent/order/fill/reconciliation
rows remain source of truth.
