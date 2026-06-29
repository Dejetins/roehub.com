# Stage 01: Notifications Schema, Domain, Ports

Дата: `2026-06-29`

Статус: `completed-local`

Acceptance boundary: Stage `01` добавляет provider-neutral foundation для bounded context `notifications`. На момент этого отчета реализация прошла локальные focused gates и real-boundary schema smoke, но еще требует публикации в `main`, GitHub CI и `macstudio` sync перед переводом в `accepted`.

## User Required Before Start

Nothing.

Telegram token, admin chat id, smoke password, cookies, exchange credentials или другие secrets не требовались и не запрашивались. Для DB boundary использовался host-local env source на `macstudio`; значения DSN/credentials не выводились.

## Checkout And Branch

| Field | Value |
|---|---|
| Checkout path | `/Users/daniildegtyarev/Projects/roehub.com` |
| Branch | `main` |
| Branch/worktree/stash workflow | not created |
| Unrelated dirty work observed | yes: untracked `docs/architecture/market_data/market-data-live-tail-repair-v1.md` and `docs/architecture/market_data/market-data-live-tail-repair-v1-stage-reports/`; not staged or modified |

## Scope

Implemented:

- additive Alembic revision `20260629_0037` after `20260618_0036`;
- provider-neutral tables for notification events, routes, deliveries, attempts, Telegram updates and report runs;
- immutable domain objects for `NotificationEvent`, `NotificationRoute`, `NotificationDelivery`, `NotificationDeliveryAttempt`, `TelegramUpdate` and `NotificationReportRun`;
- `NotificationRepository` application port scaffold;
- migration and domain tests covering additive DDL, status enums, dedupe keys, route separation and secret-like field rejection.

Not implemented in this stage:

- source routing from Strategy/Live Execution/admin/report facts;
- dispatcher worker, provider adapter wiring, Telegram sends or bot polling;
- API/DTO/UI changes;
- production migration apply.

## Schema Evidence

Transactional schema smoke on `macstudio` executed the new migration against a disposable schema through real Postgres, inspected the created objects, and rolled back the transaction.

| Evidence | Result |
|---|---|
| Tables | `notification_deliveries`, `notification_delivery_attempts`, `notification_events`, `notification_report_runs`, `notification_routes`, `notification_telegram_updates` |
| Indexes | `idx_notification_deliveries_pending`, `idx_notification_deliveries_route_created`, `idx_notification_delivery_attempts_delivery`, `idx_notification_events_category_created`, `idx_notification_events_owner_created`, `idx_notification_report_runs_owner_period`, `idx_notification_routes_admin`, `idx_notification_routes_owner_channel`, `idx_notification_telegram_updates_status` plus primary/unique indexes |
| Constraints sample | delivery source cardinality, delivery status, attempt request/response hash, event owner/recipient, event payload JSON, route owner separation, report period/status, Telegram update idempotency |
| Cleanup | `transaction_rollback`; no persistent production schema change |

## Validation

| Check | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps/migrations/test_notifications_context_sql.py` | passed: `7 passed` |
| `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps/migrations` | passed: `45 passed` |
| `uv run ruff check src/trading/contexts/notifications tests/unit/contexts/notifications tests/unit/apps/migrations/test_notifications_context_sql.py` | passed |
| `uv run pyright src/trading/contexts/notifications tests/unit/contexts/notifications` | passed |
| `uv run ruff check .` | passed |
| `uv run pyright` | passed |
| `uv run pytest -q -ra` | passed: `1395 passed`, 3 existing `httpx` deprecation warnings |
| Real-boundary schema smoke on `macstudio` transactional disposable schema | passed |
| `uv run python -m tools.docs.generate_docs_index --check` | locally polluted by unrelated untracked market-data docs after scoped README pruning; CI on clean `main` must provide final docs-index evidence |

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `none` | No API routes or payloads changed. |
| DTO schema | `none` | No DTOs changed. |
| Ports | `compatible-change` | Adds `NotificationRepository`; no existing port signatures changed. |
| Persisted schema | `compatible-change` | Adds `notification_*` tables/indexes/constraints; no existing tables, columns or constraints changed. |
| Config/defaults | `none` | No config files changed. |
| Request/cache/persistence identity | `compatible-change` | Adds notification `dedupe_key` identity for future intake; existing identities unchanged. |
| Service-call semantics | `none` | No service calls added. |
| External side effects | `none` | No Telegram or provider send path is wired. |
| Logs/metrics/audit/redaction | `compatible-change` | Adds redaction-oriented domain and schema constraints; no runtime logging changed. |
| Alerts/runbooks | `none` | Runbooks are later stages. |
| Browser-visible behavior | `none` | No web UI changes. |
| Performance | `unknown` | Dispatcher/query hot paths are not implemented yet. |

## File Manifest

| Path | Action | Reason | Contract impact |
|---|---|---|---|
| `alembic/versions/20260629_0037_notifications_context_v1.py` | created | Add provider-neutral notifications schema. | `compatible-change` persisted schema |
| `src/trading/contexts/notifications/__init__.py` | created | Expose notifications bounded context surface. | `compatible-change` ports |
| `src/trading/contexts/notifications/domain/__init__.py` | created | Export domain objects. | `compatible-change` ports |
| `src/trading/contexts/notifications/domain/notification.py` | created | Add domain validation and redaction invariants. | `compatible-change` ports/domain |
| `src/trading/contexts/notifications/application/__init__.py` | created | Application package entrypoint. | `compatible-change` ports |
| `src/trading/contexts/notifications/application/ports/__init__.py` | created | Port package entrypoint. | `compatible-change` ports |
| `src/trading/contexts/notifications/application/ports/notification_repository.py` | created | Add repository port scaffold. | `compatible-change` ports |
| `tests/unit/contexts/notifications/test_notification_domain.py` | created | Domain coverage for dedupe, statuses, route separation and redaction. | `none` |
| `tests/unit/apps/migrations/test_notifications_context_sql.py` | created | Migration structure and redaction constraint coverage. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/01-notifications-schema-domain-ports.md` | created | Stage `01` report. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` | modified | Record Stage `01` local result and evidence. | `none` |

## Residual Risks

- Stage `01` is not `accepted` until the diff is published to `main`, CI is green, and `macstudio` checkout/smoke evidence is recorded.
- Future stages still need repository adapter implementations, route-decision logic, dispatcher leases/retries/unknown handling, metrics and provider wiring.
- The current schema intentionally avoids cross-context FKs so isolated migration proof is possible; source ownership and ACL checks must be enforced in later adapters/use cases.
