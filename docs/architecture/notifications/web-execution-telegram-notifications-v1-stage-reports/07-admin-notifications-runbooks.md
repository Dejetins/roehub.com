# Stage 07: Admin Notifications And Runbooks

Дата: `2026-06-29`

Статус: `accepted`

Acceptance boundary: Stage `07` adds provider-neutral admin critical/alert/report routing, bounded metrics, Prometheus alert rules and an operational runbook. Implementation commit `43444b2fdf1de86dcec7d7e939e32f4be15b3097` is published to `main`; GitHub CI/deploy passed; `macstudio` checkout is synchronized to the same commit and production smoke passed.

## User Required Before Start

Required only to choose or confirm a real admin Telegram recipient. Synthetic admin drill required nothing.

No Telegram token, chat id, admin recipient value, password, cookie or provider payload was required or printed.

## Scope

Implemented:

- `NotificationAdminDrillService` for synthetic `admin_critical`, `admin_alert` and `admin_report` drills;
- admin-only route lookup and delivery creation through existing notifications repository contracts;
- log-only dispatcher proof that admin events create delivery attempts;
- bounded API monitoring helpers for admin notification totals, unknown deliveries, pending age, retry/rate-limit, worker health and missed report schedules;
- macOS Prometheus alert rules for Stage `07` notification admin conditions;
- production bootstrap installation of the new alert rules;
- runbook `docs/runbooks/notifications-admin-alerts.md`.

Not implemented in this stage:

- real Telegram admin recipient canary;
- production SQL-backed admin recipient setup UI;
- automatic escalation to a real provider.

## Admin Drill Behavior

| Behavior | Result |
|---|---|
| Admin route separation | Synthetic drill lists only active `recipient_kind=admin` routes. |
| User leakage guard | Active user routes for admin categories do not receive admin drill deliveries. |
| Categories | Covers `admin_critical`, `admin_alert`, `admin_report`. |
| Delivery | Creates pending delivery rows with `event_id` and admin route id. |
| Attempt proof | `NotificationDispatcher` sends through `LogOnlyNotificationProvider`, creating attempt rows. |
| Provider boundary | No real Telegram provider is required. |

## Alert Coverage

| Alert | Source metric | Severity |
|---|---|---|
| `NotificationsCriticalUnknownDelivery` | `notifications_delivery_unknown_total` | `critical` |
| `NotificationsDispatcherPendingOld` | `notifications_pending_oldest_age_seconds` | `warning` |
| `NotificationsWorkerDown` | `notifications_worker_up` | `critical` |
| `NotificationsRetry429High` | `notifications_deliveries_retry_total{reason="rate_limited"}` | `warning` |
| `NotificationsMissedReportSchedule` | `notifications_report_schedule_missed_total` | `warning` |

## Real Boundary Evidence

Synthetic admin drill executed with in-memory repository adapter and `LogOnlyNotificationProvider`:

| Evidence | Result |
|---|---|
| Admin events | `events=3` |
| Admin deliveries | `deliveries=3` |
| Dispatcher sends | `sent=3` |
| Attempts | `attempts=3` |
| Route kind | `route_kinds=admin` |
| Smoke line | `stage07_admin_drill_smoke=ok events=3 deliveries=3 sent=3 attempts=3 drill_deliveries=3 categories=admin_alert,admin_critical,admin_report route_kinds=admin` |

## Validation

| Check | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/notifications/test_admin_notifications.py tests/unit/apps/api/test_notification_admin_monitoring.py tests/unit/infra/test_monitoring_assets.py` | passed: `9 passed` |
| `uv run pytest -q tests/unit/contexts/notifications tests/unit/infra tests/unit/apps/api/test_notification_admin_monitoring.py` | passed: `60 passed` |
| `uv run ruff check src/trading/contexts/notifications tests/unit/contexts/notifications tests/unit/infra tests/unit/apps/api/test_notification_admin_monitoring.py apps/api/monitoring.py` | passed |
| `uv run pyright src/trading/contexts/notifications tests/unit/contexts/notifications tests/unit/infra tests/unit/apps/api/test_notification_admin_monitoring.py apps/api/monitoring.py` | passed |
| Synthetic admin drill smoke | passed: `stage07_admin_drill_smoke=ok ... route_kinds=admin` |
| `uv run python -m tools.docs.generate_docs_index --check` | clean CI passed in run `28397126835`; local dirty checkout still contains unrelated untracked `market-data-live-tail-repair-v1` docs |
| GitHub CI | passed: run `28397126835` for `43444b2fdf1de86dcec7d7e939e32f4be15b3097` |
| GitHub deploy | passed: Backend `28397233003`, Web `28397233010`, App Image `28397233012` |
| `macstudio` sync/smoke | passed: `/Users/daniildegtyarev/Projects/roehub.com` fast-forwarded to `43444b2fdf1de86dcec7d7e939e32f4be15b3097`; `/opt/roehub/app/scripts/macos/smoke_prod.sh` passed |

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `none` | No HTTP route changed. |
| DTO schema | `none` | No API DTO changed. |
| Application exports | `compatible-change` | Adds admin drill service/result/facts exports. |
| Ports | `none` | Uses existing repository methods. |
| Persisted schema | `none` | No migration changed. |
| Config/defaults | `compatible-change` | Adds repo-managed Prometheus rule file to macOS production config/bootstrap. |
| External service calls | `none` | No provider network call added. |
| External side effects | `compatible-change` | When invoked with active fake/log admin routes, drill creates admin event/delivery rows. |
| Browser-visible behavior | `none` | No UI changed. |
| Ops/alerts | `compatible-change` | Adds bounded metrics helpers, alert rules and runbook. |

## File Manifest

| Path | Action | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/notifications/application/admin_notifications.py` | created | Add synthetic admin drill service and facts. | `compatible-change` application surface |
| `src/trading/contexts/notifications/application/__init__.py` | modified | Export admin drill surface. | `compatible-change` application export |
| `src/trading/contexts/notifications/__init__.py` | modified | Export admin drill surface at context root. | `compatible-change` context export |
| `apps/api/monitoring.py` | modified | Add bounded notification admin metrics helpers. | `compatible-change` metrics |
| `infra/macos/prometheus/prometheus.prod.yml` | modified | Load notification admin alert rules. | `compatible-change` ops config |
| `infra/macos/prometheus/rules/notifications-admin.rules.yml` | created | Add Stage `07` notification admin alerts. | `compatible-change` alerts |
| `scripts/macos/bootstrap_native_prod.sh` | modified | Install notification admin alert rules on Mac Studio. | `compatible-change` deploy/bootstrap |
| `docs/runbooks/notifications-admin-alerts.md` | created | Add diagnosis, redaction, replay and escalation runbook. | `compatible-change` ops docs |
| `tests/unit/contexts/notifications/test_admin_notifications.py` | created | Cover admin-only drill and log-only attempts. | `none` |
| `tests/unit/apps/api/test_notification_admin_monitoring.py` | created | Cover bounded metric exposure. | `none` |
| `tests/unit/infra/test_monitoring_assets.py` | modified | Cover alert rules and bootstrap installation. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/07-admin-notifications-runbooks.md` | created | Stage `07` report. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` | modified | Record Stage `07` local implementation and evidence. | `none` |
| `docs/architecture/README.md` | modified | Add Stage `07` report and runbook to docs index. | `none` |

## Residual Risks

- Real admin Telegram recipient confirmation remains deferred to Stage `09` canary or explicit host-local setup.
- Metrics helpers are bounded and tested, but production worker integration for all gauges/counters remains future wiring.
- Prometheus rule deployment is repo-managed; runtime reload proof is limited to CI/deploy plus `macstudio` production smoke, not a live alert firing drill.
