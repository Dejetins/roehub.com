# Runbook - Notifications Admin Alerts

## Purpose

This runbook covers Stage `07` notifications admin alerts, admin-only synthetic drills and replay policy. It applies to `admin_critical`, `admin_alert` and `admin_report` categories.

## Secret Handling

- Do not paste Telegram bot tokens, chat ids, cookies, provider payloads or raw credentials into issues, docs, logs or chat.
- Admin recipient configuration must come from host-local env/config or persisted admin-owned routes.
- Evidence may report only key presence, redacted references or stable hashes.

## Critical Unknown Delivery

Alert: `NotificationsCriticalUnknownDelivery`.

Diagnosis:

- Find the notification delivery by internal sanitized ids only.
- Inspect `status`, `last_error_code`, attempt count and provider key.
- Confirm whether the source category is critical or admin-only.
- Check provider health and recent timeout/rate-limit patterns.

Replay policy:

- Do not blindly retry `unknown` critical or trade-related deliveries.
- Manual replay must create a new delivery linked in operator notes to the unknown delivery id.
- Replays require an operator decision after checking whether the provider may already have accepted the message.

## Stale Pending Deliveries

Alert: `NotificationsDispatcherPendingOld`.

Diagnosis:

- Check `notifications_pending_oldest_age_seconds` by provider, channel and severity.
- Check dispatcher worker health and recent retry/dead-letter volume.
- Confirm route status is not `requires_rebind`, `paused` or `disabled`.

Action:

- Drain through the dispatcher path only.
- Do not update delivery rows manually except as part of a documented repair.

## Worker Down

Alert: `NotificationsWorkerDown`.

Diagnosis:

- Check Mac Studio launchd and Monit state for the named worker.
- Inspect worker logs under the managed Roehub log path.
- Verify `/metrics` after restart.

Action:

- Restart through the managed Mac Studio service path.
- Keep real Telegram canary disabled until worker health is stable.

## High Retry Rate

Alert: `NotificationsRetry429High`.

Diagnosis:

- Inspect retry reason and provider key.
- Confirm `retry_after_seconds` is respected.
- Check whether a schedule/report burst created unexpected volume.

Action:

- Let bounded backoff drain first.
- Avoid manual replay until the provider pressure falls.

## Missed Report Schedule

Alert: `NotificationsMissedReportSchedule`.

Diagnosis:

- Check scheduler clock and timezone source.
- Inspect the route `schedule_json`.
- Verify the report-run dedupe key for the affected period.

Action:

- Re-run scheduler for the affected period.
- Confirm no duplicate `NotificationReportRun` is created for the same user/report type/period/scope.

## Stage Evidence

- Stage report: `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/07-admin-notifications-runbooks.md`.
- Ledger: `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md`.
