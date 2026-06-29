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

## Manual Replay

Use replay only after the operator has inspected the original delivery, attempt rows and provider health.

Required checks:

- Confirm the original delivery status and sanitized `last_error_code`.
- Confirm whether the source category is trade-related, critical or admin-only.
- Check whether the provider may already have accepted the message.
- Link the new delivery to the original delivery id in operator notes.

Do not mutate the original `unknown` delivery into `pending`. Keep it as evidence and create a new delivery for the replay decision.

## Stale Pending Deliveries

Alert: `NotificationsDispatcherPendingOld`.

Diagnosis:

- Check `notifications_pending_oldest_age_seconds` by provider, channel and severity.
- Check dispatcher worker health and recent retry/dead-letter volume.
- Confirm route status is not `requires_rebind`, `paused` or `disabled`.

Action:

- Drain through the dispatcher path only.
- Do not update delivery rows manually except as part of a documented repair.

## Route Disable Or Rebind

Use this when a Telegram route is invalid, unsafe, mis-scoped or blocked by provider errors.

Diagnosis:

- Inspect the route by internal ids only.
- Confirm `recipient_kind`, `user_id`, `channel_key`, `provider_key`, `mode`, `category_filter`, `scope_filter_json` and `status`.
- For 400/403-style provider errors, prefer `requires_rebind` over blind retry.
- For operator-forced shutdown, set the route to `disabled` or `paused` according to the incident scope.

Action:

- Rebind requires a fresh web-generated one-time code or an authenticated admin/user action.
- Do not copy raw Telegram chat ids into notes, tickets or chat.
- After rebind, run a `log_only` or approved canary proof before enabling broad real-provider traffic.

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

## Canary Rollback

Use this when a real Telegram canary or provider expansion produces unexpected delivery state, route leakage, provider errors or user/operator confusion.

Rollback actions:

- Disable or pause the canary route first.
- Keep the dispatcher in `log_only` provider mode unless the incident owner explicitly approves another bounded provider mode.
- Stop any real-provider worker path that was enabled only for the canary.
- Preserve delivery, attempt, route and report-run rows for investigation.
- Do not delete ledger rows or proof rows as a rollback mechanism.

Recovery criteria:

- `smoke_prod.sh` passes on `macstudio`.
- Dispatcher metrics are reachable.
- No unexplained `unknown`, `dead_letter` or stale pending growth remains.
- A fresh user-approved canary recipient is available before another real Telegram send.

## Stage Evidence

- Stage report: `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/07-admin-notifications-runbooks.md`.
- Stage `09` canary report: `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/09-mac-studio-production-canary.md`.
- Stage `11` final closure report: `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/11-final-docs-and-main-closure.md`.
- Ledger: `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md`.

## Stage 09 Production Canary

Run log-only matrix on Mac Studio from the authoritative checkout after the target revision is deployed. Use the production virtualenv and production config:

```bash
cd /Users/daniildegtyarev/Projects/roehub.com
set -a
source /Users/daniildegtyarev/.config/roehub/roehub.env
set +a
export STRATEGY_PG_DSN="${STRATEGY_PG_DSN:-${POSTGRES_DSN}}"
export NOTIFICATIONS_PG_DSN="${NOTIFICATIONS_PG_DSN:-${STRATEGY_PG_DSN}}"
/opt/roehub/app/.venv/bin/python scripts/notifications/stage09_production_canary.py \
  --config /opt/roehub/app/configs/prod/notifications.yaml \
  --mode log-only-matrix
```

Check real Telegram readiness without sending a message:

```bash
cd /Users/daniildegtyarev/Projects/roehub.com
/opt/roehub/app/.venv/bin/python scripts/notifications/stage09_production_canary.py \
  --config /opt/roehub/app/configs/prod/notifications.yaml \
  --mode real-readiness
```

The readiness output may report only booleans and counts. Do not print or paste token values, raw chat ids or provider payloads. A real Telegram canary is allowed only after the user approves the test/admin recipient and confirms receipt.
