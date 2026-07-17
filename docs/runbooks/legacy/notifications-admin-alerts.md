# Runbook - Notifications Admin Alerts

## Purpose

This runbook covers notification provider instances, admin alerts, synthetic drills and explicit replay policy. It applies to installation-wide and organization-owned providers, including critical and trading categories.

## Secret Handling

- Do not paste Telegram bot tokens, chat ids, cookies, provider payloads or raw credentials into issues, docs, logs or chat.
- Provider credentials and Telegram recipients must be stored behind typed OpenBao references. Raw bot tokens and raw chat ids are not runtime environment inputs.
- Evidence may report only key presence, redacted references or stable hashes.

## Шлюз исходящего доступа Telegram

Production-диспетчер на `macstudio` получает доступ к `api.telegram.org` через
закрытый SSH-туннель до host-local SSH-псевдонима `roehub-telegram-egress`.
Конкретный адрес шлюза и путь к ключу хранятся только в `~/.ssh/config` на
хосте; публичный прокси-порт не используется:

- Squid на шлюзе слушает только `127.0.0.1:3128`;
- ACL Squid разрешает только `api.telegram.org`; другие адреса должны получать
  `403 Forbidden`;
- отдельная учетная запись `roehub-tunnel` может переадресовывать только
  `127.0.0.1:3128`;
- `launchd`-служба `com.roehub.telegram-egress-tunnel` на `macstudio` публикует
  прокси только локально как `127.0.0.1:18180`;
- `notification-dispatcher` использует локальный прокси только при наличии
  host-local ключа `ROEHUB_NOTIFICATIONS_TELEGRAM_PROXY_URL`;
- токен экземпляра бота разрешается только через его типизированную ссылку
  OpenBao и отдельную service identity; значение токена не передается в
  окружение процесса, SSH-конфигурацию, `plist`, runbook или журналы туннеля.

Шаблон host-local SSH-конфигурации находится в
`infra/macos/ssh/roehub-telegram-egress.conf.example`. Заполненный файл, адрес
шлюза и закрытый ключ не должны попадать в Git.

Проверка шлюза с `macstudio`:

```bash
ssh macstudio 'ssh roehub-telegram-egress true'
```

Проверка постоянного туннеля и доступа к Telegram с `macstudio`:

```bash
ssh macstudio 'launchctl print gui/$(id -u)/com.roehub.telegram-egress-tunnel'
ssh macstudio 'curl -4sS -o /dev/null -w "%{http_code}\n" \
  --proxy http://127.0.0.1:18180 \
  --connect-timeout 5 --max-time 15 \
  https://api.telegram.org'
```

Ожидаемый HTTP-код для корневого адреса Telegram — `302`. Эта проверка не
отправляет сообщение и не раскрывает токен.

Восстановление после обрыва:

```bash
ssh macstudio 'launchctl kickstart -k \
  gui/$(id -u)/com.roehub.telegram-egress-tunnel'
ssh <gateway-admin-alias> 'systemctl restart squid'
```

После восстановления повторить проверку HTTP-кода. Не включать реальную
Telegram-доставку, пока `real-readiness` не подтверждает один согласованный
тестовый маршрут.

Проверка метрик Telegram API:

```bash
ssh macstudio 'curl -fsS http://127.0.0.1:9210/metrics | grep notifications_telegram_api_'
```

- `notifications_telegram_api_up` равно `1`, если последний `getMe` завершился
  успешно;
- `notifications_telegram_api_probe_latency_seconds` показывает задержку
  последней проверки;
- `notifications_telegram_api_last_success_unixtime` хранит время последнего
  успешного ответа;
- `notifications_telegram_api_probe_total{result=...}` считает успешные и
  неуспешные проверки без идентификаторов пользователей, чатов и токенов.

Каждый результат доставки записывается в системный журнал диспетчера только с
внутренними `delivery_id`, `route_id`, `provider`, `status`, `attempt_count` и
нормализованным `error_code`. Адрес получателя, текст сообщения и ответ
Telegram в журнал не выводятся.

Проверка управляемой ротации журналов:

```bash
ssh macstudio 'launchctl print gui/$(id -u)/com.roehub.notification-log-rotation'
ssh macstudio '/opt/roehub/app/scripts/macos/rotate_notification_logs.sh'
```

## Пользовательские счетчики

`/ui/account/notifications/scoped` возвращает счетчик только для текущего
аутентифицированного пользователя. Он строится из delivery ledger и учитывает
только успешные отправки через `telegram_bot_api`; `log_only`, административные
маршруты и доставки других пользователей исключены.

Полная пользовательская привязка не включается этим rollout. Ее канонический
источник остается в Identity; нельзя создавать параллельную таблицу Notifications
без совместной миграции consumer-ов и production `telegram_bot_worker`.

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

## Provider Instance Degraded

Alert: `NotificationsProviderInstanceDegraded`.

Diagnosis:

- Use only the sanitized `provider` and `provider_instance` metric labels.
- Inspect the bounded provider health code and the affected instance's recent `sent`, `retry`, `unknown` and `dead_letter` outcomes.
- Confirm whether the failure is credential, transport, timeout or provider-side degradation without printing secret values or provider payloads.
- Verify that other organization or installation instances remain independent.

Action:

- Pause or disable only the affected instance when continued attempts are unsafe.
- Do not reroute critical or trading notifications from an organization bot to the installation-wide bot.
- Preserve `unknown` deliveries and use explicit replay only after checking possible provider acceptance.
- Restore the instance through its OpenBao reference and rerun the bounded health probe before resuming traffic.

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
  --mode log-only-matrix \
  --organization-id <organization-uuid> \
  --owner-user-id <owner-user-uuid>
```

Check real Telegram readiness without sending a message:

```bash
cd /Users/daniildegtyarev/Projects/roehub.com
/opt/roehub/app/.venv/bin/python scripts/notifications/stage09_production_canary.py \
  --config /opt/roehub/app/configs/prod/notifications.yaml \
  --mode real-readiness \
  --organization-id <organization-uuid> \
  --owner-user-id <owner-user-uuid> \
  --provider-instance-id <provider-instance-uuid>
```

The readiness output may report only booleans and counts. Do not print or paste token values, raw chat ids or provider payloads. A real Telegram canary is allowed only after the user approves the test/admin recipient and confirms receipt.
