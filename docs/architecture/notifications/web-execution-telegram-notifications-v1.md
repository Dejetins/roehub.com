# Notifications v1 - Web execution Telegram delivery and reports

Документ фиксирует целевую архитектуру bounded context `notifications`: пользовательские и админские уведомления, Telegram bot delivery, команды статистики и scheduled portfolio reports. План не меняет существующие runtime-контракты сам по себе; реализация должна идти stage-gated через ledger.

## Статус

| Поле | Значение |
|---|---|
| `status` | `plan-ready-local` |
| `created_at` | `2026-06-22` |
| `owner` | `Roehub agents / notifications` |
| `prompt_pack` | `.codex/agents/generated/web-execution-telegram-notifications-v1/` |
| `ledger` | `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` |
| `scope` | provider-neutral notifications, Telegram bot, user/admin alerts, day/week/month stats, weekly/monthly reports |

## Branch And Prompt Execution Contract

All Notifications v1 stages run in the normal repository checkout:

`/Users/daniildegtyarev/Projects/roehub.com`

The only working branch for this plan is:

`codex/web-execution-telegram-notifications-v1`

Future executor prompts must not create per-stage branches or sibling worktrees. They must include and follow:

`.codex/agents/generated/web-execution-telegram-notifications-v1/00-branch-and-stage-execution-contract.md`

Before any edit, each executor must verify `git status --short --branch` in `/Users/daniildegtyarev/Projects/roehub.com`. If the checkout is not on `codex/web-execution-telegram-notifications-v1`, the executor must switch the same checkout to that branch. If unrelated dirty work exists, it must be preserved explicitly before switching; it must not be mixed into Notifications v1 commits.

## Цель

Сделать отдельный bounded context `notifications`, чтобы:

- Strategy, Live Execution, ML/RL, ops и будущие producer'ы не зависели от Telegram напрямую.
- Пользователь мог получать сделки, сигналы, critical-only alerts, отчеты и статистику через Telegram.
- Администратор получал отдельные critical alerts, обычные ops alerts и summary reports.
- Подключение нового delivery-сервиса сводилось к новому provider adapter, конфигу и шаблонам, без переписывания producer contexts.
- Все уведомления были трассируемыми: источник, decision, delivery attempt, provider result, retry/unknown/dead-letter, redaction proof.

## Не Цель v1

- Mainnet trading enablement.
- Замена execution gateway или strategy producer.
- Прямая отправка биржевых ордеров из Telegram.
- Хранение raw Telegram token, raw provider payload, cookies, exchange credentials или chat secrets в docs/logs.
- Полная аналитическая витрина portfolio performance, если исходные ledgers еще не дают complete PnL. В таких местах v1 обязан показывать `partial` или `unavailable`.

## Текущий Срез

| Область | Что Уже Есть | Пробел |
|---|---|---|
| Identity Telegram | `identity_telegram_channels` хранит `user_id`, `chat_id`, `is_confirmed`, `confirmed_at`; есть resolver confirmed chat. | Нет полноценного Telegram bot binding flow, durable inbound updates и command handling. |
| Account preferences | `identity_integrations` поддерживает `telegram/discord/slack` и mode `off/alerts/critical`; `identity_notification_preferences` хранит channel keys и mode `off/on/critical`. | Нет категорий `signals`, `trades`, `reports`, scoped preferences по strategy/exchange/market_type, weekly/monthly schedule. |
| Strategy Telegram | Есть Strategy-specific `TelegramNotifier`, event types `signal/trade_open/trade_close/failed`, policy и adapter; prod config читает token env, dev `log_only`. | На текущем live runner фактически публикуется только `failed`; нет общей delivery queue, attempts, retry, stats commands, админ route. |
| Live Execution notifications | Есть `execution_notification_outbox` и domain notification types `producer_rejected/fill/unknown/kill_switch/terminal`, severity, redacted labels, UI list. | Outbox не доставляет во внешние каналы; нет dispatcher claim/send/mark retry, provider state, Telegram/user preference decision. |
| Portfolio/statistics data | Есть `strategy_signals`, paper orders/fills/accounting, execution orders/events/fills/funding, exchange account snapshots. | Нет единого stats query service для day/week/month, strategy/exchange filters и report snapshot quality state. |
| Admin alerts | Есть отдельные metrics/logs по worker'ам и execution outbox semantics. | Нет first-class admin notification recipients, category routing, escalation policy и report delivery ledger. |

## Current-State Evidence Sources

| Claim area | Source paths |
|---|---|
| Confirmed Telegram channel binding | `migrations/postgres/0001_identity_v1.sql`; `src/trading/contexts/strategy/adapters/outbound/acl/identity/confirmed_telegram_chat_binding_resolver.py` |
| Existing account notification preferences | `migrations/postgres/0006_identity_account_settings_v1.sql`; `apps/api/routes/ui_account.py`; `apps/api/dto/ui_account.py` |
| Strategy-specific Telegram notifier | `src/trading/contexts/strategy/application/ports/telegram_notifier.py`; `src/trading/contexts/strategy/application/services/telegram_notification_policy.py`; `src/trading/contexts/strategy/adapters/outbound/messaging/telegram/telegram_bot_api_notifier.py`; `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`; `configs/prod/strategy.yaml`; `configs/dev/strategy.yaml` |
| Current strategy runtime notification use | `src/trading/contexts/strategy/application/services/live_runner.py` |
| Live Execution notification facts/outbox | `src/trading/contexts/live_execution/domain/notification.py`; `src/trading/contexts/live_execution/application/use_cases/execution_ingress.py`; `src/trading/contexts/live_execution/application/ports/execution_intent_repository.py`; `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/execution_intent_repository.py`; `alembic/versions/20260603_0030_execution_notifications_producers_v1.py`; `apps/api/routes/ui_execution.py`; `apps/api/dto/ui_execution.py` |
| Existing signal/execution/accounting stats sources | `alembic/versions/20260531_0018_strategy_signals_v1.py`; `alembic/versions/20260531_0022_capital_reservation_paper_accounting_v1.py`; `alembic/versions/20260531_0027_testnet_order_adapters_v1.py`; `alembic/versions/20260602_0029_execution_reconciliation_pitr_v1.py`; `alembic/versions/20260531_0020_exchange_account_projection_config_guard_v1.py` |
| Accepted Live Execution outbox scope | `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/16-producer-integrations-notifications.md`; `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/13-notifications-runbooks.md` |

## Целевая Граница Bounded Context

`notifications` владеет только notification lifecycle:

- intake normalized events from source contexts;
- decision: recipient, category, severity, route, suppression, schedule;
- rendering through templates;
- delivery attempts through providers;
- inbound bot commands;
- report run lifecycle;
- metrics, alerts and audit trail for notification delivery.

`notifications` не владеет trading truth:

- `strategy` остается source of truth для strategy specs, runs, signals and strategy runtime state;
- `live_execution` остается source of truth для intents, orders, fills, reconciliation and execution notification source facts;
- `identity` остается source of truth для users, confirmed Telegram chat binding, account settings and admin/user identity;
- market/exchange/account projection contexts остаются source of truth для balances, positions, exchange account snapshots and freshness.

Dependency direction:

```text
strategy/live_execution/ml/ops/reporting
        |
        | source facts through ports or durable source tables
        v
notifications application
        |
        +-- identity ACL: users, confirmed channels, preferences, admins
        +-- source ACLs: strategy/execution/account stats reads
        +-- provider adapters: telegram_bot_api, log_only, future email/webhook/push
```

Producer contexts may emit source facts or existing outbox rows. They must not call Telegram directly after the migration stage except through an explicitly temporary fallback flag.

## Domain Model

### NotificationEvent

Generic fact owned by `notifications`.

| Field | Contract |
|---|---|
| `event_id` | UUID generated by notifications or deterministic source hash for idempotent intake. |
| `owner_user_id` | Nullable for admin-only/platform events; required for user events. |
| `recipient_kind` | `user`, `admin`, `both`. |
| `source_context` | `strategy`, `live_execution`, `rl_trading`, `market_data`, `ops`, `identity`, `notifications`. |
| `source_event_type` | Source-specific stable type, for example `producer_fill` or `strategy_signal`. |
| `category` | Stable product category, see category table below. |
| `severity` | `info`, `warning`, `critical`. |
| `scope_json` | Redacted structured scope: strategy id, exchange, market type, symbol, run id, connection id when allowed. |
| `payload_json` | Redacted bounded payload for rendering and stats links. No secrets, no raw provider payloads. |
| `dedupe_key` | Stable source/event/category/recipient key. |
| `occurred_at` | Source event time. |
| `created_at` | Intake time. |

### NotificationRoute

Route/preference decision for one recipient and category.

| Field | Contract |
|---|---|
| `route_id` | UUID. |
| `recipient_kind` | `user` or `admin`. |
| `user_id` | Nullable only for platform admin broadcast route templates. |
| `channel_key` | `telegram`, future `email`, `webhook`, `push`, `in_app`. |
| `provider_key` | `telegram_bot_api`, `log_only`, future provider id. |
| `mode` | `off`, `critical_only`, `trades`, `signals`, `reports`, `all`. |
| `category_filter` | Explicit allowed categories. |
| `scope_filter_json` | Optional strategy/exchange/market_type/symbol filters. |
| `schedule_json` | Weekly/monthly report schedule and timezone. |
| `status` | `active`, `paused`, `requires_rebind`, `disabled`. |

### NotificationDelivery

Durable delivery queue row.

| Field | Contract |
|---|---|
| `delivery_id` | UUID. |
| `event_id` | Nullable when delivery is for report or command response. |
| `report_run_id` | Nullable when event delivery. |
| `command_id` | Nullable when event/report delivery. |
| `route_id` | Chosen route. |
| `provider_key` | Delivery adapter key. |
| `channel_key` | Product channel. |
| `recipient_address_ref` | Provider address reference, never raw token. Telegram chat id may be stored in DB but logs/docs must redact it. |
| `template_key` | Stable renderer key. |
| `rendered_payload_json` | Bounded rendered message metadata; raw provider request is not stored by default. |
| `status` | `pending`, `claimed`, `sent`, `failed`, `retry`, `dead_letter`, `suppressed`, `unknown`. |
| `attempt_count` | Incremented by dispatcher. |
| `next_attempt_at` | Backoff schedule. |
| `lease_until` | Claim lease. |
| `last_error_code` | Sanitized provider/application error code. |
| `provider_message_id` | Provider result id when available. |
| `created_at`, `sent_at` | Delivery lifecycle timestamps. |

### NotificationDeliveryAttempt

Append-only attempt log:

- `attempt_id`
- `delivery_id`
- `provider_key`
- `started_at`
- `finished_at`
- `status`
- `http_status` when applicable
- `error_code`
- `retry_after_seconds`
- `redacted_request_hash`
- `redacted_response_hash`

### TelegramUpdate

Durable inbound bot update:

- `telegram_update_id` unique;
- `received_at`;
- `chat_id_ref`;
- `user_id` when binding is known;
- `command_name`;
- `command_args_json`;
- `status`: `pending`, `handled`, `ignored`, `failed`, `dead_letter`;
- `idempotency_key`.

### NotificationReportRun

Scheduled or requested summary:

- `report_run_id`;
- `owner_user_id`;
- `report_type`: `portfolio_weekly`, `portfolio_monthly`, `stats_on_demand`;
- `period_start`, `period_end`;
- `scope_json`;
- `quality_status`: `complete`, `partial`, `unavailable`;
- `status`: `pending`, `rendered`, `sent`, `failed`, `suppressed`;
- `dedupe_key`.

## Categories And Modes

| Category | Default severity | User mode coverage | Admin coverage | Source |
|---|---:|---|---|---|
| `strategy_run_failed` | `warning` | `critical_only`, `all` | warning alert | strategy |
| `strategy_signal` | `info` | `signals`, `all` | optional summary only | strategy |
| `trade_fill` | `info` | `trades`, `all` | optional summary only | live_execution / paper accounting |
| `execution_rejected` | `warning` | `critical_only`, `trades`, `all` | warning alert | live_execution |
| `execution_terminal` | `warning` | `critical_only`, `trades`, `all` | warning alert | live_execution |
| `execution_unknown` | `critical` | `critical_only`, `all` | critical alert | live_execution |
| `kill_switch` | `critical` | `critical_only`, `all` | critical alert | live_execution |
| `portfolio_report` | `info` | `reports`, `all` | optional summary only | notifications/stats |
| `stats_response` | `info` | command response only | no | notifications/stats |
| `system_alert` | `warning` | `critical_only`, `all` when user-impacting | alert | ops |
| `admin_critical` | `critical` | no | critical alert | ops/notifications |
| `admin_alert` | `warning` | no | alert | ops/notifications |
| `admin_report` | `info` | no | report | notifications/ops |

Preference compatibility with current identity tables:

- Existing `telegram=off` suppresses Telegram route.
- Existing `risk_alerts=critical` maps to `critical_only` for risk, kill switch and unknown execution categories.
- Existing `trade_fills=on` maps to `trades`.
- Existing `daily_report` remains a legacy alias; v1 adds explicit weekly and monthly schedule in notifications-owned settings.
- Existing `/ui/account/notifications` response must remain compatible. New scoped preferences should use additive routes/endpoints instead of changing the current DTO shape in place.

## Provider Plug-in Contract

Provider adapters implement:

```python
class NotificationProviderAdapter(Protocol):
    provider_key: str

    async def send(
        self,
        delivery: NotificationDelivery,
        message: RenderedNotificationMessage,
    ) -> ProviderDeliveryResult: ...
```

`ProviderDeliveryResult`:

- `status`: `sent`, `retry`, `failed`, `unknown`;
- `provider_message_id`;
- `retry_after_seconds`;
- `error_code`;
- `redacted_request_hash`;
- `redacted_response_hash`.

Adapters in v1:

- `log_only` - deterministic local/test adapter, default for synthetic checks.
- `telegram_bot_api` - Telegram Bot API `sendMessage`.

Future adapters:

- `email`;
- `webhook`;
- `push`;
- `in_app`.

Adding a new provider should require:

1. Adapter implementation.
2. Config entry under `notifications.providers.<provider_key>`.
3. Renderer/template capability mapping.
4. Focused provider tests.
5. Stage-ledger update with contract classification.

It should not require changes in `strategy`, `live_execution`, `ml` or other source contexts.

## Telegram Bot Contract

### Binding

Binding is web-initiated and bot-confirmed:

1. User opens web settings and requests Telegram binding code.
2. Platform creates short-lived one-time binding code, stored hashed with TTL and owner.
3. User sends `/start <code>` to the bot.
4. Bot validates code and writes/updates confirmed Telegram channel through identity ACL.
5. `notifications` records command handling and sends confirmation delivery.

The bot must not accept arbitrary chat takeover by user id text. Rebind requires fresh one-time code or an authenticated web action.

### Commands

| Command | Result |
|---|---|
| `/start <code>` | Confirm Telegram channel binding. |
| `/stats today` | User portfolio stats for current day. |
| `/stats week` | User portfolio stats for current calendar/user-timezone week. |
| `/stats month` | User portfolio stats for current calendar/user-timezone month. |
| `/strategy <id_or_name> today|week|month` | Stats filtered to one strategy. |
| `/exchange <exchange_or_connection> today|week|month` | Stats filtered to one exchange/connection. |
| `/settings` | Current Telegram notification modes and report schedule. |
| `/critical_only` | Set Telegram route to critical-only. |
| `/signals_on` / `/signals_off` | Toggle signal notifications. |
| `/reports weekly on|off` | Toggle weekly portfolio report. |
| `/reports monthly on|off` | Toggle monthly portfolio report. |

Inbound update processing must be idempotent by `telegram_update_id`. Unknown commands return a bounded help response without exposing internal ids beyond user-owned scopes.

### Receiving Updates

Initial implementation should use a worker-owned polling adapter:

- `apps/worker/telegram_bot_worker`;
- no public webhook route required for first canary;
- durable update storage before command execution;
- offset only advances after durable storage.

A webhook receiver can be added later behind the same `TelegramUpdateReceiver` port if public routing is desired.

## Dispatcher Runtime

Initial services:

- `apps/worker/notification_dispatcher` - claims pending deliveries, renders, sends through provider adapters, records attempts and metrics.
- `apps/worker/telegram_bot_worker` - receives Telegram updates, stores them, handles commands, creates command response deliveries.
- Existing API/web surfaces add read/write settings and binding-code endpoints in later stages.

Claim rules:

- claim due rows with `status in ('pending', 'retry')` and `next_attempt_at <= now`;
- set `status='claimed'`, `lease_until=now + lease_ttl`;
- expired leases are reclaimable with attempt/backoff cap;
- max attempts are category/provider-specific;
- dead-letter critical delivery failures create admin notification events.

## Service Calls And Unknown State

Telegram `sendMessage` has no platform idempotency key. Therefore:

- transport timeout, connection reset or ambiguous 5xx after request send becomes `unknown`;
- `unknown` must not be blindly retried for trade/critical user messages;
- operator/admin alert is created for `unknown` critical/trade deliveries;
- manual replay must create a new delivery linked to the unknown delivery id;
- 429 uses `retry_after` and remains retryable;
- 400/403 marks route `requires_rebind` or delivery `failed` with sanitized reason;
- provider token is read only from host-local env/config and is never rendered in docs/logs/attempt rows.

Business consequence: notification delivery is at-least-once inside DB until provider boundary, but provider-visible duplicates are possible only through explicit operator replay after unknown state. Reports must include stable period header/id so a user can see if two messages describe the same period.

## Stats Query Contract

`NotificationStatsQueryService` returns `PortfolioStatsSnapshot`.

Required inputs:

- `owner_user_id`;
- `period_start`, `period_end`;
- optional `strategy_id`;
- optional `exchange_name` or `exchange_connection_id`;
- optional `market_type`;
- optional `mode`: `paper`, `testnet`, future `mainnet`.

Output:

| Field group | Sources | Quality |
|---|---|---|
| Signals | `strategy_signals` | complete if table reachable and period filter applied. |
| Execution outcomes | `execution_notification_outbox`, source/outcome links, orders/events/fills | partial until all producer paths emit unified links. |
| Paper PnL/accounting | `paper_orders`, `paper_fills`, `strategy_paper_accounting` | complete for accepted paper paths; unavailable for modes without paper accounting. |
| Testnet fills/funding | `execution_orders`, `execution_order_events`, `execution_fills`, `execution_funding_events` | partial if exchange reconciliation coverage is incomplete. |
| Portfolio snapshot | `exchange_account_snapshots`, balances, positions, open orders | partial if snapshot stale or exchange connection missing. |

Every response includes:

- `quality_status`: `complete`, `partial`, `unavailable`;
- `freshness`;
- `missing_sources`;
- period and timezone;
- explicit `N/A` for metrics that cannot be computed from accepted source ledgers.

Do not infer real portfolio PnL from incomplete testnet/order rows. Prefer a partial response over a false number.

## Scheduled Reports

Weekly and monthly report runs are owned by `notifications`:

- scheduler creates `NotificationReportRun` for each active route and period;
- dedupe key is `owner_user_id + report_type + period_start + period_end + scope`;
- report run queries stats snapshot and creates delivery;
- missed schedules are alertable;
- per-user timezone is read from identity/account settings when available, otherwise platform default is explicit in the report.

## Admin Notifications

Admin routes are separate from user routes:

- `admin_critical`: kill-switch, execution unknown, dispatcher stuck, critical delivery unknown/failure, worker down.
- `admin_alert`: high retry/429 rate, route disable spike, stale pending rows, missed report schedule.
- `admin_report`: daily/weekly ops summary if enabled.

Admin recipient config must be host-local or persisted as admin-owned routes, never hardcoded in source files. Admin alerts should include source context, category, sanitized ids, counts, time window and runbook link.

## Observability

Metrics:

- `notifications_events_total{source_context,category,severity}`;
- `notifications_routes_decisions_total{channel,provider,category,decision}`;
- `notifications_deliveries_total{provider,channel,status,reason}`;
- `notifications_delivery_latency_seconds{provider,channel,category}`;
- `notifications_pending_oldest_age_seconds{provider,channel,severity}`;
- `notifications_delivery_unknown_total{provider,channel,category}`;
- `notifications_report_runs_total{report_type,status}`;
- `telegram_updates_total{status,reason}`;
- `telegram_commands_total{command,status}`;
- `admin_notifications_total{category,severity,status}`.

Alerts:

- critical: oldest pending critical delivery above threshold;
- critical: any unknown critical/trade delivery unless explicitly acknowledged;
- critical: dispatcher or Telegram bot worker down;
- warning: retry/429/failed rate above threshold;
- warning: missed weekly/monthly report schedule;
- warning: route `requires_rebind` spike.

## Redaction Rules

No notification path may log or document:

- provider tokens;
- auth headers;
- cookies;
- exchange credentials;
- raw Telegram update payloads with full chat/user ids;
- raw provider responses when they contain PII;
- message text that contains secrets.

Allowed evidence:

- counts;
- delivery ids;
- source ids when already non-secret internal ids;
- hashes of redacted requests/responses;
- provider status/error code;
- chat id suffix or hash only when needed.

## Contract Impact

| Surface | Classification | Rationale |
|---|---|---|
| Public API | `compatible-change` | Additive settings, binding and stats endpoints. Existing `/ui/account/notifications` schema remains stable. |
| DTO schema | `compatible-change` | New DTOs for scoped preferences, Telegram binding status, stats snapshots and delivery state. Existing DTO fields are not removed. |
| Ports | `compatible-change` | New `notifications` ports and read-only ACLs into identity/strategy/live_execution/account data. |
| Persisted schema | `compatible-change` | New additive tables and indexes. Existing `execution_notification_outbox` is source input, not mutated into delivery queue. |
| Config | `compatible-change` | New optional `notifications.*` config. Existing `strategy.telegram` remains until migration/deprecation stage. |
| Service calls | `compatible-change` | Telegram send calls move to a dedicated worker with explicit timeout/retry/unknown semantics. |
| Side effects | `compatible-change` | New user/admin messages are opt-in/preferences-controlled; synthetic stages default to `log_only`. |
| Browser-visible behavior | `compatible-change` | Settings UI grows Telegram binding/status and scoped notification controls. Existing settings routes stay available. |
| Logs/metrics/alerts | `compatible-change` | Additive metrics and alert rules; no raw secret payloads. |
| Runtime topology | `compatible-change` | New optional workers behind config/launchd stages; no existing worker must depend on them for trading correctness in early rollout. |

## Implementation Stages

Detailed ledger: `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md`.

High-level sequence:

1. Stage `00` - baseline and plan freeze.
2. Stage `01` - notifications schema/domain/ports.
3. Stage `02` - source router and synthetic event matrix.
4. Stage `03` - dispatcher and provider plug-in contract.
5. Stage `04` - Telegram binding and inbound bot commands.
6. Stage `05` - stats query service for day/week/month and scopes.
7. Stage `06` - weekly/monthly report scheduler.
8. Stage `07` - admin notifications, alerts and runbooks.
9. Stage `08` - web settings UI integration.
10. Stage `09` - Mac Studio production canary.
11. Stage `10` - migrate/deprecate direct Strategy Telegram notifier after canary evidence.

## Synthetic Test Account Matrix

Until every producer path is ready to emit live notifications, stages must use synthetic calls or rows on a test account and record evidence per category:

| Notification type | Synthetic proof |
|---|---|
| `strategy_run_failed` | Synthetic strategy run failure source row or router fixture. |
| `strategy_signal` | Synthetic `strategy_signals` row or router fixture. |
| `trade_fill` | Synthetic execution/paper fill source fixture and delivery row. |
| `execution_rejected` | Synthetic `producer_rejected` source fact. |
| `execution_terminal` | Synthetic `producer_terminal` source fact. |
| `execution_unknown` | Synthetic `producer_unknown` source fact plus admin escalation. |
| `kill_switch` | Synthetic `producer_kill_switch` source fact plus admin escalation. |
| `portfolio_report` weekly | Synthetic report run over stats fixture. |
| `portfolio_report` monthly | Synthetic report run over stats fixture. |
| `stats_response` day/week/month | Synthetic Telegram command update and command response delivery. |
| `admin_critical` | Synthetic critical ops event and admin route delivery. |
| `admin_alert` | Synthetic warning ops event and admin route delivery. |
| `admin_report` | Synthetic ops summary report run. |

Each proof must include DB rows, route decision, delivery/attempt state, metrics, redaction check and final status. Real Telegram provider proof is reserved for canary stages and must not print raw token or raw chat id.
