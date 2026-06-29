# Stage 10: Strategy Telegram Migration

Дата: `2026-06-29`

Статус: `completed-local`

Checkout path: `/Users/daniildegtyarev/Projects/roehub.com`

Branch: `main`

User required before start: `nothing`; Stage `09` was accepted in the ledger before implementation. No Telegram token, chat id, cookie, provider payload, DSN or raw credential was printed. No branch, worktree or stash workflow was created.

Unrelated dirty changes observed and excluded: foreign Market Data hunk in `docs/architecture/README.md`, live-execution ledger, market-data ledger and untracked market-data Stage `02` report.

Acceptance boundary: Stage `10` migrates Strategy `failed` notification routing from direct Strategy Telegram delivery to the `notifications` context while keeping direct Strategy Telegram adapters as rollback-only modes. The stage is not accepted until the implementation revision is on `main`, CI/deploy pass, Mac Studio checkout/runtime are synchronized, `smoke_prod.sh` passes and a post-main production runtime proof verifies the Strategy notifications mode without real Telegram send.

## Scope

Implemented locally:

- added `NotificationsTelegramNotifier`, a Strategy outbound adapter that preserves the existing Strategy `TelegramNotifier` port and policy/debounce path but records a `NotificationEvent` plus pending `NotificationDelivery` rows through `notifications`;
- extended Strategy Telegram config mode validation to accept `notifications`, `log_only` and `telegram`;
- switched dev/test/prod `strategy.yaml` defaults to `mode: "notifications"`;
- wired `strategy_live_runner` `notifications` mode to `PostgresNotificationRepository` with `NOTIFICATIONS_PG_DSN`, `STRATEGY_PG_DSN`, `POSTGRES_DSN` fallback order;
- kept `log_only` and `telegram` modes as rollback paths with existing metrics hooks and token fail-fast behavior;
- added integration-style unit proof that a controlled Strategy live-runner failure creates `notifications` event/delivery rows without Telegram send.

Not implemented locally:

- removing old direct Strategy Telegram adapters;
- enabling real Telegram provider send from Strategy runtime;
- changing notification schema or user settings API;
- changing dispatcher production provider mode.

## Migration Behavior

| Behavior | Result |
|---|---|
| Main mode | `strategy.telegram.mode=notifications` queues into `notifications` instead of calling Bot API. |
| Strategy failure | Existing Strategy failed-run policy builds the legacy message, then `NotificationsTelegramNotifier` maps it to `category=strategy_run_failed`, `severity=warning`. |
| Delivery | Active user notification routes create pending deliveries with the route provider key. No provider send happens inside Strategy. |
| Rollback | Set `strategy.telegram.mode=telegram` for old direct Bot API path or `log_only` for previous log-only path. Existing token fail-fast for `telegram` remains. |
| Metrics compatibility | Existing `strategy_telegram_notify_total`, `strategy_telegram_notify_errors_total`, `strategy_telegram_notify_skipped_total` remain wired; in `notifications` mode they mean queued/skipped/error at the Strategy handoff boundary. |

## Local Validation

| Check | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/strategy/application/test_strategy_live_runner.py::test_live_runner_queues_failed_notification_through_notifications_context tests/unit/contexts/strategy/adapters/test_notifications_telegram_notifier.py tests/unit/contexts/strategy/adapters/test_strategy_live_runner_runtime_config.py` | passed: `12 passed` |
| `uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/notifications tests/unit/apps` | passed: `508 passed, 3 warnings` |
| `uv run ruff check src/trading/contexts/strategy src/trading/contexts/notifications apps/worker/strategy_live_runner tests/unit/contexts/strategy tests/unit/contexts/notifications` | passed |
| `uv run pyright src/trading/contexts/strategy src/trading/contexts/notifications apps/worker/strategy_live_runner tests/unit/contexts/strategy tests/unit/contexts/notifications` | passed |

## Proof Boundary

Local proof boundary: focused integration harness in `tests/unit/contexts/strategy/application/test_strategy_live_runner.py::test_live_runner_queues_failed_notification_through_notifications_context`.

The proof runs a controlled Strategy live-runner failure through the real Strategy policy and the new notifications-backed adapter. It verifies:

- Strategy run reaches `failed`;
- `NotificationEvent.source_context=strategy`;
- `NotificationEvent.category=strategy_run_failed`;
- `NotificationEvent.source_event_type=failed`;
- pending `NotificationDelivery` is created for the active route;
- no direct Telegram provider call is used.

Post-main production runtime proof remains pending for acceptance.

## Artifact Review

Review mode: cold self-review fallback. Independent review was not available in the current environment.

Verdict: local Stage `10` artifacts are ready for implementation commit and post-main proof. Fixed blocker: direct Strategy Telegram remains as explicit rollback-only modes rather than being removed or silently disabled. Follow-up check: after deployment, prove Mac Studio checkout/runtime sync, `smoke_prod.sh`, and a bounded Strategy notifications-mode runtime smoke without real Telegram send. Residual risk: pending `telegram_bot_api` deliveries will not be claimed while production notification dispatcher remains in `log_only` provider mode, which is expected until final rollout approval.

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `none` | No HTTP route or DTO changed. |
| DTO schema | `none` | No API DTO changed. |
| Ports | `compatible-change` | Existing Strategy `TelegramNotifier` port is reused; a new outbound adapter implementation is added. |
| Persisted schema | `none` | Uses existing Stage `01` notification tables; no migration changed. |
| Config/defaults | `compatible-change` | Adds `notifications` as valid Strategy telegram mode and switches dev/test/prod defaults to it. Old `log_only` and `telegram` modes remain valid rollback values. |
| Service-call semantics | `compatible-change` | Strategy no longer calls Telegram directly in default config; it writes durable notifications rows and leaves provider send to dispatcher. |
| External side effects | `compatible-change` | Default side effect changes from direct send/log to DB event/delivery queueing. No real Telegram send is done by Strategy in `notifications` mode. |
| Logs/metrics/redaction | `compatible-change` | Existing Strategy telegram metrics remain but represent handoff queueing in `notifications` mode. No secret values are logged. |
| Alerts/runbooks | `compatible-change` | Old Strategy Telegram doc now documents fallback state; Notifications runbook remains the dispatcher/send source of truth. |
| Browser-visible behavior | `none` | No UI changed. |
| Performance | `unknown` | Strategy failure path adds one notification event and route/delivery query on failure; no hot-path benchmark was run because normal candle processing is unchanged. |

## File Manifest

| Path | Action | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/strategy/adapters/outbound/messaging/telegram/notifications_telegram_notifier.py` | created | Queue Strategy notifications through `notifications`. | `compatible-change` adapter |
| `src/trading/contexts/strategy/adapters/outbound/messaging/telegram/__init__.py` | modified | Export new adapter. | `compatible-change` export |
| `src/trading/contexts/strategy/adapters/outbound/messaging/__init__.py` | modified | Export new adapter. | `compatible-change` export |
| `src/trading/contexts/strategy/adapters/outbound/__init__.py` | modified | Export new adapter. | `compatible-change` export |
| `src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py` | modified | Accept `notifications` mode in live-runner config. | `compatible-change` config |
| `src/trading/contexts/strategy/adapters/outbound/config/strategy_runtime_config.py` | modified | Accept `notifications` mode in source config. | `compatible-change` config |
| `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py` | modified | Wire `notifications` mode to Postgres notifications repository. | `compatible-change` runtime |
| `configs/dev/strategy.yaml` | modified | Switch default Strategy notification mode to `notifications`. | `compatible-change` config |
| `configs/test/strategy.yaml` | modified | Switch default Strategy notification mode to `notifications`. | `compatible-change` config |
| `configs/prod/strategy.yaml` | modified | Switch production Strategy notification mode to `notifications`. | `compatible-change` config |
| `tests/unit/contexts/strategy/adapters/test_notifications_telegram_notifier.py` | created | Cover adapter event/delivery queueing and skipped-route behavior. | `none` |
| `tests/unit/contexts/strategy/application/test_strategy_live_runner.py` | modified | Add controlled live-runner failure proof through notifications. | `none` |
| `tests/unit/contexts/strategy/adapters/test_strategy_live_runner_runtime_config.py` | modified | Update expected default mode. | `none` |
| `docs/architecture/strategy/strategy-telegram-notifier-best-effort-policy-v1.md` | modified | Document Stage `10` migration/fallback state. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/10-strategy-telegram-migration.md` | created | Stage `10` local report. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` | modified | Record Stage `10` local implementation state. | `none` |

## Residual Risks

- Stage `10` is `completed-local`, not accepted, until post-main production runtime proof is recorded.
- Direct `telegram` rollback still exists and can send real Telegram messages if explicitly configured with token and active bindings.
- Production dispatcher currently runs in `log_only` provider mode; pending `telegram_bot_api` deliveries are intentionally not claimed until final rollout approval.
