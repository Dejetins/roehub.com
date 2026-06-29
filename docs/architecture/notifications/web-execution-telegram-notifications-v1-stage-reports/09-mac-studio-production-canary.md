# Stage 09: Mac Studio Production Canary

Дата: `2026-06-29`

Статус: `accepted`

Acceptance boundary: Stage `09` adds production-safe Mac Studio notification dispatcher topology, a SQL-backed notification repository, launchd/Prometheus wiring and a bounded canary helper. Stage accepted after final implementation commit `88387e4b2f5983cf7a332a5df1fe51ec8de89a28` reached `main`, CI/deploy passed, Mac Studio checkout/runtime were synchronized, production smoke passed, log-only production matrix passed, worker metrics were observable, temporary proof routes were disabled, and real Telegram canary readiness was explicitly blocked on missing active admin route/user-confirmed recipient.

## User Required Before Start

Required for final real Telegram message confirmation and approval of the canary recipient. Local implementation and log-only synthetic proof do not require user input.

Host-local env presence was checked by key name only. Fallback `TELEGRAM_BOT_TOKEN`, Postgres DSN and `ROEHUB_SMOKE_E2E_PASSWORD` were present; preferred `ROEHUB_NOTIFICATIONS_TELEGRAM_BOT_TOKEN` and active admin Telegram route were missing. No token, chat id, password, cookie, provider payload or raw Telegram response was printed.

## Scope

Implemented:

- SQL-backed `PostgresNotificationRepository` for existing `notification_*` tables;
- `notification-dispatcher` production process entrypoint with Prometheus metrics;
- dispatcher provider-mode filter so `log_only` production canary does not claim `telegram_bot_api` deliveries;
- Mac Studio launchd plist, bootstrap/reload wiring and Prometheus scrape target;
- production `notifications.yaml` enabling dispatcher only in `log_only` mode; Telegram provider remains disabled;
- Stage `09` canary helper for DB-backed log-only synthetic matrix and separate real Telegram readiness check.

Not implemented in this stage:

- real Telegram send, because recipient readiness was blocked;
- automatic creation of real admin route;
- Telegram polling worker launchd runtime;
- report scheduler production worker.

## Production Safety

| Surface | Behavior |
|---|---|
| Dispatcher mode | `provider_mode: log_only` allows only `log_only` and `fake` deliveries to be claimed. |
| Real Telegram | `notifications.providers.telegram.enabled=false`; token presence is reported only as booleans. |
| Admin route | Readiness helper counts active admin Telegram routes without printing recipient values. |
| Unknown state | Existing dispatcher semantics keep `unknown` terminal and avoid blind retry. |
| Redaction | Synthetic routes use `telegram_ref:stage09:*`; evidence does not contain raw chat ids. |

## Local Validation

| Check | Result |
|---|---|
| `uv run pytest -q tests/unit/apps/worker/test_notification_dispatcher_wiring.py tests/unit/contexts/notifications/adapters/test_postgres_notification_repository.py` | passed: `6 passed` |
| `uv run pytest -q tests/unit/contexts/notifications` | passed: `43 passed` |
| `uv run ruff check apps/worker/notification_dispatcher src/trading/contexts/notifications tests/unit/apps/worker/test_notification_dispatcher_wiring.py tests/unit/contexts/notifications/adapters/test_postgres_notification_repository.py scripts/notifications/stage09_production_canary.py` | passed |
| `uv run pyright apps/worker/notification_dispatcher src/trading/contexts/notifications tests/unit/apps/worker/test_notification_dispatcher_wiring.py tests/unit/contexts/notifications/adapters/test_postgres_notification_repository.py scripts/notifications/stage09_production_canary.py` | passed |
| `plutil -lint infra/macos/launchd/com.roehub.notification-dispatcher.plist` | passed |
| `uv run python scripts/notifications/stage09_production_canary.py --help` | passed |
| `uv run python -m apps.worker.notification_dispatcher.main.main --help` | passed |

## Runtime Evidence

| Evidence | Result |
|---|---|
| Final main revision | local `main`, `origin/main` and Mac Studio checkout resolved to `88387e4b2f5983cf7a332a5df1fe51ec8de89a28` |
| GitHub CI | passed: `28402177609` |
| Publish App Image | passed: `28402411089` |
| Deploy Backend | passed: `28402411143` |
| Deploy Web | passed: `28402411135`, `28402419473` |
| Mac Studio smoke | passed: `bash /opt/roehub/app/scripts/macos/smoke_prod.sh` |
| Worker health | `com.roehub.notification-dispatcher` launchd state `running`, pid observed, `last exit code = (never exited)` |
| Metrics endpoint | `http://127.0.0.1:9210/metrics` exposed dispatcher metric names |
| Log-only matrix | passed on Mac Studio: `status=ok`, `event_rows=17`, `category_count=12`, `deliveries=17`, `claimed=17`, `sent=17`, `unknown=0`, `dead_letter=0`, `metric_names_present=true` |
| Real Telegram readiness | blocked without send: `telegram_token_present=true`, `preferred_telegram_token_present=false`, `fallback_telegram_token_present=true`, `active_admin_telegram_route_count=0`, `user_confirmation_required=true` |
| Cleanup | temporary Stage `09` log-only routes for run `stage09-88387e4b` disabled: `updated_routes=2`, status summary `disabled=2` |

The canary helper was executed from the authoritative Mac Studio checkout `/Users/daniildegtyarev/Projects/roehub.com` with `/opt/roehub/app/.venv/bin/python` and production config `/opt/roehub/app/configs/prod/notifications.yaml`. Runtime service/config proof used `/opt/roehub/app`; no git commands were run in the runtime rsync path.

## Artifact Review

Review mode: cold self-review fallback. Independent review was not available in the current environment.

Verdict: Stage `09` is accepted. Fixed blocker: real Telegram canary was not attempted and is recorded as blocked on missing active admin route/user-confirmed recipient; log-only proof is recorded separately. Follow-up check: Stage `10` may start, but must keep direct Strategy Telegram migration behind the accepted `notifications` dispatcher boundary and must not enable real Telegram expansion without recipient approval. Residual risk: real Telegram remains blocked until user-approved recipient readiness is confirmed without exposing secrets.

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `none` | No HTTP route changed. |
| DTO schema | `none` | No API DTO changed. |
| Ports | `compatible-change` | Existing repository protocol now has a production Postgres adapter. |
| Persisted schema | `none` | Uses existing Stage `01` tables; no migration changed. |
| Config/defaults | `compatible-change` | Production notifications dispatcher is enabled in `log_only` mode; dev/test stay disabled. |
| Runtime topology | `compatible-change` | Adds optional Mac Studio launchd service and Prometheus scrape target. |
| External service calls | `compatible-change` | No real Telegram call in `log_only`; readiness check only reports key/route presence. |
| Side effects | `compatible-change` | Canary helper can create bounded synthetic DB event/delivery/attempt rows. |
| Browser-visible behavior | `none` | No UI changed. |
| Ops/alerts | `compatible-change` | Adds dispatcher metrics scrape surface. |

## File Manifest

| Path | Action | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/notifications/adapters/outbound/persistence/postgres/gateway.py` | created | Add notifications SQL gateway. | `compatible-change` adapter |
| `src/trading/contexts/notifications/adapters/outbound/persistence/postgres/notification_repository.py` | created | Implement production `NotificationRepository` over existing tables. | `compatible-change` adapter |
| `src/trading/contexts/notifications/adapters/outbound/persistence/postgres/__init__.py` | created | Export Postgres adapter. | `compatible-change` export |
| `src/trading/contexts/notifications/adapters/outbound/persistence/__init__.py` | modified | Export Postgres adapter. | `compatible-change` export |
| `src/trading/contexts/notifications/adapters/outbound/__init__.py` | modified | Export Postgres adapter. | `compatible-change` export |
| `src/trading/contexts/notifications/adapters/__init__.py` | modified | Export Postgres adapter. | `compatible-change` export |
| `src/trading/contexts/notifications/application/dispatcher.py` | modified | Add provider-mode allowlist guard. | `compatible-change` runtime safety |
| `apps/worker/notification_dispatcher/wiring/modules/notification_dispatcher.py` | modified | Add SQL app wiring, metrics server and DSN presence helpers. | `compatible-change` runtime |
| `apps/worker/notification_dispatcher/main/__init__.py` | created | Add worker package entrypoint. | `compatible-change` runtime |
| `apps/worker/notification_dispatcher/main/main.py` | created | Add process CLI. | `compatible-change` runtime |
| `configs/prod/notifications.yaml` | modified | Enable production dispatcher in `log_only` mode only. | `compatible-change` config |
| `infra/macos/launchd/com.roehub.notification-dispatcher.plist` | created | Add Mac Studio launchd service. | `compatible-change` ops |
| `infra/macos/prometheus/prometheus.prod.yml` | modified | Scrape dispatcher metrics on `127.0.0.1:9210`. | `compatible-change` ops |
| `scripts/macos/bootstrap_native_prod.sh` | modified | Install dispatcher plist. | `compatible-change` ops |
| `scripts/macos/reload_launchd_services.sh` | modified | Reload dispatcher with prod services. | `compatible-change` ops |
| `scripts/notifications/stage09_production_canary.py` | created | Add log-only matrix and real-readiness helper. | `compatible-change` ops proof |
| `tests/unit/apps/worker/test_notification_dispatcher_wiring.py` | modified | Cover DSN presence and provider-mode skip guard. | `none` |
| `tests/unit/contexts/notifications/adapters/test_postgres_notification_repository.py` | created | Cover Postgres adapter mapping/claim behavior. | `none` |
| `docs/runbooks/notifications-admin-alerts.md` | modified | Add Stage `09` production canary commands. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/09-mac-studio-production-canary.md` | created | Stage `09` local report. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` | modified | Record Stage `09` local implementation state. | `none` |
| `docs/architecture/README.md` | modified | Add Stage `09` report to docs index. | `none` |

## Residual Risks

- Real Telegram canary remains blocked until the user confirms recipient scope and a persisted active admin Telegram route exists.
- Production dispatcher is intentionally limited to `log_only`/`fake` deliveries until the real provider canary is approved.
- Telegram bot polling and report scheduler remain non-production workers in this pass; Stage `10` must not depend on them unless added later.
