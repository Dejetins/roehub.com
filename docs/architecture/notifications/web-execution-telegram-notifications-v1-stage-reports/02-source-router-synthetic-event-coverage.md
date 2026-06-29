# Stage 02: Source Router Synthetic Event Coverage

Дата: `2026-06-29`

Статус: `accepted`

Acceptance boundary: Stage `02` добавляет синтетический source router для перевода Strategy, Live Execution, report/stats и admin facts в provider-neutral `NotificationEvent`, route decisions и fake/log delivery candidates. Stage accepted после публикации в `main`: implementation commit `0934dee11c12c70abc52ee3fcfa427ca5d1cd204`, green CI/deploy evidence и `macstudio` checkout/smoke синхронизированы.

## User Required Before Start

Nothing.

Telegram token, admin chat id, smoke password, cookies, exchange credentials или другие secrets не требовались и не запрашивались. Хостовая DB-проверка использовала host-local env source на `macstudio`; DSN/credentials не выводились.

## Checkout And Branch

| Field | Value |
|---|---|
| Checkout path | `/Users/daniildegtyarev/Projects/roehub.com` |
| Branch | `main` |
| Branch/worktree/stash workflow | not created |
| Unrelated dirty work observed | yes: existing `.codex/*`, RL calibration files and untracked market-data docs were present; Stage `02` staging must include only notifications/docs scoped files |

## Scope

Implemented:

- `NotificationSourceRouter` for synthetic source facts, generic `NotificationEvent`, route decisions and fake/log delivery candidates;
- provider-neutral synthetic fact matrix for Strategy, Live Execution, report/stats and admin notification categories;
- in-memory repository adapter for application-level synthetic flow tests;
- user/admin route separation and mode filtering for `critical_only`, `signals`, `trades`, `reports`, `all` and `off`;
- redaction rejection for secret-like synthetic payload keys before event creation.

Not implemented in this stage:

- Telegram provider calls, Telegram token usage, bot polling or inbound commands;
- durable dispatcher leases/retries/unknown/dead-letter behavior;
- real report-run lifecycle or stats query service;
- changes to existing Strategy or Live Execution producer contracts.

## Synthetic Coverage

| Type | Category | Stage `02` synthetic evidence |
|---|---|---|
| Strategy run failed | `strategy_run_failed` | source fact `strategy/failed` -> event -> `critical_only`/`all` user route decision -> log-only delivery/attempt |
| Strategy signal | `strategy_signal` | source fact `strategy/signal` -> event -> `signals`/`all` user route decision -> log-only delivery/attempt |
| Trade fill | `trade_fill` | source fact `live_execution/producer_fill` -> event -> `trades`/`all` user route decision -> log-only delivery/attempt |
| Execution rejected | `execution_rejected` | source fact `live_execution/producer_rejected` -> event -> `critical_only`/`trades`/`all` user route decision -> log-only delivery/attempt |
| Execution terminal | `execution_terminal` | source fact `live_execution/producer_terminal` -> event -> `critical_only`/`trades`/`all` user route decision -> log-only delivery/attempt |
| Execution unknown | `execution_unknown` | source fact `live_execution/producer_unknown` -> event -> user/admin route separation; user modes `critical_only`/`all`, no `trades` leakage |
| Kill switch | `kill_switch` | source fact `live_execution/producer_kill_switch` -> event -> user/admin route separation; user modes `critical_only`/`all`, no `trades` leakage |
| Weekly portfolio report | `portfolio_report` | source fact `notifications/portfolio_weekly` -> event -> `reports`/`all` user route decision -> log-only delivery/attempt |
| Monthly portfolio report | `portfolio_report` | source fact `notifications/portfolio_monthly` -> event -> `reports`/`all` user route decision -> log-only delivery/attempt |
| Day stats command | `stats_response` | source fact `notifications/stats_today` -> event -> `all` user route decision -> log-only delivery/attempt |
| Week stats command | `stats_response` | source fact `notifications/stats_week` -> event -> `all` user route decision -> log-only delivery/attempt |
| Month stats command | `stats_response` | source fact `notifications/stats_month` -> event -> `all` user route decision -> log-only delivery/attempt |
| Strategy stats command | `stats_response` | source fact `notifications/strategy_stats_week` with `scope=strategy` -> event -> `all` user route decision -> log-only delivery/attempt |
| Exchange stats command | `stats_response` | source fact `notifications/exchange_stats_month` with `scope=exchange` -> event -> `all` user route decision -> log-only delivery/attempt |
| Admin critical | `admin_critical` | source fact `ops/admin_critical` -> admin-only event -> admin route decision -> log-only delivery/attempt |
| Admin alert | `admin_alert` | source fact `ops/admin_alert` -> admin-only event -> admin route decision -> log-only delivery/attempt |
| Admin report | `admin_report` | source fact `notifications/admin_report` -> admin-only event -> admin route decision -> log-only delivery/attempt |

Report-run and stats snapshot materialization remain later-stage responsibilities (`05` and `06`). Stage `02` proves synthetic source-to-event routing only.

## Real Boundary Evidence

`macstudio` host-local Postgres smoke executed against a transactional disposable schema:

| Evidence | Result |
|---|---|
| Schema | `stage02_notifications_5a02860e24dd` |
| DDL | Stage `01` migration loaded by file path and applied under disposable `search_path` |
| Rows | user event + admin event, user route + admin route, two deliveries, two attempts |
| Counts | `user_events=1`, `admin_events=1`, `deliveries=2`, `attempts=2` |
| Cleanup | `rollback=ok`; no persistent production table writes |

## Validation

| Check | Result |
|---|---|
| `uv run pytest -q tests/unit/contexts/notifications` | passed: `9 passed` |
| `uv run ruff check src/trading/contexts/notifications tests/unit/contexts/notifications` | passed |
| `uv run pyright src/trading/contexts/notifications tests/unit/contexts/notifications` | passed |
| Real-boundary synthetic flow evidence on `macstudio` disposable schema | passed: user/admin event-route-delivery-attempt rows read back, then rolled back |
| `uv run python -m tools.docs.generate_docs_index --check` | local check failed because the dirty checkout contains unrelated untracked market-data docs; in-memory diff confirmed only those unrelated market-data entries are missing after the scoped Stage `02` README entry |
| GitHub CI `28391601667` for `0934dee11c12c70abc52ee3fcfa427ca5d1cd204` | passed; static, docs-index, migrations and all test shards green |
| GitHub deploy runs for `0934dee11c12c70abc52ee3fcfa427ca5d1cd204` | `Deploy Backend` run `28391849966`, `Deploy Web` run `28391850036`, and `Publish App Image` run `28391850009` passed |
| `macstudio` checkout sync and smoke | `git pull --ff-only origin main` reached `0934dee11c12c70abc52ee3fcfa427ca5d1cd204`; `bash /opt/roehub/app/scripts/macos/smoke_prod.sh` passed |

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public API | `none` | No routes or payloads changed. |
| DTO schema | `none` | No DTOs changed. |
| Ports | `compatible-change` | Adds source-router application surface and in-memory repository adapter; existing port signatures unchanged. |
| Persisted schema | `none` | No new migration or schema change in this stage. |
| Config/defaults | `none` | No config files changed. |
| Source producer contracts | `none` | Strategy and Live Execution code was read only; no producer signatures changed. |
| Service-call semantics | `none` | No external provider calls added. |
| External side effects | `none` | Synthetic delivery uses `log_only`/`fake` candidates only. |
| Logs/metrics/audit/redaction | `compatible-change` | Adds redaction guard coverage before event payload creation. |
| Browser-visible behavior | `none` | No web UI changes. |
| Performance | `unknown` | Dispatcher/backlog hot path is Stage `03`; this stage is in-memory synthetic routing only. |

## File Manifest

| Path | Action | Reason | Contract impact |
|---|---|---|---|
| `src/trading/contexts/notifications/application/source_router.py` | created | Add synthetic source fact to event/route/delivery candidate mapping. | `compatible-change` application surface |
| `src/trading/contexts/notifications/adapters/__init__.py` | created | Export adapter package. | `compatible-change` application surface |
| `src/trading/contexts/notifications/adapters/outbound/__init__.py` | created | Export outbound adapter package. | `compatible-change` application surface |
| `src/trading/contexts/notifications/adapters/outbound/persistence/__init__.py` | created | Export persistence adapter package. | `compatible-change` application surface |
| `src/trading/contexts/notifications/adapters/outbound/persistence/in_memory_notification_repository.py` | created | Add fake/log repository for synthetic flow tests. | `compatible-change` test/support adapter |
| `src/trading/contexts/notifications/application/__init__.py` | modified | Export source-router objects. | `compatible-change` application surface |
| `src/trading/contexts/notifications/__init__.py` | modified | Export source-router objects at context boundary. | `compatible-change` application surface |
| `tests/unit/contexts/notifications/test_source_router.py` | created | Cover full synthetic matrix, route separation, preference modes and redaction rejection. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/02-source-router-synthetic-event-coverage.md` | created | Stage `02` report. | `none` |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` | modified | Record Stage `02` local result and matrix evidence. | `none` |
| `docs/architecture/README.md` | modified | Add Stage `02` report to docs index. | `none` |

## Residual Risks

- Stage `03` must replace fake/log delivery candidates with dispatcher claim/retry/unknown/dead-letter behavior; current attempts are synthetic proof objects only.
- Real Telegram provider, token, chat binding and admin recipient canary are intentionally deferred to later gated stages.
- Report-run and stats data quality semantics are not implemented here; Stage `02` only proves source fact coverage and route decisions.
