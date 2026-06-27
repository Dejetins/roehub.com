# Stage 00: Baseline And Plan Freeze

Дата: `2026-06-22`

Статус: `completed-local`

Acceptance boundary: этот отчет фиксирует локально готовый architecture/stage plan на `main`. Это не production-runtime acceptance для будущих code stages.

## User Required Before Start

Nothing.

Stage `00` является docs-only планированием. Telegram token, chat id, admin recipient, Keycloak password, exchange credentials или другие секреты не нужны и не запрашивались.

## Scope

Stage `00` зафиксировал:

- отдельный bounded context `notifications`;
- provider-neutral delivery contract;
- Telegram bot binding and command contract;
- user modes: critical-only, signals, trades, reports and all;
- admin critical/alert/report routing;
- day/week/month stats and strategy/exchange filters;
- weekly/monthly portfolio reports;
- synthetic notification matrix for future implementation stages;
- contract-impact baseline and stage ledger;
- main-only prompt execution contract for all future Notifications v1 stages.

## Main And Prompt Execution Contract

All future Notifications v1 work must use:

- checkout: `/Users/daniildegtyarev/Projects/roehub.com`;
- branch: `main`;
- prompt contract: `.codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md`.

No future stage prompt may create a branch, per-stage branch, sibling worktree, temporary checkout, local coordination folder, or stash-based workflow by default. If a future executor finds unrelated dirty work in the checkout, it must stage only the explicitly scoped files or report the blocker.

## Current-State Facts

| Area | Observed fact | Gap closed by plan |
|---|---|---|
| Identity | Confirmed Telegram channel storage exists through `identity_telegram_channels`; current account settings have coarse integration/preferences modes. | Plan adds binding-code flow, scoped routes and report schedule without breaking current DTOs. |
| Strategy | Strategy-specific Telegram notifier exists and prod config can use Telegram; current live runner only publishes failure notification through this path. | Plan moves delivery behind notifications dispatcher and keeps Strategy direct path as temporary fallback until migration stage. |
| Live Execution | Execution notification outbox exists for producer rejected/fill/unknown/kill-switch/terminal, with redacted labels and UI listing. | Plan treats it as source facts and adds separate provider delivery queue/attempt lifecycle. |
| Stats | Signals, paper accounting, execution orders/fills/funding and exchange snapshots exist as source ledgers. | Plan adds stats query service with `complete/partial/unavailable` quality state instead of inferred PnL. |
| Admin alerts | Metrics/logs exist across workers, but admin notification recipients and escalation are not first-class. | Plan adds admin recipient kind, categories, route config and synthetic drills. |

## Evidence Sources

| Area | Source paths |
|---|---|
| Identity Telegram binding | `migrations/postgres/0001_identity_v1.sql`; `src/trading/contexts/strategy/adapters/outbound/acl/identity/confirmed_telegram_chat_binding_resolver.py` |
| Account notification settings | `migrations/postgres/0006_identity_account_settings_v1.sql`; `apps/api/routes/ui_account.py`; `apps/api/dto/ui_account.py` |
| Strategy Telegram path | `src/trading/contexts/strategy/application/ports/telegram_notifier.py`; `src/trading/contexts/strategy/application/services/telegram_notification_policy.py`; `src/trading/contexts/strategy/adapters/outbound/messaging/telegram/telegram_bot_api_notifier.py`; `src/trading/contexts/strategy/application/services/live_runner.py`; `apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py`; `configs/prod/strategy.yaml`; `configs/dev/strategy.yaml` |
| Live Execution notification outbox | `src/trading/contexts/live_execution/domain/notification.py`; `src/trading/contexts/live_execution/application/use_cases/execution_ingress.py`; `src/trading/contexts/live_execution/application/ports/execution_intent_repository.py`; `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/execution_intent_repository.py`; `alembic/versions/20260603_0030_execution_notifications_producers_v1.py`; `apps/api/routes/ui_execution.py`; `apps/api/dto/ui_execution.py` |
| Stats source ledgers | `alembic/versions/20260531_0018_strategy_signals_v1.py`; `alembic/versions/20260531_0022_capital_reservation_paper_accounting_v1.py`; `alembic/versions/20260531_0027_testnet_order_adapters_v1.py`; `alembic/versions/20260602_0029_execution_reconciliation_pitr_v1.py`; `alembic/versions/20260531_0020_exchange_account_projection_config_guard_v1.py` |

## Business Impact

| Layer | Impact |
|---|---|
| User value | Users can receive execution/signal/risk notifications and request portfolio stats from Telegram after implementation stages. |
| Operator value | Admins get critical alerts and reports through the same audited delivery machinery, separate from user routes. |
| Release safety | New context decouples producer code from Telegram and allows `log_only` synthetic stages before real provider canary. |
| Supportability | Delivery attempts, unknown state, dead letters and route decisions become inspectable instead of best-effort-only logs. |
| Money safety | Stage `00` opens no exchange path and changes no trading execution behavior. Future stages must not allow Telegram to submit orders. |

## Architecture Decisions

| Decision | Rationale |
|---|---|
| Create bounded context `notifications`. | Strategy/live_execution/ML/ops should publish facts, not know provider-specific delivery. |
| Keep `execution_notification_outbox` as source input, not final delivery queue. | Its existing semantics are execution-domain notification facts, not provider attempt lifecycle. |
| Start with polling Telegram bot worker, webhook later. | Polling avoids public route/webhook rollout before command/storage contracts are proven. |
| Use `log_only`/fake provider for synthetic stages. | Lets every notification type be tested without a real Telegram send or token exposure. |
| Treat Telegram send timeout as `unknown`. | Telegram `sendMessage` does not provide a platform idempotency key; blind retry can duplicate trade/critical messages. |
| Preserve current `/ui/account/notifications` contract. | Existing web/API consumers should not break while scoped preferences are added. |

## Synthetic Test Matrix

The future implementation ledger must record a proof row for each type:

- `strategy_run_failed`;
- `strategy_signal`;
- `trade_fill`;
- `execution_rejected`;
- `execution_terminal`;
- `execution_unknown`;
- `kill_switch`;
- weekly `portfolio_report`;
- monthly `portfolio_report`;
- day/week/month `stats_response`;
- strategy-scoped stats response;
- exchange-scoped stats response;
- `admin_critical`;
- `admin_alert`;
- `admin_report`.

Required evidence per type:

- source fact or synthetic fixture;
- normalized notification event;
- route/preference decision;
- delivery row;
- delivery attempt;
- provider result through fake/log or canary Telegram adapter;
- metrics counter;
- redaction check;
- final status.

## Contract Impact

| Surface | Classification | Stage `00` fact |
|---|---|---|
| Public API | `none` now; planned `compatible-change` | No API code changed in Stage `00`. Future stages add endpoints. |
| DTO schema | `none` now; planned `compatible-change` | No DTO code changed. |
| Ports | `none` now; planned `compatible-change` | New ports are documented only. |
| Persisted schema | `none` now; planned `compatible-change` | No migrations in Stage `00`. |
| Config/defaults | `none` now; planned `compatible-change` | No config files changed. |
| External service calls | `none` now; planned `compatible-change` | No Telegram call was made. |
| Side effects | `none` | Docs-only. |
| Browser-visible behavior | `none` now; planned `compatible-change` | No web UI changed. |
| Logs/metrics/alerts | `none` now; planned `compatible-change` | Metrics/alerts are planned only. |
| Performance | `none` now; future `unknown` until Stage `03`/`05` measurement | Docs-only. |

## Validation

| Check | Result | Notes |
|---|---|---|
| Current-state code/docs inventory | passed | Identity, Strategy Telegram, Live Execution notification outbox and stats ledgers were inspected before writing the plan. |
| Docs index | passed | `uv run python -m tools.docs.generate_docs_index` updated `docs/architecture/README.md`; `uv run python -m tools.docs.generate_docs_index --check` passed. |
| Cold-head review | passed after fixes | `architecture-review/references/cold-head-plan-prompt-pack-review.md`, cold self-review fallback. |
| Runtime sync | N/A | Docs-only local branch. |
| Browser QA | N/A | No browser-visible code changed. |

## Cold-Head Review Receipt

Cold-head review: completed
Mode: cold self-review fallback
Review scope: `docs/architecture/notifications/web-execution-telegram-notifications-v1.md`, `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md`, `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/00-baseline-and-plan-freeze.md`, `.codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md`, `.codex/agents/generated/web-execution-telegram-notifications-v1/01-notifications-schema-domain-ports.md`..`11-final-docs-and-main-closure.md`, `.codex/PLANS.md`, `docs/architecture/README.md`
Review instructions: architecture-review/references/cold-head-plan-prompt-pack-review.md
Verdict: Release after fixes
Blockers fixed: added source-path evidence for current-state claims, updated Stage `00` validation and file manifest from draft/planned to actual local results, replaced the earlier branch contract with mandatory `main` execution, added access/user-presence matrix and full Stage `01`-`11` prompt pack.
Local follow-up check: completed
Residual risks: schema/table names, stats query cost, host-local Telegram canary setup, Strategy direct notifier migration and final product rollout approval remain future-stage risks.

## File Manifest

| Path | Action | Reason | Contract impact |
|---|---|---|---|
| `docs/architecture/notifications/web-execution-telegram-notifications-v1.md` | created | Architecture plan for notifications bounded context. | `none` now, planned compatible changes. |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/web-execution-telegram-notifications-v1-stage-ledger.md` | created | Stage ledger and synthetic notification proof matrix. | `none` now, planned compatible changes. |
| `docs/architecture/notifications/web-execution-telegram-notifications-v1-stage-reports/00-baseline-and-plan-freeze.md` | created | Stage `00` local report. | `none`. |
| `.codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md` | created | Mandatory main-branch prompt execution/access contract for future stages. | `none`. |
| `.codex/PLANS.md` | modified | Add compact active workstream checkpoint. | `none`. |
| `docs/architecture/README.md` | modified | Regenerated architecture docs index. | `none`. |

## Residual Risks

- The first code stage still needs exact table/index naming and migration design.
- Stats quality depends on source ledger completeness; unavailable fields must stay explicit.
- Telegram provider canary needs host-local token/admin route setup without exposing secrets.
- Strategy direct Telegram path must remain controlled until migration Stage `10`.
