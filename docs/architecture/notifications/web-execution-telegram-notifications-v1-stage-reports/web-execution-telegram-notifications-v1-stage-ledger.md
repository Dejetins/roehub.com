# Notifications v1 - журнал выполнения stages

Единый handoff-документ для плана `docs/architecture/notifications/web-execution-telegram-notifications-v1.md`.

## Статус Документа

| Поле | Значение |
|---|---|
| `plan_doc` | `docs/architecture/notifications/web-execution-telegram-notifications-v1.md` |
| `ledger_status` | `active` |
| `current_stage` | `00` |
| `updated_at` | `2026-06-27` |
| `owner` | `Roehub agents / notifications executors` |
| `branch` | `main` |
| `checkout_path` | `/Users/daniildegtyarev/Projects/roehub.com` |
| `prompt_contract` | `.codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md` |

## Правила Обновления

| Правило | Требование |
|---|---|
| Обязательность | Каждый stage обновляет этот ledger после validation и до финального отчета. |
| Источник фактов | Писать только проверенные факты: tests, runtime calls, DB evidence, browser QA, CI, deploy/smoke или явно помеченные blockers. |
| Статусы | Использовать `pending`, `in_progress`, `completed-local`, `accepted`, `blocked`, `skipped`, `superseded`. |
| Local не accepted | `completed-local` означает, что работа готова в обычном локальном checkout `/Users/daniildegtyarev/Projects/roehub.com` на ветке `main`, но еще не имеет полной delivery/runtime evidence если она применима. `accepted` разрешен только после согласованного delivery path, main evidence и runtime/browser evidence когда они применимы. |
| Tests не acceptance | Unit/integration/static checks обязательны как gates, но non-trivial stage accepted только после real-boundary/e2e evidence по затронутой поверхности. |
| Секреты | Не писать secrets, tokens, cookies, passphrases, ciphertext, raw provider payloads, HMAC, API keys или credentials. Telegram token/chat ids не выводить в docs/logs. |
| Synthetic proof | До готовности всех producer paths использовать синтетические source facts/fixtures на тестовом аккаунте и фиксировать type-by-type evidence. |
| User required before start | Каждый stage до implementation явно фиксирует `User required before start: ...`; если нужны ключи/артефакты/доступы, executor останавливается и не просит secrets в чате. |
| Provider boundary | Реальный Telegram provider включать только в canary stage. До этого использовать `log_only` или fake adapter. |
| Unknown state | Любой provider timeout/ambiguous send фиксировать как `unknown` и не делать blind retry для trade/critical user messages. |
| Mac Studio | Git-команды на `macstudio` только в `/Users/daniildegtyarev/Projects/roehub.com`; runtime checks допускаются в `/opt/roehub/app`. |
| Branch lifecycle | Работа ведется только в обычном checkout `/Users/daniildegtyarev/Projects/roehub.com` на ветке `main`. Не создавать branches, per-stage branches, sibling worktrees, temporary checkouts, local coordination folders или stash-based workflow для этого плана. |
| Prompt contract | Каждый будущий executor prompt для этого плана обязан включать `.codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md` в required context и проверить ветку до чтения широкого контекста или edits. |
| Unrelated dirty work | Если checkout не на `main` или есть unrelated dirty changes, executor не создает branch/worktree/stash workaround; он stage'ит только явно scoped files или сообщает blocker. |
| User presence | Каждый stage обязан зафиксировать `User required before start: ...`; реальные Telegram token/admin recipient/binding/canary действия требуют пользователя только в stages `04`, `07`, `09` и финальном rollout sign-off. |
| File manifest | Каждый stage report обязан фиксировать `Created / Modified / Deleted / Reason / Contract impact`. |
| Docs index | При изменении markdown docs обязательно обновить или проверить `docs/architecture/README.md` через `uv run python -m tools.docs.generate_docs_index --check`. |

## Stage Status

| Stage | Статус | Stage report | Validation depth | Ключевой результат | User required before start | Blocker | Next stage allowed |
|---|---|---|---|---|---|---|---|
| `00` Baseline and plan freeze | blocked | `00-baseline-and-plan-freeze.md` | docs/current-state/code-contract inventory + docs-only CI + host readiness audit | Target architecture, provider-neutral contracts, synthetic notification matrix, main-only execution contract and staged prompt pack are published on `main` at `5aad584d069d5020d19775ab24dce333cbeb7801`, but Stage `00` is not accepted because `macstudio` git-checkout sync is blocked by unrelated dirty RL files that overlap `origin/main`. | nothing before local planning; host-checkout owner action needed for sync | `macstudio` checkout at `/Users/daniildegtyarev/Projects/roehub.com` is behind `origin/main` and `git merge --ff-only --no-commit origin/main` aborts because dirty RL files would be overwritten | no |
| `01` Notifications schema/domain/ports | pending | `01-notifications-schema-domain-ports.md` | migration/domain/unit tests | Add `notifications` bounded context tables, domain objects, repositories and ports without provider side effects. | nothing | Stage `00` delivery/review decision | no |
| `02` Source router synthetic event coverage | pending | `02-source-router-synthetic-event-coverage.md` | unit/integration synthetic matrix | Map strategy/live_execution/admin/report source facts into generic notification events and delivery decisions using `log_only`. | nothing | Stage `01 accepted` | no |
| `03` Dispatcher and provider plug-in contract | pending | `03-dispatcher-provider-plugin-contract.md` | dispatcher lease/retry/unknown tests + fake/log provider | Implement delivery dispatcher, provider adapter interface, attempts, retries, unknown/dead-letter and metrics. Telegram adapter stays feature-flagged. | nothing for fake/log provider; host-local Telegram token only for optional local canary | Stage `02 accepted` | no |
| `04` Telegram binding and inbound bot commands | pending | `04-telegram-binding-inbound-commands.md` | bot update idempotency + command fixtures | Implement one-time binding code, durable Telegram updates, polling worker and `/stats`/settings commands behind safe config. | Telegram bot token in host-local env for real provider smoke; do not paste token in chat | Stage `03 accepted` | no |
| `05` Stats query service day/week/month | pending | `05-stats-query-service-day-week-month.md` | query fixtures + partial/unavailable assertions | Provide portfolio/strategy/exchange stats snapshots for day/week/month with quality status. | nothing | Stage `04 accepted` for bot command integration; query service may be built earlier if isolated | no |
| `06` Weekly/monthly report scheduler | pending | `06-scheduled-reports.md` | scheduler/dedupe/report-run tests | Create weekly/monthly report runs per user route and deliver through dispatcher. | nothing | Stage `05 accepted` | no |
| `07` Admin notifications and ops runbooks | pending | `07-admin-notifications-runbooks.md` | synthetic admin critical/alert/report drill | Add admin route config, admin categories, alert thresholds, metrics and runbooks. | admin recipient route through host-local config or admin UI; do not paste chat ids in docs | Stage `03 accepted`; preferably Stage `06 accepted` for reports | no |
| `08` Web settings UI integration | pending | `08-web-settings-ui-integration.md` | API/browser QA with smoke account | Add Telegram binding status, scoped modes and report schedule to settings UI/API. | smoke account access via existing host-local flow | Stage `04 accepted` and Stage `06 accepted` | no |
| `09` Mac Studio production canary | pending | `09-mac-studio-production-canary.md` | Mac Studio workers + log_only then one test Telegram/admin route | Prove production topology, metrics, redaction, worker health, synthetic matrix, and bounded real Telegram canary. | test Telegram bot token/chat binding/admin route through host-local config/UI | Stage `08 accepted` | no |
| `10` Migrate/deprecate direct Strategy Telegram notifier | pending | `10-strategy-telegram-migration.md` | strategy failure notification parity + fallback toggle | Route Strategy direct Telegram notifications through `notifications`; keep temporary rollback flag. | nothing | Stage `09 accepted` | no |
| `11` Final docs and delivery closure | pending | `11-final-docs-and-main-closure.md` | docs/prompt closure + CI/deploy/readiness evidence | Close docs/runbooks/prompt pack, verify ledger consistency, and record final rollout state. | user sign-off only if expanding beyond smoke/test recipients | Stage `10 accepted` | no |

## Synthetic Notification Coverage Matrix

Every implementation stage that changes event routing or delivery must update this table with exact evidence.

| Type | Category | Required proof | Current status | Evidence |
|---|---|---|---|---|
| Strategy run failed | `strategy_run_failed` | synthetic strategy failure source fact -> event -> route decision -> delivery attempt | planned | TBD |
| Strategy signal | `strategy_signal` | synthetic `strategy_signals` row or fixture -> event -> signal-mode delivery | planned | TBD |
| Trade fill | `trade_fill` | synthetic paper/execution fill source -> event -> trades-mode delivery | planned | TBD |
| Execution rejected | `execution_rejected` | synthetic `producer_rejected` source -> warning delivery | planned | TBD |
| Execution terminal | `execution_terminal` | synthetic `producer_terminal` source -> warning delivery | planned | TBD |
| Execution unknown | `execution_unknown` | synthetic `producer_unknown` source -> critical user delivery + admin escalation | planned | TBD |
| Kill switch | `kill_switch` | synthetic `producer_kill_switch` source -> critical user delivery + admin escalation | planned | TBD |
| Weekly portfolio report | `portfolio_report` | report run with weekly period, dedupe key, snapshot quality, delivery | planned | TBD |
| Monthly portfolio report | `portfolio_report` | report run with monthly period, dedupe key, snapshot quality, delivery | planned | TBD |
| Day stats command | `stats_response` | synthetic `/stats today` update -> command -> stats snapshot -> response delivery | planned | TBD |
| Week stats command | `stats_response` | synthetic `/stats week` update -> command -> stats snapshot -> response delivery | planned | TBD |
| Month stats command | `stats_response` | synthetic `/stats month` update -> command -> stats snapshot -> response delivery | planned | TBD |
| Strategy stats command | `stats_response` | synthetic `/strategy <id> week` update with ownership filter | planned | TBD |
| Exchange stats command | `stats_response` | synthetic `/exchange <connection> month` update with ownership filter | planned | TBD |
| Admin critical | `admin_critical` | synthetic critical ops event -> admin route -> delivery/attempt/metric | planned | TBD |
| Admin alert | `admin_alert` | synthetic warning ops event -> admin route -> delivery/attempt/metric | planned | TBD |
| Admin report | `admin_report` | synthetic admin summary report run -> delivery | planned | TBD |

## Contract Impact Summary

| Surface | Expected classification | Notes |
|---|---|---|
| Public API | `compatible-change` | Additive notification settings, Telegram binding and stats endpoints. Existing account notification DTO remains stable. |
| Ports | `compatible-change` | Additive `notifications` ports and read-only ACLs into identity, strategy, live_execution and account stats. |
| DTO schema | `compatible-change` | New DTOs only unless a future stage explicitly migrates existing DTOs. |
| Persisted schema | `compatible-change` | Additive `notifications_*` tables; existing execution outbox remains source fact table. |
| Config/defaults | `compatible-change` | Additive `notifications.*`; initial default `enabled=false` or `log_only` for safe rollout. |
| External service calls | `compatible-change` | Telegram calls move behind provider adapter with timeout/retry/unknown semantics. |
| Side effects | `compatible-change` | User/admin messages are preference-controlled and canary-gated. Synthetic stages use fake/log provider. |
| Browser-visible behavior | `compatible-change` | Settings UI additions only; existing flows remain. |
| Alerts/runbooks | `compatible-change` | Additive alerting and runbook surfaces. |
| Performance | `unknown` until Stage `03` | Dispatcher/backlog latency and stats query cost need measured gates. |

## Business Impact Summary

| Layer | Expected impact | Notes |
|---|---|---|
| User notifications | Users can opt into Telegram critical-only alerts, signals, trades and reports. | Must be preference-controlled and reversible. |
| User self-service stats | Users can request day/week/month portfolio, strategy and exchange stats from Telegram. | Must show `partial`/`unavailable` instead of invented metrics. |
| Operator/admin alerts | Admins get separate critical/alert/report routing. | Admin route must never reuse user route accidentally. |
| Trading/money boundary | No order submit or exchange credential path is added by notifications. | Telegram commands are read/settings/report commands only in v1. |
| Support/debugging | Delivery state, attempts, retries, unknown and dead letters become inspectable. | Redaction and no-secret evidence remain mandatory. |
| Runtime risk | New workers add operational surface. | Rollout starts with `log_only`, then bounded Telegram canary. |

## Checks And Evidence

| Stage | Local gates | Real-boundary / e2e evidence | Result | Evidence path |
|---|---|---|---|---|
| `00` | `uv run python -m tools.docs.generate_docs_index --check` passed in the original Stage `00` pass; GitHub CI docs-index drift check passed for `5aad584d069d5020d19775ab24dce333cbeb7801` in run `28288227251` | Docs-only CI passed; `Deploy Backend` run `28288236320` skipped the deploy job; `macstudio` git-checkout sync is blocked by unrelated dirty RL files overlapping `origin/main` | blocked | `00-baseline-and-plan-freeze.md` |
| `01` | migration/domain/unit/focused ruff/pyright | DB migration apply/rollback or test DB proof | TBD | Stage report |
| `02` | synthetic router tests for every category in matrix | DB rows for events/routes/deliveries through fake/log provider | TBD | Stage report |
| `03` | dispatcher lease/retry/unknown/dead-letter tests | fake/log provider backlog drain and metrics proof | TBD | Stage report |
| `04` | bot command/update idempotency tests | bounded Telegram binding smoke if token/chat provided through host-local env/UI | TBD | Stage report |
| `05` | stats query fixtures and quality-status tests | query proof against synthetic account fixtures | TBD | Stage report |
| `06` | scheduler/dedupe/report rendering tests | weekly/monthly synthetic report delivery rows | TBD | Stage report |
| `07` | admin event tests and alert-rule checks | admin synthetic critical/alert/report drill | TBD | Stage report |
| `08` | API tests and browser QA | authenticated settings UI proof with smoke account | TBD | Stage report |
| `09` | full focused gates + docs index | Mac Studio worker/metrics/log_only/Telegram canary proof | TBD | Stage report |
| `10` | strategy notification parity tests | Strategy failure event routed through notifications with fallback proof | TBD | Stage report |

## Current Blockers And Risks

| Risk | Severity | Current handling |
|---|---|---|
| Direct Strategy Telegram notifier can drift from new notifications policy. | medium | Keep as fallback until Stage `10`; route new work through notifications. |
| Existing account notification preferences are too coarse. | medium | Keep DTO stable and add scoped routes/preferences in new endpoints/tables. |
| Telegram send unknown state can duplicate messages if blindly retried. | high | Mark unknown, alert admin, require explicit replay for trade/critical messages. |
| Stats sources are incomplete for some modes. | medium | Stats service must report `partial`/`unavailable`; no inferred PnL. |
| Provider secrets can leak through logs/evidence. | high | Redaction rules and stage checks forbid raw token/chat/provider payload output. |
| Admin routing can accidentally notify users. | high | Separate admin recipient kind, route table and synthetic proof. |
| `macstudio` git checkout is behind `origin/main` and dirty on unrelated RL files. | high | Do not fast-forward, reset, stash, or overwrite those files from this prompt pack. Stage `01` remains blocked until the checkout owner resolves/publishes/preserves the RL changes or explicitly accepts a narrower runtime-only sync boundary. |

## Change Log

| Date | Stage | Change | Evidence |
|---|---|---|---|
| 2026-06-22 | `00` | Created Notifications v1 plan and stage ledger; fixed provider-neutral architecture, Telegram bot command contract, stats/report scope and synthetic notification matrix. Earlier branch-specific wording was superseded by the 2026-06-27 `main` contract. | `web-execution-telegram-notifications-v1.md`; this ledger |
| 2026-06-22 | superseded branch contract | Historical note only: an earlier branch contract existed and is no longer valid. Use the 2026-06-27 `main` prompt pack contract instead. | superseded by `.codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md` |
| 2026-06-27 | main prompt pack | User corrected execution policy: all Notifications v1 work must run on `main`. Updated plan/ledger/prompt contract, added user-presence/access matrix, and prepared full Stage `01`-`11` prompt pack. | `.codex/agents/generated/web-execution-telegram-notifications-v1/`; this ledger |
| 2026-06-27 | `00` delivery audit | Reclassified Stage `00` from `completed-local` to `blocked`: `origin/main` is at `5aad584d069d5020d19775ab24dce333cbeb7801` and docs-only CI passed, but `macstudio` git-checkout sync cannot be safely completed because the remote checkout is behind `origin/main` and `git merge --ff-only --no-commit origin/main` aborts on dirty RL files overlapping `origin/main`. | `gh run view 28288227251`; `gh run view 28288236320`; `ssh macstudio git -C /Users/daniildegtyarev/Projects/roehub.com status --short --branch`; dirty/target path intersection |
