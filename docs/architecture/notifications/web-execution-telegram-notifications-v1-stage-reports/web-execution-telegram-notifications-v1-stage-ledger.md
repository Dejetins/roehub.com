# Notifications v1 - журнал выполнения stages

Единый handoff-документ для плана `docs/architecture/notifications/web-execution-telegram-notifications-v1.md`.

## Статус Документа

| Поле | Значение |
|---|---|
| `plan_doc` | `docs/architecture/notifications/web-execution-telegram-notifications-v1.md` |
| `ledger_status` | `active` |
| `current_stage` | `05` |
| `updated_at` | `2026-06-29` |
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
| `00` Baseline and plan freeze | accepted | `00-baseline-and-plan-freeze.md` | docs/current-state/code-contract inventory + docs-only CI + host/runtime smoke audit | Target architecture, provider-neutral contracts, synthetic notification matrix, main-only execution contract and staged prompt pack are published on `main` at `bcb8bf9096c5349a392adab0f4a815abcc850792`; local checkout, `origin/main`, `macstudio` git checkout and production smoke are synchronized. | nothing | none | yes |
| `01` Notifications schema/domain/ports | accepted | `01-notifications-schema-domain-ports.md` | migration/domain/unit tests + transactional Postgres schema smoke + CI/deploy/host sync | Added additive `notification_*` schema, domain objects and `NotificationRepository` port; local full gates, real Postgres transactional schema smoke, GitHub CI/deploy and `macstudio` smoke passed. | nothing | none | yes |
| `02` Source router synthetic event coverage | accepted | `02-source-router-synthetic-event-coverage.md` | unit/integration synthetic matrix + transactional host-local Postgres disposable schema smoke + CI/deploy/host sync | Added synthetic source router, route decisions and fake/log delivery candidates for every matrix row; local gates, `macstudio` disposable-schema DB smoke, GitHub CI/deploy and `macstudio` smoke passed. | nothing | none | yes |
| `03` Dispatcher and provider plug-in contract | accepted | `03-dispatcher-provider-plugin-contract.md` | dispatcher lease/retry/unknown/dead-letter tests + fake/log provider + composition-root smoke + CI/deploy/host sync | Added delivery dispatcher, provider adapter interface, attempts, retries, unknown/dead-letter, suppression and metrics. Telegram adapter stays disabled by default and feature-flagged. | nothing for fake/log provider; host-local Telegram token only for later real canary | none | yes |
| `04` Telegram binding and inbound bot commands | accepted | `04-telegram-binding-inbound-commands.md` | bot update idempotency + command fixtures + API binding tests + synthetic command smoke + CI/deploy/host sync | Added one-time hashed binding code flow, redacted Telegram update mapper, idempotent command handler, account API endpoints and disabled-by-default worker shell. Real Telegram `/start` smoke skipped. | Telegram bot token in host-local env only for optional real provider smoke; synthetic tests require nothing | none | yes |
| `05` Stats query service day/week/month | completed-local | `05-stats-query-service-day-week-month.md` | query fixtures + partial/unavailable assertions + seeded ACL smoke | Added `NotificationStatsQueryService` for portfolio/strategy/exchange day/week/month snapshots, explicit quality status, missing sources, owner filters and Telegram command rendering when injected. | nothing | awaiting commit/CI/deploy/host sync before acceptance | no |
| `06` Weekly/monthly report scheduler | pending | `06-scheduled-reports.md` | scheduler/dedupe/report-run tests | Create weekly/monthly report runs per user route and deliver through dispatcher. | nothing | Stage `05 accepted` | no |
| `07` Admin notifications and ops runbooks | pending | `07-admin-notifications-runbooks.md` | synthetic admin critical/alert/report drill | Add admin route config, admin categories, alert thresholds, metrics and runbooks. | admin recipient route through host-local config or admin UI; do not paste chat ids in docs | Stage `03 accepted`; preferably Stage `06 accepted` for reports | no |
| `08` Web settings UI integration | pending | `08-web-settings-ui-integration.md` | API/browser QA with smoke account | Add Telegram binding status, scoped modes and report schedule to settings UI/API. | smoke account access via existing host-local flow | Stage `04 accepted` and Stage `06 accepted` | no |
| `09` Mac Studio production canary | pending | `09-mac-studio-production-canary.md` | Mac Studio workers + log_only then one test Telegram/admin route | Prove production topology, metrics, redaction, worker health, synthetic matrix, and bounded real Telegram canary. | test Telegram bot token/chat binding/admin route through host-local config/UI | Stage `08 accepted` | no |
| `10` Migrate/deprecate direct Strategy Telegram notifier | pending | `10-strategy-telegram-migration.md` | strategy failure notification parity + fallback toggle | Route Strategy direct Telegram notifications through `notifications`; keep temporary rollback flag. | nothing | Stage `09 accepted` | no |
| `11` Final docs and delivery closure | pending | `11-final-docs-and-main-closure.md` | docs/prompt closure + CI/deploy/readiness evidence | Close docs/runbooks/prompt pack, verify ledger consistency, and record final rollout state. | user sign-off only if expanding beyond smoke/test recipients | Stage `10 accepted` | no |

## Synthetic Notification Coverage Matrix

Every implementation stage that changes event routing or delivery must update this table with exact evidence.

Stage `03` delivery lifecycle evidence applies to every routed delivery candidate in the matrix: `test_dispatcher_claims_pending_delivery_and_marks_sent`, `test_dispatcher_schedules_retry_until_attempt_budget_is_exhausted`, `test_dispatcher_marks_unknown_without_blind_retry`, `test_dispatcher_dead_letters_missing_provider`, and composition-root smoke `stage03_dispatcher_smoke=ok claimed=3 sent=1 unknown=1 dead_letter=1 attempts=3`.

| Type | Category | Required proof | Current status | Evidence |
|---|---|---|---|---|
| Strategy run failed | `strategy_run_failed` | synthetic strategy failure source fact -> event -> route decision -> delivery attempt | covered-stage02-synthetic | `test_synthetic_matrix_routes_every_type_to_fake_log_delivery_candidate`; `test_router_applies_user_preference_modes`: `critical_only`/`all` user route |
| Strategy signal | `strategy_signal` | synthetic `strategy_signals` row or fixture -> event -> signal-mode delivery | covered-stage02-synthetic | `test_synthetic_matrix_routes_every_type_to_fake_log_delivery_candidate`; `test_router_applies_user_preference_modes`: `signals`/`all` user route |
| Trade fill | `trade_fill` | synthetic paper/execution fill source -> event -> trades-mode delivery | covered-stage02-synthetic | `test_synthetic_matrix_routes_every_type_to_fake_log_delivery_candidate`; `test_router_applies_user_preference_modes`: `trades`/`all` user route |
| Execution rejected | `execution_rejected` | synthetic `producer_rejected` source -> warning delivery | covered-stage02-synthetic | `test_synthetic_matrix_routes_every_type_to_fake_log_delivery_candidate`; `test_router_applies_user_preference_modes`: `critical_only`/`trades`/`all` user route |
| Execution terminal | `execution_terminal` | synthetic `producer_terminal` source -> warning delivery | covered-stage02-synthetic | `test_synthetic_matrix_routes_every_type_to_fake_log_delivery_candidate`; `test_router_applies_user_preference_modes`: `critical_only`/`trades`/`all` user route |
| Execution unknown | `execution_unknown` | synthetic `producer_unknown` source -> critical user delivery + admin escalation | covered-stage02-synthetic | `test_synthetic_matrix_routes_every_type_to_fake_log_delivery_candidate`; `test_router_proves_user_admin_route_separation`; `test_router_applies_user_preference_modes`: `critical_only`/`all`, no `trades` leakage |
| Kill switch | `kill_switch` | synthetic `producer_kill_switch` source -> critical user delivery + admin escalation | covered-stage02-synthetic | `test_synthetic_matrix_routes_every_type_to_fake_log_delivery_candidate`; `test_router_proves_user_admin_route_separation`; `test_router_applies_user_preference_modes`: `critical_only`/`all`, no `trades` leakage |
| Weekly portfolio report | `portfolio_report` | report run with weekly period, dedupe key, snapshot quality, delivery | covered-stage02-synthetic | `notifications/portfolio_weekly` source fact -> `reports`/`all` user route; real report-run lifecycle remains Stage `06` |
| Monthly portfolio report | `portfolio_report` | report run with monthly period, dedupe key, snapshot quality, delivery | covered-stage02-synthetic | `notifications/portfolio_monthly` source fact -> `reports`/`all` user route; real report-run lifecycle remains Stage `06` |
| Day stats command | `stats_response` | synthetic `/stats today` update -> command -> stats snapshot -> response delivery | covered-stage05-stats | `test_stats_query_covers_day_week_month_periods_with_complete_quality`; `test_bound_stats_commands_render_stats_service_snapshot` |
| Week stats command | `stats_response` | synthetic `/stats week` update -> command -> stats snapshot -> response delivery | covered-stage05-stats | `test_stats_query_covers_day_week_month_periods_with_complete_quality`; `stage05_stats_smoke=ok` |
| Month stats command | `stats_response` | synthetic `/stats month` update -> command -> stats snapshot -> response delivery | covered-stage05-stats | `test_stats_query_covers_day_week_month_periods_with_complete_quality`; `stage05_stats_smoke=ok` |
| Strategy stats command | `stats_response` | synthetic `/strategy <id> week` update with ownership filter | covered-stage05-stats | `test_strategy_and_exchange_filters_are_owner_scoped`; `test_bound_stats_commands_render_stats_service_snapshot`; foreign owner returns `unavailable` |
| Exchange stats command | `stats_response` | synthetic `/exchange <connection> month` update with ownership filter | covered-stage05-stats | `test_strategy_and_exchange_filters_are_owner_scoped`; `test_bound_stats_commands_render_stats_service_snapshot`; foreign owner rows are filtered out |
| Admin critical | `admin_critical` | synthetic critical ops event -> admin route -> delivery/attempt/metric | covered-stage02-synthetic | `test_synthetic_matrix_routes_every_type_to_fake_log_delivery_candidate`; `test_router_proves_user_admin_route_separation`; host DB smoke inserted/read admin event-route-delivery-attempt |
| Admin alert | `admin_alert` | synthetic warning ops event -> admin route -> delivery/attempt/metric | covered-stage02-synthetic | `test_synthetic_matrix_routes_every_type_to_fake_log_delivery_candidate`; admin-only route category |
| Admin report | `admin_report` | synthetic admin summary report run -> delivery | covered-stage02-synthetic | `notifications/admin_report` source fact -> admin route; real report-run lifecycle remains Stage `06`/`07` |

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
| `00` | `uv run python -m tools.docs.generate_docs_index --check` passed in the original Stage `00` pass; GitHub CI passed for `bcb8bf9096c5349a392adab0f4a815abcc850792` in run `28289331189` | `origin/main`, local checkout and `macstudio` git checkout all resolved to `bcb8bf9096c5349a392adab0f4a815abcc850792`; `Deploy Backend` run `28289627030` succeeded; `bash /opt/roehub/app/scripts/macos/smoke_prod.sh` passed on `macstudio` | accepted | `00-baseline-and-plan-freeze.md` |
| `01` | Focused pytest/ruff/pyright passed; full `uv run ruff check .`, `uv run pyright`, `uv run pytest -q -ra` passed locally; final CI run `28390654585` passed for `14fe0d9a12cdd2e9bc3bf1974de085fc67b2bf63` | `macstudio` host-local Postgres transactional disposable schema smoke created and inspected 6 `notification_*` tables plus indexes/constraints, then rolled back; `macstudio` checkout reached `14fe0d9a12cdd2e9bc3bf1974de085fc67b2bf63` and `smoke_prod.sh` passed | accepted | `01-notifications-schema-domain-ports.md` |
| `02` | `uv run pytest -q tests/unit/contexts/notifications` passed: `9 passed`; `uv run ruff check src/trading/contexts/notifications tests/unit/contexts/notifications` passed; `uv run pyright src/trading/contexts/notifications tests/unit/contexts/notifications` passed; final CI run `28391601667` passed for `0934dee11c12c70abc52ee3fcfa427ca5d1cd204` | `macstudio` host-local Postgres transactional disposable schema `stage02_notifications_5a02860e24dd`: Stage `01` migration applied under disposable `search_path`, user/admin event-route-delivery-attempt rows inserted/read, `rollback=ok`; deploy runs `28391849966`, `28391850036`, `28391850009` passed; `macstudio` checkout reached `0934dee11c12c70abc52ee3fcfa427ca5d1cd204` and `smoke_prod.sh` passed | accepted | `02-source-router-synthetic-event-coverage.md` |
| `03` | `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps` passed: `374 passed, 3 warnings`; ruff and pyright prompt-scope checks passed; final CI run `28392951099` passed for `ad7cd18b140bb4f7cd40436dc6994779bc322591` | composition-root in-memory repository smoke drained backlog through dispatcher providers: `claimed=3`, `sent=1`, `unknown=1`, `dead_letter=1`, `attempts=3`; deploy runs `28393209157`, `28393209162`, `28393220376`, `28393209150` passed; `macstudio` checkout reached `ad7cd18b140bb4f7cd40436dc6994779bc322591` and `smoke_prod.sh` passed | accepted | `03-dispatcher-provider-plugin-contract.md` |
| `04` | `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps/api tests/unit/apps` passed: `244 passed`; `tests/unit/apps/worker` broad worker pass also passed; prompt-scope ruff and pyright passed; final CI run `28394484608` passed for `e98b26d32efa21bf48694d6e7d6911e9822a43fb` | Synthetic binding/command smoke passed: `stage04_telegram_smoke=ok ... code_stored_as_hash=True`; real Telegram `/start` smoke skipped; deploy runs `28394731330`, `28394731324`, `28394731348` passed; `macstudio` checkout reached `e98b26d32efa21bf48694d6e7d6911e9822a43fb` and `smoke_prod.sh` passed | accepted | `04-telegram-binding-inbound-commands.md` |
| `05` | `uv run pytest -q tests/unit/contexts/notifications` passed: `37 passed`; prompt-scope ruff and pyright passed | Seeded ACL stats smoke passed: `stage05_stats_smoke=ok portfolio_quality=complete strategy_quality=complete foreign_quality=unavailable ... owner_filtered=True` | completed-local | `05-stats-query-service-day-week-month.md` |
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
| Stage `01` introduces additive schema and repository contracts. | medium | Require migration/domain tests plus disposable DB or repository migration-harness evidence before acceptance. |

## Change Log

| Date | Stage | Change | Evidence |
|---|---|---|---|
| 2026-06-22 | `00` | Created Notifications v1 plan and stage ledger; fixed provider-neutral architecture, Telegram bot command contract, stats/report scope and synthetic notification matrix. Earlier branch-specific wording was superseded by the 2026-06-27 `main` contract. | `web-execution-telegram-notifications-v1.md`; this ledger |
| 2026-06-22 | superseded branch contract | Historical note only: an earlier branch contract existed and is no longer valid. Use the 2026-06-27 `main` prompt pack contract instead. | superseded by `.codex/agents/generated/web-execution-telegram-notifications-v1/00-main-and-stage-execution-contract.md` |
| 2026-06-27 | main prompt pack | User corrected execution policy: all Notifications v1 work must run on `main`. Updated plan/ledger/prompt contract, added user-presence/access matrix, and prepared full Stage `01`-`11` prompt pack. | `.codex/agents/generated/web-execution-telegram-notifications-v1/`; this ledger |
| 2026-06-27 | `00` delivery audit | Reclassified Stage `00` from `completed-local` to `blocked`: `origin/main` is at `5aad584d069d5020d19775ab24dce333cbeb7801` and docs-only CI passed, but `macstudio` git-checkout sync cannot be safely completed because the remote checkout is behind `origin/main` and `git merge --ff-only --no-commit origin/main` aborts on dirty RL files overlapping `origin/main`. | `gh run view 28288227251`; `gh run view 28288236320`; `ssh macstudio git -C /Users/daniildegtyarev/Projects/roehub.com status --short --branch`; dirty/target path intersection |
| 2026-06-29 | `00` delivery audit refresh | Accepted Stage `00` after the host sync blocker was no longer present: local checkout, `origin/main` and `macstudio` git checkout all resolved to `bcb8bf9096c5349a392adab0f4a815abcc850792`, GitHub CI/deploy were green, and `smoke_prod.sh` passed on `macstudio`. Stage `01` is now allowed. | `gh run view 28289331189`; `gh run view 28289627030`; `ssh macstudio git -C /Users/daniildegtyarev/Projects/roehub.com status --short --branch`; `bash /opt/roehub/app/scripts/macos/smoke_prod.sh` |
| 2026-06-29 | `01` local implementation | Completed local Stage `01` foundation: additive notifications schema, domain objects, repository port and focused tests. Stage remains `completed-local` until main/CI/macstudio delivery evidence is recorded. | `01-notifications-schema-domain-ports.md`; focused pytest/ruff/pyright; `macstudio` transactional schema smoke |
| 2026-06-29 | `01` acceptance | Accepted Stage `01` after local full gates, transactional Postgres schema smoke, code commit `51172c21e3c0f5d8d7f022b2693acb502173fd05`, docs-index fix commit `14fe0d9a12cdd2e9bc3bf1974de085fc67b2bf63`, green final CI and `macstudio` checkout/smoke. Stage `02` is now allowed. | `gh run view 28390654585`; `ssh macstudio git -C /Users/daniildegtyarev/Projects/roehub.com pull --ff-only origin main`; `bash /opt/roehub/app/scripts/macos/smoke_prod.sh` |
| 2026-06-29 | `02` local implementation | Completed local Stage `02`: synthetic source router, in-memory repository adapter, full notification matrix, route separation, preference-mode checks and redaction rejection. Stage remains `completed-local` until commit/CI/macstudio sync evidence is recorded. | `02-source-router-synthetic-event-coverage.md`; `uv run pytest -q tests/unit/contexts/notifications`; focused ruff/pyright; `macstudio` transactional disposable schema smoke |
| 2026-06-29 | `02` acceptance | Accepted Stage `02` after implementation commit `0934dee11c12c70abc52ee3fcfa427ca5d1cd204`, green CI, deploy runs and `macstudio` checkout/smoke. Stage `03` is now allowed. | `gh run view 28391601667`; `gh run view 28391849966`; `gh run view 28391850036`; `gh run view 28391850009`; `bash /opt/roehub/app/scripts/macos/smoke_prod.sh` |
| 2026-06-29 | `03` local implementation | Completed local Stage `03`: delivery dispatcher, provider port, fake/log and gated Telegram provider, safe configs, metrics and composition-root backlog smoke. Stage remains `completed-local` until commit/CI/deploy/`macstudio` sync evidence is recorded. | `03-dispatcher-provider-plugin-contract.md`; `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps`; prompt-scope ruff/pyright; `stage03_dispatcher_smoke=ok` |
| 2026-06-29 | `03` acceptance | Accepted Stage `03` after implementation commit `ad7cd18b140bb4f7cd40436dc6994779bc322591`, green CI, deploy runs and `macstudio` checkout/smoke. Stage `04` is now allowed by ledger; Stage `04` still must satisfy its user-required Telegram token precondition before real-provider work. | `gh run view 28392951099`; `gh run view 28393209157`; `gh run view 28393209162`; `gh run view 28393220376`; `gh run view 28393209150`; `bash /opt/roehub/app/scripts/macos/smoke_prod.sh` |
| 2026-06-29 | `04` local implementation | Completed local Stage `04`: one-time hashed Telegram binding code flow, binding API, redacted inbound update mapper, idempotent command handler, command matrix, fail-closed strategy/exchange scope checks and disabled worker shell. Stage remains `completed-local` until commit/CI/deploy/`macstudio` sync evidence is recorded. | `04-telegram-binding-inbound-commands.md`; `uv run pytest -q tests/unit/contexts/notifications tests/unit/apps/api tests/unit/apps`; prompt-scope ruff/pyright; `stage04_telegram_smoke=ok` |
| 2026-06-29 | `04` acceptance | Accepted Stage `04` after implementation commit `e98b26d32efa21bf48694d6e7d6911e9822a43fb`, green CI, deploy runs and `macstudio` checkout/smoke. Stage `05` is now allowed by ledger; Stage `05` requires no user presence before start. | `gh run view 28394484608`; `gh run view 28394731330`; `gh run view 28394731324`; `gh run view 28394731348`; `bash /opt/roehub/app/scripts/macos/smoke_prod.sh` |
| 2026-06-29 | `05` local implementation | Completed local Stage `05`: stats query service, stats ACL reader, day/week/month windows, complete/partial/unavailable quality behavior, owner filters and Telegram stats command rendering when a stats service is injected. Stage remains `completed-local` until commit/CI/deploy/`macstudio` sync evidence is recorded. | `05-stats-query-service-day-week-month.md`; `uv run pytest -q tests/unit/contexts/notifications`; prompt-scope ruff/pyright; `stage05_stats_smoke=ok` |
