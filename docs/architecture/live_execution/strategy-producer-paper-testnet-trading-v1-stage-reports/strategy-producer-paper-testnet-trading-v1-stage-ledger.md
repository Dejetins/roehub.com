# Strategy Producer Paper/Testnet Trading v1 — журнал выполнения stages

Единый handoff-документ для плана `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md`.

## Статус Документа

| Поле | Значение |
|---|---|
| `plan_doc` | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md` |
| `prompt_pack` | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/` |
| `ledger_status` | `active` |
| `current_stage` | `03` |
| `updated_at` | `2026-06-17` |
| `owner` | `Roehub agents / implementation executors` |

## Правила Обновления

| Правило | Требование |
|---|---|
| Обязательность | Каждый stage обновляет этот документ после validation и до финального отчета. |
| Источник фактов | Писать только проверенные факты: tests, runtime calls, DB evidence, browser QA, CI, benchmark, deploy/smoke или явно помеченные blockers. |
| Статусы | Использовать только `pending`, `in_progress`, `accepted`, `blocked`, `skipped`, `superseded`. |
| Tests не acceptance | Unit/integration tests, lint, type checks и docs check обязательны как gates, но non-trivial stage accepted только после real-boundary/e2e evidence. |
| Секреты | Не писать secrets, tokens, cookies, passphrases, ciphertext, raw provider payloads, HMAC, API keys или credentials. |
| Следующий stage | Заполнять handoff так, чтобы следующий executor не перечитывал весь чат. |
| Blocked state | Если stage не принят, следующий зависимый stage не стартует, кроме repair/unblock/supersede prompt. |
| Pre-start user requirements | Каждый stage до implementation явно фиксирует `User required before start: ...`; если нужны ключи/артефакты/доступы, executor останавливается и не просит secrets в чате. |
| GitHub publish | После successful validation и перед финальным статусом executor использует `github:yeet`/`publish-ci-deploy` discipline: `gh --version`, `gh auth status`, scope diff, safe stage/commit/push. |
| Branch lifecycle | Ветка допустима только как временная delivery branch. Не создавать per-stage ветку без причины; если branch/PR созданы, они должны быть доставлены в `main`, затем local/remote branch должны быть удалены после доказательства, что `main` содержит изменения. |
| Main delivery | Draft PR, pushed branch или local branch не равны delivery в `main`; stage `accepted` только после evidence, что изменения доставлены в `origin/main` или утвержденный main-branch delivery path. |
| Publish/deploy | Если stage публикуется или деплоится, фиксировать branch/commit/PR/checks/main SHA/branch cleanup/deploy/smoke/host sync. |
| Mac Studio | Git-команды на `macstudio` только в `/Users/daniildegtyarev/Projects/roehub.com`; runtime checks допускаются в `/opt/roehub/app`. |
| Host sync | Для runtime/code stages accepted требует Mac Studio checkout sync evidence и runtime smoke; для docs-only stages нужно явно записать `runtime sync N/A` с причиной. |
| 24h gate | Stage `12` accepted только после фактических 24 часов логируемого наблюдения. |
| File manifest | Каждый stage report обязан фиксировать `Created / Modified / Deleted / Reason / Contract impact`; любой файл вне prompt expected paths должен иметь отдельное объяснение. |

## Stage Status

| Stage | Статус | Prompt / task | Stage report | Validation depth | Ключевой результат | Blocker | Next stage allowed |
|---|---|---|---|---|---|---|---|
| `01` Baseline and handoff freeze | accepted | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/01-baseline-handoff-freeze.md` | `01-baseline-handoff-freeze.md` | target_runtime + browser_runtime | Runtime baseline reconciled with accepted gateway Stage `17`: API, Postgres, Redis, Monit, Prometheus and browser evidence collected; mainnet submit remains blocked; temporary branch delivery was resolved through `main`. | none | yes |
| `02` Backtest-to-strategy launch UI | accepted | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/02-backtest-launch-ui.md` | `02-backtest-launch-ui.md` | browser_runtime + API/DB + CI/deploy/runtime SQL | Backtest variants can launch `paper`/`testnet` strategy profiles through the UI/API with sanitized `$50` BTCUSDT config, provenance/run metadata, fail-closed blocked reasons, main delivery, CI/deploy, Mac Studio sync, smoke, and runtime SQL proof. | none | yes |
| `03` Scenario matrix and compatibility | blocked | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/03-scenario-matrix-compatibility.md` | `03-scenario-matrix-compatibility.md` | API/DB + runtime readiness | Local scenario matrix implementation complete: additive API endpoint, durable SQL table, readiness-backed rows, spot-short unsupported branch, futures-short real-order-capable marker. | Main delivery, CI/deploy, Mac Studio checkout sync, and runtime smoke pending. | no |
| `04` BTCUSDT market readiness | pending | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/04-btcusdt-market-readiness.md` | `04-btcusdt-market-readiness.md` | Redis/ClickHouse/API/browser | TBD | TBD | no |
| `05` Safe testnet exchange binding | pending | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/05-safe-testnet-exchange-binding.md` | `05-safe-testnet-exchange-binding.md` | exchange testnet account reads + DB/API | TBD | TBD | no |
| `06` Supervised strategy producer | pending | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/06-supervised-strategy-producer.md` | `06-supervised-strategy-producer.md` | target_runtime + Monit/Prometheus | TBD | TBD | no |
| `07` Paper full branch coverage | pending | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/07-paper-full-branch-coverage.md` | `07-paper-full-branch-coverage.md` | API/DB/Redis/browser | TBD | TBD | no |
| `08` Manual entry and manual exit | pending | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/08-manual-entry-exit.md` | `08-manual-entry-exit.md` | browser_runtime + execution ledger | TBD | TBD | no |
| `09` Real testnet representative orders | pending | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/09-real-testnet-representative-orders.md` | `09-real-testnet-representative-orders.md` | real testnet exchange + DB/Redis/metrics | TBD | TBD | no |
| `10` Strategy UI status and journal | pending | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/10-strategy-ui-status-journal.md` | `10-strategy-ui-status-journal.md` | browser_runtime + API | TBD | TBD | no |
| `11` Rate limits and load harness | pending | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/11-rate-limits-load-harness.md` | `11-rate-limits-load-harness.md` | load + metrics + Redis | TBD | TBD | no |
| `12` 24h supervised soak | pending | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/12-supervised-24h-soak.md` | `12-supervised-24h-soak.md` | 24h target_runtime | TBD | TBD | no |
| `13` Notifications and operator runbooks | pending | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/13-notifications-runbooks.md` | `13-notifications-runbooks.md` | outbox/API/metrics/runbook drill | TBD | TBD | no |
| `14` Final readiness and docs closure | pending | `.codex/agents/generated/strategy-producer-paper-testnet-trading-v1/14-final-readiness-docs-closure.md` | `14-final-readiness-docs-closure.md` | docs + CI/deploy/readiness | TBD | TBD | no |

## Что Обязательно Знать Дальше

| Stage | Факт / решение / ограничение | Почему важно следующему stage | Evidence |
|---|---|---|---|
| plan | Новый цикл не является Stage `18` старого gateway-плана. Старый gateway используется как accepted foundation. | Executors не должны снова проектировать `exchange-execution`; они должны использовать принятый source-event/risk/dispatch/order/reconciliation path. | `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md` |
| plan | Scope: `paper` + `testnet`, Binance + Bybit, spot + futures, `BTCUSDT`, `$50` на strategy allocation, mainnet вне scope. | Любая попытка mainnet submit или non-BTCUSDT acceptance является blocker. | `strategy-producer-paper-testnet-trading-v1.md` |
| plan | Futures short допускается только как verify-only safe isolated `1x`; auto-config вне scope. | Stage `05`/`09` должны блокировать unknown/mismatch config вместо изменения настроек на бирже. | `strategy-producer-paper-testnet-trading-v1.md` |
| plan | Spot short в v1 является blocked/unsupported branch без отдельного margin/borrow продукта. | Stage `03`/`07`/`09` не должны создавать фейковый real spot-short order; acceptance для этой ветки — корректный block reason. | `strategy-producer-paper-testnet-trading-v1.md` |
| plan | Strategy producer v1 переиспользует `apps/worker/strategy_live_runner`; новый app/process требует доказанного blocker и отдельного architecture update. | Stage `06` не должен молча создать второй producer runtime с другим lifecycle/metrics/ownership. | `strategy-producer-paper-testnet-trading-v1.md` |
| plan | 24h acceptance gate обязателен. | Stage `12` нельзя заменить коротким smoke. | `strategy-producer-paper-testnet-trading-v1.md` |
| plan | Каждый stage до implementation должен явно сказать, требуется ли что-то от пользователя: ключи через UI, market artifacts, SSH/env доступы или `nothing`. | Следующий executor не должен начинать работу с неявным ожиданием пользовательских секретов или внешних артефактов. | `strategy-producer-paper-testnet-trading-v1.md` |
| plan | `github:yeet`/branch/PR используются только как временный delivery path. Draft PR, pushed branch или local branch сами по себе не являются `main`/production delivery. | Stage нельзя принимать как fully delivered, если нет main SHA/CI/deploy/host-sync evidence, branch cleanup evidence если branch использовалась, или явного docs-only `N/A`. | `strategy-producer-paper-testnet-trading-v1.md` |
| `01` | `User required before start: nothing`. | Stage `01` не ждет пользовательских ключей, секретов или артефактов; все evidence собрано через existing SSH/runtime/browser access. | `01-baseline-handoff-freeze.md` |
| `01` | Current Mac Studio checkout `/Users/daniildegtyarev/Projects/roehub.com` is on `main` and matches `origin/main` at `3117fae9` during inventory. | Следующий executor должен отличать runtime baseline от local docs changes until delivery is unblocked. | `01-baseline-handoff-freeze.md` |
| `01` | Old gateway Stage `01` baseline is historical drift: current runtime now has accepted Stage `02`-`17` routes/tables/streams. | Do not use the old "absent live-execution" baseline as current fact; use accepted Stage `17` plus this report. | `01-baseline-handoff-freeze.md`; `live-execution-universal-order-gateway-v1-iteration-ledger.md` |
| `01` | `strategy_live_runner` code/config exists, but no current Mac Studio launchd/Monit process was observed. | Stage `06` owns supervision of the reused producer or must document a blocker before a new process. | `01-baseline-handoff-freeze.md` |
| `01` | Monit reports `roehub_backtest_job_runner` as `Not monitored`, while launchd and Prometheus show it running/up. `backtest-artifact-publisher` remains Prometheus `up=0`. | Treat as current ops inventory/drift, not proof that strategy-producer Stage `01` failed. | `01-baseline-handoff-freeze.md` |
| `01` | Browser/API baseline: `/settings`, `/backtests`, and `/strategies` render authenticated; relevant UI APIs return `200`; disposable smoke session was revoked to active count `0`. | Stage `02` can rely on route availability only after delivery unblock; empty/degraded dashboard states for users without strategies are valid read-model states. | `01-baseline-handoff-freeze.md` |
| `02` | Launch endpoint is intentionally allowlisted to `paper`/`testnet`, `BTCUSDT`, `spot`/`futures`, `fixed_quote`/`fixed_equity_pct`, `single_position_cap`, `long`/`short`; `mainnet` and API secret fields are not accepted by this flow. | Stage `03` should extend compatibility/readiness by preserving these fail-closed defaults instead of broadening the UI/API silently. | `02-backtest-launch-ui.md`; `apps/api/routes/strategies.py` |
| `02` | Testnet launch requires an existing `exchange_connection_id`; spot short is blocked; allocation below `$10` is blocked. | Stage `03`/`05` should treat these as expected branch outcomes until explicit exchange/account readiness work changes them. | `02-backtest-launch-ui.md`; `tests/unit/apps/api/test_strategies_routes.py` |
| `03` | `User required before start: nothing`. | Stage `03` implementation did not require user secrets, keys, external artifacts, or chat-provided credentials. | `03-scenario-matrix-compatibility.md` |
| `03` | Scenario matrix rows are discovered from current contracts: `paper`/`testnet`, `LiveStrategyProfileSizingMethod`, `single_position_cap`, and `canonical_variant_params.execution.direction_mode`; for `long_short_reversal` this yields `8` rows per source market type. | Stage `04`/`05`/`07` should consume these rows instead of inventing new scenario branches. | `03-scenario-matrix-compatibility.md`; `src/trading/contexts/strategy/application/use_cases/scenario_matrix.py` |
| `03` | Testnet spot-short is explicitly `blocked`/`unsupported` with `spot_short_not_supported`; futures short is `real_order_capable` but still launch-blocked by `exchange_connection_required` until exchange binding and isolated `1x` guard are proven. | Stage `05` and `09` must not fake spot-short real orders and must verify futures account/config before submit. | `03-scenario-matrix-compatibility.md`; `tests/unit/contexts/strategy/application/test_strategy_use_cases.py` |

## Контракты, Миграции И Совместимость

| Stage | API / DTO | Persistence | Config / env | Browser-visible | Ops / runtime | Compatibility / rollback |
|---|---|---|---|---|---|---|
| `01` | none unless drift found | none | inventory only | inventory only | inventory only | no runtime changes expected |
| `02` | compatible-change launch DTO/read model | possible additive launch/profile rows | feature flags only | launch UI changes | no new service | disable launch UI/route flag |
| `03` | compatible-change matrix/readiness DTO | additive matrix evidence | none or feature flag | matrix/readiness display | no new service | matrix can be ignored without data loss |
| `04` | compatible-change readiness fields | additive readiness/provisioning rows | BTCUSDT config | readiness display | market-data dependency | disable launch when readiness unavailable |
| `05` | compatible-change exchange binding/readiness | additive projection/config evidence | testnet-only guards | exchange readiness display | exchange-control/account reads | fail closed, no auto-config rollback needed |
| `06` | compatible-change producer controls | heartbeat/checkpoint rows if needed | service env/admin switch/allowlists | producer status | new supervised service | stop service + admin switch off |
| `07` | compatible-change paper coverage views | additive paper/accounting/coverage rows | paper mode config | paper outcomes | no exchange submit | disable paper producer, retain ledger |
| `08` | compatible-change manual source endpoints | source/intent/order/outbox rows | manual action flags | manual buttons | uses existing execution path | disable manual action flags |
| `09` | compatible-change testnet scenario controls | orders/fills/reconciliation rows | testnet scenario flags | testnet outcomes | real testnet exchange calls | stop producer, cancel/close testnet positions |
| `10` | compatible-change UI read models | none/additive read evidence | none | dashboard/journal changes | no new service | revert UI/read model fields |
| `11` | compatible-change load endpoints/tools | load run evidence rows | limiter config | possible status display | load harness/metrics | stop harness, lower limits |
| `12` | none/compatible evidence APIs | soak evidence rows/logs | soak config | final status page/report | 24h runtime | stop services, archive soak evidence |
| `13` | compatible-change notification read models | outbox/event rows | alert/runbook config | notification status | Prometheus/Monit | disable alert rules/outbox delivery hooks |
| `14` | docs only unless drift repair | docs only unless drift repair | docs only | docs only | CI/deploy evidence | docs rollback via git |

## Проверки И Evidence

| Stage | Local gates | Real-boundary / e2e evidence | Result | Evidence path / note | Tests-only exception | Residual risk |
|---|---|---|---|---|---|---|
| `01` | `python -m tools.docs.generate_docs_index --check` passed after docs index regeneration | SSH/API/SQL/Redis/Monit/Prometheus/browser inventory collected on Mac Studio/public Roehub | accepted | `01-baseline-handoff-freeze.md` | none | Docs-only runtime sync is `N/A`; Mac Studio git checkout sync is required after main push. |
| `02` | `uv run ruff check apps/api apps/web src/trading/contexts/strategy tests`; `uv run pyright apps/api src/trading/contexts/strategy tests`; `uv run pytest -q tests/unit/apps/api tests/unit/apps/web tests/unit/contexts/strategy`; docs index check all passed | Playwright success flow passed locally with launch POST `200`, redirect `/strategies`, dashboard `200`, console `0`; blocked testnet/no-exchange modal displayed `exchange_connection_required`; CI, Deploy Backend, Publish App Image, and Deploy Web succeeded; Mac Studio checkout fast-forwarded to `762ef6cb`; `scripts/macos/smoke_prod.sh` exited `0`; runtime DB has Alembic `20260617_0031` and `testnet` in both strategy mode constraints | accepted | `02-backtest-launch-ui.md`; screenshots under `output/playwright/backtest-launch-ui-*.png`; CI run `27649915820`; deploy runs `27650024744`, `27650024742`, `27650101271` | none | Stage `03` must still prove scenario compatibility and exchange readiness separately; this stage only launches sanitized paper/testnet configs. |
| `03` | `python -m compileall -q ...`; focused scenario/API/migration tests; `IDENTITY_FAIL_FAST=false uv run pytest -q tests/unit/contexts/strategy tests/unit/contexts/backtest tests/unit/apps` (`753 passed, 3 warnings`); `uv run ruff check src/trading/contexts/strategy src/trading/contexts/backtest apps tests`; `uv run pyright src/trading/contexts/strategy src/trading/contexts/backtest apps tests` | FastAPI TestClient route call against `/top` public variant key; SQL migration contract; compatibility/readiness service calls inside matrix service | blocked on delivery | `03-scenario-matrix-compatibility.md` | none | Default local broad pytest fails when `IDENTITY_FAIL_FAST=true` and `KEYCLOAK_BASE_URL` is absent; main delivery/CI/deploy/Mac Studio smoke still pending. |
| `04` | focused readiness tests | Redis/ClickHouse/API/browser readiness for BTCUSDT | TBD | Stage report | none | TBD |
| `05` | focused exchange guard tests | Binance/Bybit testnet account/config reads + block proof | TBD | Stage report | none | TBD |
| `06` | service tests, lint, type check | launchd/Monit/Prometheus/health/restart proof | TBD | Stage report | none | TBD |
| `07` | paper/accounting tests | every matrix row through paper API/DB/Redis/browser | TBD | Stage report | none | TBD |
| `08` | API/UI/idempotency tests | Playwright manual entry/exit + ledger proof | TBD | Stage report | none | TBD |
| `09` | adapter/scenario tests | real testnet orders/fills/status/reconciliation | TBD | Stage report | none | TBD |
| `10` | UI/API tests, node check | Playwright desktop/mobile, network/console/DOM secret scan | TBD | Stage report | none | TBD |
| `11` | load harness unit tests | controlled testnet-mode load metrics, Redis lag, rate limiter evidence | TBD | Stage report | none | TBD |
| `12` | smoke before soak | 24h logged runtime evidence | TBD | Stage report | none | TBD |
| `13` | alert/runbook/outbox tests | outbox rows, Prometheus rule check, runbook drill | TBD | Stage report | none | TBD |
| `14` | docs index, final gates | CI/deploy/readiness if code/docs delivered | TBD | Stage report | none | TBD |

## Publish / Deploy Handoff

| Stage | Branch | Commit | PR | Checks before push | Deploy/runtime status | Notes |
|---|---|---|---|---|---|---|
| `01` | temporary `codex/stage01-baseline-handoff-freeze`, then `main` | branch content fast-forwarded into `main`; final pushed main SHA in executor final report | `https://github.com/Dejetins/roehub.com/pull/28` superseded by direct main delivery | docs index passed; inventory collected; branch cleanup required after push | runtime sync `N/A` docs-only; Mac Studio git checkout sync required | No code/runtime changes. Stage accepted only after branch cleanup evidence is reported. |
| `02` | `main` | `762ef6cbdc95b8f0b969cdb20cef5e7dfb6300a0` (`Implement backtest launch UI`) | none | local gates, Playwright browser proof, `gh auth status`, CI run `27649915820` | Deploy Backend `27650024744`, Publish App Image `27650024742`, final Deploy Web `27650101271`, Mac Studio checkout sync to `762ef6cb`, prod smoke `0`, runtime SQL Alembic `20260617_0031` | Direct main delivery after local validation. Stage accepted. |
| `03` | local `main` worktree | pending | pending | local gates and docs-index check complete; `github:yeet`/`publish-ci-deploy` preflight pending | pending main delivery, CI/deploy, Mac Studio checkout sync, runtime smoke | Stage remains blocked until prompt-required delivery evidence is recorded. |
| `04` | TBD | TBD | TBD | local gates + readiness probes + `github:yeet` preflight | TBD | Main delivery evidence required before accepted. |
| `05` | TBD | TBD | TBD | local gates + testnet account reads + `github:yeet` preflight | TBD | User supplies keys through UI, not prompt/docs; main delivery evidence required before accepted. |
| `06` | TBD | TBD | TBD | local gates + service smoke + `github:yeet` preflight | TBD | launchd/Monit and Mac Studio host-sync evidence required. |
| `07` | TBD | TBD | TBD | local gates + paper e2e + `github:yeet` preflight | TBD | Full matrix proof and main delivery evidence required. |
| `08` | TBD | TBD | TBD | local gates + browser/manual e2e + `github:yeet` preflight | TBD | Manual action proof and main delivery evidence required. |
| `09` | TBD | TBD | TBD | local gates + testnet matrix + `github:yeet` preflight | TBD | Real testnet orders; host-sync/runtime evidence required. |
| `10` | TBD | TBD | TBD | local gates + browser QA + `github:yeet` preflight | TBD | UI status/journal and main delivery evidence required. |
| `11` | TBD | TBD | TBD | local gates + load run + `github:yeet` preflight | TBD | Metrics and host-sync/runtime evidence required. |
| `12` | TBD | TBD | TBD | pre-soak smoke + `github:yeet` preflight | TBD | 24h gate and host-sync/runtime evidence required. |
| `13` | TBD | TBD | TBD | local gates + alert/runbook checks + `github:yeet` preflight | TBD | Outbox delivery channels out of scope; main delivery evidence still required. |
| `14` | TBD | TBD | TBD | docs/gates/CI + `github:yeet` preflight | TBD | Final closure requires main delivery and host-sync status for the cycle. |

## Blockers

| Stage | Blocker | Severity | Owner / next action | Resolved evidence | Next stage allowed |
|---|---|---|---|---|---|
| `03` | Main delivery, CI/deploy, Mac Studio checkout sync, and runtime smoke not yet recorded for the code/runtime stage. | blocker | Stage `03` executor: run `github:yeet`/`publish-ci-deploy` discipline after final local docs-index check, without staging unrelated pre-existing `docs/architecture/ml/` changes. | pending | no |

## Change Log

| Date | Stage | Change | Evidence |
|---|---|---|---|
| 2026-06-16 | plan | Created separate plan and ledger for paper/testnet strategy producer cycle. | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md` |
| 2026-06-16 | plan | Added mandatory pre-start user requirement disclosure and delivery contract: `github:yeet` publish, main-branch evidence, and Mac Studio host-sync evidence before `accepted`. | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1.md` |
| 2026-06-16 | `01` | Added Stage `01` baseline handoff report and marked the stage `blocked` on main delivery, not on runtime evidence. | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/01-baseline-handoff-freeze.md` |
| 2026-06-16 | `01` | Regenerated architecture docs index for the new Stage `01` report. | `docs/architecture/README.md` |
| 2026-06-16 | `01` | Opened draft PR `https://github.com/Dejetins/roehub.com/pull/28`; stage remains blocked until main delivery evidence exists. | `codex/stage01-baseline-handoff-freeze` |
| 2026-06-16 | `01` | Updated delivery contract so temporary branches/PRs must be delivered to `main` and deleted before acceptance; fast-forwarded Stage `01` branch content into `main` and marked Stage `01` accepted pending final push/cleanup evidence in the executor report. | `01-baseline-handoff-freeze.md`; prompt pack `01`-`14` |
| 2026-06-17 | `02` | Started Stage `02`; recorded `User required before start: nothing` and narrowed the concrete planned file list before implementation edits. | `02-backtest-launch-ui.md` |
| 2026-06-17 | `02` | Implemented local backtest-to-strategy launch UI/API path and collected local gates + Playwright success/blocked evidence; marked stage blocked on GitHub delivery prerequisite because `gh auth status` timed out on keyring. | `02-backtest-launch-ui.md`; `gh auth status` |
| 2026-06-17 | `02` | Unblocked GitHub auth, committed and pushed `762ef6cb` to `main`; CI, backend deploy, app image publish, and web deploy succeeded; Mac Studio checkout/runtime smoke and production SQL constraint proof collected; marked Stage `02` accepted and opened Stage `03`. | `02-backtest-launch-ui.md`; CI `27649915820`; Deploy Backend `27650024744`; Publish App Image `27650024742`; Deploy Web `27650101271` |
| 2026-06-17 | `03` | Started Stage `03`; recorded `User required before start: nothing`, verified Stage `02` accepted, and narrowed expected broad paths to a concrete file list before implementation edits. | `03-scenario-matrix-compatibility.md` |
| 2026-06-17 | `03` | Implemented local scenario matrix API/use case/persistence/migration/tests and marked Stage `03` blocked on prompt-required main delivery, CI/deploy, Mac Studio checkout sync, and runtime smoke. | `03-scenario-matrix-compatibility.md`; `src/trading/contexts/strategy/application/use_cases/scenario_matrix.py`; `alembic/versions/20260617_0032_strategy_variant_scenario_matrix_v1.py` |
