# Mainnet Real-Money Trading v1 — журнал выполнения stages

Единый handoff-документ для выполнения плана через prompt pack или Codex Goal mode.

## Статус Документа

| Поле | Значение |
|---|---|
| `plan_doc` | `docs/architecture/live_execution/mainnet-real-money-trading-v1.md` |
| `prompt_pack_dir` | `.codex/agents/generated/mainnet-real-money-trading-v1/` |
| `stage_ledger` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/mainnet-real-money-trading-v1-stage-ledger.md` |
| `execution_mode` | `goal_driven` |
| `ledger_status` | `active` |
| `current_stage` | `00` |
| `updated_at` | `2026-07-03` |
| `owner` | `Roehub agents / implementation executors` |

## Режим Исполнения

| Режим | Правило |
|---|---|
| `goal_driven` | Executor может продолжать stage за stage только пока этот ledger явно разрешает следующий stage. Остановка обязательна при `blocked`, отсутствии evidence, несвязанном `plan_doc` / `prompt_pack_dir` / `stage_ledger` или необходимости user action. |
| `manual_sequential` | Допустим для ручного запуска одного stage; после stage executor обновляет ledger и сообщает следующий prompt. |
| `GOAL.md` | Не является обязательным артефактом; Codex Goal mode исполняется поверх трех linked artifacts. |

## Правила Обновления

| Правило | Требование |
|---|---|
| Обязательность | Каждый stage обновляет этот ledger после validation и до финального отчета. |
| Источник фактов | Писать только проверенные факты: runtime calls, DB evidence, browser QA, Prometheus, Monit, CI/deploy, exchange calls или явно помеченные blockers. |
| Tests не acceptance | `ruff`, `pyright`, `pytest` обязательны для touched code, но real-money stage не принимается без real-boundary evidence. |
| Статусы | Только `pending`, `in_progress`, `accepted`, `blocked`, `skipped`, `superseded`. |
| User required before start | Каждый stage фиксирует, что требуется от пользователя до старта. Если требуется действие пользователя, stage не стартует до подтверждения. |
| Telegram blocker | Ни один stage после `00` не может быть accepted, пока Stage `01` не докажет Telegram host readiness после буквального user confirmation. |
| Money side effects | Real orders допустимы только в stages `06`, `07`, `09` и только при accepted prerequisites, caps, scoped allowlist and canary window. |
| No blind retry | После unknown provider state любой retry запрещен до provider lookup/reconciliation. |
| Secrets | Не писать API keys, secrets, tokens, cookies, passphrases, ciphertext, HMAC, signed payloads, raw Authorization headers, raw sensitive provider payloads. |
| Metrics journal | Все новые mainnet Prometheus metrics должны быть отражены в stage report и `docs/runbooks/prod-dashboard-metrics-reference-ru.md`. |
| Proof boundary | Для Mac Studio использовать labels `target_host_readiness_pre_main`, `read_only_existing_runtime_smoke`, `post_main_production_runtime_proof` или `N/A`. |
| Post-main proof | `post_main_production_runtime_proof` засчитывается только когда target revision уже в `origin/main`, GitHub Actions/CI зеленые, deploy/sync в `/opt/roehub/app` выполнен из verified main checkout и после этого собраны runtime/API/browser/service доказательства. |
| Pre-main proof limits | `target_host_readiness_pre_main` и `read_only_existing_runtime_smoke` могут проверять только текущий host/runtime или prerequisites. Они не доказывают, что changed code работает в production. |
| Branch/worktree/stash | По умолчанию работа на `main`; branch, worktree, stash или отдельная папка допустимы только при explicit user approval. |
| Parallel main | Грязный `main` из параллельных чатов не blocker. Stage владеет только scoped files/hunks. |
| Scoped staging | Запрещены broad staging/unstaging/commit commands. Перед commit/push требуется `git diff --cached --name-status` и marker `ROEHUB_SCOPED_STAGING_REVIEWED=1`. |
| Publish/deploy | Accepted implementation stage должен фиксировать main commit, CI/deploy, Mac Studio sync/runtime proof, если stage меняет runtime/code. |
| Docs index | Если менялись Markdown docs, фиксировать `uv run python -m tools.docs.generate_docs_index --check` или regenerate + check. |

## Business Impact

| Область | Обязательное понимание для executor |
|---|---|
| Money-moving risk | Этот plan впервые разрешает реальные mainnet side effects. Любой неверный submit, retry, futures config или auto-close может привести к реальной потере денег. |
| User trust | UI/alert/report не должны показывать success, если provider state неизвестен, alert не доставлен или reconciliation не matched. |
| Scope containment | Даже после accepted stages mainnet остается включенным только для явно принятых scoped canary surfaces; broad rollout является отдельным планом. |
| Fees/slippage | Canary `<=15 USDT` может создать реальные комиссии, slippage и dust; stage reports должны фиксировать фактические fill/slippage/fee facts без персональных или секретных данных. |

## Service Calls And Redaction Coverage

| Surface | Покрытие в плане и prompts | Redaction / safety rule |
|---|---|---|
| UI/API | `/settings` и `/strategies` readiness, launch, status, kill switch, browser/API proof where stage requires it. | Не писать session cookies, raw credentials, raw form payloads или секретные user identifiers. |
| DB | Source events, intents, orders, fills, reconciliation, risk/caps, notification/outbox evidence. | SQL evidence должен маскировать sensitive ids where possible; raw provider payloads не включать в docs. |
| Redis | Dispatch stream, pending/retry/DLQ/backpressure checks before and after canaries. | Не писать raw signed payloads, tokens or secrets in Redis samples. |
| `exchange-control` | Credential custody remains isolated; `exchange-execution` is the only decrypt/use boundary. | Raw API keys, secrets, passphrases, ciphertext, HMAC and Authorization headers are forbidden in reports/log excerpts. |
| Binance/Bybit | Read-only readiness, futures config read-back, market order submit, status/fill/reconciliation only in allowed stages. | Store only masked connection ids/suffixes and normalized status; no raw provider sensitive payloads. |
| Telegram/user alerts | User-alert delivery proof is mandatory after the host blocker is solved. | Bot tokens, chat ids and raw user notification destinations are forbidden in reports. |
| Metrics/logging | Mainnet metrics must be Prometheus/Grafana-ready and reflected in `docs/runbooks/prod-dashboard-metrics-reference-ru.md`. | No high-cardinality raw order ids, user ids, key suffixes, provider payloads or secrets in metric labels. |

## Stage Status

| Stage | Статус | Prompt / task | Stage report | Validation depth | Proof boundary | Ключевой результат | Blocker | Next stage allowed |
|---|---|---|---|---|---|---|---|---|
| `00` Baseline and hard-block manifest | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/00-baseline-hard-block-manifest.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/00-baseline-hard-block-manifest.md` | docs/runtime inventory | `target_host_readiness_pre_main` for read-only checks, otherwise `N/A` | TBD | none yet | no |
| `01` User prerequisite and Telegram gate | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/01-user-prerequisites-telegram-gate.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/01-user-prerequisites-telegram-gate.md` | runtime/user prerequisite gate | `read_only_existing_runtime_smoke` until changed code exists | TBD | Telegram host access unresolved until user says it is solved and runtime proves it | no |
| `02` Mainnet exchange connections readiness | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/02-mainnet-exchange-connections-readiness.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/02-mainnet-exchange-connections-readiness.md` | exchange read-only + DB/API/browser | `post_main_production_runtime_proof` if code changes; otherwise `read_only_existing_runtime_smoke` | TBD | waits for `01 accepted` and user-provided keys/funding/IP allowlist | no |
| `03` Mainnet risk caps and kill-switch policy | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/03-risk-caps-kill-switch-policy.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/03-risk-caps-kill-switch-policy.md` | API/DB/browser/metrics | `post_main_production_runtime_proof` | TBD | capital allocation manifest unresolved | no |
| `04` Mainnet metrics alerts and user-alert contract | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/04-metrics-alerts-user-alert-contract.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/04-metrics-alerts-user-alert-contract.md` | Prometheus/runbook/notification runtime | `post_main_production_runtime_proof` | TBD | waits for `01 accepted` Telegram readiness | no |
| `05` Mainnet adapter enablement no-submit proof | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/05-mainnet-adapter-enablement-no-submit.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/05-mainnet-adapter-enablement-no-submit.md` | runtime/Redis/DB/metrics | `post_main_production_runtime_proof` | TBD | waits for `02-04 accepted` | no |
| `06` Futures config and market-order guard | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/06-futures-config-market-order-guard.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/06-futures-config-market-order-guard.md` | exchange config read-back + no-submit guard | `post_main_production_runtime_proof` | TBD | waits for user futures config approval and no open orders/positions preflight | no |
| `07` Real mainnet ops canary matrix | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/07-real-mainnet-ops-canary-matrix.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/07-real-mainnet-ops-canary-matrix.md` | real exchange orders + auto-close + reconciliation | `post_main_production_runtime_proof` | TBD | waits for explicit bounded canary window approval | no |
| `08` Strategy producer live-mode enablement | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/08-strategy-producer-live-mode-enablement.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/08-strategy-producer-live-mode-enablement.md` | API/DB/Redis/Monit/Prometheus/browser | `post_main_production_runtime_proof` | TBD | waits for `07 accepted` | no |
| `09` Strategy-driven mainnet canaries | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/09-strategy-driven-mainnet-canaries.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/09-strategy-driven-mainnet-canaries.md` | real strategy signal -> real order/fill/reconcile/alert | `post_main_production_runtime_proof` | TBD | waits for user-approved scoped automatic strategy window | no |
| `10` Closure cleanup and go-no-go record | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/10-closure-cleanup-go-no-go.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/10-closure-cleanup-go-no-go.md` | browser/API/DB/Redis/Prometheus/Monit/docs | `post_main_production_runtime_proof` | TBD | waits for `09 accepted` and zero unsafe residual state | no |

## Что Обязательно Знать Дальше

| Stage | Факт / решение / ограничение | Почему важно следующему stage | Evidence |
|---|---|---|---|
| plan | Mainnet real-money trading is a separate plan, not paper/testnet closure. | Executors must not treat accepted testnet as money-moving approval. | `docs/architecture/live_execution/mainnet-real-money-trading-v1.md` |
| plan | Telegram user alert delivery is mandatory and currently a hard blocker until user confirms host connectivity is solved. | No real-money stage after `00` may be accepted without `01 accepted`. | User clarification 2026-07-03; this ledger. |
| plan | First canary market order cap is `15 USDT`; canary positions must close immediately. | Stages `06`, `07`, `09` must reject anything above cap or without auto-close proof. | `mainnet-real-money-trading-v1.md` |
| plan | Budget statement has ambiguity: `20 USDT` per market but `60 USDT` total. | Stage `03` must create/validate explicit capital allocation manifest before submit. | User clarification 2026-07-03; plan open blockers. |
| plan | Mainnet orders are automatic under supervised agent control, not per-order manual approvals. | Stages must rely on scoped allowlists, caps, kill switch, alerts, and canary windows. | User clarification 2026-07-03. |
| plan | Futures default is isolated `1x`; user may set futures params, but platform config mutation is explicit pre-submit stage only. | Stage `06` owns account config and read-back; order submit must not hidden-auto-config. | User clarification 2026-07-03. |

## Контракты, Миграции И Совместимость

| Stage | API / DTO | Persistence | Config / env | Browser-visible | Ops / runtime | Compatibility / rollback |
|---|---|---|---|---|---|---|
| `00` | none | none | none | none | none | docs-only; no rollback beyond docs |
| `01` | compatible-change if readiness API needed | possible notification/readiness audit | Telegram readiness env presence only; no token output | possible readiness display | Telegram runtime check | blocked until user confirms; no trading |
| `02` | compatible-change to exchange readiness details | account projection/readiness rows | no raw secrets | `/settings` readiness | read-only provider calls | no submit; disable readiness if mismatch |
| `03` | compatible-change to risk/cap APIs | risk/cap/audit tables likely | kill switches/cap config | `/strategies` blocked/ready states | metrics | cap config can be disabled |
| `04` | compatible-change to notification/read models | notification evidence rows | Telegram/provider config readiness | alert status | Prometheus rules/runbooks | keep mainnet disabled if alerts fail |
| `05` | internal API/run-once contract | heartbeat/observation/order guard rows | mainnet-capable adapter mode with submit gate | none | exchange-execution readiness | default gate closed |
| `06` | config command API | futures config audit rows | scoped canary token/config | `/settings` futures config status | exchange config calls | restore user config only by explicit operator action |
| `07` | ops canary endpoint/command | orders/fills/reconciliation/notifications | canary scope env/config | `/strategies` outcome | real mainnet orders | auto-close + kill switch rollback |
| `08` | strategy live mode API/readiness | producer/risk audit rows | `live` allowlist config | `/strategies` live status | producer runtime | disable live mode allowlist |
| `09` | strategy-driven execution | full source->order ledger | scoped canary strategy config | `/strategies` outcome | real mainnet strategy loop | stop run/kill switch/auto-close |
| `10` | none or compatible read model | final evidence rows | mainnet remains scoped | final proof | closure checks | disable all non-production scopes |

## Проверки И Evidence

| Stage | Local gates | Real-boundary / e2e evidence | Proof boundary | Result | Evidence path / note | Tests-only exception | Residual risk |
|---|---|---|---|---|---|---|---|
| `00` | docs index, diff check | read-only current-state inventory | `target_host_readiness_pre_main` if used | TBD | stage report | not allowed | TBD |
| `01` | docs/checks if changed | Telegram readiness + user confirmation | `read_only_existing_runtime_smoke` or `post_main_production_runtime_proof` if code changed | TBD | stage report | not allowed | Telegram host/VPN outside scope |
| `02` | focused gates if code changed | mainnet read-only provider/API/DB/browser | `post_main_production_runtime_proof` if code changed | TBD | stage report | not allowed | provider/account restrictions may differ |
| `03` | focused gates | risk/cap API/DB/browser/metrics | `post_main_production_runtime_proof` | TBD | stage report | not allowed | capital ambiguity until resolved |
| `04` | promtool/docs/tests | Prometheus query + notification runtime/readiness | `post_main_production_runtime_proof` | TBD | stage report | not allowed | Telegram provider still may degrade |
| `05` | focused gates | exchange-execution no-submit proof | `post_main_production_runtime_proof` | TBD | stage report | not allowed | gate misconfig risk |
| `06` | focused gates | real futures config read-back/no-submit | `post_main_production_runtime_proof` | TBD | stage report | not allowed | exchange account mode constraints |
| `07` | focused gates | real mainnet canary orders and close | `post_main_production_runtime_proof` | TBD | stage report | not allowed | real slippage/fee/loss possible |
| `08` | focused gates | producer live allowlist no-broad proof | `post_main_production_runtime_proof` | TBD | stage report | not allowed | broad enablement misconfig risk |
| `09` | focused gates | real strategy signal/order/fill/alert | `post_main_production_runtime_proof` | TBD | stage report | not allowed | signal timing may need bounded wait |
| `10` | docs/gates | no residual orders/positions/unknowns + browser proof | `post_main_production_runtime_proof` | TBD | stage report | not allowed | future expansion remains separate |

## File Manifest

| Stage | Created | Modified | Deleted | Outside expected paths | Outside-path justification | Foreign changes excluded | Mixed files / hunk status |
|---|---|---|---|---|---|---|---|
| `00` | Stage report | Plan/ledger/docs index if needed | none | none expected | N/A | TBD | none |
| `01` | Stage report | notification/runbook/config if needed | none | none expected | N/A | TBD | none |
| `02` | Stage report | exchange readiness code/docs/UI if needed | none | none expected | N/A | TBD | none |
| `03` | Stage report | risk/cap code/docs/UI/config | none | none expected | N/A | TBD | none |
| `04` | Stage report | Prometheus/runbooks/notifications/dashboard docs | none | none expected | N/A | TBD | none |
| `05` | Stage report | exchange-execution/config/runtime docs | none | none expected | N/A | TBD | none |
| `06` | Stage report | futures config command/code/docs | none | none expected | N/A | TBD | none |
| `07` | Stage report | canary harness/docs/runtime configs | none | none expected | N/A | TBD | none |
| `08` | Stage report | strategy producer live mode/UI/docs/config | none | none expected | N/A | TBD | none |
| `09` | Stage report | strategy canary harness/docs/UI evidence | none | none expected | N/A | TBD | none |
| `10` | Final report | plan/ledger/docs index/runbooks | none | none expected | N/A | TBD | none |

## Publish / Deploy Handoff

| Stage | Branch / worktree / stash status | Scoped staging evidence | Commit | PR | Checks before push | Deploy/runtime status | Docs index evidence | Notes |
|---|---|---|---|---|---|---|---|---|
| all | `main`; no branch/worktree/stash unless explicitly approved | `git diff --cached --name-status`: required before any commit | TBD | N/A unless user requests | focused gates + docs index | runtime/code stages require post-main proof | required for Markdown | Use `publish-ci-deploy` only after accepted validation and scoped staging. |

## Blockers

| Stage | Blocker | Severity | Owner / next action | Resolved evidence | Next stage allowed |
|---|---|---|---|---|---|
| `01` | Telegram host access to `api.telegram.org` not confirmed solved by user. | blocker | User resolves host/VLESS outside this plan; executor then proves readiness without secrets. | TBD | no |
| `03` | Capital allocation conflict: `20 USDT` per market and `60 USDT` total do not align across four market surfaces. | blocker | User/operator records explicit allocation manifest. | TBD | no |
| `07` | Real money canary causes fees/slippage and possible loss. | high | User explicitly approves bounded canary window before start. | TBD | no |
| `09` | Strategy-driven real orders are automatic under agent supervision. | high | User explicitly approves scoped automatic strategy window. | TBD | no |

## Change Log

| Date | Stage | Change | Evidence |
|---|---|---|---|
| `2026-07-03` | plan | Created mainnet real-money trading plan and ledger with Telegram hard blocker, `15 USDT` canary cap, market-order-only policy, auto-close and goal-driven execution mode. | `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`; this ledger |
