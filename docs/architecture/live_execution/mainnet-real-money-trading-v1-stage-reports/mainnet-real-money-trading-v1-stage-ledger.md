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
| Money side effects | Real orders допустимы только в stages `10`-`15` и `18`, только при accepted prerequisites, caps, scoped allowlist, one-shot approval and canary window. |
| No blind retry | После unknown provider state любой retry запрещен до provider lookup/reconciliation. |
| Auto-close | Market canary close requires durable close order evidence. `cancel_after_submit` is not accepted as close evidence. |
| Secrets | Не писать API keys, secrets, tokens, cookies, passphrases, ciphertext, HMAC, signed payloads, raw Authorization headers, raw sensitive provider payloads. |
| Metrics journal | Все новые mainnet Prometheus metrics должны быть отражены в stage report и `docs/runbooks/prod-dashboard-metrics-reference-ru.md`. Histogram base names use seconds and do not include `_bucket`. |
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
| Money-moving risk | Этот plan впервые разрешает реальные mainnet side effects. Любой неверный submit, retry, futures config, close order или kill-switch polarity может привести к реальной потере денег. |
| User trust | UI/alert/report не должны показывать success, если provider state неизвестен, alert не доставлен или reconciliation не matched. |
| Scope containment | Даже после accepted stages mainnet остается включенным только для явно принятых scoped canary surfaces; broad rollout является отдельным планом. |
| Fees/slippage | Canary `<=15 USDT` может создать реальные комиссии, slippage и dust; stage reports должны фиксировать фактические fill/slippage/fee facts без персональных или секретных данных. |

## Service Calls And Redaction Coverage

| Surface | Покрытие в плане и prompts | Redaction / safety rule |
|---|---|---|
| UI/API | `/settings` и `/strategies` readiness, launch, status, kill switch, browser/API proof where stage requires it. | Не писать session cookies, raw credentials, raw form payloads или секретные user identifiers. |
| DB | Source events, intents, order pairs, fills, reconciliation, risk/caps, approval tokens, notification/outbox evidence. | SQL evidence должен маскировать sensitive ids where possible; raw provider payloads не включать в docs. |
| Redis | Dispatch stream, pending/retry/DLQ/backpressure checks before and after canaries. | Не писать raw signed payloads, tokens or secrets in Redis samples. |
| `exchange-control` | Credential custody remains isolated; `exchange-execution` is the only decrypt/use boundary. | Raw API keys, secrets, passphrases, ciphertext, HMAC and Authorization headers are forbidden in reports/log excerpts. |
| Binance/Bybit REST | Read-only readiness, futures config read-back, market submit, close submit, status/fill/reconciliation only in allowed stages. | Store only masked connection ids/suffixes and normalized status; no raw provider sensitive payloads. |
| Binance/Bybit private stream or REST fallback | Order/fill/position event freshness, reconnect/backfill or bounded polling proof. | Do not log raw stream payloads; summarize event ids/status and redact account/order sensitive fields. |
| Telegram/user alerts | User-alert delivery proof is mandatory after the host blocker is solved. | Bot tokens, chat ids and raw user notification destinations are forbidden in reports. |
| Metrics/logging | Mainnet metrics must be Prometheus/Grafana-ready and reflected in `docs/runbooks/prod-dashboard-metrics-reference-ru.md`. | No high-cardinality raw order ids, user ids, key suffixes, provider payloads or secrets in metric labels. |

## Stage Status

| Stage | Статус | Prompt / task | Stage report | Validation depth | Proof boundary | Ключевой результат | Blocker | Next stage allowed |
|---|---|---|---|---|---|---|---|---|
| `00` Baseline hard-block and stale-copy manifest | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/00-baseline-hard-block-stale-copy-manifest.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/00-baseline-hard-block-stale-copy-manifest.md` | docs/runtime inventory | `target_host_readiness_pre_main if read-only host checks are used, otherwise N/A` | TBD | none yet | no |
| `01` User prerequisite and Telegram gate | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/01-user-prerequisites-telegram-gate.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/01-user-prerequisites-telegram-gate.md` | runtime/user prerequisite gate | `read_only_existing_runtime_smoke until changed code exists` | TBD | waits for user action | no |
| `02` Mainnet exchange connections read-only readiness | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/02-mainnet-exchange-connections-readiness.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/02-mainnet-exchange-connections-readiness.md` | mainnet read-only provider/API/DB/browser | `post_main_production_runtime_proof if code changes; otherwise read_only_existing_runtime_smoke` | TBD | waits for user action | no |
| `03` Kill-switch semantics and stale reason repair | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/03-kill-switch-semantics-stale-reason-repair.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/03-kill-switch-semantics-stale-reason-repair.md` | API/DTO/domain/docs/tests/browser if visible | `post_main_production_runtime_proof for changed code` | TBD | waits for Stage `02 accepted` | no |
| `04` Mainnet risk caps, capital manifest and approval schema | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/04-risk-caps-capital-manifest-approval-schema.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/04-risk-caps-capital-manifest-approval-schema.md` | API/DB/browser/metrics | `post_main_production_runtime_proof` | TBD | waits for user action | no |
| `05` Mainnet metrics, alerts, dashboard and user-alert contract | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/05-metrics-alerts-user-alert-contract.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/05-metrics-alerts-user-alert-contract.md` | Prometheus/runbook/notification runtime | `post_main_production_runtime_proof` | TBD | waits for user action | no |
| `06` Mainnet adapter enablement behind fail-closed no-submit mode | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/06-mainnet-adapter-capable-no-submit.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/06-mainnet-adapter-capable-no-submit.md` | runtime/Redis/DB/metrics | `post_main_production_runtime_proof` | TBD | waits for Stage `05 accepted` | no |
| `07` Open/close order-pair lifecycle and auto-close model | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/07-open-close-order-pair-lifecycle.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/07-open-close-order-pair-lifecycle.md` | domain/API/DB/Redis tests and no-submit runtime proof | `post_main_production_runtime_proof` | TBD | waits for Stage `06 accepted` | no |
| `08` Futures account config and market-order guard | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/08-futures-config-market-order-guard.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/08-futures-config-market-order-guard.md` | exchange config read-back plus no-submit guard | `post_main_production_runtime_proof` | TBD | waits for user action | no |
| `09` Private stream or REST fallback and reconciliation semantics | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/09-private-stream-reconciliation-semantics.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/09-private-stream-reconciliation-semantics.md` | runtime/provider/API/DB/metrics no-submit or safe-readiness evidence | `post_main_production_runtime_proof` | TBD | waits for Stage `08 accepted` | no |
| `10` Real mainnet ops canary: Binance spot long | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/10-ops-canary-binance-spot-long.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/10-ops-canary-binance-spot-long.md` | real exchange order/fill/close/reconcile/alert/browser/API | `post_main_production_runtime_proof` | TBD | waits for user action | no |
| `11` Real mainnet ops canary: Bybit spot long | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/11-ops-canary-bybit-spot-long.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/11-ops-canary-bybit-spot-long.md` | real exchange order/fill/close/reconcile/alert/browser/API | `post_main_production_runtime_proof` | TBD | waits for Stage `10 accepted` | no |
| `12` Real mainnet ops canary: Binance futures long | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/12-ops-canary-binance-futures-long.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/12-ops-canary-binance-futures-long.md` | real futures order/fill/close/reconcile/alert | `post_main_production_runtime_proof` | TBD | waits for Stage `11 accepted` | no |
| `13` Real mainnet ops canary: Binance futures short | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/13-ops-canary-binance-futures-short.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/13-ops-canary-binance-futures-short.md` | real futures order/fill/close/reconcile/alert | `post_main_production_runtime_proof` | TBD | waits for Stage `12 accepted` | no |
| `14` Real mainnet ops canary: Bybit futures long | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/14-ops-canary-bybit-futures-long.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/14-ops-canary-bybit-futures-long.md` | real futures order/fill/close/reconcile/alert | `post_main_production_runtime_proof` | TBD | waits for Stage `13 accepted` | no |
| `15` Real mainnet ops canary: Bybit futures short | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/15-ops-canary-bybit-futures-short.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/15-ops-canary-bybit-futures-short.md` | real futures order/fill/close/reconcile/alert | `post_main_production_runtime_proof` | TBD | waits for Stage `14 accepted` | no |
| `16` Ops canary matrix closure | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/16-ops-canary-matrix-closure.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/16-ops-canary-matrix-closure.md` | DB/Redis/Prometheus/Monit/browser/API | `post_main_production_runtime_proof` | TBD | waits for Stage `15 accepted` | no |
| `17` Strategy producer live-mode contract and no-order enablement | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/17-strategy-live-mode-contract-no-order.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/17-strategy-live-mode-contract-no-order.md` | API/DB/Redis/Monit/Prometheus/browser no-order proof | `post_main_production_runtime_proof` | TBD | waits for Stage `16 accepted` | no |
| `18` Strategy-driven mainnet canaries per market | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/18-strategy-driven-mainnet-canaries.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/18-strategy-driven-mainnet-canaries.md` | live candles/strategy signal/real orders/fills/reconcile/alerts/browser/API/metrics | `post_main_production_runtime_proof` | TBD | waits for user action | no |
| `19` Closure cleanup and go/no-go record | pending | `.codex/agents/generated/mainnet-real-money-trading-v1/19-closure-cleanup-go-no-go.md` | `docs/architecture/live_execution/mainnet-real-money-trading-v1-stage-reports/19-closure-cleanup-go-no-go.md` | browser/API/DB/Redis/Prometheus/Monit/docs | `post_main_production_runtime_proof` | TBD | waits for Stage `18 accepted` | no |

## Что Обязательно Знать Дальше

| Stage | Факт / решение / ограничение | Почему важно следующему stage | Evidence |
|---|---|---|---|
| plan | All GPT-5.5 Pro diagnostic findings are accepted; no finding was rejected. | Executors must not reopen these decisions unless new evidence proves a conflict. | User attachment 2026-07-03; plan adoption table. |
| plan | Mainnet real-money trading is a separate plan, not paper/testnet closure. | Executors must not treat accepted testnet as money-moving approval. | `docs/architecture/live_execution/mainnet-real-money-trading-v1.md` |
| plan | Telegram user alert delivery is mandatory and remains a hard blocker until user confirms host connectivity is solved. | No real-money stage after `00` may be accepted without `01 accepted`. | User clarification 2026-07-03; this ledger. |
| plan | `kill_switch_active=true` means emergency stop; ambiguous `kill_switch_open` semantics must be repaired before risk/caps stages. | Prevents inverted safety gates. | Stage `03`; diagnostics. |
| plan | First canary market order cap is `15 USDT`; canary positions must close immediately through open/close pair lifecycle. | Stages `10`-`18` must reject anything above cap or without auto-close proof. | Plan hard decisions. |
| plan | Budget statement has ambiguity: `20 USDT` per market but `60 USDT` total. | Stage `04` must create/validate explicit capital allocation manifest before submit. | User clarification 2026-07-03; plan open blockers. |
| plan | Futures default is isolated `1x`; platform config mutation is explicit pre-submit stage only. | Stage `08` owns account config and read-back; order submit must not hidden-auto-config. | User clarification 2026-07-03. |
| plan | Private stream readiness must be full lifecycle or explicit REST polling fallback. | Testnet auth probe is not enough for mainnet submit safety. | Stage `09`; official exchange docs. |
| plan | Futures reconciliation splits order/fill, position and funding. | Funding pending must not block short-lived canary closure if fill/position matched. | Stage `09`; diagnostics. |
| plan | Real ops canaries are sequential row gates. | No next row starts until previous row has close/reconcile/alert/no residual proof. | Stages `10`-`16`. |
| plan | Strategy-driven mainnet orders are automatic under supervised agent control, not per-order manual approvals. | Stages rely on scoped allowlists, one-shot token, caps, kill switch, alerts and stop-after-first-canary. | User clarification 2026-07-03; Stage `18`. |

## Контракты, Миграции И Совместимость

| Stage | API / DTO | Persistence | Config / env | Browser-visible | Ops / runtime | Compatibility / rollback |
|---|---|---|---|---|---|---|
| `00` | none or compatible readiness read model | none or readiness evidence rows | read-only/prerequisite only | settings/readiness where relevant | no money-moving side effects | fail closed / no submit |
| `01` | none or compatible readiness read model | none or readiness evidence rows | read-only/prerequisite only | settings/readiness where relevant | no money-moving side effects | fail closed / no submit |
| `02` | none or compatible readiness read model | none or readiness evidence rows | read-only/prerequisite only | settings/readiness where relevant | no money-moving side effects | fail closed / no submit |
| `03` | compatible-change or breaking if renaming public DTO requires migration | risk reason audit may change | kill switch config names/semantics | status/reason copy | operator stop semantics | requires docs/tests migration and stale reason mapping |
| `04` | risk/cap/approval APIs | capital manifest, canary scope, caps, audit | cap/approval config | blocked/ready states | risk gate | default disabled until manifest accepted |
| `05` | metrics/read models only | notification/metric evidence rows if needed | Prometheus/alerts | alert/status surfaces | monitoring and notifications | no order submit |
| `06` | internal readiness/no-submit mode | heartbeat/observation/order guard rows | mainnet-capable no-submit adapter mode | none | exchange-execution | default gate closed |
| `07` | open/close lifecycle commands/read models | order-pair lifecycle tables/fields | auto-close policy | outcome/status | close failure kill switch | requires migration/backfill or no legacy mainnet rows |
| `08` | futures config command | futures config audit rows | isolated 1x/user-selected config | futures config readiness | exchange config calls | no hidden submit-time mutation |
| `09` | stream/fallback/reconciliation read models | private stream/fallback/reconciliation rows | stream/fallback thresholds | status if exposed | stream/reconciliation workers | fallback is explicit and scoped |
| `10` | ops canary command/read model | orders/fills/reconciliation/notifications/runtime artifacts | canary scope token | outcome proof | real mainnet order side effects | auto-close + kill switch rollback |
| `11` | ops canary command/read model | orders/fills/reconciliation/notifications/runtime artifacts | canary scope token | outcome proof | real mainnet order side effects | auto-close + kill switch rollback |
| `12` | ops canary command/read model | orders/fills/reconciliation/notifications/runtime artifacts | canary scope token | outcome proof | real mainnet order side effects | auto-close + kill switch rollback |
| `13` | ops canary command/read model | orders/fills/reconciliation/notifications/runtime artifacts | canary scope token | outcome proof | real mainnet order side effects | auto-close + kill switch rollback |
| `14` | ops canary command/read model | orders/fills/reconciliation/notifications/runtime artifacts | canary scope token | outcome proof | real mainnet order side effects | auto-close + kill switch rollback |
| `15` | ops canary command/read model | orders/fills/reconciliation/notifications/runtime artifacts | canary scope token | outcome proof | real mainnet order side effects | auto-close + kill switch rollback |
| `16` | ops canary command/read model | orders/fills/reconciliation/notifications/runtime artifacts | canary scope token | outcome proof | real mainnet order side effects | auto-close + kill switch rollback |
| `17` | strategy live-mode readiness/contract | producer/risk audit rows | live allowlist no-order | live status | strategy producer | no strategy order yet |
| `18` | strategy-driven execution | full source->order ledger | scoped canary strategy config | outcome proof | real strategy mainnet loop | stop run/kill switch/auto-close |
| `19` | none or compatible read model | final evidence rows | mainnet remains scoped | final proof | closure checks | disable all non-accepted scopes |

## Проверки И Evidence

| Stage | Local gates | Real-boundary / e2e evidence | Proof boundary | Result | Evidence path / note | Tests-only exception | Residual risk |
|---|---|---|---|---|---|---|---|
| `00` | focused gates + docs index if docs changed | Foundation reconciled, current mainnet hard-blocks listed, stale reason/copy drift inventoried, prompt pack checked, no runtime mutation. | `target_host_readiness_pre_main if read-only host checks are used, otherwise N/A` | TBD | stage report | not allowed | docs-only baseline but must not claim readiness beyond observed facts |
| `01` | focused gates + docs index if docs changed | User confirmation and runtime Telegram readiness are proven without secrets; if not, stage is blocked and all money stages stay closed. | `read_only_existing_runtime_smoke until changed code exists` | TBD | stage report | not allowed | Telegram/VLESS setup is out of scope and must not be performed by executor |
| `02` | focused gates + docs index if docs changed | Read-only Binance/Bybit spot/futures readiness, trade permission, no withdrawal, IP restriction and balance buckets are proven; no order submit. | `post_main_production_runtime_proof if code changes; otherwise read_only_existing_runtime_smoke` | TBD | stage report | not allowed | Readiness must not expose credentials or send orders |
| `03` | focused gates + docs index if docs changed | kill_switch_active/execution_enabled semantics are fixed or explicitly mapped, stale stage reason/copy drift is repaired, docs/runbooks/tests agree; no order submit. | `post_main_production_runtime_proof for changed code` | TBD | stage report | not allowed | Unsafe kill-switch polarity can enable trading when operator expects stop |
| `04` | focused gates + docs index if docs changed | Persisted manifest/scope/order/open-exposure/gross/daily-loss/fee-reserve caps, one-shot approval token and risk audit math are proven; no order submit. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | Abstract booleans are insufficient for mainnet risk control |
| `05` | focused gates + docs index if docs changed | Prometheus base metric names in seconds, alerts, dashboard reference, user-alert contract and runbooks are proven; no order submit. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | Metrics must not use invalid histogram base names or mixed units |
| `06` | focused gates + docs index if docs changed | exchange-execution can run mainnet-capable no-submit mode, but submit is blocked without scoped one-shot canary approval; no real order. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | Adding mainnet base URLs without no-submit guard is unsafe |
| `07` | focused gates + docs index if docs changed | Durable open/close pair lifecycle, close linkage, spot filled-qty close, futures reduce-only close, partial-fill and close-failure states are proven without mainnet submit. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | cancel_after_submit is not close proof for market orders |
| `08` | focused gates + docs index if docs changed | Binance/Bybit futures config is set/read back, no open orders/positions preflight passes, fresh config snapshot is linked to canary scope; no strategy order yet. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | Hidden config mutation inside submit is forbidden |
| `09` | focused gates + docs index if docs changed | Mainnet private order/fill stream lifecycle or explicit REST polling fallback is accepted; order/fill, position and funding reconciliation are separated. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | Testnet auth probe is not mainnet private stream proof |
| `10` | focused gates + docs index if docs changed | Binance spot buy market <=15 USDT opens, close sell by filled base qty, alert/reconciliation/latency/slippage pass, no residual exposure. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | first real mainnet order; stop on unknown or close failure |
| `11` | focused gates + docs index if docs changed | Bybit spot buy market <=15 USDT opens, close sell by filled base qty, alert/reconciliation/latency/slippage pass, no residual exposure. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | real mainnet order; stop on unknown or close failure |
| `12` | focused gates + docs index if docs changed | Binance futures long open/close reduce-only market <=15 USDT, isolated 1x, fill/position reconciliation matched, funding not blocking. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | real futures exposure; reduce-only close must work |
| `13` | focused gates + docs index if docs changed | Binance futures short open/close reduce-only market <=15 USDT, isolated 1x, fill/position reconciliation matched, funding not blocking. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | real futures short exposure; reduce-only close must work |
| `14` | focused gates + docs index if docs changed | Bybit futures long open/close reduce-only market <=15 USDT, isolated 1x, fill/position reconciliation matched, funding not blocking. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | real futures exposure; reduce-only close must work |
| `15` | focused gates + docs index if docs changed | Bybit futures short open/close reduce-only market <=15 USDT, isolated 1x, fill/position reconciliation matched, funding not blocking. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | real futures short exposure; reduce-only close must work |
| `16` | focused gates + docs index if docs changed | Matrix-wide residual state is clean, no unknown/retry/DLQ growth, alerts delivered, metrics complete, budget/gross/fee audit recorded. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | must prove the whole matrix is safe before strategy live mode |
| `17` | focused gates + docs index if docs changed | Live-mode contract defines sizing, profile binding, canary token propagation, fan-out guard, stop-after-first-canary and restart/dedup semantics; no strategy-driven real order yet. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | live mode must not fan out or create unexpected orders |
| `18` | focused gates + docs index if docs changed | One real strategy signal per required market surface executes automatically under allowlist/caps, auto-closes, records candle-to-fill-to-alert latency, and stops after first canary per scope. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | automatic real-money strategy execution; stop on first unknown |
| `19` | focused gates + docs index if docs changed | No unexpected open orders/positions, no unknown/DLQ/retry growth, metrics/dashboard/alerts verified, ledgers updated, mainnet expansion remains scoped. | `post_main_production_runtime_proof` | TBD | stage report | not allowed | closure must not broaden mainnet access |

## File Manifest

| Stage | Created | Modified | Deleted | Outside expected paths | Outside-path justification | Foreign changes excluded | Mixed files / hunk status |
|---|---|---|---|---|---|---|---|
| `00` | Stage report | Stage report; plan/ledger/docs index if needed | none | none expected | N/A | TBD | none |
| `01` | Stage report | Stage report; notification runbook/config/docs if needed | none | none expected | N/A | TBD | none |
| `02` | Stage report | Exchange readiness code/docs/UI/report/ledger if needed | none | none expected | N/A | TBD | none |
| `03` | Stage report | Risk/domain/API/UI/runbook/tests/stage report/ledger | none | none expected | N/A | TBD | none |
| `04` | Stage report | Risk/cap code, migrations, docs, UI/config/report/ledger | none | none expected | N/A | TBD | none |
| `05` | Stage report | Prometheus rules, dashboard metrics reference, notification/runbook docs, report/ledger | none | none expected | N/A | TBD | none |
| `06` | Stage report | exchange-execution/config/runtime docs/report/ledger | none | none expected | N/A | TBD | none |
| `07` | Stage report | live_execution domain/application/migrations/tests/runbook/report/ledger | none | none expected | N/A | TBD | none |
| `08` | Stage report | futures config command/code/docs/tests/report/ledger | none | none expected | N/A | TBD | none |
| `09` | Stage report | exchange-execution adapters, reconciliation code, metrics, runbooks, report/ledger | none | none expected | N/A | TBD | none |
| `10` | Stage report | Stage report/runtime artifacts/ledger; canary harness docs if needed | none | none expected | N/A | TBD | none |
| `11` | Stage report | Stage report/runtime artifacts/ledger | none | none expected | N/A | TBD | none |
| `12` | Stage report | Stage report/runtime artifacts/ledger | none | none expected | N/A | TBD | none |
| `13` | Stage report | Stage report/runtime artifacts/ledger | none | none expected | N/A | TBD | none |
| `14` | Stage report | Stage report/runtime artifacts/ledger | none | none expected | N/A | TBD | none |
| `15` | Stage report | Stage report/runtime artifacts/ledger | none | none expected | N/A | TBD | none |
| `16` | Stage report | Stage report/runtime artifacts/ledger/runbooks if needed | none | none expected | N/A | TBD | none |
| `17` | Stage report | strategy-live-runner/live_execution/apps/api/apps-web/runbooks/report/ledger | none | none expected | N/A | TBD | none |
| `18` | Stage report | strategy canary harness/docs/UI evidence/runtime artifacts/report/ledger | none | none expected | N/A | TBD | none |
| `19` | Stage report | Final report; plan/ledger/docs index/runbooks | none | none expected | N/A | TBD | none |

## Publish / Deploy Handoff

| Stage | Branch / worktree / stash status | Scoped staging evidence | Commit | PR | Checks before push | Deploy/runtime status | Docs index evidence | Notes |
|---|---|---|---|---|---|---|---|---|
| all | `main`; no branch/worktree/stash unless explicitly approved | `git diff --cached --name-status`: required before any commit | TBD | N/A unless user requests | focused gates + docs index | runtime/code stages require post-main proof | required for Markdown | Use `publish-ci-deploy` only after accepted validation and scoped staging. |

## Blockers

| Stage | Blocker | Severity | Owner / next action | Resolved evidence | Next stage allowed |
|---|---|---|---|---|---|
| `01` | Telegram host access to `api.telegram.org` not confirmed solved by user. | blocker | User resolves host/VLESS outside this plan; executor then proves readiness without secrets. | TBD | no |
| `03` | Kill switch semantic drift around `kill_switch_open`. | blocker | Executor repairs or explicitly maps semantics before caps/submit stages. | Stage `03 accepted` | no |
| `04` | Capital allocation conflict: `20 USDT` per market and `60 USDT` total do not align across four market surfaces. | blocker | User/operator records explicit allocation manifest with open exposure, gross notional and fee/slippage reserve. | Stage `04 accepted` | no |
| `07` | Auto-close lifecycle missing from mainnet model. | blocker | Executor implements/proves open/close pair lifecycle before real orders. | Stage `07 accepted` | no |
| `09` | Full private stream lifecycle or explicit REST fallback is not accepted yet. | blocker | Executor proves stream/fallback/reconciliation semantics. | Stage `09 accepted` | no |
| `10` | Real money canary causes fees/slippage and possible loss. | high | User explicitly approves bounded canary window before start. | TBD | no |
| `18` | Strategy-driven real orders are automatic under agent supervision. | high | User explicitly approves scoped automatic strategy window. | TBD | no |

## Diagnostic Adoption Checklist

| Finding | Status | Notes |
|---|---|---|
| Kill switch semantic drift | accepted into plan | Stage `03`. |
| Auto-close lifecycle missing | accepted into plan | Stage `07`. |
| Futures config guard | accepted into plan | Stage `08`. |
| Risk/caps schema too abstract | accepted into plan | Stage `04`. |
| Capital allocation conflict | accepted into plan | Stage `04`. |
| Private stream readiness gap | accepted into plan | Stage `09`. |
| Futures funding reconciliation drift | accepted into plan | Stage `09`. |
| Strategy live-mode contract gap | accepted into plan | Stage `17` and Stage `18`. |
| Stage `07` too large | accepted into plan | Split into Stages `10`-`16`. |
| Metrics names with `_bucket` | accepted into plan | Stage `05`; base histogram names without `_bucket`. |
| Stale reason/copy drift | accepted into plan | Stage `00` inventory and Stage `03` repair. |
| Disagreed findings | none | No recommendation was rejected. |

## Change Log

| Date | Stage | Change | Evidence |
|---|---|---|---|
| `2026-07-03` | plan | Created mainnet real-money trading plan and ledger with Telegram hard blocker, `15 USDT` canary cap, market-order-only policy, auto-close and goal-driven execution mode. | `docs/architecture/live_execution/mainnet-real-money-trading-v1.md`; this ledger |
| `2026-07-03` | plan hardening | Adopted GPT-5.5 Pro diagnostic findings: kill-switch semantics, risk/caps schema, open/close lifecycle, futures config guard, private stream/fallback, reconciliation split, sequential ops canaries, metric naming and stale reason sweep. | User diagnostic attachment; updated plan; this ledger |
