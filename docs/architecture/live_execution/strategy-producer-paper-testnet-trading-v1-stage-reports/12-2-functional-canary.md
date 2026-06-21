# Stage 12.2: Functional Canary

Статус: `blocked`.

Дата проверки: `2026-06-21`.

## Pre-Start

User required before start: none beyond the provided prompt. The canary used existing Mac Studio/runtime/browser access and stopped short of any acceptance downgrade or secret request if runtime evidence could not be produced safely.

Stage `12.1` gate: `accepted` in `strategy-producer-paper-testnet-trading-v1-stage-ledger.md`. The selected subject from Stage `12.1` was:

| Field | Value |
|---|---|
| owner `user_id` | `ab094ba2-61d7-4fbf-be8f-cbad9f351572` |
| strategy `id` | `ee15e181-309f-478e-8726-04a299f1292f` |
| profile `id` | `5103b2db-5211-4f62-9e0e-a23605de9b41` |
| Stage `12.1` run `id` | `bb5d31ca-3f82-4237-bfb1-25cc9d6bf156` |
| connection `id` | `8cec780c-c19c-4781-bd22-2af1d592039d` |
| mode / exchange / market | `testnet` / `binance` / `spot` |
| instrument | `binance:spot:BTCUSDT` |
| strategy spec | `MA(3,5)`, `1m`, `spot`, `BTCUSDT` |

No source code change was planned or made for this gate. The gate was runtime evidence plus report/ledger/docs-index synchronization.

## Concrete File List Before Edits

| Path | Planned action | Reason |
|---|---:|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-2-functional-canary.md` | create | Record canary evidence and blocked decision. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | update | Mark `12.2 blocked` and keep `12.3` closed. |
| `docs/architecture/README.md` | generated check/update if required | Required docs index consistency after adding a stage report. |
| `output/playwright/stage12-2-functional-canary/*` | create evidence artifacts | Production `/strategies` browser proof and request/console summaries. |
| `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist` | host-runtime reapply | Reapply the Stage `12.1` scoped producer allowlist after runtime drift showed the producer disabled. |
| `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist.stage12-2-pre-canary-backup` | host-runtime backup | Recovery copy before reapplying the non-secret LaunchAgent environment variables. |

## Decision

Final decision: `blocked`.

Stage `12.2` did not prove functional canary acceptance. A replacement run was started and produced a pre-window burst, but the declared 31-minute canary window produced:

| Metric | In-window result |
|---|---:|
| new strategy signals | `0` |
| new actionable strategy signals | `0` |
| new execution source events | `0` |
| new execution intents | `0` |
| new execution orders | `0` |
| mainnet orders | `0` |
| Redis pending requests | `0` |

The producer and exchange-execution health endpoints stayed `ready`, but producer cycle/poll metrics did not advance during the canary. That combination is not acceptance for an active strategy runtime. Stage `12.3` remains blocked until Stage `12.2` is repaired and rerun successfully.

## Бизнес-Читаемое Резюме

Stage `12.2` должен был доказать не просто зеленый статус сервиса, а живую работу strategy producer в течение 30-60 минут: новые сигналы, новые source events, отсутствие mainnet/order side effects и видимость состояния в `/strategies`.

Проверка показала безопасный, но неприемлемый результат. Система осталась в Testnet, не создала mainnet/order side effects и корректно отображала выбранную стратегию в UI, но после начального всплеска до окна canary не было ни одного нового сигнала или source event. Поэтому следующий stage нельзя начинать: иначе burst/resource gate будет измерять не активную нагрузку, а фактически простаивающий или зависший producer.

## Business Impact

| Layer | Impact |
|---|---|
| Operator confidence | Stage `12.2` prevented a false-positive soak. The UI/API could show a `running` Testnet strategy, but runtime evidence proved that the active loop did not keep producing new signals/source events during the declared canary window. |
| Release risk | The next burst/resource gate remains closed. Accepting this gate would let later stages measure an idle or stale producer, which is the exact failure mode the Stage `12.1`/`12.2` split was designed to catch. |
| Money safety | Safety boundaries held: mode stayed `testnet`, `exchange-execution` stayed `adapter_mode=testnet`, mainnet orders stayed `0`, Redis dispatch streams did not grow, and no execution orders were created. |
| Customer-visible behavior | `/strategies` can show the selected strategy and pre-window journal rows, but those rows are not proof of ongoing canary progress. Operators need a repaired run before relying on the status as active runtime evidence. |
| Operations | The blocked result points to a repairable runtime-progress/readiness gap: health can remain ready while cycle/poll metrics are stale. Future gates should require explicit in-window deltas, not only ready status. |

## Runtime Preparation Evidence

The Stage `12.1` selected run was no longer usable at canary start:

| Check | Result |
|---|---|
| Stage `12.1` run `bb5d31ca-...` | state `failed` |
| selected profile | still `ready`, mode `testnet` |
| failure class | local ClickHouse read path refused connection during gap repair |
| historical rows | the failed run had prior signals, but no execution intents or orders |

Runtime drift was also found before the canary: the LaunchAgent had lost the non-secret Stage `12.1` producer enablement/allowlist environment. The canary re-applied only the scoped settings:

| Setting | Runtime value |
|---|---|
| `ROEHUB_EXECUTION_STRATEGY_PRODUCER_ENABLED` | `true` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOW_ALL` | `false` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOWED_MODES` | `paper,testnet` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOWED_USER_IDS` | `ab094ba2-61d7-4fbf-be8f-cbad9f351572` |
| `ROEHUB_STRATEGY_PRODUCER_ALLOWED_STRATEGY_IDS` | `ee15e181-309f-478e-8726-04a299f1292f` |

Recovery file: `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist.stage12-2-pre-canary-backup`.

After reapply, producer readiness showed `ready=True`, `enabled=True`, `allow_all=False`, allowed modes `paper,testnet`, and allowlist counts `1/1`. `exchange-execution` readiness remained `ready` with `adapter_mode=testnet`.

The first API restart attempt was correctly blocked by stale capital projection. The account projection was refreshed through the existing exchange-control path, then the strategy run API started replacement run `ec1aeb3c-dfeb-4684-a459-8dc9f736414b`, which reached `running`.

Immediately before the declared canary window, the replacement run produced a one-time burst:

| Surface | Pre-window value |
|---|---:|
| total signals on replacement run | `16` |
| actionable signals on replacement run | `3` |
| source events on replacement run | `16` |
| execution intents | `0` |
| execution orders | `0` |
| paper orders/fills/accounting | `0` |

That burst is useful diagnostic evidence, but it is not counted as Stage `12.2` acceptance because it occurred before the declared window and did not continue.

## Canary Window

| Field | Value |
|---|---|
| declared start UTC | `2026-06-21T15:56:26.695105+00:00` |
| declared end UTC | `2026-06-21T16:27:26.984444+00:00` |
| declared start Moscow | `2026-06-21 18:56:26+03` |
| declared end Moscow | `2026-06-21 19:27:26+03` |
| duration | `1860s` / `31m` |
| run under observation | `ec1aeb3c-dfeb-4684-a459-8dc9f736414b` |

Snapshots were collected before, during, and after the window at roughly five-minute intervals. The key values were unchanged from start to finish:

| Surface | Start | End | Result |
|---|---:|---:|---|
| run state | `running` | `running` | stayed active but made no progress |
| run checkpoint | `2026-06-19T02:45:00+03:00` | same | stale |
| signals since canary start | `0` | `0` | failed active-signal requirement |
| actionable signals since canary start | `0` | `0` | failed active-signal requirement |
| source events since canary start | `0` | `0` | failed execution-source-event requirement |
| total signals on replacement run | `16` | `16` | no in-window delta |
| total source events on replacement run | `16` | `16` | no in-window delta |
| total intents | `0` | `0` | expected for current testnet producer behavior; no dispatch occurred |
| total orders | `0` | `0` | no order side effect |
| mainnet orders since start | `0` | `0` | safety boundary held |
| Redis `execution.requests.v1` length | `41` | `41` | no dispatch growth |
| Redis pending for `exchange-execution.v1` | `0` | `0` | no stuck execution requests |
| Redis retry / DLQ lengths | `1` / `2` | `1` / `2` | no new retry/DLQ growth |
| `strategy_live_runner_iterations_total` | `70.0` | `70.0` | producer loop metric frozen |
| `strategy_live_runner_iteration_errors_total` | `0.0` | `0.0` | no surfaced loop error |
| `strategy_producer_polled_runs` | `0.0` | `0.0` | no active polling metric delta |
| `strategy_producer_active_instruments` | `0.0` | `0.0` | no active instrument metric delta |
| strategy producer health | `ready` | `ready` | ready status did not prove progress |
| exchange-execution health | `ready` | `ready` | execution service remained available |
| strategy-live-runner RSS | `68144 KiB` | `70736 KiB` | bounded process growth |
| exchange-execution RSS | `97136 KiB` | `97200 KiB` | stable |

Producer signal counters also stayed frozen during the window:

| Metric | Value |
|---|---:|
| `strategy_producer_source_event_total{mode="testnet", outcome="warmup"}` | `4` |
| `strategy_producer_source_event_total{mode="testnet", outcome="no_signal"}` | `9` |
| `strategy_producer_source_event_total{mode="testnet", outcome="signal"}` | `3` |
| `strategy_signal_total{mode="testnet", signal="close"}` | `2` |
| `strategy_signal_total{mode="testnet", signal="open"}` | `1` |
| `strategy_signal_total{mode="testnet", signal="no_signal"}` | `9` |
| `strategy_signal_total{mode="testnet", signal="warmup"}` | `4` |
| `strategy_producer_signal_lag_seconds` | `230877.84196209908` |

## Browser and API Evidence

Production browser proof was collected against `https://roehub.com/strategies?strategy_id=ee15e181-309f-478e-8726-04a299f1292f` with a temporary authenticated session for the selected owner. The session was revoked after proof; active recent proof sessions returned `0`.

| Artifact | Result |
|---|---|
| `output/playwright/stage12-2-functional-canary/strategies-prod-blocked-canary.png` | Screenshot captured the selected strategy page. |
| `output/playwright/stage12-2-functional-canary/strategies-prod.snapshot.txt` | Page title `Strategies | Roehub`; selected strategy `BTCUSDT 1m [MA(3,5)] #288FDBB3`; mode `testnet`; producer/runtime `running`; profile `ready`. |
| `output/playwright/stage12-2-functional-canary/strategies-prod.console-error.txt` | `Total messages: 0 (Errors: 0, Warnings: 0)`. |
| `output/playwright/stage12-2-functional-canary/strategies-prod.requests.txt` | Dashboard API request returned `200`. |

The `/strategies` dashboard showed the pre-window `06/21, 06:53 PM` source-event/signal rows and the replacement run as `running`. It did not show new in-window source-event progression. Direct API evidence matched the UI:

| Field | Value |
|---|---|
| API status | `200` |
| selected strategy | `ee15e181-309f-478e-8726-04a299f1292f` |
| selected status | `live` |
| live profile mode | `testnet` |
| readiness | `ready`, `testnet_ready_recent_auth_and_connection` |
| runtime run state | `running` |
| runtime producer status | `running` |
| runtime run id | `ec1aeb3c-dfeb-4684-a459-8dc9f736414b` |
| signal count in dashboard response | `20` |
| latest signal `created_at` | `2026-06-21T15:53:57.795000Z` |
| latest signal outcome / reason | `no_signal` / `ma_cross_no_change` |
| latest intent/order status | `null` / `null` |

## Root-Cause Notes for Repair

Evidence supports two separate issues:

1. The Stage `12.1` run failed before this canary after a local ClickHouse read path refused connection during gap repair.
2. The replacement run processed one initial historical-candle burst, then the producer cycle/poll metrics stopped advancing while `/health/ready` remained ready.

The next repair should investigate the live runner loop, candle/gap repair behavior, and readiness semantics. A rerun should start from a fresh, explicitly declared canary window and require in-window signal/source-event deltas before opening Stage `12.3`.

## Contract Impact

| Dimension | Impact | Notes |
|---|---:|---|
| Public API contract | `none` | No route, payload, or response schema changed. Existing API calls were used for proof. |
| Port contract | `none` | No port/interface changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration or schema change. |
| Config schema/defaults | `none` | No repo config schema/default changed. |
| Host runtime config | `compatible-change` | Re-applied the existing scoped LaunchAgent producer enablement/allowlist for one selected `paper,testnet` subject. |
| Request hash / cache key / persistence identity | `none` | No identity semantics changed. |
| Service-call semantics | `none` | Existing API, exchange-control, DB, Redis, health, metrics, and browser paths were used. |
| External side effects / unknown-state semantics | `compatible-change` | Refreshed a Testnet account projection and started one replacement Testnet strategy run; no mainnet path enabled. |
| Logs / metrics / audit / report semantics | `compatible-change` | Added blocked canary evidence and ledger handoff. |
| Alert/runbook semantics | `none` | No alert or runbook changed. |
| Benchmark / rollout gate impact | `compatible-change` | Stage `12.3` remains closed; Stage `12.2` must be repaired and rerun. |
| Performance risk on verified hot path | `unknown` | Runtime stayed bounded in RSS, but the producer progress freeze requires repair before longer gates. |
| Browser-visible behavior | `compatible-change` | `/strategies` visibly showed the replacement run and pre-window journal rows, but no accepted in-window progression. |

## Conditional Service-Call Coverage / Условное Покрытие Service-Call

Эта таблица перечисляет только вызовы, реально затронутые Stage `12.2`. Поверхности вне scope явно помечены `N/A`, чтобы не создавать ложное впечатление, что они были проверены.

| Caller / callee | Purpose | Evidence | Failure behavior |
|---|---|---|---|
| Authenticated API -> Postgres | Start replacement run and read dashboard state. | Run `ec1aeb3c-...` reached `running`; dashboard returned `200`. | Initial run start correctly blocked on stale capital projection until account projection was refreshed. |
| Runtime proof script -> exchange-control | Refresh selected Testnet account projection. | Projection became fresh with `account_state_read_ok` before retrying the run start. | Would block canary start because allocation/risk preflight would remain stale. |
| `strategy-live-runner` -> Postgres / market data | Poll active run and create signals/source events. | Pre-window burst created `16` source events; declared window created `0`. | Stage `12.2` blocked; repair required before rerun. |
| `exchange-execution` -> Redis / Postgres / exchange-control / exchange adapters | Ensure execution service is available and still Testnet-bound. | Health ready, adapter mode `testnet`, Redis pending `0`, mainnet orders `0`. | Would block or abort canary if readiness/pending/mainnet boundary failed. |
| Browser/Web -> API | Prove `/strategies` renders the selected active Testnet state. | Production Playwright proof returned `200`, no console errors, selected run visible. | Browser-visible stage would remain blocked without this proof. |
| N/A -> notification delivery | Notification delivery is outside Stage `12.2`. | N/A | Covered by later notification/runbook stages. |

## Logging and Redaction Coverage

| Surface | Coverage | Result |
|---|---|---|
| Stage report / ledger | Manual redaction review for secrets, tokens, cookies, DSNs, passwords, exchange keys, signed payloads, raw provider payloads, and raw session values. | Passed; only sanitized IDs, counts, metrics, timestamps, and local artifact paths are recorded. |
| Runtime logs | Log evidence was summarized instead of copied verbatim where it could include process arguments or connection material. | Passed; the report records the failure class without raw command lines or secret-bearing values. |
| Browser proof | Temporary authenticated session was used for the selected owner and revoked after proof. | Passed; no cookie/session value is written to artifacts or this report; active recent proof sessions returned `0`. |
| Provider payloads | Raw exchange-control/provider responses are not part of this report. | Explicit `N/A`; only projection status/count/timestamp evidence is included. |
| Notification delivery logs | Notification delivery is outside Stage `12.2`. | Explicit `N/A`. |

## Quality Gates

| Gate | Result | Evidence |
|---|---:|---|
| Runtime canary | failed | 31-minute window produced `0` new signals and `0` new source events; producer cycle metrics froze. |
| Mainnet boundary | passed | `mainnet_orders_since_start=0`; `exchange-execution` remained `adapter_mode=testnet`. |
| Redis execution boundary | passed | `execution.requests.v1` length, retry, DLQ, and group pending did not grow. |
| Production browser proof | passed for blocked decision | `/strategies` rendered selected Testnet run, dashboard API `200`, console errors `0`, temporary session revoked. |
| Secret scan by inspection | passed | Report records only sanitized IDs, metrics, and paths; no tokens, cookies, DSNs, passwords, exchange keys, or raw credentials. |
| `python -m tools.docs.generate_docs_index --check` | passed | `OK: /Users/daniildegtyarev/Projects/roehub.com/docs/architecture/README.md is up-to-date.` |
| Cold-head artifact review | passed | Cold self-review fallback completed after docs-index check; see receipt below. |

## Cold-Head Review

| Field | Result |
|---|---|
| Cold-head review | completed |
| Mode | cold self-review fallback; subagent tools are present, but the available tool policy only permits spawning when the user explicitly requests delegation/subagents. |
| Verdict | artifact handoff is ready as a `blocked` Stage `12.2` report; the blocked decision is supported by in-window runtime deltas, browser/API proof, safety-boundary evidence, docs-index check, and ledger update. |
| Blockers confirmed | The report distinguishes the pre-window burst from the declared canary window; records `0` in-window signals/source events; keeps Stage `12.3` closed; documents root-cause directions without overstating the fix. |
| Residual risks | The replacement canary run may still need explicit stop/cleanup before rerun. The root cause is not repaired in this stage. Host runtime remains scoped to one selected `paper,testnet` subject, but producer progress/readiness semantics require investigation. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-2-functional-canary.md` | none | none | Record Stage `12.2` blocked canary report. | `compatible-change`: report/rollout gate evidence. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Mark `12.2 blocked`; keep `12.3` closed. | `compatible-change`: ledger/handoff. |
| none | `docs/architecture/README.md` | none | Regenerate architecture docs index for the new Stage `12.2` report. | `none`: generated docs index. |
| `output/playwright/stage12-2-functional-canary/strategies-prod-blocked-canary.png`; `output/playwright/stage12-2-functional-canary/strategies-prod.snapshot.txt`; `output/playwright/stage12-2-functional-canary/strategies-prod.console-error.txt`; `output/playwright/stage12-2-functional-canary/strategies-prod.requests.txt` | none | none | Production browser evidence for the blocked canary decision. | `none`: local evidence artifacts. |
| `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist.stage12-2-pre-canary-backup` | none | none | Host-local recovery copy before reapplying producer runtime override. | `none`: backup only. |
| none | `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.strategy-live-runner.plist` | none | Reapply scoped `paper,testnet` producer allowlist after runtime drift. | `compatible-change`: host runtime config only, no repo default change. |

## Handoff

Stage `12.2` is blocked. Stage `12.3` must not start.

Repair/rerun requirements:

- investigate why the Stage `12.1` run failed on the local ClickHouse read path;
- investigate why replacement run `ec1aeb3c-dfeb-4684-a459-8dc9f736414b` processed a pre-window burst and then stopped advancing producer cycle/poll metrics while health stayed ready;
- decide whether to stop or replace the current canary run before rerun;
- rerun Stage `12.2` with a fresh declared 30-60 minute canary window;
- require in-window strategy signal/source-event deltas, stable Redis/execution boundaries, browser/API proof, docs index check, and cold-head artifact review before acceptance.
