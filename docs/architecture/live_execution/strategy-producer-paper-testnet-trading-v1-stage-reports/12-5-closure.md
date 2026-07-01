# Stage 12.5: Closure

Статус: `accepted`.

Дата проверки: `2026-07-02`.

## Pre-Start

User required before start: nothing for the original Stage `12.5` closure checks. The closure used existing local checkout access, existing `macstudio` SSH/runtime access, and the host-local smoke password source `/Users/daniildegtyarev/.config/roehub/roehub.env:ROEHUB_SMOKE_E2E_PASSWORD`.

User required before runtime repair/restart: explicit user authorization was received in chat: `делай runtime repair/restart для strategy_id=ee15e181-309f-478e-8726-04a299f1292f, только testnet/paper, без mainnet, сначала read-only preflight, затем безопасно восстанови freshness и rerun 12.`

No password, cookie, token, DSN, exchange key, raw credential, raw provider payload, raw session value, or secret-bearing browser state was printed or written to this report.

Previous stage ledger gate: `strategy-producer-paper-testnet-trading-v1-stage-ledger.md` was read before edits. It records:

| Gate | Ledger status | Evidence |
|---|---|---|
| `12.1` Readiness gate | `accepted` | Selected Testnet subject, scoped producer enablement, API/DB/Redis/Monit/Prometheus/RSS readiness, no mainnet order growth. |
| `12.2` Functional canary | `accepted` | `32m03s` accepted rerun with `+32` signals and `+32` execution source events, Redis pending/lag `0`, browser/API proof, no intents/orders/mainnet rows. |
| `12.3` Burst/resource gate | `accepted` | Controlled `180` `testnet` strategies, `passed=true`, `violations=[]`, Redis pending `0`, no retry/DLQ growth, no production intent/order/mainnet deltas, resource recovery passed. |
| `12.4` Sustained 6h soak | `accepted` | Fixed collector artifact `20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757`, `21600s`, `7` snapshots, final `360/360/360` candles/signals/source events, process rows non-empty, browser/API proof passed. |

The old monolithic `12-supervised-6h-soak.md` remains `superseded` historical negative evidence and is not counted as Stage `12` acceptance.

## Closure Decision

Final Stage `12` decision after runtime repair/restart and rerun: `accepted`.

Простыми словами: исходный `12.5` blocker был реальным, но он больше не актуален. Старый selected run `c2138129-a14a-40b3-bcf0-9ff4cf5a5757` был безопасно остановлен через application use case, Redis candle pending был очищен до `0`, account projection был обновлен read-only через `exchange-control`, и новый scoped testnet/paper run `c665f9e7-b4a6-4ede-83ee-b33a311f0ef4` стал `running` и снова пишет свежие `StrategySignal` / `ExecutionSourceEvent`.

Stage `13` is now allowed to start. Stage `12` is accepted only through the split-gate chain `12.1` + `12.2` + `12.3` + `12.4` + this accepted `12.5`; the old monolithic `12` report remains superseded.

## Business Impact

| Layer | Impact |
|---|---|
| Operator confidence | The selected strategy is no longer a false-green stale row: browser/API, DB, Redis, and producer metrics now agree that the scoped strategy is `live`/`running` and fresh. |
| Release risk | Stage `13` can start from accepted Stage `12` evidence instead of layering notifications/runbooks on top of a stale runtime. |
| Money safety | No order submit/status/cancel path was invoked during repair; execution orders in the last 24h stayed `0`, mainnet orders stayed `0`, and unknown orders stayed `0`. |
| Customer-visible behavior | `/strategies?strategy_id=...` authenticates through Keycloak and shows the selected strategy as running. The top-level dashboard still says `degraded` only because five read-model panels are intentionally unavailable, not because selected runtime freshness failed. |
| Operations | The next operational concern moves from Stage `12` cleanup to Stage `13` notification/runbook coverage. The accepted current run is `c665f9e7-b4a6-4ede-83ee-b33a311f0ef4`. |

## Runtime Repair Summary

Read-only preflight found:

| Surface | Preflight result |
|---|---|
| selected strategy | `ee15e181-309f-478e-8726-04a299f1292f` |
| owner user | `ab094ba2-61d7-4fbf-be8f-cbad9f351572` |
| selected run before repair | `c2138129-a14a-40b3-bcf0-9ff4cf5a5757`, state `running` |
| latest selected-run signal/source event before repair | `2026-07-01 03:58:00+03` / `2026-07-01 03:58:00.078234+03` |
| recent freshness before repair | `signals_30m=0`, `source_events_30m=0` |
| Redis candle group before repair | `pending=1`, lag observed as `655` during preflight |
| execution stream safety before repair | execution pending `0`, retry `1`, DLQ `2` |
| order safety before repair | orders in last 24h `0`, mainnet orders `0`, unknown orders `0` |

Safe repair actions performed:

| Step | Action | Boundary |
|---|---|---|
| stop stale run | Used `RestartStrategyUseCase` from runtime wiring; old run moved to `stopping` and then `stopped` with restart metadata `drained`. | Application use case only; no manual SQL state rewrite. |
| clear candle debt | Live runner drained the stale pending candle after the run moved out of active processing. | Redis candle pending became `0`; no execution stream mutation. |
| account projection refresh | Ran `scripts.live_execution.sync_exchange_account_projection` through `exchange-control` with local-only `ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL=http://127.0.0.1:9205` and host-local internal token. | Read-only account-state sync; `environment=testnet`, `open_order_count=0`, no order submit. |
| start successor run | Used `RunStrategyUseCase` under `ROEHUB_ENV=prod`; new run `c665f9e7-b4a6-4ede-83ee-b33a311f0ef4` was created in `starting` and advanced to `running`. | Scoped owner/strategy, `paper,testnet` producer allowlist, no mainnet. |
| account guard correction | Re-ran account projection sync with `--min-notional 0`, matching dashboard readiness requirements and replacing the earlier stricter diagnostic guard. | Read-only correction; no provider mutation. |

Root cause notes:

| Finding | Resolution |
|---|---|
| The strategy was not "not launched"; the stale run was `running` but blocked on freshness after earlier Redis/ClickHouse/Postgres transient failures and candle repair debt. | Old run stopped/drained; new run started cleanly and advanced checkpoint. |
| First standalone repair attempt used `configs/dev/strategy.yaml`, so Redis readiness probed host `redis` instead of production `127.0.0.1`. | Repair command was rerun with `ROEHUB_ENV=prod`, matching `launchd` runtime context. |
| The first successor path hit `capital_projection_stale`. | Account projection was refreshed through `exchange-control`; the new run then started successfully. |
| Dashboard account readiness briefly showed `min_notional_below_requirement` because diagnostic syncs with `--min-notional 10` / `50` wrote stricter guard results. | Final sync used dashboard-compatible `--min-notional 0`; browser proof then showed `account_readiness_status=fresh`. |

## Evidence Index

| Stage | Accepted report / artifact | Closure use |
|---|---|---|
| `12.1` | `12-1-readiness-gate.md` | Confirms readiness gate passed before canary/burst/soak. |
| `12.2` | `12-2-functional-canary.md` | Confirms functional producer canary passed on a fresh rerun. |
| `12.3` | `12-3-burst-resource-gate.md` | Confirms controlled burst/resource gate passed before sustained soak. |
| `12.4` | `12-4-sustained-6h-soak.md`; `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757/latest_status.json`; `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757/browser_api_proof.json` | Confirms `12.4` accepted only through the fixed collector artifact, not the old blocked candidate. |
| `12.5` superseded blocked proof | `output/playwright/stage12-5-closure-rerun/strategies-browser-api-proof.json` | Historical proof that auth worked but runtime freshness was stale before repair. Superseded by the repair rerun. |
| `12.5` accepted browser/runtime rerun | `output/playwright/stage12-5-closure-runtime-repair-rerun/strategies-browser-api-proof.json`; `output/playwright/stage12-5-closure-runtime-repair-rerun/console-errors.txt` | Fresh authenticated `/strategies?strategy_id=...` proof after repair; dashboard API `200`; console errors `0`; selected run `running`; signal/account/market readiness passed. |

## Fresh Browser Proof

Target:

```text
https://roehub.com/strategies?strategy_id=ee15e181-309f-478e-8726-04a299f1292f
```

Authentication contract:

| Field | Value |
|---|---|
| username | `smoke_e2e_keycloak` |
| password source | `/Users/daniildegtyarev/.config/roehub/roehub.env:ROEHUB_SMOKE_E2E_PASSWORD` |
| secret logged | `false` |
| login result | passed |

Browser/API result:

| Surface | Result |
|---|---|
| final URL | `https://roehub.com/strategies?strategy_id=ee15e181-309f-478e-8726-04a299f1292f` |
| final title | `Strategies | Roehub` |
| `/api/ui/strategies/dashboard?strategy_id=...` | `200` |
| final console errors | `0` |
| selected run state | `running` |
| selected strategy status | `live` |
| signal journal | `ready`, `20` latest rows |
| compatibility readiness | `ready` |
| market data state | `ready` |
| account readiness | `fresh`, age `36s` in final proof |
| source statuses | `available=9`, `degraded=0`, `unavailable=5` |
| sanitized JSON artifact | `output/playwright/stage12-5-closure-runtime-repair-rerun/strategies-browser-api-proof.json` |
| sanitized console artifact | `output/playwright/stage12-5-closure-runtime-repair-rerun/console-errors.txt` |

`refresh_status` remains `degraded` because the dashboard intentionally includes five not-yet-migrated read-model panels: `strategy_paper_accounting`, `market_candles`, `strategy_stat_projections`, `execution_fills`, and `strategy_events`. The final proof has `degraded_sources=[]`; this is recorded as residual dashboard projection debt, not as a Stage `12.5` freshness blocker.

## Cleanup And Runtime Freshness

Runtime health after repair:

| Surface | Result |
|---|---|
| `strategy_producer` `/health/ready` | `ready=true`, `enabled=true`, `allow_all=false`, allowed modes `paper,testnet`, allowlist counts `1/1`. |
| producer metrics | `strategy_producer_polled_runs=1`, `strategy_live_runner_iterations_total=2588`, `strategy_live_runner_messages_read_total=2756`, `strategy_live_runner_messages_acked_total=689`, `strategy_live_runner_iteration_errors_total=1` historical repair error retained. |
| stale collector/temp processes | none found for `stage12`, `soak`, `12-4`, or Playwright proof patterns on `macstudio`. |

Redis and DB cleanup:

| Surface | Final result | Decision |
|---|---:|---|
| active runs for selected strategy | `1` | pass |
| current run | `c665f9e7-b4a6-4ede-83ee-b33a311f0ef4`, state `running` | pass |
| current run checkpoint | `2026-07-02 00:58:00+03` | pass |
| latest current-run signal/source event | `2026-07-02 00:59:00.859+03` / `2026-07-02 00:59:00.889629+03` | pass |
| `StrategySignal` rows in last 30m for current run | `689` | pass |
| linked `ExecutionSourceEvent` rows in last 30m for current run | `689` | pass |
| `XPENDING md.candles.1m.binance:spot:BTCUSDT strategy.live_runner.v1` | `0` | pass |
| candle group lag | `0` | pass |
| `XPENDING execution.requests.v1 exchange-execution.v1` | `0` | pass |
| `XLEN execution.requests.v1` | `41` | unchanged accepted baseline |
| `XLEN execution.requests.retry.v1` | `1` | unchanged accepted baseline |
| `XLEN execution.requests.dlq.v1` | `2` | unchanged accepted baseline |
| account projection | `fresh`, `environment=testnet`, `open_order_count=0`, `filter_count=1` | pass |
| execution intents in last 24h | `0` | pass |
| execution orders in last 24h | `0` | pass |
| mainnet orders in last 24h | `0` | pass |
| unknown orders in last 24h | `0` | pass |
| paper orders in last 24h | `0` | pass |
| pending reconciliation rows | `30` | pre-existing debt; no new orders in last 24h |

## Contract Impact

| Dimension | Impact | Notes |
|---|---:|---|
| Public API contract | `none` | No route, payload, or response schema changed. |
| Port contract | `none` | No interface changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration or schema change. |
| Config schema/defaults | `none` | No repo config default changed. Runtime repair used existing product env and existing `launchd` context. |
| Request hash / cache key / persistence identity | `none` | No identity semantics changed. |
| Browser-visible behavior | `compatible-change` | `/strategies` proof now shows the selected strategy `live`/`running` with fresh signal/account/market readiness. Known unmigrated panels still keep top-level `refresh_status=degraded`. |
| Runtime/ops gate | `compatible-change` | Ledger/report now mark `12.5 accepted`; Stage `13` opens. |
| Performance risk on verified hot path | `none` | No code path changed. |

## Conditional Service-Call Coverage And Mutation Boundary

| Caller / callee | Purpose | Evidence | Boundary |
|---|---|---|---|
| Browser / Keycloak | Authenticate `smoke_e2e_keycloak` through the required production auth path. | Authenticated `/strategies?strategy_id=...` proof passed. | Password read from host-local env source; no secret printed or stored. |
| Browser / `/api/ui/strategies/dashboard` | Prove selected strategy runtime state through the authenticated UI/API surface. | Dashboard API `200`, selected run `running`, signal journal `ready`, account readiness `fresh`, market data `ready`, console errors `0`. | Read-only UI/API proof. |
| Repair command / `RestartStrategyUseCase` | Stop/drain stale selected run. | Old run `c2138129-...` moved to `stopped`; Redis candle pending cleared. | Application use case; no manual DB state rewrite. |
| Repair command / `sync_exchange_account_projection` | Refresh testnet account projection through `exchange-control`. | `status=fresh`, `environment=testnet`, `open_order_count=0`. | Read-only account-state; no submit/cancel/config provider mutation. |
| Repair command / `RunStrategyUseCase` | Start clean selected run. | New run `c665f9e7-...` reached `running` and wrote fresh signals/source events. | Scoped user/strategy, `paper,testnet` only. |
| N/A / exchange adapters | No order submit/status/cancel was attempted during repair. | execution orders in last 24h `0`, mainnet orders `0`, unknown orders `0`. | Money-safety boundary preserved. |

## Monitoring And Runbook Boundary

| Surface | Status |
|---|---|
| Alert changes | `N/A`; no Prometheus, Monit, Grafana, or alert file changed in this stage. |
| Runbook changes | `N/A`; Stage `13` owns notification/runbook work after this accepted closure. |
| Monitoring evidence | Producer health/metrics, Redis execution/candle state, DB signal/source-event/account/order counters were read after repair. |
| Required follow-up | Stage `13` may start. Preserve the known dashboard projection debt separately: top-level `refresh_status=degraded` can remain until unmigrated panels are implemented, but selected runtime freshness is no longer blocked. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-5-closure.md` | none | Convert Stage `12.5` from blocked closure to accepted repair/rerun closure with browser, cleanup, and runtime evidence. | `compatible-change`: docs/handoff status only. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Mark `12.5 accepted`, set `current_stage=13`, allow Stage `13`. | `compatible-change`: ledger/handoff only. |
| none | `docs/architecture/README.md` | none | Regenerated architecture docs index so `12-5-closure.md` status is `accepted`. | `none`: generated documentation index only. |
| `output/playwright/stage12-5-closure-runtime-repair-rerun/strategies-browser-api-proof.json` | none | none | Sanitized authenticated browser/API proof after repair. | `none`: local ignored evidence artifact. |
| `output/playwright/stage12-5-closure-runtime-repair-rerun/console-errors.txt` | none | none | Sanitized console summary for accepted browser proof. | `none`: local ignored evidence artifact. |
| `output/playwright/stage12-5-closure-runtime-repair-rerun/login-run-code-*.txt` | none | none | Sanitized local debug trail for safe secret-entry automation; contains no password value. | `none`: local ignored evidence artifact. |

Files outside expected prompt paths: local `output/playwright/stage12-5-closure-runtime-repair-rerun/*` artifacts are ignored evidence artifacts and are not intended for commit.

## Quality Gates

| Gate | Result | Evidence |
|---|---:|---|
| Previous stage ledger gate | passed | Ledger has `12.1`, `12.2`, `12.3`, and `12.4` as `accepted`. |
| Runtime repair preflight | passed | User explicitly authorized scoped testnet/paper repair; preflight read DB/Redis/health before mutation. |
| Runtime freshness rerun | passed | New run `c665f9e7-...` is `running`, `signals_30m=689`, `source_events_30m=689`, Redis candle pending/lag `0/0`. |
| Fresh browser proof | passed | Keycloak login succeeded; `/strategies?strategy_id=...` loaded; dashboard API `200`; selected run `running`; console errors `0`; account readiness `fresh`. |
| Cleanup proof | passed | No stale collector/temp process; execution pending `0`; retry/DLQ stayed `1/2`; no order/mainnet/unknown growth. |
| Mainnet/unknown safety | passed | Execution orders in last 24h `0`; mainnet orders in last 24h `0`; unknown orders in last 24h `0`. |
| Secret/redaction inspection | passed | Actual `ROEHUB_SMOKE_E2E_PASSWORD` value was searched across the report, ledger, and `stage12-5-closure-runtime-repair-rerun` artifacts; leaks `0`. |
| `uv run python -m tools.docs.generate_docs_index --check` | passed | `docs/architecture/README.md` regenerated and verified up to date. |
| `python -m tools.docs.generate_docs_index --check` | passed | Non-uv docs index check also verified `docs/architecture/README.md` up to date. |
| `uv run ruff check .` | passed | `All checks passed!` |
| `uv run pyright` | passed | `0 errors, 0 warnings, 0 informations`. |
| `uv run pytest -q -ra` | passed | `1473 passed, 3 warnings`. |
| `git diff --check` | passed | No whitespace errors. |
| Direct-main delivery | executor final handoff | Final commit/push/CI/Mac Studio evidence is recorded by the executor final response because the artifact cannot contain its own final commit SHA before commit. |

## Cold-Head Review

| Field | Result |
|---|---|
| Cold-head review | completed |
| Mode | cold self-review fallback; no separate independent subagent tool was active in the available tool list for this turn. |
| Review scope | Stage `12.5` accepted closure report, stage ledger update, proof boundary, file manifest, Stage `13` gating. |
| Review instructions | `architecture-review/references/cold-head-plan-prompt-pack-review.md` |
| Verdict | Release |
| Blockers fixed | Added business impact, explicit conditional service-call/mutation boundary, quality-gate results, and a plain explanation that dashboard `refresh_status=degraded` is known unmigrated-panel debt rather than selected runtime freshness failure. |
| Local follow-up check | completed |
| Residual risks | Top-level dashboard `refresh_status=degraded` remains until five known panels are migrated; account projection freshness has a short runtime age window and may need refresh in future browser drills; delivery SHA is recorded by the executor final handoff. |

## Next Action

Stage `13` may start.

The next executor should use this accepted closure as the handoff and should not reopen Stage `12` unless new runtime evidence regresses. Stage `13` should treat the following as current facts:

| Fact | Handoff |
|---|---|
| current selected strategy | `ee15e181-309f-478e-8726-04a299f1292f` |
| current accepted run | `c665f9e7-b4a6-4ede-83ee-b33a311f0ef4` |
| allowed runtime scope | `paper,testnet`; no mainnet |
| source-event path | fresh: `689` signals/source events in the final 30-minute proof window |
| Redis cleanup | candle pending/lag `0/0`; execution pending `0`; retry/DLQ `1/2` unchanged |
| browser state | authenticated `/strategies?strategy_id=...` proof passed; selected strategy is `live`/`running` |
| dashboard residual | five unmigrated dashboard panels keep top-level `refresh_status=degraded`; this is projection debt, not selected runtime freshness failure |
