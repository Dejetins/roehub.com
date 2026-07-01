# Stage 12.5: Closure

Статус: `blocked`.

Дата проверки: `2026-07-01`.

## Pre-Start

User required before start: nothing. Stage used existing local checkout access, existing `macstudio` SSH/runtime access, and the host-local smoke password source `/Users/daniildegtyarev/.config/roehub/roehub.env:ROEHUB_SMOKE_E2E_PASSWORD`. No password, cookie, token, DSN, exchange key, raw credential, raw provider payload, or session value was printed or written to this report.

Previous stage ledger gate: `strategy-producer-paper-testnet-trading-v1-stage-ledger.md` was read before edits. It records:

| Gate | Ledger status | Evidence |
|---|---|---|
| `12.1` Readiness gate | `accepted` | Selected Testnet subject, scoped producer enablement, API/DB/Redis/Monit/Prometheus/RSS readiness, no mainnet order growth. |
| `12.2` Functional canary | `accepted` | `32m03s` accepted rerun with `+32` signals and `+32` execution source events, Redis pending/lag `0`, browser/API proof, no intents/orders/mainnet rows. |
| `12.3` Burst/resource gate | `accepted` | Controlled `180` `testnet` strategies, `passed=true`, `violations=[]`, Redis pending `0`, no retry/DLQ growth, no production intent/order/mainnet deltas, resource recovery passed. |
| `12.4` Sustained 6h soak | `accepted` | Fixed collector artifact `20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757`, `21600s`, `7` snapshots, final `360/360/360` candles/signals/source events, process rows non-empty, browser/API proof passed. |

The old monolithic `12-supervised-6h-soak.md` remains `superseded` historical negative evidence and is not counted as Stage `12` acceptance.

## Closure Decision

Final Stage `12` decision: `blocked`, not accepted.

Простыми словами: предыдущие подгейты Stage `12` действительно приняты, но финальное закрытие нельзя подписывать, пока операторский браузерный вход не проходит, а выбранная стратегия выглядит активной только по статусу строки, но не производит свежие signals/source events. Это именно тот случай, где closure должен защитить следующий этап от ложного `green` статуса.

Stage `12.5` cannot close the Stage `12` chain because the required fresh closure evidence failed closed on two independent acceptance surfaces:

| Surface | Result | Closure impact |
|---|---|---|
| Fresh `/strategies` browser proof with `smoke_e2e_keycloak` | blocked: Keycloak returned `Invalid username or password` using the required host-local `ROEHUB_SMOKE_E2E_PASSWORD` source; final `/strategies` dashboard API returned `401`. | Blocks closure. |
| Cleanup/runtime freshness | blocked: selected run `c2138129-a14a-40b3-bcf0-9ff4cf5a5757` is still `running`, but no `StrategySignal` or linked `ExecutionSourceEvent` rows appeared in the last 30 minutes; Redis candle group has `pending=1`, `lag=454`. | Blocks closure. |
| Stale collector/temp process cleanup | passed: no stale `stage12`/`12.4` collector or unexpected active temp proof process was found. | Not blocking. |
| Mainnet/unknown safety | passed: no mainnet orders in the last 24h; unknown orders total `0`. | Not blocking. |
| Execution retry/DLQ | compatible with prior baseline: execution pending `0`, retry stream `1`, DLQ stream `2`. | Not blocking by itself. |
| Reconciliation debt | unchanged pre-existing debt: pending/running reconciliation rows `30`. | Not blocking by itself; no new order rows in last 24h. |

Stage `13` remains closed until a later repair/rerun produces accepted `12.5` browser proof and cleanup/runtime freshness.

## Evidence Index

| Stage | Accepted report / artifact | Closure use |
|---|---|---|
| `12.1` | `12-1-readiness-gate.md` | Confirms readiness gate passed before canary/burst/soak. |
| `12.2` | `12-2-functional-canary.md` | Confirms functional producer canary passed on a fresh rerun. |
| `12.3` | `12-3-burst-resource-gate.md` | Confirms controlled burst/resource gate passed before sustained soak. |
| `12.4` | `12-4-sustained-6h-soak.md`; `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757/latest_status.json`; `/opt/roehub/state/live_execution/stage12-4-sustained-6h-soak/20260630T162058Z-stage07-fixed-process-rerun-c2138129-a14a-40b3-bcf0-9ff4cf5a5757/browser_api_proof.json` | Confirms `12.4` accepted only through the fixed collector artifact, not the old blocked candidate. |
| `12.5` browser attempt | `output/playwright/stage12-5-closure/strategies-browser-api-proof.json`; `output/playwright/stage12-5-closure/strategies-closure-body.txt` | Records blocked fresh browser proof; no secrets are present. |

## Business Impact

| Layer | Impact |
|---|---|
| Operator confidence | Stage `12` cannot be reported as closed because the operator-facing `/strategies` proof is currently unauthenticated and the selected active run is stale. This prevents a false-positive handoff into notifications/runbooks. |
| Release risk | Stage `13` remains closed. Starting notification/runbook work from a stale selected strategy would hide a runtime freshness issue behind downstream docs. |
| Money safety | The no-mainnet boundary still holds: no mainnet orders appeared in the last 24h, unknown orders total is `0`, and exchange-execution remains `adapter_mode=testnet`. |
| Customer-visible behavior | The browser surface cannot be accepted because `smoke_e2e_keycloak` cannot authenticate with the required host-local password source. Users/operators would not have fresh verified `/strategies` status for closure. |
| Operations | Cleanup must handle or intentionally stop/restart the stale selected run and Redis candle consumer debt before a closure rerun. |

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
| login result | blocked by Keycloak form error `Invalid username or password`. |

Browser/API result:

| Surface | Result |
|---|---|
| final URL | `https://roehub.com/login?next=%2Fstrategies` |
| final title | `Sign in to Roehub | Roehub` |
| `/api/ui/strategies/dashboard?strategy_id=...` | `401` |
| final console errors | `1`, the dashboard API `401` after unauthenticated redirect |
| request failures | `0` |
| accepted screenshot | none; screenshot capture did not complete after the blocked unauthenticated state, and this is not accepted browser proof |
| sanitized body text artifact | `output/playwright/stage12-5-closure/strategies-closure-body.txt` |
| sanitized JSON artifact | `output/playwright/stage12-5-closure/strategies-browser-api-proof.json` |

The previous Stage `12.4` browser/API proof remains valid for `12.4`, but it is not a fresh closure proof for `12.5`.

## Cleanup And Runtime Freshness

Read-only cleanup commands on `macstudio` found no stale collector/temp process matching `stage12.*collector`, `stage 12.*collector`, `soak.*collector`, `stage12-4`, `12-4.*collector`, `playwright.*stage12`, or `node.*stage12`.

Runtime health:

| Surface | Result |
|---|---|
| `strategy_producer` `/health/ready` | `ready=true`, `enabled=true`, `allow_all=false`, allowed modes `paper,testnet`, allowlist counts `1/1`. |
| `exchange-execution` `/health/ready` | `status=ready`, `adapter_mode=testnet`, dependencies ready, execution pending `0`. |
| Mac Studio git checkout | `/Users/daniildegtyarev/Projects/roehub.com` on `main`, clean, but stale at `0700a92b` before this docs handoff; no runtime code changed by this stage. |

Redis and DB cleanup:

| Surface | Result | Decision |
|---|---:|---|
| `XPENDING execution.requests.v1 exchange-execution.v1` | `0` | pass |
| `XLEN execution.requests.v1` | `41` | unchanged accepted baseline |
| `XLEN execution.requests.retry.v1` | `1` | unchanged accepted baseline |
| `XLEN execution.requests.dlq.v1` | `2` | unchanged accepted baseline |
| `XPENDING md.candles.1m.binance:spot:BTCUSDT strategy.live_runner.v1` | `1` | block cleanup freshness |
| candle group lag | `454` | block cleanup freshness |
| selected run state | `running` | not sufficient by itself |
| selected run checkpoint | `2026-07-01 03:57:00+03:00` | stale relative to closure check |
| `StrategySignal` rows in last 30m for selected run | `0` | block cleanup freshness |
| linked `ExecutionSourceEvent` rows in last 30m for selected run | `0` | block cleanup freshness |
| latest signal/source event for selected run | `2026-07-01 03:58:00+03:00` / `2026-07-01 03:58:00.078234+03:00` | stale |
| mainnet orders in last 24h | `0` | pass |
| unknown orders total | `0` | pass |
| pending/running reconciliation rows | `30` | pre-existing debt, no new order rows in last 24h |
| execution orders in last 24h | `0` | pass |

## Contract Impact

| Dimension | Impact | Notes |
|---|---:|---|
| Public API contract | `none` | No route, payload, or response schema changed. |
| Port contract | `none` | No interface changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration or schema change. |
| Config schema/defaults | `none` | No repo config default changed. |
| Request hash / cache key / persistence identity | `none` | No identity semantics changed. |
| Browser-visible behavior | `unknown` | Required authenticated proof is blocked by smoke account credential failure. |
| Runtime/ops gate | `compatible-change` | Ledger/report now record `12.5 blocked`; Stage `13` stays closed. |
| Performance risk on verified hot path | `none` | No code path changed. |

## Conditional Service-Call Coverage

This stage is evidence and handoff only. It did not add code, alter service call behavior, submit orders, mutate exchange state, rotate credentials, or change retry/unknown-state semantics.

| Caller / callee | Purpose | Evidence | Failure behavior |
|---|---|---|---|
| Browser / Keycloak | Authenticate `smoke_e2e_keycloak` through the required production auth path. | Keycloak form returned `Invalid username or password`; no accepted authenticated session was created. | Blocks Stage `12.5`; do not bypass with a guessed secret. |
| Browser / `/api/ui/strategies/dashboard` | Prove `/strategies` selected strategy runtime state through the authenticated UI/API surface. | Dashboard API returned `401` after unauthenticated redirect. | Blocks Stage `12.5`; previous `12.4` proof is historical, not fresh closure proof. |
| Runtime proof script / Postgres | Read selected run freshness, recent signals/source events, mainnet/unknown/reconciliation counters. | Read-only DB evidence showed stale selected run, `0` recent signals/source events, mainnet `0`, unknown `0`, reconciliation pending `30`. | Blocks cleanup freshness while preserving no-mainnet safety. |
| Runtime proof script / Redis | Read execution and candle consumer state. | Execution pending `0`, retry `1`, DLQ `2`; candle group pending `1`, lag `454`. | Blocks cleanup freshness until debt is cleared or intentionally classified by a repair/rerun. |
| Runtime proof script / health endpoints | Check producer and exchange-execution readiness. | Producer ready/scoped; exchange-execution ready/testnet. | Readiness alone is not enough for closure because freshness/browser proof failed. |
| N/A / exchange adapters | No order submit/status/cancel was attempted. | `execution_orders` in last 24h `0`; no mainnet order growth. | Any future repair that submits orders must be a separate explicit gate. |

## Monitoring And Runbook Boundary

| Surface | Status |
|---|---|
| Alert changes | `N/A`; no Prometheus, Monit, Grafana, or alert file changed in this stage. |
| Runbook changes | `N/A`; the next action is a repair/rerun handoff, not a new operator runbook. |
| Monitoring evidence | Producer/exchange-execution health endpoints were read; Redis/DB counters were read; no monitoring config was changed. |
| Required follow-up | Repair stale selected-run/candle consumer freshness and smoke auth before rerunning closure. If this requires a durable operational procedure, Stage `13`/runbook work must wait until `12.5 accepted`. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-5-closure.md` | none | none | Stage `12.5` closure report with blocked decision and evidence. | `compatible-change`: stage status/handoff only. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Mark `12.5 blocked`; keep Stage `13` closed. | `compatible-change`: ledger/handoff only. |
| `output/playwright/stage12-5-closure/strategies-browser-api-proof.json` | none | none | Sanitized blocked browser/API proof artifact. | `none`: local ignored evidence artifact. |
| `output/playwright/stage12-5-closure/strategies-closure-body.txt` | none | none | Sanitized body text from blocked unauthenticated browser state. | `none`: local ignored evidence artifact. |
| none | `docs/architecture/README.md` if regenerated | none | Required docs index consistency after adding report. | `none`: generated documentation index only. |

Files outside expected prompt paths: local `output/playwright/stage12-5-closure/*` artifacts are ignored evidence artifacts and are not intended for commit.

## Quality Gates

| Gate | Result | Evidence |
|---|---:|---|
| Previous stage ledger gate | passed | Ledger has `12.1`, `12.2`, `12.3`, and `12.4` as `accepted`; `12.5` was open before this run. |
| Fresh browser proof | blocked | Keycloak rejected `smoke_e2e_keycloak` using required host-local password source; dashboard API returned `401`. |
| Cleanup proof | blocked | Selected run stale; no signal/source-event rows in last 30m; candle group `pending=1`, `lag=454`. |
| Mainnet/unknown safety | passed | Mainnet orders in last 24h `0`; unknown orders total `0`; execution orders in last 24h `0`. |
| Secret/redaction inspection | passed | Report and generated JSON/body artifacts contain no password, cookie, token, DSN, exchange key, raw credential, session value, or provider payload. |
| `python -m tools.docs.generate_docs_index --check` | passed | Initial check found `docs/architecture/README.md` out of date; `python -m tools.docs.generate_docs_index` regenerated it; repeat `--check` passed. |
| Direct-main delivery | pending | Blocked report/ledger must still be published as scoped docs handoff if local gates pass. |

## Cold-Head Review

| Field | Result |
|---|---|
| Cold-head review | completed |
| Mode | cold self-review fallback; subagent delegation is not used because the active tool policy permits subagents only after an explicit user request. |
| Review scope | Stage `12.5` blocked closure report, stage ledger updates, docs index update, proof boundary, file manifest, Stage `13` gating. |
| Review instructions | `architecture-review/references/cold-head-plan-prompt-pack-review.md` |
| Verdict | Release after fixes for the blocked handoff artifact. Stage `12.5` itself remains `blocked`. |
| Blockers fixed | Added Russian business-readable closure explanation, business impact, conditional service-call coverage, monitoring/runbook `N/A` boundary, file manifest, docs-index evidence, and explicit Stage `13` closed handoff. |
| Local follow-up check | completed: `python -m tools.docs.generate_docs_index --check`, `uv run python -m tools.docs.generate_docs_index --check`, `git diff --check`, `uv run ruff check .`, `uv run pyright`, and `uv run pytest -q -ra` passed. |
| Residual risks | The real stage blockers remain unresolved by design: smoke Keycloak auth fails, selected run freshness is stale, and Redis candle pending/lag debt exists. Delivery SHA is recorded by the executor final handoff because the artifact cannot contain its own final commit hash before commit. |

## Next Action

Do not start Stage `13`.

The next executor should repair or refresh the smoke Keycloak credential source and restore selected-run cleanup/freshness before rerunning `12.5`. A valid rerun must show:

| Required rerun evidence | Acceptance boundary |
|---|---|
| `smoke_e2e_keycloak` can authenticate through Keycloak with the required host-local password source. | Fresh `/strategies` page and dashboard API proof returns authenticated `200`. |
| selected strategy runtime is fresh or intentionally stopped/cleaned up. | No stale active run with no recent signals/source events; Redis candle pending/lag within accepted thresholds. |
| no stale collectors/temp processes. | Read-only process inventory clean. |
| Redis retry/DLQ/mainnet/unknown/reconciliation deltas remain within accepted thresholds. | No new debt beyond known accepted/pre-existing baselines. |
| docs index and direct-main scoped delivery complete. | Ledger/report on `origin/main`; Stage `13` only opens after `12.5 accepted`. |
