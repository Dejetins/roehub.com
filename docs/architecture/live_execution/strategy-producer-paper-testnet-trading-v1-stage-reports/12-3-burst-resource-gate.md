# Stage 12.3: Burst/Resource Gate

Статус: `accepted`.

Дата acceptance: `2026-06-26`.

## Pre-Start

User required before start: nothing. Stage used existing Mac Studio SSH/runtime access, existing runtime env source `/Users/daniildegtyarev/.config/roehub/roehub.env`, and no secrets from chat.

Previous stage ledger gate: `strategy-producer-paper-testnet-trading-v1-stage-ledger.md` marks Stage `12.2` as `accepted`, `current_stage=12.3`, and records the accepted `2026-06-26` canary rerun on run `d87917a1-...` with `+32` signals and `+32` execution source events over `32m03s`. Stage `12.3` may start only after that accepted functional canary; this burst does not replace Stage `12.2` evidence.

Selected active subject carried from Stage `12.2`:

| Field | Value |
|---|---|
| owner `user_id` | `ab094ba2-61d7-4fbf-be8f-cbad9f351572` |
| strategy `id` | `ee15e181-309f-478e-8726-04a299f1292f` |
| run `id` | `d87917a1-1d72-49a8-b5c5-e40290bd3096` |
| mode / exchange / market | `testnet` / `binance` / `spot` |
| instrument | `binance:spot:BTCUSDT` |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-3-burst-resource-gate.md` | none | none | Stage `12.3` evidence, thresholds, decision, and handoff. | `none`; documentation/evidence only. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Mark Stage `12.3 accepted`, open Stage `12.4`, and record validation/delivery handoff. | `compatible-change`; staged workflow status only. |
| none | `docs/architecture/README.md` if docs index changes | none | Required docs index regeneration/check after adding a stage report. | `none`; generated documentation index only. |

Files outside prompt expected paths: none planned. Host-local transient Monit files under `/tmp/roehub-stage12-3-monit-final.*` were created only to run an isolated Monit evidence pass with the existing `/opt/homebrew/etc/monitrc`; they were removed by the collector trap and are not repository artifacts.

## Burst Configuration

The accepted burst reused the Stage `11` harness and did not create a new load path.

Command on Mac Studio from `/Users/daniildegtyarev/Projects/roehub.com`:

```bash
/opt/homebrew/bin/uv run python -m apps.exchange_execution.load_harness \
  --strategies 180 \
  --read-count 25 \
  --rate-limit-per-second 60 \
  --rate-limit-burst 20 \
  --pretty
```

Runtime env was sourced from `/Users/daniildegtyarev/.config/roehub/roehub.env`; reports do not include env values. Redis/DB commands used explicit tool paths `/opt/homebrew/bin/redis-cli` and `/opt/homebrew/bin/psql`.

Safety boundary:

| Boundary | Result |
|---|---|
| Load mode mix | `testnet=180`, `paper=0`, `mainnet=0`. |
| Real provider HTTP load | None; Stage `11` controlled adapters only. |
| Mainnet side effect | No new mainnet rows; historical `mainnet_orders_total=2` stayed unchanged. |
| Secrets/redaction | No passwords, cookies, tokens, DSNs, exchange keys, raw provider payloads, or session values are recorded. |

## Thresholds

| Metric | Threshold |
|---|---:|
| Harness result | `passed=true`, `violations=[]` |
| Strategy count | `>= 180` for this burst run |
| Mode mix | `testnet=100%` |
| Main retry/DLQ | `0 / 0` |
| Redis pending final | `0` |
| Redis max pending | `<= read_count` (`25`) |
| Queue lag | p95 `<= 15000ms`, p99 `<= 20000ms` |
| Signal-to-source p99 | `<= 250ms` |
| Source-to-intent p99 | `<= 250ms` |
| Risk p99 | `<= 50ms` |
| Dispatch p99 | `<= 500ms` |
| Adapter submit/status/cancel p99 | `<= 25ms` |
| Limiter wait | total `> 0s`, p99 `<= 250ms` |
| Ack/fill latency | p95 `<= 15000ms`, p99 `<= 20000ms` |
| Reconciliation pending | `0` |
| Harness CPU seconds | `<= 20s` |
| Harness RSS delta | `<= 256MB` |
| Host/resource recovery band | post CPU/RSS/Redis/DB values must return to baseline-adjacent band with no queue debt, no new unknown orders, and no sustained saturation. |

## Harness Result

Final Mac Studio burst result: `passed=true`, `violations=[]`, `rc=0`, stderr empty.

| Metric | Result | Threshold | Decision |
|---|---:|---:|---|
| Strategies | `180` | `>=180` | pass |
| Mode mix | `testnet=180`, `paper=0` | `testnet=100%` | pass |
| Requests / submitted / acked | `180 / 180 / 180` | submitted equals strategy count | pass |
| Guard / adapter / quarantined | `0 / 0 / 0` | `0` | pass |
| Main retry / DLQ | `0 / 0` | `0 / 0` | pass |
| Orders by exchange | `binance=90`, `bybit=90` | representative mix | pass |
| Orders by environment | `testnet=180` | no mainnet | pass |
| Fills / reconciliation | `fills=180`, `matched=180`, pending `0` | pending `0` | pass |
| Redis pending final / max pending | `0 / 25` | `0 / <=25` | pass |
| Queue lag | p95 `13599.579985ms`, p99 `14195.401678ms` | `15000ms / 20000ms` | pass |
| Signal-to-source p99 | `0.053323ms` | `250ms` | pass |
| Source-to-intent p99 | `0.100534ms` | `250ms` | pass |
| Risk p99 | `0.00521ms` | `50ms` | pass |
| Dispatch p99 | `0.036279ms` | `500ms` | pass |
| Adapter submit/status/cancel p99 | `2ms / 1ms / 1ms` | `25ms` | pass |
| Limiter wait total / p99 / max | `10.313829s / 16.169754ms / 16.458292ms` | `>0s / <=250ms` | pass |
| Harness CPU / RSS delta | `0.670141s / 2.203125MB` | `20s / 256MB` | pass |

Controlled probes:

| Probe | Result |
|---|---|
| Backpressure | `result=retry`, `reason=dispatch_backpressure`, `retry_count=1`, `request_count=0`, `dlq_count=0`. |
| Retry budget | `result=dlq`, `reason=retry_budget_exhausted`, `request_count=0`, `dlq_count=0`. |

## Baseline / During / Post Snapshots

Snapshot times:

| Label | UTC | Moscow |
|---|---|---|
| baseline | `2026-06-26T20:56:50.476595+00:00` | `2026-06-26 23:56:50+03` |
| during | `2026-06-26T20:56:54.622690+00:00` | `2026-06-26 23:56:54+03` |
| post | `2026-06-26T20:57:13.294597+00:00` | `2026-06-26 23:57:13+03` |

Host Prometheus/resource values:

| Surface | Baseline | During | Post | Delta / decision |
|---|---:|---:|---:|---|
| Prometheus host CPU busy % | `11.237694535596088` | `11.237694535596088` | `11.40034765174529` | `+0.162653pp`; no saturation. |
| Prometheus load1 | `1.8671875` | `1.8671875` | `1.90576171875` | `+0.038574`; low. |
| Prometheus memory free bytes | `39091109888` | `39091109888` | `38954008576` | `-137101312` bytes, about `130.75MiB`; acceptable on 64GB host. |
| Redis memory bytes | `1275437648` | `1275437648` | `1275541632` | `+103984` bytes; stable. |
| Redis connected clients | `7` | `7` | `7` | no growth. |
| Postgres `roehub` backends | `4` | `4` | `4` | no growth. |
| `redis_up` / `pg_up` | `1 / 1` | `1 / 1` | `1 / 1` | exporters healthy. |
| `up{job="strategy-producer"}` / `up{job="exchange-execution"}` | `1 / 1` | `1 / 1` | `1 / 1` | Prometheus scrape healthy. |
| `exchange_execution_ready{status="ready"}` | `1` | `1` | `1` | ready throughout. |

Process RSS/CPU snapshots:

| Process | Baseline RSS MB / CPU % | During RSS MB / CPU % | Post RSS MB / CPU % | Decision |
|---|---:|---:|---:|---|
| `strategy_live_runner` | `61.984 / 0.0` | `61.984 / 1.3` | `62.000 / 0.0` | recovered, RSS delta `+0.016MB`. |
| `exchange_execution` runtime process | `94.250 / 1.8` | `94.250 / 1.5` | `94.250 / 0.1` | stable. |
| transient harness processes | `0` | about `82.578MB` combined `uv` + harness Python | `0` | cleaned up after burst. |
| Redis process | `1254.359 / 0.1` | `1254.359 / 0.1` | `1254.500 / 0.3` | stable, delta `+0.141MB`. |
| Postgres process group | `109.641 / 0.2` | `109.641 / 0.5` | `109.656 / 9.6` | RSS stable; CPU point sample is not sustained saturation. |

Redis/runtime queues:

| Surface | Baseline | During | Post | Decision |
|---|---:|---:|---:|---|
| `XPENDING execution.requests.v1 exchange-execution.v1` | `0` | `0` | `0` | pass |
| `XLEN execution.requests.v1` | `41` | `41` | `41` | no production queue growth |
| `XLEN execution.requests.dlq.v1` | `2` | `2` | `2` | no DLQ growth |
| `XPENDING md.candles.1m.binance:spot:BTCUSDT strategy.live_runner.v1` | `0` | `0` | `0` | pass |
| candle group lag | `0` | `0` | `0` | pass |
| candle group entries-read | `136444` | `136444` | `136445` | active canary continued |

Database state:

| Surface | Baseline | During | Post | Delta / decision |
|---|---:|---:|---:|---|
| selected run state | `running` | `running` | `running` | pass |
| selected checkpoint | `23:55:00+03` | `23:55:00+03` | `23:56:00+03` | active strategy continued |
| selected run signals total | `97` | `97` | `98` | `+1`; burst did not replace canary |
| selected run actionable signals | `26` | `26` | `26` | no unexpected action burst |
| selected run source events total | `86` | `86` | `87` | `+1`; canary path alive |
| selected run recorded source events | `21` | `21` | `21` | unchanged in this short window |
| execution intents total | `122` | `122` | `122` | no production intent growth |
| execution orders total | `35` | `35` | `35` | no production order growth |
| historical mainnet orders total | `2` | `2` | `2` | no new mainnet rows |
| unknown orders total | `0` | `0` | `0` | pass |

Producer/exchange metrics:

| Metric | Baseline | During | Post | Decision |
|---|---:|---:|---:|---|
| `strategy_live_runner_iterations_total` | `333` | `335` | `340` | advances |
| `strategy_live_runner_iteration_errors_total` | `0` | `0` | `0` | pass |
| `strategy_live_runner_messages_read_total` | `17` | `17` | `18` | active |
| `strategy_live_runner_messages_acked_total` | `17` | `17` | `18` | active |
| `strategy_producer_ready` | `1` | `1` | `1` | pass |
| `strategy_producer_allow_all` | `0` | `0` | `0` | scoped |
| `strategy_producer_polled_runs` | `1` | `1` | `1` | selected run polled |
| `strategy_producer_active_instruments` | `1` | `1` | `1` | selected instrument active |
| `strategy_producer_source_events_total{outcome="signal"}` | `4` | `4` | `4` | stable in short window |
| `strategy_producer_source_events_total{outcome="no_signal"}` | `13` | `13` | `14` | active no-signal event |
| `exchange_execution_redis_pending` | `0` | `0` | `0` | pass |
| `exchange_execution_clock_drift_ms` | `0.141` | `0.081` | `0.086` | within `1000ms` guard |

Monit evidence:

| Surface | Baseline / during / post result |
|---|---|
| Monit summary | `MacStudioDaniil OK`; `roehub_strategy_live_runner OK`; `roehub_exchange_execution OK`; OpenBao, market-data workers, Keycloak, exchange-control, and backtest job runner also `OK`. |
| `roehub_strategy_live_runner` | `status OK`, monitored, pid `4988`, memory `62.0MB`, metrics and `/health/ready` ports responsive. |
| `roehub_exchange_execution` | `status OK`, monitored, pid `2902`, memory `94.2MB`, metrics and `/health/ready` ports responsive. |

Monit caveat: the default `monit -c /opt/homebrew/etc/monitrc summary` control path initially refused `127.0.0.1:2812` because its default runtime state was stale. For this gate, I used the same config with explicit transient pid/state files:

```bash
/opt/homebrew/opt/monit/bin/monit \
  -c /opt/homebrew/etc/monitrc \
  -p /tmp/roehub-stage12-3-monit-final.pid \
  -s /tmp/roehub-stage12-3-monit-final.state \
  -d 30
```

That isolated daemon produced the Monit status above during the burst and was then stopped. This is accepted as Monit service-health evidence for Stage `12.3`, but Stage `12.4` should either repair the default Homebrew Monit control path before the 6h soak or explicitly declare the same `-p/-s` evidence mode before starting.

## Commands And PromQL

Representative exact commands/queries used:

```bash
ssh macstudio 'zsh -s' <<'REMOTE'
cd /Users/daniildegtyarev/Projects/roehub.com
set -a
source /Users/daniildegtyarev/.config/roehub/roehub.env
set +a
/opt/homebrew/bin/uv run python -m apps.exchange_execution.load_harness --strategies 180 --read-count 25 --rate-limit-per-second 60 --rate-limit-burst 20 --pretty
/opt/homebrew/bin/redis-cli XPENDING execution.requests.v1 exchange-execution.v1
/opt/homebrew/bin/redis-cli XPENDING md.candles.1m.binance:spot:BTCUSDT strategy.live_runner.v1
/opt/homebrew/bin/redis-cli XINFO GROUPS md.candles.1m.binance:spot:BTCUSDT
/opt/homebrew/bin/psql "$POSTGRES_DSN" -X -A -t -c '<sanitized count query>'
curl -fsS http://127.0.0.1:9207/health/ready
curl -fsS http://127.0.0.1:9206/health/ready
curl -fsS http://127.0.0.1:9207/metrics
curl -fsS http://127.0.0.1:9206/metrics
REMOTE
```

PromQL:

```promql
100 * (1 - avg(rate(node_cpu_seconds_total{job="node-exporter",mode="idle"}[5m])))
node_load1{job="node-exporter"}
node_memory_free_bytes{job="node-exporter"}
redis_up{job="redis-exporter"}
redis_memory_used_bytes{job="redis-exporter"}
redis_connected_clients{job="redis-exporter"}
pg_up{job="postgres-exporter"}
pg_stat_database_numbackends{job="postgres-exporter",datname="roehub"}
max(up{job="strategy-producer"})
max(up{job="exchange-execution"})
max(exchange_execution_ready{job="exchange-execution",status="ready"})
```

Sanitized DB count shape:

```sql
select json_build_object(
  'selected_run', (... from strategy_runs where run_id = '<selected-run>'),
  'active_runs_selected_strategy', (...),
  'signals_selected_run_total', (...),
  'source_events_selected_run_total', (... source_ref_json->>'strategy_run_id' = '<selected-run>'),
  'execution_intents_total', (...),
  'execution_orders_total', (...),
  'mainnet_orders_total', (... where environment='mainnet'),
  'unknown_orders_total', (... where status in ('unknown','submit_unknown','reconcile_unknown'))
);
```

## Decision

Final decision: `accepted`.

The controlled burst passed with `violations=[]`; CPU/RAM/process RSS/Redis/DB/Prometheus/Monit snapshots were collected before, during, and after the burst; Redis pending returned/remained `0`; retry/DLQ/unknown/reconciliation deltas stayed within thresholds; and the Stage `12.2` active canary subject remained running and continued to advance. Stage `12.4` may start after this report and ledger are delivered.

## Contract Impact

| Dimension | Impact | Notes |
|---|---:|---|
| Public API contract | `none` | No API changed. |
| Port contract | `none` | No interface changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration/table change. |
| Config schema/defaults | `none` | No repo config changed. |
| Host runtime config | `compatible-change` | Started a transient Monit daemon with existing config and temporary pid/state for evidence; stopped it after collection. Existing services stayed unchanged. |
| Request hash/cache/persistence identity | `none` | No identity semantics changed. |
| Service-call semantics | `none` | Existing health, metrics, Redis, DB, and controlled harness paths were used. |
| External side effects / unknown-state semantics | `none` | No real provider HTTP load, no production queue growth, no new orders/intents/mainnet rows. |
| Logs/metrics/report semantics | `compatible-change` | Adds Stage `12.3` evidence report and ledger status. |
| Browser-visible behavior | `none` | No browser/UI behavior changed. |

## Quality Gates

| Gate | Result | Evidence |
|---|---:|---|
| Stage `12.2` prerequisite | passed | Ledger marks `12.2 accepted`, `current_stage=12.3` before this run. |
| Controlled burst | passed | Mac Studio Stage `11` harness `180` strategies: `passed=true`, `violations=[]`. |
| Resource recovery | passed | Post CPU/RSS/Redis/DB values returned to baseline-adjacent band; no queue debt. |
| Redis/DB safety | passed | execution pending `0`, candle pending/lag `0`, no new intents/orders/mainnet/unknown rows. |
| Monit | passed with caveat | Same `/opt/homebrew/etc/monitrc` using transient pid/state reported `OK` for target services; default Monit control path drift is a Stage `12.4` preflight risk. |
| Docs index | passed | `python -m tools.docs.generate_docs_index --check` -> `OK: ... docs/architecture/README.md is up-to-date.` |
| Cold-head artifact review | passed | Cold self-review fallback completed after docs-index check; pending receipt rows were fixed before delivery. |
| Broad ruff | passed | `uv run ruff check .` -> `All checks passed!` |
| Broad pyright | passed | `uv run pyright` -> `0 errors, 0 warnings, 0 informations`; pyright reported a newer upstream version available. |
| Broad pytest | passed | `uv run pytest -q -ra` -> `1384 passed, 3 warnings`; warnings are existing `httpx` per-request cookie deprecations in web route tests. |
| GitHub publish preflight | passed | `gh --version` -> `2.85.0`; `gh auth status` authenticated account `Dejetins` with token value redacted by GitHub CLI output. |

## Cold-Head Review

| Field | Result |
|---|---|
| Cold-head review | completed |
| Mode | cold self-review fallback; subagent tooling was discovered, but its tool policy allows spawning only after an explicit user request for delegation/subagents. |
| Review scope | Stage `12.3` report, stage ledger, docs index, evidence boundary, redaction, file manifest, threshold decisions, Monit caveat, and Stage `12.4` handoff. |
| Verdict | Release after fixes. |
| Blockers fixed | Replaced pending docs-index/cold-review rows with passed receipts; kept delivery pending until direct-main publish evidence exists. |
| Local follow-up check | completed: docs index check passed; `git diff --check` passed; redaction scan found only policy/env-var names and negative statements, not raw secret values. |
| Residual risks | Default Monit control path drift remains a Stage `12.4` preflight risk; Stage `12.3` burst does not replace the sustained 6h soak. |

## Delivery Status

Scoped docs/report delivery uses direct-main publish discipline. Local gates passed before commit; runtime code deploy is `N/A` because no repo runtime code changed. The exact final pushed SHA, GitHub Actions status, Mac Studio checkout sync, and production smoke result are recorded in the executor final handoff.

## Handoff To Stage 12.4

Stage `12.4` sustained 6h soak may start only after this Stage `12.3` report/ledger is delivered. Required preflight for `12.4`:

| Check | Required result |
|---|---|
| Stage ledger | `12.3 accepted`, `current_stage=12.4`. |
| Active selected strategy | run `d87917a1-...` still `running`, or a replacement is explicitly created and recorded before the 6h window. |
| Producer scope | enabled, `allow_all=false`, modes `paper,testnet`, selected owner/strategy allowlist only unless the next prompt explicitly changes scope. |
| Redis | execution pending `0`, candle pending/lag acceptable. |
| Monit | repair default control path or predeclare the explicit transient pid/state evidence mode before starting 6h observation. |
| Proof boundary | The 6h soak must prove sustained active strategy runtime; it must not rely on this short burst as substitute evidence. |

## Blockers And Residual Risk

| Item | Severity | Impact | Next action |
|---|---:|---|---|
| Default Monit control path drift | medium | `monit -c /opt/homebrew/etc/monitrc summary` initially refused `127.0.0.1:2812`; Stage `12.3` used an isolated same-config pid/state daemon for evidence, but the default Homebrew Monit daemon path should be repaired before long soak. | Stage `12.4` preflight should repair or explicitly declare monitoring mode before starting the 6h timer. |
| Stage `12.3` is not a functional canary | expected | Burst harness uses controlled adapters and does not prove real provider order execution or replace Stage `12.2`. | Keep Stage `12.2` canary and Stage `09` real testnet order proof as separate evidence boundaries. |
