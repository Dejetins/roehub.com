# Stage 12: Supervised 6h Soak

Статус: `blocked`.

Дата старта подготовки: `2026-06-20`.

Плановое окно soak: `2026-06-19T22:35:00Z` .. `2026-06-20T04:35:00Z` (`2026-06-20 01:35` .. `07:35 MSK`).

## Pre-Start

User required before start: nothing.

Stage `11` gate: `accepted` в `strategy-producer-paper-testnet-trading-v1-stage-ledger.md`; Stage `12` may start. Blockers before start: `none`.

Mac Studio source checkout was fast-forwarded to current `origin/main` docs-only commit `921711e222b9afbaec9fa8460470965072529275`. Runtime code remains the accepted Stage `11` code path; the new commit only refactors prompt/plan/ledger docs from 24h to 6h.

## Scope

Stage `12` is an acceptance observation gate, not a new implementation stage. It must prove six elapsed hours of stable runtime supervision and resource behavior, then run one controlled amplified-load interval using the accepted Stage `11` harness.

Current production producer config is intentionally fail-closed:

- `strategy.producer.enabled=false`
- `allow_all=false`
- allowed modes: `paper,testnet`
- allowed user count: `0`
- allowed strategy count: `0`

Therefore this stage does not enable producer admin switches, broaden allowlists, or start mainnet/live behavior. Baseline strategy count is the observed production baseline: `0` running strategy runs. The controlled burst is `testnet`-mode only through `apps.exchange_execution.load_harness`, with no real provider HTTP path and no mainnet path.

## Business Impact

Этот gate отвечает на операционный вопрос: можно ли держать текущий paper/testnet execution stack включенным под supervision без скрытого накопления очередей, unknown-state долгов и resource saturation. Для продукта это не добавляет новую пользовательскую функцию, но снижает риск следующего шага: оператор видит, что Mac Studio runtime, Redis transport, Postgres ledgers, Monit и Prometheus остаются стабильными в течение полного окна наблюдения и controlled burst возвращается в нормальный ресурсный профиль.

Если Stage `12` не проходит, следующий stage нельзя начинать как “почти готовый”: результатом должен быть blocker с конкретным ресурсным, очередным, reconciliation или browser/runtime симптомом.

## Operational Boundaries

| Surface | Stage `12` rule |
|---|---|
| Service calls | N/A for new outbound contracts: no new service integration is added. Existing calls are read-only health/metrics/DB/Redis probes plus the accepted in-memory Stage `11` harness. |
| Auth/secrets | No chat-provided secrets, raw DSNs, cookies, API keys, provider payloads, or Redis/Postgres credentials are recorded. Host-local env is used only inside Mac Studio commands. |
| Logging/redaction | Durable raw collector output is local to `/tmp/roehub-stage12-soak-20260619T2235Z.log`; the repo report stores sanitized summaries only. |
| Retry/idempotency | No production replay or retry is initiated. Existing Redis retry/DLQ baselines are observed for no increase. |
| Alerts/runbooks | Existing Monit configs, Prometheus rules, `strategy-live-worker.md`, `exchange-execution.md`, and dashboard metric references are used. No new alert rule is introduced. |

## Concrete File List Before Edits

Prompt указал broad paths. До runtime/validation scope сужен до:

| Path | Planned action | Reason |
|---|---:|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-supervised-6h-soak.md` | create/update | Stage report, pre-start plan, thresholds, evidence snapshots, final decision. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify after validation | Stage status, handoff, publish/deploy evidence. |
| `docs/architecture/README.md` | check/update if generated index changes | Required docs index gate after Markdown changes. |
| `apps/worker/strategy_live_runner` | no planned edit | Existing producer runtime and metrics are sufficient; production remains fail-closed. |
| `apps/exchange_execution` | no planned edit | Accepted Stage `11` load harness is reused as-is. |
| `infra/scripts/monit` | no planned edit | Existing Monit configs are used as evidence. |
| `infra/macos/prometheus` | no planned edit | Existing Prometheus/node-exporter/service metrics are used as evidence. |
| `docs/runbooks/mac-studio-monitoring-plan.md` | no planned edit | Existing resource metric references are sufficient. |
| `docs/runbooks/prod-dashboard-metrics-reference-ru.md` | no planned edit | Existing host CPU/load/memory PromQL is sufficient. |

Files outside prompt expected paths: none planned.

## Run Plan And Thresholds

| Item | Predeclared value |
|---|---|
| Start time | `2026-06-19T22:35:00Z` / `2026-06-20 01:35 MSK` |
| Planned end time | `2026-06-20T04:35:00Z` / `2026-06-20 07:35 MSK` |
| Baseline strategy count | `0` running strategies; producer admin disabled and allowlists empty. |
| Controlled burst strategy count | `180` synthetic `testnet` strategies via Stage `11` harness. |
| Burst plan | Start `2026-06-20T00:35:00Z`; expected command duration `< 5m`; post-burst observation until `2026-06-20T00:50:00Z`. |
| Mode mix | Baseline runtime: `paper/ready=30`, `testnet/blocked=2`, no running strategies. Burst: `testnet=100%`; `paper` is observed as ready profile baseline only. |
| Exchanges / markets | Burst harness covers Binance + Bybit, spot + futures, `BTCUSDT`; no real provider HTTP and no mainnet. |
| Kill switches | Producer admin remains disabled; exchange-execution adapter mode must stay `testnet`; Monit can stop `com.roehub.strategy-live-runner` and `com.roehub.exchange-execution`; any mainnet attempt or unknown new order state blocks acceptance. |
| Cleanup plan | Do not create production orders/positions; verify Redis pending remains `0`; verify DB counts do not grow in unknown/retry/DLQ categories beyond baseline; no temp repo artifacts retained. |
| Resource threshold | Host CPU busy p95 over snapshots `< 85%`; load1/load5 must not show sustained saturation; memory free must stay `> 15%`; process RSS for producer, exchange-execution, and API must not grow by `> 256 MiB` from baseline. |
| Queue/state threshold | Redis pending `0` at baseline/post/final; retry stream length and DLQ length must not increase except controlled harness internal probe, which does not touch production Redis; DB `execution_reconciliation_runs.status='pending'` and unknown outbox count must not increase. |
| Burst threshold | Harness result `passed=true`, `violations=[]`, `retry_count=0`, `dlq_count=0`, `redis_pending_final=0`, limiter wait observed and p99 `<=250ms`, queue lag p99 `<=20s`, CPU `<=20s`, RSS delta `<=256MiB`. |

## Monitoring Commands Before Start

| Surface | Command / query |
|---|---|
| Monit summary | `/opt/homebrew/opt/monit/bin/monit -c /opt/homebrew/etc/monitrc summary` |
| Strategy producer readiness | `curl -fsS http://127.0.0.1:9207/health/ready` |
| Exchange-execution readiness | `curl -fsS http://127.0.0.1:9206/health/ready` |
| Prometheus selected jobs | `up{job=~"node-exporter|strategy-producer|exchange-execution|blackbox-http"}` |
| Host CPU busy | `100 * (1 - avg(rate(node_cpu_seconds_total{job="node-exporter",mode="idle"}[5m])))` |
| Host load | `node_load1{job="node-exporter"}`, `node_load5{job="node-exporter"}` |
| Host memory free | `100 * node_memory_free_bytes{job="node-exporter"} / node_memory_total_bytes{job="node-exporter"}` |
| Strategy metrics | `curl -fsS http://127.0.0.1:9207/metrics` filtered to `strategy_producer_*` and `strategy_live_runner_*` |
| Exchange metrics | `curl -fsS http://127.0.0.1:9206/metrics` filtered to `exchange_execution_*`, `execution_reconciliation_total`, limiter and latency metrics |
| Process CPU/RSS | `pgrep -f <process-pattern>` plus `ps -o pid,ppid,etime,%cpu,rss,command -p <pid>` for strategy-live-runner, exchange-execution, and apps-api |
| Redis transport | `XLEN execution.requests.v1`, `XPENDING execution.requests.v1 exchange-execution.v1`, `XLEN execution.requests.retry.v1`, `XLEN execution.requests.dlq.v1` |
| DB state | Counts by `strategy_runs.state`, `strategy_live_profiles.mode/readiness_status`, `execution_intents.status`, `execution_orders.environment/status`, `execution_reconciliation_runs.status`, unknown outbox rows, and latest exchange-execution heartbeat. |
| Controlled burst | `uv run python -m apps.exchange_execution.load_harness --strategies 180 --read-count 25 --rate-limit-per-second 60 --rate-limit-burst 20 --pretty` |

Prometheus service process metrics `process_cpu_seconds_total` and `process_resident_memory_bytes` are not exposed for `strategy-producer`/`exchange-execution`; process-level CPU/RSS evidence therefore uses the accepted repository `ps` sampling method also used by prior benchmark/resource evidence.

## Pre-Start Baseline

| Surface | Value |
|---|---|
| Mac Studio checkout | `main` at `921711e222b9afbaec9fa8460470965072529275`, clean vs `origin/main`. |
| API / producer / exchange readiness | API `/health` returned `ok`; strategy producer `/health/ready` returned `ready=true`, producer disabled, empty allowlists; exchange-execution `/health/ready` returned `status=ready`, `adapter_mode=testnet`. |
| Monit | `roehub_strategy_live_runner`, `roehub_exchange_execution`, `roehub_exchange_control`, `roehub_keycloak`, `roehub_openbao`, market-data workers, and `roehub_backtest_job_runner` all `OK`. |
| Prometheus | `up=1` for `node-exporter`, `strategy-producer`, `exchange-execution`, blackbox API/OpenAPI/ClickHouse probes; known unrelated `backtest-artifact-publisher` target remains outside Stage `12` acceptance. |
| Host resource snapshot | CPU busy `46.731%`, load1 `5.8125`, load5 `6.4141`, memory free `60.251%`. |
| Process snapshot | `strategy-live-runner` pid `86135`, RSS `68784 KiB`; `exchange-execution` pid `86126`, RSS `97792 KiB`; `apps-api` pid `86144`, RSS `235232 KiB`. |
| Strategy DB baseline | `strategy_runs`: `failed=9`, `stopped=27`; no `running` rows. `strategy_live_profiles`: `paper/ready=30`, `testnet/blocked=2`, `monitor_only/ready=6`, `live/ready=2`, `live/blocked=2`. |
| Execution DB baseline | `execution_source_events=183`; `execution_intents`: `accepted=2`, `dispatched=40`, `quarantined=1`, `recorded=6`, `rejected=72`, `retry=1`; `execution_orders`: `mainnet/guard_rejected=2`, `testnet/adapter_error=9`, `testnet/cancelled=18`, `testnet/guard_rejected=5`, `testnet/status_checked=1`; `execution_reconciliation_runs`: `matched=7`, `pending=30`; unknown outbox rows `1`. |
| Historical pending/unknown reason | Pending reconciliation: `adapter_error_reconciliation_pending/adapter_error=18`, `funding_reconciliation_pending/status_checked/new=12`; unknown outbox row is historical `2026-06-03 ... producer_unknown/stage16_unknown_state`. Acceptance threshold is no increase during this stage. |
| Redis baseline | `execution.requests.v1` length `41`, consumer pending `0`, retry length `1`, DLQ length `2`. |

## Evidence Snapshots

| Snapshot | Planned time | Status | Summary |
|---|---:|---:|---|
| Baseline / pre-burst | `2026-06-19T22:35:00Z` | passed | Monit OK, API/producer/exchange readiness OK, Prometheus `up=1`, host CPU busy `44.996%` at `22:35Z` / `46.999%` at `22:36Z`, load1 about `6.1`, memory free `56.8-57.1%`, producer/admin disabled with `0` polled runs, Redis pending `0`, request length `41`, retry `1`, DLQ `2`, DB counts unchanged. Initial collector missed Redis due non-login `PATH`; supplemental baseline at `22:36Z` used explicit Homebrew paths and completed Redis/DB evidence. |
| During burst | `2026-06-20T00:35:00Z` .. `00:50:00Z` | blocked | Collector reached `BURST START 2026-06-20T00:35:00Z`, but `uv` was not available in the non-login SSH PATH, so the Stage `11` harness command did not execute. The immediate snapshot still showed Monit OK, API/producer/exchange readiness OK, Prometheus `up=1`, CPU busy `41.934%`, load1 `5.9307`, load5 `5.8530`, memory free `54.969%`, producer disabled with `0` polled runs, Redis pending `0`, retry `1`, DLQ `2`, and DB counts unchanged. |
| Post-burst | `2026-06-20T01:05:00Z` | not collected | The collector exited after the failed burst command, before the scheduled post-burst snapshot. |
| Final | `2026-06-20T04:35:00Z` | not collected | Fresh SSH access to `macstudio` failed after the collector exit, so final host/browser/runtime proof could not be collected. |

## Validation Log

| Check | Result | Evidence |
|---|---:|---|
| Stage `11` accepted gate | passed | Ledger row marks Stage `11` `accepted`, blocker `none`, next stage allowed `yes`. |
| Pre-soak docs index | passed | `python -m tools.docs.generate_docs_index --check` -> `OK: docs/architecture/README.md is up-to-date`. |
| Resource telemetry discovery | passed | Existing Prometheus node-exporter host CPU/load/memory, Monit process supervision, service metrics, and `ps` process CPU/RSS are available. |
| Baseline collector | passed with recovered path issue | Durable log: `/tmp/roehub-stage12-soak-20260619T2235Z.log`. The first baseline captured Monit/Prometheus/process evidence and stopped on `redis-cli` path lookup. Restarted collector at `2026-06-19T22:36:00Z` with `/opt/homebrew/bin/redis-cli` and `/opt/homebrew/bin/psql`; Redis/DB baseline completed before the later burst command failure. |
| Controlled burst | blocked | At `2026-06-20T00:35:00Z`, remote shell reported `run_burst:3: command not found: uv`; the `180`-strategy harness did not run, so controlled burst acceptance evidence is missing. |
| SSH recovery | blocked | After collector exit, fresh `ssh macstudio`, explicit key, `studio-vpn`, `studio-lan`, and Tailscale SSH attempts failed with public-key/authentication errors or host-key wrapper issues. No final runtime or browser proof was collected. |

## File Manifest

| Created | Modified | Deleted | Reason | Contract impact |
|---|---|---|---|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-supervised-6h-soak.md` | none | none | Stage `12` pre-start plan, thresholds, runtime evidence, final decision. | none: documentation/evidence artifact. |
| none | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | none | Blocked Stage `12` handoff and Stage `13` gate protection. | none: documentation handoff. |
| none | `docs/architecture/README.md` | none | Docs index consistency after adding the Stage `12` report. | none: documentation index. |

## Decision

Final decision: `blocked`.

Blockers:

- Controlled burst did not run because `uv` was missing from the non-login SSH PATH used by the collector. Recovery path: run the collector with an explicit `uv` path or source the Mac Studio login/runtime environment before the burst command.
- SSH access to Mac Studio failed after the collector exited, so the gate could not be resumed, corrected, or finalized. Recovery path: restore SSH/Tailscale SSH auth, then rerun the full 6-hour Stage `12` gate from a fresh pre-start baseline.
- Final browser status was not captured because runtime access was lost before the end of the window.

Acceptance impact: Stage `12` is not accepted. Stage `13` remains blocked.

## Rerun Attempt 2 Pre-Start

User required before start: nothing.

Stage `11` gate: `accepted` в `strategy-producer-paper-testnet-trading-v1-stage-ledger.md`; Stage `12` rerun may start as a repair/unblock run. Blockers before rerun start: `none`.

Preflight at `2026-06-20T21:45:53Z`: `ssh macstudio` succeeded; Mac Studio host `MacStudioDaniil`; `/Users/daniildegtyarev/Projects/roehub.com` is on `main` at `921711e222b9afbaec9fa8460470965072529275`; `/opt/homebrew/bin/uv`, `/opt/homebrew/bin/redis-cli`, `/opt/homebrew/bin/psql`, and Monit are available; API `/health`, strategy producer `/health/ready`, and exchange-execution `/health/ready` returned healthy/ready. Remote git checkout has unrelated market-data/backtest working-tree edits; Stage `12` collector will not modify them.

Concrete file list before rerun edits remains:

| Path | Planned action | Reason |
|---|---:|---|
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/12-supervised-6h-soak.md` | update | Record rerun pre-start, thresholds, evidence snapshots, final decision. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify after validation | Stage status, handoff, publish/deploy evidence. |
| `docs/architecture/README.md` | update only if generated index changes | Required docs index consistency after Markdown changes. |

Files outside prompt expected paths: none planned. Existing unrelated market-data/backtest edits in the local and Mac Studio checkouts are out of scope and must not be staged or reverted for this gate.

Rerun plan:

| Item | Predeclared value |
|---|---|
| Start time | `2026-06-20T22:00:00Z` / `2026-06-21 01:00 MSK` |
| Planned end time | `2026-06-21T04:00:00Z` / `2026-06-21 07:00 MSK` |
| Baseline strategy count | Re-read at baseline; expected `0` running strategy runs with producer admin disabled and allowlists empty. |
| Controlled burst strategy count | `180` synthetic `testnet` strategies via accepted Stage `11` harness. |
| Burst plan | Start `2026-06-21T00:00:00Z`; expected command duration `< 5m`; post-burst observation at about `2026-06-21T00:15:00Z`. |
| Mode mix | Baseline runtime observed in DB; burst is `testnet=100%`; paper profiles are observed as readiness baseline only. |
| Exchanges / markets | Burst harness covers Binance + Bybit, spot + futures, `BTCUSDT`; no real provider HTTP and no mainnet. |
| Runtime path fix | Collector invokes `/opt/homebrew/bin/uv`, `/opt/homebrew/bin/redis-cli`, and `/opt/homebrew/bin/psql` explicitly and runs repository commands under `zsh -lc` from `/Users/daniildegtyarev/Projects/roehub.com`. |
| Kill switches | Producer admin remains disabled; exchange-execution adapter mode must stay `testnet`; Monit can stop `com.roehub.strategy-live-runner` and `com.roehub.exchange-execution`; any mainnet attempt or unknown new order state blocks acceptance. |
| Cleanup plan | Do not create production orders/positions; verify Redis pending remains `0`; verify DB counts do not grow in unknown/retry/DLQ categories beyond baseline; no repo temp artifacts retained. |
| Resource threshold | Host CPU busy p95 over snapshots `< 85%`; load1/load5 must not show sustained saturation; memory free must stay `> 15%`; process RSS for producer, exchange-execution, and API must not grow by `> 256 MiB` from baseline. |
| Queue/state threshold | Redis pending `0` at baseline/post/final; retry stream length and DLQ length must not increase; DB `execution_reconciliation_runs.status='pending'` and unknown outbox count must not increase. |
| Burst threshold | Harness result `passed=true`, `violations=[]`, `retry_count=0`, `dlq_count=0`, `redis_pending_final=0`, limiter wait observed and p99 `<=250ms`, queue lag p99 `<=20s`, CPU `<=20s`, RSS delta `<=256MiB`. |

Rerun collector log: `/tmp/roehub-stage12-soak-rerun-20260620T2200Z.log`.
