# Stage 13: Exchange-Execution Process Skeleton

Stage 13 adds the supervised `exchange-execution` process boundary with
health, metrics, Redis observation, disabled-adapter guards, heartbeat rows,
DLQ quarantine for poison/non-dispatchable messages, launchd/Monit/Prometheus
assets, and the operator runbook.

Date: 2026-05-31.

Status: accepted after direct-main deploy, follow-up native-service install
repair, Monit reload repair, CI/deploy success, and Mac Studio runtime proof.

## Scope

Included:

- separate `apps/exchange_execution` FastAPI process with `/health`,
  `/health/ready`, `/metrics`, and a local internal `run-once` diagnostic;
- exchange-execution process config under `configs/{dev,test,prod}`;
- Redis consumer group observation for `execution.requests.v1` through
  `exchange-execution.v1`;
- adapter-disabled mode that records valid dispatched/accepted intents but does
  not acknowledge them or call an exchange adapter;
- poison/non-dispatchable message handling that records a durable observation,
  writes a DLQ marker, and acks only after the durable observation succeeds;
- heartbeat and request-observation tables;
- bounded Prometheus metrics for readiness, dependency state, Redis stream
  length, pending count, clock drift, observations, DLQ and acknowledgements;
- launchd, Monit and Prometheus scrape/alert assets;
- runbook `docs/runbooks/exchange-execution.md`;
- focused process, Redis adapter, app and migration tests.

Out of scope:

- no exchange SDK/API call, credential decrypt, signed payload or order submit;
- no mainnet or testnet order submission;
- no private stream lifecycle;
- no order/fill/reconciliation ledger yet;
- no browser-visible surface change.

## Local Evidence

| Gate | Command | Result |
|---|---|---|
| Focused tests | `uv run pytest -q tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/contexts/live_execution/adapters/test_redis_exchange_execution_consumer.py tests/unit/apps/exchange_execution/test_app.py tests/unit/apps/migrations/test_exchange_execution_process_skeleton_sql.py` | `6 passed`. |
| Focused ruff | `uv run ruff check apps/exchange_execution src/trading/contexts/live_execution tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/contexts/live_execution/adapters/test_redis_exchange_execution_consumer.py tests/unit/apps/exchange_execution/test_app.py tests/unit/apps/migrations/test_exchange_execution_process_skeleton_sql.py` | Passed. |
| Focused pyright | `uv run pyright apps/exchange_execution src/trading/contexts/live_execution tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/contexts/live_execution/adapters/test_redis_exchange_execution_consumer.py tests/unit/apps/exchange_execution/test_app.py` | `0 errors`. |
| Required ruff scope | `uv run ruff check apps/exchange_execution src/trading/contexts/live_execution tests` | Passed. |
| Required pyright scope | `uv run pyright apps/exchange_execution src/trading/contexts/live_execution tests` | `0 errors`. |
| Required unit/apps scope | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps` | `329 passed, 3 warnings`. |
| Required integration path | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps tests/integration` | Blocked locally because `tests/integration` does not exist in this checkout; no tests ran for that literal path. |
| Full ruff | `uv run ruff check .` | Passed. |
| Full pyright | `uv run pyright` | `0 errors`. |
| Full local suite | `uv run pytest -q -ra` | `1080 passed, 3 warnings`. |
| Docs index | `uv run python -m tools.docs.generate_docs_index --check` | Passed. |
| Local process smoke | `ROEHUB_ENV=dev STRATEGY_FAIL_FAST=false ROEHUB_EXCHANGE_EXECUTION_CONFIG=configs/dev/exchange_execution.yaml uv run python -m apps.exchange_execution.main.main --host 127.0.0.1 --port 19206` plus `curl /health/ready`, `curl /metrics`, `POST /internal/v1/run-once` | Process started, `/health/ready` returned HTTP `200` with `status=degraded`, `adapter_disabled_stage13`, `postgres=ready`, `redis_consumer_disabled`, backpressure/DLQ/clock drift unavailable because dev consumer disabled; `/metrics` exposed `exchange_execution_ready`, dependency gauges and `exchange_execution_adapter_disabled 1.0`; `run-once` returned HTTP `409 consumer_disabled`. |
| Local Redis CLI | `redis-cli -h 127.0.0.1 -p 6379 PING` | Blocked on workstation: `redis-cli` is not installed. Target runtime proof must use Mac Studio. |
| Native service scripts | `bash -n scripts/macos/bootstrap_native_prod.sh scripts/macos/bootstrap_native_test.sh scripts/macos/reload_launchd_services.sh` | Passed. |
| Native service regression | `uv run pytest -q tests/unit/infra/test_native_service_assets.py tests/unit/tools/test_ci_route_changes.py` | `7 passed`. |
| Native service ruff | `uv run ruff check tests/unit/infra/test_native_service_assets.py tests/unit/tools/test_ci_route_changes.py` | Passed. |
| Native service pyright | `uv run pyright tests/unit/infra/test_native_service_assets.py tests/unit/tools/test_ci_route_changes.py` | `0 errors`. |
| Secret grep | `rg -n "(api_key|apikey|secret|token|cookie|authorization|passphrase|signature|ciphertext)" apps/exchange_execution src/trading/contexts/live_execution/adapters/outbound/redis/exchange_execution_consumer.py src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/13-exchange-execution-process-skeleton.md docs/runbooks/exchange-execution.md infra/macos/launchd/com.roehub.exchange-execution.plist infra/scripts/monit/roehub-exchange-execution.monitrc configs/prod/exchange_execution.yaml` | Only policy/report wording matched; no secret values, cookies, raw authorization headers, signed payloads or ciphertexts were present. |
| Whitespace | `git diff --check` | Passed. |

## Runtime Evidence

Direct-main delivery:

- implementation commit `35255ae5`;
- native service install follow-up `b12651f6`;
- Monit reload workflow follow-up `488db042`;
- CI `26722743278`, `26722919238`, and full matrix CI `26723039066`
  succeeded;
- Publish App Image `26722819985`, `26722959212`, and `26723120694`
  succeeded;
- Deploy Backend `26722819989`, `26722959240`, `26723120686`, and
  workflow-dispatch `26723143386` succeeded;
- Deploy Web `26722819995`, `26722824151`, `26722959214`,
  `26722963442`, and `26723120684` succeeded.

Mac Studio runtime proof:

- native assets installed:
  `/Users/daniildegtyarev/Library/LaunchAgents/com.roehub.exchange-execution.plist`
  and `/opt/homebrew/etc/monit.d/roehub-exchange-execution.monitrc`;
- `launchctl list` showed `com.roehub.exchange-execution` loaded with pid
  `78878`;
- `/opt/homebrew/bin/monit -c /opt/homebrew/etc/monitrc summary` showed
  `roehub_exchange_execution OK`;
- `GET http://127.0.0.1:9206/health` returned
  `{"status":"ok","service":"exchange-execution"}`;
- `GET http://127.0.0.1:9206/health/ready` returned HTTP `200` with
  `status=degraded`, `status_reason=adapter_disabled_stage13`,
  `adapter_mode=disabled`, `consumer_enabled=1`, `fail_fast=1`,
  Redis ready, DLQ ready, backpressure ready, clock drift ready and Postgres
  heartbeat recorded;
- `POST http://127.0.0.1:9206/internal/v1/run-once` returned HTTP `200`
  with `read_count=0`, `observed_count=0`, `acked_count=0`,
  `reason=adapter_disabled_no_submit`;
- `/metrics` exposed `exchange_execution_ready`,
  `exchange_execution_dependency_ready`, `exchange_execution_redis_stream_length`,
  `exchange_execution_redis_pending`, `exchange_execution_clock_drift_ms`, and
  `exchange_execution_adapter_disabled 1.0`;
- Prometheus active target for `job="exchange-execution"` was `health=up`,
  scrape URL `http://127.0.0.1:9206/metrics`, and
  `up{job="exchange-execution"} == 1`;
- Postgres showed one heartbeat row and one observation row; latest heartbeat
  was `exchange-execution|degraded|adapter_disabled_stage13`;
- `scripts/macos/smoke_prod.sh` completed successfully after the final backend
  deploy;
- no exchange adapter, credential decrypt, signed payload, submit, cancel,
  amend or order status call was introduced or invoked.

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | none | No public API or browser route is added. The process exposes local-only health, metrics and internal run-once diagnostics on `127.0.0.1:9206`. |
| Ports/DTO | compatible-change | Adds exchange-execution process repository/consumer ports and domain DTOs under `live_execution`. |
| Persistence | compatible-change | Adds heartbeat and request-observation tables. Existing execution source/intent/risk/dispatch tables are unchanged. |
| Redis | compatible-change | Adds the first consumer-side observation of the existing Stage 12 stream/group contract. Valid disabled-adapter observations are not acked; poison/non-dispatchable messages are DLQed and acked after durable observation. |
| Config | compatible-change | Adds `configs/{dev,test,prod}/exchange_execution.yaml`; prod binds `127.0.0.1:9206` and keeps `adapter_mode=disabled`. |
| Runtime/Ops | compatible-change | Adds launchd, Monit, Prometheus scrape and alert assets for `exchange-execution`. |
| UI/browser | none | No visible page or user workflow changed. |
| Metrics/logs/redaction | compatible-change | Adds bounded process metrics without user, strategy, connection, stream id, token, cookie, signed payload or secret-bearing labels. |
| External side effects | none | No exchange adapter, credential decrypt, signed payload, submit, cancel, amend or order status call exists in this stage. |

## Rollback

Stop and unload `com.roehub.exchange-execution`, remove its Monit entry, and
remove the Prometheus scrape/alert entries. The heartbeat and observation
tables are additive and can remain for audit. Valid Redis request messages are
not acked while adapters are disabled, so rollback does not silently lose
money-boundary work.

## Handoff To Stage 14

Stage `14` can rely on a separate supervised process boundary, disabled-adapter
health semantics, heartbeat/observation persistence, Redis group/DLQ handling,
and bounded metrics. It must add testnet-only native Binance/Bybit adapters,
credential resolution inside the exchange-execution boundary, exchange
server-time checks, limiter integration, private stream lifecycle skeleton and
mainnet hard-block evidence before any submit path is accepted.
