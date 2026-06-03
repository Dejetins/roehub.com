---
doc: live-execution-stage-17-production-readiness-e2e-deploy-proof
stage: "17"
status: blocked
canonical_plan: docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md
ledger: docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md
---

# Stage 17: Production Readiness E2E Deploy Proof

Status: `blocked`.

Stage 17 delivered production-readiness monitoring/runbook assets and repaired
the `exchange-execution` pending-message self-heal path on `main`, but the
stage is not accepted. The required single fresh full safe journey from
backtest variant through browser-visible strategy outcome, source event, risk,
Redis, testnet fill, reconciliation, notification, latency and slippage was not
completed in this run.

Previous stage: Stage `16` is accepted in the iteration ledger.

## Blocker

The remaining acceptance blocker is fresh full-path proof, not local tests or
deployment.

- The production process is configured with `cancel_after_submit=true`.
- Prior accepted Stage `15` market-fill evidence proved that a Bybit market
  order can fill, but the post-fill cancel attempt records a known
  `exchange_ret_code_170213` adapter error and a pending reconciliation row.
- Running a new fill smoke under that same runtime setting would intentionally
  create a reconciliation-pending alert window, which conflicts with this
  stage's requirement for no unexplained pending/DLQ/unknown state after the
  E2E run.
- No separate explicit canary approval or runtime contract currently allows
  a bounded market-fill smoke with `cancel_after_submit=false`.

Required before acceptance: define and approve a safe Stage 17 fill canary
variant, most likely a testnet-only market-fill smoke that disables
`cancel_after_submit` for the bounded canary or an equivalent order path that
produces a fill without creating the known cancel-after-fill adapter error.
Then rerun Playwright/API/DB/Redis/exchange/metrics evidence as one correlated
journey and record source-to-fill latency plus slippage from runtime facts.

## Files Changed

Code:

- `src/trading/contexts/live_execution/application/ports/exchange_execution_consumer.py`
- `src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py`
- `src/trading/contexts/live_execution/adapters/outbound/redis/exchange_execution_consumer.py`

Config/ops:

- `infra/macos/prometheus/prometheus.prod.yml`
- `infra/macos/prometheus/rules/live-execution-stage17.rules.yml`
- `infra/scripts/monit/roehub-exchange-execution.monitrc`
- `scripts/macos/bootstrap_native_prod.sh`

Docs:

- `docs/runbooks/exchange-execution.md`
- `docs/runbooks/mac-studio-monitoring-plan.md`
- this report
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md`

Tests:

- `tests/unit/contexts/live_execution/adapters/test_redis_exchange_execution_consumer.py`
- `tests/unit/contexts/live_execution/test_exchange_execution_process.py`
- `tests/unit/infra/test_monitoring_assets.py`

Unrelated dirty backtest docs in the working tree were left untouched.

## Implementation Summary

- Added `ExchangeExecutionConsumer.read_pending_requests()` and implemented it
  for the Redis consumer using `XREADGROUP ... 0`.
- `ExchangeExecutionProcessService.run_once()` now reads pending messages for
  its own consumer before new `>` messages, so process restarts can self-heal
  old pending entries through the normal durable-state path.
- Added Stage 17 production Prometheus rule file and bootstrap installation
  into `/opt/roehub/config/prometheus.rules/`.
- Added alert/runbook coverage for DLQ, clock drift, private stream, dispatch
  backpressure/rate pressure, reconciliation pending, PITR not verified and
  unknown state. Rules include severity, owner, escalation and runbook action.
- Added runbook canary, rollback, kill-switch and alert action sections.
- Fixed a production-discovered pending-drain bug: synthetic unsupported
  exchange prefixes such as `codexstage16:*` now produce a durable request
  observation and Redis ack without inserting an invalid `execution_orders`
  row that violates the existing `binance|bybit` order-ledger constraint.

## Real-Boundary Evidence

Direct-main publish/deploy:

- Commit `dac40caa` pushed to `main`; CI `26904424782`, Publish App Image
  `26904532691`, Deploy Backend `26904532675`, Deploy Web `26904532673` and
  follow-up Deploy Web `26904545441` succeeded.
- Follow-up fix commit `88ef1fe0` pushed to `main`; CI `26904770456`,
  Publish App Image `26904959252`, Deploy Backend `26904956680` and Deploy Web
  `26904960071` succeeded.

Mac Studio runtime after `88ef1fe0`:

- `/opt/roehub/app` contains the pending-reader fix.
- `POST http://127.0.0.1:9206/internal/v1/run-once` returned
  `read_count=0`, `acked_count=0`, `reason=testnet_adapter_processed` after
  the background consumer had already drained pending messages.
- `GET http://127.0.0.1:9206/health/ready` returned
  `status=ready`, `status_reason=all_dependencies_ready`,
  `adapter_mode=testnet`, Redis `pending_count=0`, DLQ stream length `1`,
  `ledger_pitr=pitr_restore_verified`, clock drift `0.069 ms` and Postgres
  heartbeat ready.
- `XPENDING execution.requests.v1 exchange-execution.v1` returned `0`.
- `XLEN execution.requests.dlq.v1` stayed `1`; the only latest DLQ entry is the
  pre-existing Stage 12 `stage12_poison_probe`.
- SQL observations for old pending ids:
  - `1780251280150-0`: historical `adapter_disabled/adapter_disabled_stage13`
    followed by `guard_rejected/exchange_connection_not_found` in testnet.
  - `1780438818573-0`: `guard_rejected/exchange_adapter_not_enabled` in
    testnet.
- SQL `execution_orders` for Stage 16 synthetic intent
  `e1c709c4-a27a-4f0f-8e77-f93019957e36` returned count `0`, proving the fix
  avoided the invalid `codexstage16` order row.
- SQL `execution_orders` for the older Stage 12 Binance intent recorded
  `guard_rejected/exchange_connection_not_found`, preserving durable truth.

Monitoring/runtime:

- Prometheus loaded group `live-execution-production-readiness` with all seven
  Stage 17 alerts inactive:
  `LiveExecutionDlqGrowing`, `LiveExecutionClockDriftUnsafe`,
  `LiveExecutionPrivateStreamMissingForSubmit`,
  `LiveExecutionDispatchBackpressure`, `LiveExecutionReconciliationPending`,
  `LiveExecutionPitrNotVerified`, `LiveExecutionUnknownState`.
- Prometheus query `up{job="exchange-execution"}` returned `1`.
- Monit with explicit control file
  `/opt/homebrew/etc/monitrc` showed `roehub_exchange_execution`,
  `roehub_exchange_control` and `roehub_openbao` as `OK`.
- `bash /opt/roehub/app/scripts/macos/smoke_prod.sh` passed.

Proof not collected:

- No fresh Stage 17 Playwright journey was executed for a new backtest variant
  through strategy run, signal, fill, reconciliation and notification.
- No fresh source-to-fill latency or slippage sample was recorded for Stage 17.
- No fresh testnet fill was submitted during Stage 17 after the canary blocker
  above was identified.

## Quality Gates

Passed before the first publish:

- `uv run pytest -q tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/contexts/live_execution/adapters/test_redis_exchange_execution_consumer.py tests/unit/contexts/live_execution/adapters/test_redis_execution_dispatch_transport.py tests/unit/apps/exchange_execution/test_app.py`
- `uv run pytest -q tests/unit/infra/test_monitoring_assets.py`
- `uv run ruff check` over touched Stage 17 files
- `uv run pyright` over touched Stage 17 files
- `uv run ruff check src apps tests`
- `uv run pyright src apps tests`
- `uv run pytest -q tests/unit` (`1093 passed, 3 warnings`)
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check` over Stage 17 files

Passed for the follow-up pending-drain fix:

- `uv run pytest -q tests/unit/contexts/live_execution/test_exchange_execution_process.py`
- `uv run pytest -q tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/contexts/live_execution/adapters/test_redis_exchange_execution_consumer.py tests/unit/apps/exchange_execution/test_app.py tests/unit/infra/test_monitoring_assets.py`
- `uv run ruff check src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py tests/unit/contexts/live_execution/test_exchange_execution_process.py`
- `uv run ruff check src apps tests`
- `uv run pyright src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py tests/unit/contexts/live_execution/test_exchange_execution_process.py`
- `uv run pyright src apps tests`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`

Repository limitation:

- The literal `tests/integration` and `tests/e2e` directories do not exist in
  this checkout. The required literal
  `uv run pytest -q tests/unit tests/integration tests/e2e` cannot run as
  written.

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public/API | none | No public route or DTO contract changed. |
| Persistence | none | No schema change. Unsupported exchange prefixes no longer attempt invalid order-row persistence. |
| Redis | compatible-change | `exchange-execution` now recovers its own pending messages before reading new messages. Stream names/group remain unchanged. |
| Config | compatible-change | Adds production Prometheus rule file and bootstrap installation path. |
| Runtime/ops | compatible-change | Adds alert/runbook coverage; pending self-heal reduces restart recovery risk. |
| UI/browser | none | No browser-visible code changed in Stage 17. Browser acceptance remains blocked. |
| Metrics/logs | compatible-change | Adds alerting over existing bounded metrics; no new unbounded labels. |
| External side effects | none for this run | No fresh testnet or mainnet order was submitted during Stage 17. Mainnet remains blocked. |

## Secrets And Redaction

Reports and runtime evidence recorded only ids, statuses, counts, timestamps,
bounded reasons and metric names. No API keys, secrets, passphrases, OpenBao
tokens, cookies, raw Authorization headers, ciphertext, raw signed provider
payloads or sensitive provider responses were written.

## Rollback

- To disable exchange side effects, set
  `configs/prod/exchange_execution.yaml` `exchange_execution.process.adapter_mode`
  to `disabled` in a reviewed rollback change and redeploy/restart
  `com.roehub.exchange-execution`.
- To stop the runtime immediately:
  `launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.exchange-execution.plist`.
- Restore with:
  `launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.exchange-execution.plist`
  and `curl -fsS http://127.0.0.1:9206/health/ready`.
- Do not delete source/intent/order/fill/reconciliation/outbox ledgers during
  rollback. Reconcile by Postgres/provider truth before any retry.

## Next Handoff

Stage 17 remains blocked for acceptance. The next attempt should start from
commits `dac40caa` and `88ef1fe0`, keep the deployed pending self-heal and
alert/runbook assets, then add or approve a bounded testnet fill canary that
does not create a known cancel-after-fill reconciliation-pending state. After
that, execute one correlated Playwright/API/DB/Redis/exchange/metrics journey
from backtest variant through UI outcome link and record source-to-fill
latency plus slippage.
