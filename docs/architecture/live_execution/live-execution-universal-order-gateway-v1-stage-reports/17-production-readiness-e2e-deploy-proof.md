---
doc: live-execution-stage-17-production-readiness-e2e-deploy-proof
stage: "17"
status: accepted
canonical_plan: docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md
ledger: docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md
---

# Stage 17: Production Readiness E2E Deploy Proof

Status: `accepted`.

Stage 17 is accepted after an approved bounded Bybit spot testnet fill canary
on the deployed Mac Studio runtime. The canary used `cancel_after_submit=false`
only for the bounded Stage 17 run, submitted no mainnet orders, printed no
secrets, used a minimal `6.0` USDT notional, restored the runtime env after the
run, and proved DB/Redis/exchange/reconciliation/notification/metrics/public
UI convergence.

Previous stage: Stage `16` is accepted in the iteration ledger.

## Approval

The canary was explicitly approved for Stage 17:

- one safe Bybit spot testnet market-fill smoke;
- `cancel_after_submit=false` only for this canary;
- no mainnet submit;
- no secrets in reports;
- minimal testnet notional;
- kill-switch/rollback ready;
- mandatory DB, Redis, exchange, reconciliation, notification, metrics and UI
  evidence after the run.

## Files Changed

Code:

- `apps/exchange_execution/main/app.py`
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

- `tests/unit/apps/exchange_execution/test_app.py`
- `tests/unit/contexts/live_execution/adapters/test_redis_exchange_execution_consumer.py`
- `tests/unit/contexts/live_execution/test_exchange_execution_process.py`
- `tests/unit/infra/test_monitoring_assets.py`

Unrelated dirty backtest docs in the working tree were left untouched.

## Implementation Summary

- Added `ExchangeExecutionConsumer.read_pending_requests()` and implemented it
  for the Redis consumer using `XREADGROUP ... 0`.
- `ExchangeExecutionProcessService.run_once()` now drains this consumer's
  pending messages before reading new `>` messages, so restarts can self-heal
  old pending entries through the normal durable-state path.
- Added Stage 17 production Prometheus rules, bootstrap installation, Monit and
  runbook alert actions for DLQ growth, clock drift, private stream, dispatch
  backpressure/rate pressure, reconciliation pending, PITR and unknown state.
- Added `ROEHUB_EXCHANGE_EXECUTION_CANCEL_AFTER_SUBMIT` as a runtime env
  override over the YAML adapter setting. The production default remained safe;
  the override was set to `false` only around the approved Stage 17 canary and
  removed by the cleanup trap.
- Fixed the production-discovered Redis dispatch race: API dispatch publishes
  the Redis message while the intent is still `dispatching`, then marks it
  `dispatched`. A live consumer may read the message in that window, so
  `exchange-execution` now accepts `dispatching` or `dispatched` when
  `risk_status=accepted`. Other statuses remain non-dispatchable.
- Fixed the earlier unsupported-exchange pending-drain bug so synthetic
  unsupported prefixes produce durable observations and Redis ack without
  inserting invalid order-ledger rows.

## Real-Boundary Evidence

Direct-main publish/deploy:

- Original Stage 17 ops implementation commit `dac40caa`; CI `26904424782`,
  Publish App Image `26904532691`, Deploy Backend `26904532675`, Deploy Web
  `26904532673` and follow-up Deploy Web `26904545441` succeeded.
- Pending-drain follow-up commit `88ef1fe0`; CI `26904770456`,
  Publish App Image `26904959252`, Deploy Backend `26904956680`, and Deploy
  Web `26904960071` succeeded.
- Canary override commit `45bba8d3`; CI `27027849660`, Publish App Image
  `27027943193`, Deploy Backend `27027943230`, Deploy Web `27027943220` and
  follow-up Deploy Web `27027952958` succeeded.
- Dispatch race fix commit `fc4d917f`; CI `27028500672`, Publish App Image
  `27028656084`, Deploy Backend `27028656105`, Deploy Web `27028656094` and
  follow-up Deploy Web `27028664901` succeeded.

Mac Studio runtime after `fc4d917f`:

- `/opt/roehub/app` contains the deployed race fix:
  `intent.status not in {"dispatching", "dispatched"}`.
- `GET http://127.0.0.1:9206/health/ready` returned `status=ready`,
  `status_reason=all_dependencies_ready`, `adapter_mode=testnet`,
  Redis `pending_count=0`, DLQ stream length `2`,
  `ledger_pitr=pitr_restore_verified`, clock drift `0.052 ms`, and Postgres
  heartbeat ready.
- `bash /opt/roehub/app/scripts/macos/smoke_prod.sh` passed.
- `XPENDING execution.requests.v1 exchange-execution.v1` returned `0`.
- `XLEN execution.requests.dlq.v1` was `2` before and after the final canary.
  The Stage 17 accepted run did not add DLQ entries. The extra DLQ over the
  historical Stage 12 entry came from the pre-fix Stage 17 race probe and is
  recorded below.

Approved canary:

- Marker: `stage17-canary-20260605-03`.
- Strategy: `c9c7ec41-1698-4c3b-8355-f654208ef1ba`.
- Signal: `4466df03-d3f3-4f58-bc91-71553de458e2`.
- Source event: `75c78ca9-fcc4-4442-97fb-03255f7ea010`.
- Intent: `459a229a-eb44-449b-a92c-021be430cbad`.
- Order: `94137594-41c2-4d6b-89f1-448ed8b672e7`.
- Exchange order id: `2230892879675746560`.
- Exchange/environment: `bybit` / `testnet`.
- Instrument/order: `bybit:spot:BTCUSDT`, market buy, quote notional `6.0`.
- Order status: `status_checked / filled`.
- Fill: one provider trade id `2100000000183760587`, price
  `63879.200000000000`, quantity `0.000093000000`, fee
  `1.67400E-7 BTC`.
- Cancel proof: `cancel_requested_at=NULL`, `cancelled_at=NULL`.
- Reconciliation: one run, `matched / spot_order_status_and_fills_matched`,
  provider status `filled`, fill count `1`, funding event count `0`,
  `spot_funding_not_applicable=1`.
- Notifications: `producer_fill` and `producer_terminal`, both
  `info / spot_order_status_and_fills_matched`.
- Source outcome: `filled / spot_order_status_and_fills_matched`.
- Redis: request stream length `14 -> 15`, pending `0 -> 0`, DLQ `2 -> 2`.
- Exchange metrics after canary included
  `exchange_execution_testnet_order_total{exchange="bybit",reason="submitted"} 1`,
  `exchange_execution_private_stream_total{exchange="bybit",reason="private_ws_auth_probe_ready"} 1`,
  `exchange_execution_submit_latency_ms_count{exchange="bybit"} 1`, and
  `exchange_execution_redis_pending 0`.
- Measured latency:
  - source -> intent: `52.534 ms`;
  - intent -> Redis dispatch: `56.396 ms`;
  - dispatch -> submitted: `1115.908 ms`;
  - submitted -> checked: `325.736 ms`;
  - source -> first fill: `1217.995 ms`;
  - adapter submit latency: `268.931291 ms`.
- Public Bybit testnet last price before the canary was `63879.2`; first fill
  price was `63879.2`; slippage versus the pre-submit last price was `0 bps`.
- The temporary canary identity session was revoked and the synthetic strategy
  run was stopped after evidence capture.

Public UI evidence:

- Playwright opened
  `https://roehub.com/strategies?strategy_id=c9c7ec41-1698-4c3b-8355-f654208ef1ba`.
- Dashboard data request returned `200` for
  `/api/ui/strategies/dashboard?refresh=initial&state=all&strategy_id=...`.
- The page showed `Execution outcomes` as `ready`.
- The canary row showed:
  `strategy_signal: filled / spot_order_status_and_fills_matched`,
  `dispatched / risk_gate_accepted`,
  `status_checked / filled`,
  `producer_terminal / spot_order_status_and_fills_matched`.
- Screenshot:
  `output/playwright/stage17-canary-20260605-03-roehub-strategies.png`.
- The public shell also showed a non-blocking current-user badge error
  `Unexpected current-user status: 502`; the strategy dashboard API and
  execution-outcome table still loaded successfully. This is a residual UI
  shell/auth edge follow-up, not a blocker for the Stage 17 execution outcome
  proof.

## Runtime Findings Fixed During Acceptance

Two pre-acceptance probes found real blockers and did not move money on
mainnet:

- `stage17-canary-20260605-01` failed before API order submission because the
  synthetic signal used invalid `strategy_signals.signal_action='buy'`.
  Constraint proof showed valid actions are `none/open/close/reduce/reverse`;
  the accepted canary used `open`, `side=buy`, `outcome=signal`.
- `stage17-canary-20260605-02` exposed the Redis dispatch race and DLQed one
  accepted intent as `intent_not_dispatchable` before any order row existed.
  The race was fixed in `fc4d917f` and covered by
  `test_testnet_adapter_accepts_dispatching_intent_after_redis_publish_race`.

## Quality Gates

Passed for the canary override commit:

- `uv run pytest -q tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py`
- `uv run ruff check apps/exchange_execution/main/app.py src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py`
- `uv run pyright apps/exchange_execution/main/app.py src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py`
- `uv run ruff check src apps tests`
- `uv run pyright src apps tests`
- `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps/exchange_execution tests/unit/apps/api/test_ui_execution_routes.py tests/unit/infra/test_monitoring_assets.py`
- `python -m tools.docs.generate_docs_index --check`
- `git diff --check`

Passed for the dispatch race fix:

- `uv run pytest -q tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py`
- `uv run ruff check src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py tests/unit/contexts/live_execution/test_exchange_execution_process.py`
- `uv run pyright src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py tests/unit/contexts/live_execution/test_exchange_execution_process.py`

Repository limitation:

- Literal `tests/integration` and `tests/e2e` directories do not exist in this
  checkout. The required literal
  `uv run pytest -q tests/unit tests/integration tests/e2e` cannot run as
  written.

## Contract Impact

| Surface | Classification | Notes |
|---|---|---|
| Public/API | none | No public route or DTO contract changed. |
| Persistence | none | No schema change; all canary facts are append-only runtime rows. |
| Redis | compatible-change | Consumer drains pending before new messages and accepts the API publish race state `dispatching/accepted`. Stream names/group are unchanged. |
| Config | compatible-change | Added production Prometheus rule installation and a bounded env override for `cancel_after_submit`; default remains safe. |
| Runtime/ops | compatible-change | Alert/runbook coverage added; canary override was temporary and restored after the run. |
| UI/browser | compatible-change | No UI code changed in Stage 17; public `/strategies` outcome table displays the new producer outcome row. |
| Metrics/logs | compatible-change | Uses existing bounded metric labels and new alert rules. |
| External side effects | compatible-change | One approved Bybit spot testnet market buy was submitted and filled. No mainnet submit occurred. |

## Secrets And Redaction

Reports and runtime evidence record only ids, statuses, counts, timestamps,
bounded reasons, metric names and non-secret provider ids. No API keys,
secrets, passphrases, OpenBao tokens, cookies, raw Authorization headers,
ciphertext, raw signed provider payloads or sensitive provider responses were
written.

## Rollback

- The immediate kill switch remains:
  `launchctl bootout gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.exchange-execution.plist`.
- Restore with:
  `launchctl bootstrap gui/$(id -u) /Users/daniildegtyarev/Library/LaunchAgents/com.roehub.exchange-execution.plist`
  and `curl -fsS http://127.0.0.1:9206/health/ready`.
- To disable exchange side effects by config, set
  `configs/prod/exchange_execution.yaml` `exchange_execution.process.adapter_mode`
  to `disabled` in a reviewed rollback change and redeploy/restart
  `com.roehub.exchange-execution`.
- Remove any emergency
  `ROEHUB_EXCHANGE_EXECUTION_CANCEL_AFTER_SUBMIT=false` env override and restart
  `com.roehub.exchange-execution`; the accepted canary cleanup already proved
  the override was not left in `/Users/daniildegtyarev/.config/roehub/roehub.env`.
- Do not delete source/intent/order/fill/reconciliation/outbox ledgers during
  rollback. Reconcile by Postgres/provider truth before any retry.

## Next Handoff

Stage `17` is accepted. Stage `18` can start from the accepted producer-neutral
path: source event -> risk -> Redis dispatch -> exchange-execution -> order/fill
ledger -> reconciliation -> notification outbox -> `/strategies` outcome link.
Mainnet remains blocked until a later stage explicitly approves mainnet submit
policy, notional limits, recent-auth/kill-switch operations and rollback drills.
