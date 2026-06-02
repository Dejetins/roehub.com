# Stage 15: Order Fill Reconciliation And PITR

Stage 15 adds append-only order events, fill and funding ledgers,
reconciliation runs, retention policy metadata and a PITR readiness gate for
the existing `exchange-execution` testnet order path.

Date: 2026-06-03.

Status: accepted. Implementation, direct-main CI/deploy, Mac Studio runtime
proof, real Bybit spot testnet fill/status/cancel reconciliation, retention
metadata and PITR restore-drill evidence are complete.

## Scope

Included:

- additive `execution_order_events`, `execution_fills`,
  `execution_funding_events`, `execution_reconciliation_runs`,
  `execution_ledger_retention_policies` and `execution_ledger_pitr_drills`
  migration;
- order status adapters that extract normalized Binance/Bybit fill facts
  without storing raw signed provider payloads;
- append-only order event recording for guard, pending, private stream
  backfill, submit, status, cancel and adapter-error decisions;
- fill/funding dedupe by provider fact id;
- reconciliation runs from status/backfill facts with explicit
  `funding_reconciliation_pending` for futures without funding facts and
  `spot_funding_not_applicable` for spot;
- `ledger.pitr_required` config with prod/test fail-closed readiness reason
  `pitr_restore_not_verified` until the configured restore-drill marker is
  present;
- bounded metrics `execution_reconciliation_total` and
  `execution_ledger_backup_restore_total`;
- runbook updates for DB diagnostics, retention and PITR.

Out of scope:

- no mainnet order submit;
- no browser/UI changes;
- no notification outbox implementation;
- no claim that futures funding/PnL is complete when provider funding facts are
  absent;
- no broad OMS or advanced order lifecycle changes.

## Local Evidence

| Gate | Command | Result |
|---|---|---|
| Focused ruff | `uv run ruff check apps/exchange_execution src/trading/contexts/live_execution tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py tests/unit/apps/migrations/test_execution_reconciliation_pitr_sql.py alembic/versions/20260602_0029_execution_reconciliation_pitr_v1.py` | Passed. |
| Focused tests | `uv run pytest -q tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py tests/unit/apps/migrations/test_execution_reconciliation_pitr_sql.py` | `11 passed`. |
| Focused pyright | `uv run pyright apps/exchange_execution src/trading/contexts/live_execution tests/unit/contexts/live_execution/test_exchange_execution_process.py tests/unit/apps/exchange_execution/test_app.py` | `0 errors`. |
| Required ruff scope | `uv run ruff check apps/exchange_execution src/trading/contexts/live_execution tests` | Passed. |
| Required pyright scope | `uv run pyright apps/exchange_execution src/trading/contexts/live_execution tests` | `0 errors`. |
| Required pytest literal path | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps tests/integration` | Blocked locally because `tests/integration` does not exist in this checkout; no tests ran for that literal command. |
| Required available unit/apps path | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/apps` | `338 passed, 3 warnings`. |
| Docs index | `uv run python -m tools.docs.generate_docs_index --check` | Passed after regenerating `docs/architecture/README.md`. |
| Whitespace | `git diff --check` | Passed. |
| Full ruff | `uv run ruff check .` | Passed. |
| Full pyright | `uv run pyright` | `0 errors`. |
| Full local suite | `uv run pytest -q -ra` | `1091 passed, 3 warnings`. |

## Runtime Evidence

Direct-main delivery:

- implementation commit `4c2fa626` pushed to `main`;
- CI run `26848950859` passed;
- Publish App Image `26849114751`, Deploy Backend `26849114746`, Deploy Web
  `26849114759` and follow-up Deploy Web `26849126853` passed;
- `scripts/macos/smoke_prod.sh` passed after deploy and confirmed launchd,
  Redis, Postgres service state, Tailscale and API unauthenticated boundary.

Mac Studio readiness and schema:

- `GET http://127.0.0.1:9206/health/ready` returned
  `status=ready`, `status_reason=all_dependencies_ready`,
  `adapter_mode=testnet`;
- dependencies included `adapter=testnet_adapters_ready`,
  `ledger_pitr=pitr_restore_verified`, Redis stream length `12`,
  pre-existing pending count `1`, DLQ length `1`, clock drift `0.057 ms` and
  Postgres heartbeat `ready`;
- Alembic head was `20260602_0029`;
- `to_regclass` found `execution_order_events`, `execution_fills`,
  `execution_funding_events`, `execution_reconciliation_runs`,
  `execution_ledger_retention_policies` and
  `execution_ledger_pitr_drills`;
- ledger counts after the probes were `execution_orders=9`,
  `execution_order_events=10`, `execution_fills=1`,
  `execution_funding_events=0`, `execution_reconciliation_runs=3`,
  `execution_ledger_retention_policies=5` and
  `execution_ledger_pitr_drills=1`.

Controlled Bybit spot testnet probes:

| Probe | Boundary | Result |
|---|---|---|
| `stage15-market-fill-probe-20260603-03` | Created an `ops_test` source event and accepted market buy intent through the same Postgres repository and Redis dispatch transport as the API path, then executed `POST /internal/v1/run-once`. | Intent persisted as `dispatched/redis_xadd_ok`; provider order id was present; status lookup returned `filled`; one normalized fill was recorded with provider trade id present, price `64625.200000000000`, quantity `0.000092000000`, fee `0.000000165600 BTC`; reconciliation recorded `matched/spot_order_status_and_fills_matched` with `fill_count=1`. The later cancel attempt returned adapter reason `exchange_ret_code_170213` because the market order was already filled, and a second reconciliation row recorded `pending/adapter_error_reconciliation_pending`. |
| `stage15-limit-cancel-probe-20260603-01` | Created an `ops_test` source event and accepted low limit buy intent through Redis dispatch, then executed `POST /internal/v1/run-once`. | `run-once` returned HTTP `200` with `read_count=1`, `observed_count=1`, `submitted_count=1`, `adapter_error_count=0`, `acked_count=1`; `execution_orders` recorded `cancelled/cancel_requested` with provider order id, submit, status and cancel timestamps present; reconciliation recorded `matched/spot_order_status_matched` with `fill_count=0`. |

Order event and observation evidence:

- market fill probe wrote append-only order events
  `submit_pending`, `private_stream_backfill`, `submitted`, `status_checked`
  and `adapter_error`;
- limit cancel probe wrote append-only order events `submit_pending`,
  `private_stream_backfill`, `submitted`, `status_checked` and `cancelled`;
- `exchange_execution_request_observations` recorded
  `adapter_error/exchange_ret_code_170213` for Redis message
  `1780436341228-0` and
  `testnet_submitted/testnet_submit_status_cancel_recorded` for Redis message
  `1780436388341-0`;
- Redis scan over the latest 15 request, retry and DLQ stream entries reported
  `0` hits for secret-like terms.

Retention and PITR proof:

- retention policies were seeded for `execution_orders`,
  `execution_order_events`, `execution_fills`, `execution_funding_events` and
  `execution_reconciliation_runs`, all with `retention_days=2555`,
  `archive_before_purge=true`, `pitr_required=true`, `status=configured`;
- production database role cannot create databases, so the restore drill used
  a real `pg_dump` / `pg_restore` into an isolated temporary local Postgres
  cluster on Mac Studio;
- `execution_ledger_pitr_drills` recorded
  `verified/stage15_temp_cluster_restore_verified`, method
  `pg_dump_pg_restore`, restore target `temporary_local_cluster`, restored row
  counts `execution_orders=7`, `execution_order_events=0`,
  `execution_fills=0`, `execution_funding_events=0`,
  `execution_reconciliation_runs=0`,
  `execution_ledger_retention_policies=5`;
- `ROEHUB_EXECUTION_PITR_VERIFIED=true` was set in the Mac Studio runtime env
  after the restore proof and `com.roehub.exchange-execution` was restarted;
  readiness then reported `ledger_pitr=pitr_restore_verified`.

Metrics:

- `/metrics` exposed
  `exchange_execution_ready{status="ready",reason="all_dependencies_ready"} 1`;
- dependency gauges included
  `exchange_execution_dependency_ready{dependency="ledger_pitr",status="ready",reason="pitr_restore_verified"} 1`;
- order/reconciliation metrics included
  `exchange_execution_testnet_order_total{exchange="bybit",reason="submitted"} 2`,
  `exchange_execution_private_stream_total{exchange="bybit",reason="private_ws_auth_probe_ready"} 2`,
  `execution_reconciliation_total{status="matched",reason="spot_order_status_and_fills_matched"} 1`,
  `execution_reconciliation_total{status="matched",reason="spot_order_status_matched"} 1`
  and
  `execution_reconciliation_total{status="pending",reason="adapter_error_reconciliation_pending"} 1`.

Residual risk:

- futures funding facts remain unproven with real provider data in Stage 15;
  the code records `funding_reconciliation_pending` for non-spot orders without
  funding facts and Stage 16/17 must not treat futures funding/PnL as complete
  until a futures funding boundary is exercised.

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | none | No public API or browser route changes. |
| Ports/DTO | compatible-change | `ExchangeExecutionOrderRepository` gains ledger/reconciliation/PITR methods; status results may carry optional fill/funding facts. |
| Persistence | compatible-change | Adds append-only money-ledger and ops metadata tables; widens `execution_orders.status` with `reconciled`. |
| Redis | none | Dispatch stream and ack-after-durable-decision semantics are unchanged. |
| Config | compatible-change | Adds `exchange_execution.ledger.pitr_required` and `pitr_verified_env`; prod/test require proof marker. |
| Runtime/Ops | compatible-change | Readiness now reports `ledger_pitr`; prod/test are not ready without restore proof. |
| UI/browser | none | No browser-visible surface changed. |
| Metrics/logs/redaction | compatible-change | Adds bounded reconciliation/PITR metrics with no user/provider ids as labels. |
| External side effects | compatible-change | No new submit path; existing testnet status checks may call provider execution/trade history for reconciliation facts. |

## Rollback

Set `exchange_execution.process.adapter_mode=disabled` and restart
`com.roehub.exchange-execution` to stop exchange side effects. Existing
`execution_orders`, order events, fills, funding events, reconciliation runs,
retention policies and PITR drill rows are additive audit data and should
remain for incident review. Do not replay unknown orders blindly; reconcile by
`client_order_id` / provider order id first.

## Secrets And Redaction

The implementation stores normalized provider order/trade identifiers and
numeric facts only. It does not write API keys, secrets, passphrases,
signatures, raw Authorization headers, OpenBao tokens, ciphertext, cookies or
raw signed provider payloads to reports, logs or metrics. Metrics labels are
bounded to status/reason/exchange.

## Handoff To Stage 16

Stage 16 can start. Producers must continue to use the single execution
ingress, risk and Redis dispatch path so `exchange-execution` records order
events, fills and reconciliation state before operator/user notification work.
Stage 16 must preserve the Stage 15 limitation that real futures funding
boundary proof is still pending and should not emit complete futures
funding/PnL notifications without that evidence.
