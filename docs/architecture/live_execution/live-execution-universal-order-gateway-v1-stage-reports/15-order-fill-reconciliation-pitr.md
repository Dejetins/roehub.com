# Stage 15: Order Fill Reconciliation And PITR

Stage 15 adds append-only order events, fill and funding ledgers,
reconciliation runs, retention policy metadata and a PITR readiness gate for
the existing `exchange-execution` testnet order path.

Date: 2026-06-03.

Status: local implementation and quality gates complete; runtime boundary
proof, direct-main delivery, CI, deploy and post-deploy proof are pending in
this working copy.

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

Pending. Stage acceptance still requires real boundary proof:

- safe testnet lifecycle or adapter simulator plus exchange state lookup for
  gap recovery;
- DB rows for order events, fills, funding events, reconciliation runs,
  retention policy and PITR drill metadata;
- private stream restart/backfill/dedupe proof;
- backup/PITR dry-run or target-host equivalent;
- direct-main CI/deploy and post-deploy smoke.

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

Stage 16 must not treat Stage 15 as accepted until runtime proof, PITR proof,
direct-main delivery and deploy evidence are appended here and in the
iteration ledger. Once accepted, producers can rely on a single dispatch path
that records order events, fills and reconciliation state before operator/user
notification work.
