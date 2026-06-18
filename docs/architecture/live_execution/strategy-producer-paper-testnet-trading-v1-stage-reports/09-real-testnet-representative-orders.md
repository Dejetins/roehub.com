# Stage 09: Real testnet representative orders

Статус: `blocked`

## Pre-Start

User required before start: active UI-created `/settings` testnet trading connections for Binance spot and Bybit futures for `smoke_e2e_keycloak` (`owner_user_id=ab094ba2-61d7-4fbf-be8f-cbad9f351572`). Do not send secrets in chat; add/rotate the missing keys through `/settings`, then rerun this stage. For the Bybit futures connection, Stage `09` restart must also verify read-only isolated margin and leverage `1x` before any short submit.

Update after Identity Stage `12`: `/settings` can now submit one physical API key
for both `Spot` and `Futures` via `market_types[]`, but the result is still
separate market-scoped `exchange_connection_id` rows. Stage `09` remains blocked
until the active ready trading rows actually exist in runtime inventory.

Stage `08` проверен в `strategy-producer-paper-testnet-trading-v1-stage-ledger.md` до implementation edits: статус `accepted`, `Next stage allowed = yes`, активных blockers нет.

## Scope

Stage `09` должен выполнить representative real testnet order coverage для `BTCUSDT`: Binance/Bybit, spot/futures, supported long/short branches и sizing groups из Stage `03` (`fixed_quote`, `fixed_equity_pct`) через existing source event -> risk -> Redis -> `exchange-execution` -> native adapter -> order/fill/reconciliation/outbox path.

Implementation не начиналась, потому что pre-start inventory показал неполное покрытие UI-created testnet connections. Prompt требует остановиться до implementation, если нужны пользовательские ключи/артефакты/доступы.

## Concrete Planned File List Before Editing

Ожидаемые broad paths `apps/exchange_execution` и `src/trading/contexts/live_execution` были сужены до документационного blocker scope до code edits:

| File | Planned action | Reason |
|---|---:|---|
| `apps/exchange_execution/main/app.py` | no code change planned | Runtime service is already ready in `testnet` adapter mode; blocker is missing connection coverage, not code health. |
| `apps/exchange_execution/adapters/native_http.py` | no code change planned | Native Binance/Bybit testnet adapters exist; no submit attempted before required connections are present. |
| `src/trading/contexts/live_execution/application/use_cases/execution_ingress.py` | no code change planned | Existing source/intent/risk path remains unchanged. |
| `src/trading/contexts/live_execution/application/use_cases/execution_dispatch.py` | no code change planned | Existing Redis dispatch path remains unchanged. |
| `src/trading/contexts/live_execution/application/use_cases/exchange_execution_process.py` | no code change planned | Existing exchange-execution process path remains unchanged. |
| `src/trading/contexts/exchange_control` | no code change planned | Missing active connections must be added through `/settings`, not by code or chat secrets. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/09-real-testnet-representative-orders.md` | create | Record pre-start requirement, runtime inventory, representative matrix, and blocked handoff. |
| `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | modify | Mark Stage `09` blocked and preserve next-executor handoff. |
| `docs/architecture/README.md` | generated update | Add the new Stage `09` report to the architecture docs index. |

## Initial Contract Impact

| Dimension | Classification | Note |
|---|---|---|
| Public API contract | `none` | No API code changed. |
| Port contract | `none` | No port changed. |
| DTO schema | `none` | No DTO changed. |
| Persisted schema | `none` | No migration/table/schema change. |
| Config schema | `none` | No env/config default changed. |
| Request hash / cache / identity | `none` | No idempotency/cache/identity behavior changed. |
| Service-call semantics | `none` | No exchange submit or mutation was attempted. |
| External side effects | `none` | No real testnet order was submitted. |
| Logs / metrics / audit / report | `compatible-change` | Adds a docs-only blocker report with sanitized runtime evidence. |
| Browser-visible behavior | `none` | No UI changed. |

## Runtime Inventory Evidence

Evidence collected on `macstudio` through read-only commands on `2026-06-18` local workspace date. The runtime user mapping is `smoke_e2e_keycloak -> identity_users.user_id ab094ba2-61d7-4fbf-be8f-cbad9f351572`. Secrets, raw credentials, tokens, signed payloads and provider payloads were not printed.

### Stage Gate

| Check | Result |
|---|---|
| Stage `08` ledger status | `accepted`; `Next stage allowed = yes`; blockers `none`. |
| Mac Studio git checkout | `/Users/daniildegtyarev/Projects/roehub.com` on `main`; observed HEAD `20135ec291303b37cc4ea74d43554974838e1961`. |
| Runtime smoke | `/opt/roehub/app/scripts/macos/smoke_prod.sh` exited `0`. |

### Exchange Connection Coverage

Required buckets are Binance spot, Binance futures, Bybit spot and Bybit futures in `testnet` with trading capability and `ready_for_trading` readiness.

| Exchange | Market | Active ready trading count | Active ids observed | Result |
|---|---|---:|---|---|
| Binance | spot | 0 | none | `blocked`: no active UI-created Binance spot testnet trading connection. |
| Binance | futures | 1 | `0b8c536b` / label `binance_testnet` | available, but no order submit attempted because full Stage `09` matrix pre-start requirements are not met. |
| Bybit | spot | 1 | `af4c90fa` | available, but no order submit attempted because full Stage `09` matrix pre-start requirements are not met. |
| Bybit | futures | 0 | none | `blocked`: no active UI-created Bybit futures testnet trading connection. |

Detailed non-secret inventory:

| Connection | Owner | Exchange | Market | Status | Readiness | Capability | Evidence |
|---|---|---|---|---|---|---|---|
| `0b8c536b` / `binance_testnet` | `ab094ba2` | Binance | futures | active | `ready_for_trading/trading_policy_ok` | `trading` | Active Stage `05` Binance futures connection. Futures short still requires fresh isolated `1x` proof before submit. |
| `af4c90fa` / `bybit_testnet` | `ab094ba2` | Bybit | spot | active | `ready_for_trading/trading_policy_ok` | `trading` | Active Stage `05` Bybit spot connection. |
| Binance spot rows | `ab094ba2` | Binance | spot | archived only | not ready | none | No active Binance spot testnet row exists for `smoke_e2e_keycloak`. |
| Bybit futures row `4cc97234` | `ab094ba2` | Bybit | futures | archived | historical `ready_for_trading` before archive | none while archived | Archived rows cannot be used for real submit. |

### Exchange-Execution Runtime

| Surface | Evidence |
|---|---|
| `/health/ready` | `status=ready`, `status_reason=all_dependencies_ready`, `adapter_mode=testnet`. |
| Adapter readiness | `testnet_adapters_ready`, `submit_enabled=1`, enabled exchange count `2`. |
| Redis readiness | request stream length `15`, pending `0`, backpressure within limit, DLQ stream length `2`, clock drift `0.064 ms`. |
| PITR guard | `ledger_pitr` ready with `pitr_restore_verified`. |
| Metrics | `exchange_execution_ready{status="ready",reason="all_dependencies_ready"} 1.0`; `exchange_execution_adapter_disabled 0.0`; Redis gauges exposed request stream `15`, DLQ `2`, pending `0`. |

### Redis Pre-Start State

| Stream / group | Evidence |
|---|---|
| `execution.requests.v1` length | `15` |
| `XPENDING execution.requests.v1 exchange-execution.v1` | `0` pending |
| `execution.requests.retry.v1` length | `1` |
| `execution.requests.dlq.v1` length | `2` |

## Representative Matrix

No real testnet submit was attempted. The result below is the required Stage `09` matrix state at pre-start blocker time.

| Exchange | Market | Direction | Sizing group | Required behavior | Result | Reason |
|---|---|---|---|---|---|---|
| Binance | spot | long | `fixed_quote` | Real spot buy/sell lifecycle with `$50` and precision/min-notional proof. | `blocked` | Missing active Binance spot testnet trading connection. |
| Binance | spot | long | `fixed_equity_pct` | Real spot buy/sell lifecycle with `$50`-bounded sizing proof. | `blocked` | Missing active Binance spot testnet trading connection. |
| Binance | spot | short | `fixed_quote` | Explicit unsupported block; no fake spot short. | `blocked` | `spot_short_not_supported`; no margin/borrow product accepted, and no active Binance spot connection. |
| Binance | spot | short | `fixed_equity_pct` | Explicit unsupported block; no fake spot short. | `blocked` | `spot_short_not_supported`; no margin/borrow product accepted, and no active Binance spot connection. |
| Binance | futures | long | `fixed_quote` | Real futures long testnet order with `$50` and final position cleanup. | `not_started` | Full Stage `09` matrix blocked before implementation due missing Binance spot and Bybit futures keys. |
| Binance | futures | long | `fixed_equity_pct` | Real futures long testnet order with `$50`-bounded sizing and cleanup. | `not_started` | Full Stage `09` matrix blocked before implementation due missing Binance spot and Bybit futures keys. |
| Binance | futures | short | `fixed_quote` | Real futures short only after fresh isolated margin `1x` read-only proof. | `not_started` | Full Stage `09` matrix blocked before implementation; fresh pre-submit futures guard still required on restart. |
| Binance | futures | short | `fixed_equity_pct` | Real futures short only after fresh isolated margin `1x` read-only proof. | `not_started` | Full Stage `09` matrix blocked before implementation; fresh pre-submit futures guard still required on restart. |
| Bybit | spot | long | `fixed_quote` | Real spot buy/sell lifecycle with `$50` and precision/min-notional proof. | `not_started` | Full Stage `09` matrix blocked before implementation due missing Binance spot and Bybit futures keys. |
| Bybit | spot | long | `fixed_equity_pct` | Real spot buy/sell lifecycle with `$50`-bounded sizing proof. | `not_started` | Full Stage `09` matrix blocked before implementation due missing Binance spot and Bybit futures keys. |
| Bybit | spot | short | `fixed_quote` | Explicit unsupported block; no fake spot short. | `blocked` | `spot_short_not_supported`; no margin/borrow product accepted. |
| Bybit | spot | short | `fixed_equity_pct` | Explicit unsupported block; no fake spot short. | `blocked` | `spot_short_not_supported`; no margin/borrow product accepted. |
| Bybit | futures | long | `fixed_quote` | Real futures long testnet order with `$50` and final position cleanup. | `blocked` | Missing active Bybit futures testnet trading connection. |
| Bybit | futures | long | `fixed_equity_pct` | Real futures long testnet order with `$50`-bounded sizing and cleanup. | `blocked` | Missing active Bybit futures testnet trading connection. |
| Bybit | futures | short | `fixed_quote` | Real futures short only after fresh isolated margin `1x` read-only proof. | `blocked` | Missing active Bybit futures testnet trading connection and fresh isolated `1x` proof. |
| Bybit | futures | short | `fixed_equity_pct` | Real futures short only after fresh isolated margin `1x` read-only proof. | `blocked` | Missing active Bybit futures testnet trading connection and fresh isolated `1x` proof. |

## Acceptance Criteria Status

| Criterion | Status | Evidence |
|---|---|---|
| Representative matrix has pass/block result for every required bucket | `partial/blocker-recorded` | Matrix above records every required exchange/market/direction/sizing row. There are no pass rows because execution stopped before implementation. |
| At least one real testnet order path per accepted representative bucket | `blocked` | No accepted bucket was executed; missing required connections prevent representative order coverage. |
| Redis ack-after-durable, pending, retry, DLQ evidence | `pre-start evidence only` | Redis pending `0`, request length `15`, retry length `1`, DLQ length `2`; no new Stage `09` message was dispatched or acked. |
| Metrics include submit latency, limiter waits, errors, private stream/reconciliation | `blocked` | Runtime exposes the metrics families, but no Stage `09` submit/reconciliation metrics were generated because no submit was attempted. |
| No mainnet submit and no secret leakage | `passed` | No submit attempted; inventory/report contains only sanitized connection ids and statuses. |

## Quality Gates

No code changed and implementation stopped before submit. Required code gates were not run because this is a pre-start blocker report, not an implemented code/runtime change.

| Gate | Result |
|---|---|
| `python -m tools.docs.generate_docs_index --check` | passed |

## File Manifest

| Action | File | Reason | Contract impact |
|---|---|---|---|
| Created | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/09-real-testnet-representative-orders.md` | Stage `09` pre-start blocker report with sanitized runtime inventory and matrix. | `none` runtime |
| Modified | `docs/architecture/live_execution/strategy-producer-paper-testnet-trading-v1-stage-reports/strategy-producer-paper-testnet-trading-v1-stage-ledger.md` | Mark Stage `09` blocked and record exact handoff. | `none` runtime |
| Modified | `docs/architecture/README.md` | Generated docs index entry for the new Stage `09` report. | `none` runtime |
| Deleted | none | No files deleted. | `none` |

## Blockers

| Blocker | Severity | Owner / next action | Acceptance impact |
|---|---|---|---|
| Missing active Binance spot testnet trading connection for Stage `09` owner. | critical | User/operator adds or rotates Binance spot testnet key through `/settings`; do not send secrets in chat. | Blocks Binance spot long proof and representative Binance spot short unsupported proof under a real connection boundary. |
| Missing active Bybit futures testnet trading connection for Stage `09` owner. | critical | User/operator adds or rotates Bybit futures testnet key through `/settings`; then Stage `09` rerun performs read-only isolated margin and leverage `1x` proof before any short. | Blocks Bybit futures long/short proof. |

## Handoff

Stage `09` is blocked before implementation. After the missing `/settings` connections are added, rerun Stage `09` from the pre-start gate:

1. If the missing physical exchange key supports both products, add or rotate it
   once through `/settings` with the required `Spot`/`Futures` checkboxes; then
   re-inventory active testnet connections and account-state projections without
   printing secrets.
2. Sync/freshen account projections for Binance spot, Binance futures, Bybit spot and Bybit futures on `BTCUSDT`.
3. Verify futures short config read-only: isolated margin, leverage `1x`, expected position mode, precision, min notional, and sufficient balance.
4. Execute representative supported testnet order buckets through source event, risk, Redis dispatch, `exchange-execution`, native adapter, order/fill/reconciliation/outbox.
5. Cancel or close any opened testnet positions, then record final account state, Redis ack/pending/retry/DLQ, metrics, and no-mainnet/no-secret proof.

No publish/deploy was performed because acceptance is blocked.
