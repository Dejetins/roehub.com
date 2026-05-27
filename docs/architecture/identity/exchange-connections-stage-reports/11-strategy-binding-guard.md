# Stage 11: Strategy Binding Guard

Дата начала: 2026-05-27.

Статус: implementation validation in progress; Stage 10E accepted and direct-main
delivered before Stage 11 work started. Финальный статус будет зафиксирован
после direct-main delivery, production API/DB/audit/metrics/browser evidence и
post-deploy проверки.

Scope: owner-scoped strategy-to-exchange-connection usage registry and lifecycle
guard for `Disconnect`/archive. Stage 11 is configuration safety only: exchange
execution, order placement, order simulation, signal-to-execution contract, risk
engine and physical deletes are out of scope.

## Prerequisite

| Requirement | Evidence | Verdict |
|---|---|---|
| Stage 10E accepted before Stage 11 starts. | Ledger marks Stage 10E `accepted`; report `10e-trading-cjm-production-readiness.md` records env-backed Bybit mainnet trade-ready proof, readonly rejection, Active-only readiness, CI/deploy success and no order placement. | Pass. |

## Implementation Contract

| Surface | Stage 11 contract |
|---|---|
| Binding model | `strategy_exchange_bindings` stores only configuration identifiers and lifecycle: `binding_id`, `owner_user_id`, `strategy_id`, `exchange_connection_id`, `usage_mode=trading`, `binding_status`, timestamps. No API key, secret, ciphertext, HMAC or exchange response body is stored. |
| Binding API | Additive account facade endpoints: `GET /api/ui/strategies/{strategy_id}/exchange-bindings`, `POST /api/ui/strategies/{strategy_id}/exchange-bindings`, `POST /api/ui/strategies/{strategy_id}/exchange-bindings/{binding_id}/disable`. |
| Owner scope | Binding creation verifies current user owns the strategy and the target exchange connection is active `ready_for_trading` with `effective_capability=trading`. |
| Lifecycle guard | `exchange-control` checks active `usage_mode=trading` bindings before disable/archive. Used connections return deterministic `409 exchange_connection_in_use`. Guard read failures fail closed as `exchange_connection_usage_guard_unavailable`. |
| Rotate | Credential rotation remains allowed for used connections because `connection_id` is stable and credential versions are replaceable. |
| Browser | `/settings` exposes `used_by_strategies_count` / `active_strategy_bindings_count`, shows usage count, and disables/explains `Disconnect` when active strategy bindings exist. |
| Audit | Redacted events: `strategy_exchange_binding_created`, `strategy_exchange_binding_disabled`, `exchange_connection_disconnect_blocked`. |
| Metrics | Bounded labels only: `exchange_connection_usage_guard_total`, `strategy_exchange_binding_total`, `exchange_connection_active_strategy_bindings`. |

## Local Evidence

| Gate | Command | Result |
|---|---|---|
| Focused pytest | `uv run pytest -q tests/unit/contexts/exchange_control/test_exchange_connection_readiness.py tests/unit/contexts/strategy/test_exchange_bindings.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations/test_strategy_exchange_bindings_sql.py` | Passed: `75 passed`, 3 existing httpx cookie deprecation warnings. |
| Required pytest | `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations` | Passed: `182 passed`, 3 existing httpx cookie deprecation warnings. |
| Ruff | `uv run ruff check src/trading/contexts/exchange_control src/trading/contexts/strategy apps/api apps/web tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations` | Passed after import ordering fix. |
| Pyright | `uv run pyright src/trading/contexts/exchange_control src/trading/contexts/strategy apps/api tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api` | Passed: `0 errors`. |
| Docs index | `python -m tools.docs.generate_docs_index --check` | Pending final docs update. |

## Runtime Evidence

Runtime acceptance is pending deployment of the Stage 11 implementation to
`main` and Mac Studio. Acceptance must prove:

| Surface | Required proof |
|---|---|
| Binding API | Real authenticated create/list/disable binding calls, not direct DB insert. |
| Guard | `POST /api/ui/account/exchange-connections/{connection_id}/disable` and `/archive` return `409 exchange_connection_in_use` while active binding exists. |
| Release | After `POST /api/ui/strategies/{strategy_id}/exchange-bindings/{binding_id}/disable`, Disconnect succeeds and the connection leaves Active. |
| DB | `strategy_exchange_bindings` lifecycle rows visible without secret-bearing columns. |
| Audit | `strategy_exchange_binding_created`, `strategy_exchange_binding_disabled`, and `exchange_connection_disconnect_blocked` rows visible with redacted metadata. |
| Metrics | `exchange_connection_usage_guard_total`, `strategy_exchange_binding_total`, and `exchange_connection_active_strategy_bindings` present. |
| Browser | Authenticated `/settings` shows usage count, blocked/explained Disconnect while used, and successful Disconnect after binding release. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `compatible-change` | Adds binding endpoints and additive usage count fields to exchange connection DTOs. Existing connection fields remain. |
| Error contract | `compatible-change` | Disconnect/archive may now intentionally return deterministic `409 exchange_connection_in_use`; this is a new conflict condition, not an unhandled failure. |
| Persistence | `compatible-change` | Adds `strategy_exchange_bindings` table and audit event enum values; no physical delete path and no secret-bearing columns. |
| Config / env | `none` | Uses existing Postgres DSNs and Stage 10 exchange-control boundary. No new required secret env vars. |
| Browser | `compatible-change` | Adds usage count and disables/explains Disconnect only when a connection is in use. |
| Ops/runtime | `compatible-change` | Adds bounded metrics and runtime acceptance queries. |
| Rollback | `forward-only` | Binding rows can be disabled; table/audit additions remain additive. No reactivation or physical delete is introduced. |

## Residual Risk / Handoff

| Risk | Status | Next action |
|---|---|---|
| Runtime acceptance | Open until deployed. | Run API/DB/audit/metrics/browser proof after direct-main deploy. |
| Rotation proof for used connection | Depends on env-backed valid credentials. | If credentials are unavailable during runtime acceptance, record this proof as partial rather than fake success. |
| Strategy UI configuration surface | Not expanded in Stage 11. | Binding API is the accepted minimal configuration surface; future UI can add strategy-side binding management. |
