# Stage 11: Strategy Binding Guard

Дата: 2026-05-27.

Статус: accepted; доставлено напрямую в `main`, CI/deploy прошли, runtime
acceptance доказал create-binding -> guard block -> release -> disconnect/archive
flow на production-like контуре без exchange execution, order placement,
симуляции ордеров или physical delete.

Scope: owner-scoped strategy-to-exchange-connection usage registry and lifecycle
guard for `Disconnect`/archive. Stage 11 is configuration safety only.

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
| Browser | `/settings` receives `used_by_strategies_count` / `active_strategy_bindings_count` and disables/explains `Disconnect` when active strategy bindings exist. |
| Audit | Redacted events: `strategy_exchange_binding_created`, `strategy_exchange_binding_disabled`, `strategy_exchange_binding_archived`, `exchange_connection_disconnect_blocked`. |
| Metrics | Bounded labels only: `exchange_connection_usage_guard_total`, `strategy_exchange_binding_total`, `exchange_connection_active_strategy_bindings`. |

## Runtime Evidence

Runtime label: `stage11_binding_guard_20260527T004541Z`.

| Surface | Evidence | Verdict |
|---|---|---|
| Trading-ready connection | Authenticated Stage 10 flow created Bybit mainnet `spot` connection `1f449df7-702d-4c11-8f4b-7afaf24e96e7` with `status=active`, `effective_capability=trading`, `connection_readiness=ready_for_trading`, reason `trading_policy_ok`. Secrets were read only from approved env-backed runtime and were not printed. | Pass. |
| Strategy binding API | `POST /api/ui/strategies/{strategy_id}/exchange-bindings` returned `201` for binding `42a891f4-59e4-42be-9bae-997055e22896`; `GET /api/ui/strategies/{strategy_id}/exchange-bindings` returned the active `usage_mode=trading` binding. No direct DB insert was used for acceptance. | Pass. |
| Account read-model | `GET /api/ui/account/exchange-connections?status=active` exposed `used_by_strategies_count=1` and `active_strategy_bindings_count=1` for the target connection. | Pass. |
| Disconnect guard | `POST /api/ui/account/exchange-connections/{connection_id}/disable` returned HTTP `409` with code `exchange_connection_in_use` while the binding was active. | Pass. |
| Archive guard | `POST /api/ui/account/exchange-connections/{connection_id}/archive` returned HTTP `409` with code `exchange_connection_in_use` while the binding was active. | Pass. |
| Rotate | `POST /api/ui/account/exchange-connections/{connection_id}/rotate` returned HTTP `200`; connection stayed `active`, `effective_capability=trading`, `connection_readiness=ready_for_trading`. The active binding did not block rotation. | Pass. |
| Release and disconnect | `POST /api/ui/strategies/{strategy_id}/exchange-bindings/{binding_id}/disable` returned HTTP `200` and binding status `disabled`; subsequent Disconnect returned HTTP `200`, then Archive returned HTTP `200`; final Active list no longer contained the target connection. | Pass. |
| DB | `strategy_exchange_bindings` showed `usage_mode=trading`, `binding_status=disabled`, `disabled_at` present, `archived_at` absent; `exchange_connections` showed the target connection as `archived/user_archived`, `effective_capability=none`, `connection_readiness=archived`. Queried columns excluded secrets. | Pass. |
| Audit | Production audit counts after acceptance included `exchange_connection_disconnect_blocked=6`, `strategy_exchange_binding_created=3`, `strategy_exchange_binding_disabled=3`; metadata is stable IDs/state/reason only, without secret-bearing values. | Pass. |
| Metrics | `/metrics` exposed `exchange_connection_usage_guard_total`, `strategy_exchange_binding_total`, `exchange_connection_active_strategy_bindings`; observed bounded labels included `action`, `result`, `reason`, `exchange`, `status` only. | Pass. |
| Browser | Authenticated Playwright `/settings` artifact `output/playwright/settings-stage11-binding-guard-20260527T004541Z.png` showed target row with `Disconnect` disabled and title `Cannot disconnect. This exchange account is used by 1 active strategies. Pause or reassign strategies first.` | Pass with note: body text assertion did not capture a visible usage string, but API counts and disabled action title proved the user-facing block/explanation. |
| Secret/artifact grep | Grep found no real trade API key or secret values in `output/playwright` or `.playwright-cli`. Historical matches were field-name-only password/API key labels. | Pass. |

## Local Evidence

| Gate | Command | Result |
|---|---|---|
| Required pytest | `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/migrations` | Passed: `185 passed`, 3 existing httpx cookie deprecation warnings. |
| Ruff | `uv run ruff check src/trading/contexts/exchange_control src/trading/contexts/strategy apps/api apps/web tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api tests/unit/apps/web tests/unit/apps/migrations` | Passed. |
| Pyright | `uv run pyright src/trading/contexts/exchange_control src/trading/contexts/strategy apps/api tests/unit/contexts/exchange_control tests/unit/contexts/strategy tests/unit/apps/api` | Passed: `0 errors`. |
| Docs index | `python -m tools.docs.generate_docs_index --check` | Passed before final report update; rerun required after this docs commit. |

Tests/lint/type checks are quality gates only. Acceptance above is based on
runtime API, browser, DB, audit and metrics evidence.

## Direct-Main Delivery

| Step | Evidence |
|---|---|
| Implementation commit | `d98d5058 Implement stage 11 strategy binding guard` pushed to `origin/main`; CI `26482295777` success; deploys Web `26482361530`, App Image `26482361548`, Backend `26482361532` success. |
| Runtime bugfix 1 | `6af34865 Fix strategy timestamp mapping for production`; CI `26483025124` success; deploys Web `26483054467`, App Image `26483054498`, Backend `26483054496` success. |
| Runtime bugfix 2 | `3fc2aca0 Normalize strategy storage timestamps`; CI `26483201653` success; deploys Web `26483235700` / `26483230581`, App Image `26483230566`, Backend `26483230586` success. |
| Bootstrap fix | `325c91c4 Run strategy binding migration in bootstrap`; CI `26483500559` success; deploys Web `26483558443` / `26483564016`, App Image `26483558444`, Backend `26483558442` success. |
| Runtime schema | Mac Studio audit constraint contains Stage 11 event types: `exchange_connection_disconnect_blocked`, `strategy_exchange_binding_created`, `strategy_exchange_binding_disabled`, `strategy_exchange_binding_archived`. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `compatible-change` | Adds binding endpoints and additive usage count fields to exchange connection DTOs. Existing connection fields remain. |
| Error contract | `compatible-change` | Disconnect/archive may intentionally return deterministic `409 exchange_connection_in_use`; this is a new conflict condition, not an unhandled failure. |
| Persistence | `compatible-change` | Adds `strategy_exchange_bindings` table and audit event enum values; no physical delete path and no secret-bearing columns. |
| Config / env | `none` | Uses existing Postgres DSNs and Stage 10 exchange-control boundary. No new required secret env vars. |
| Browser | `compatible-change` | Adds usage count data and disables/explains Disconnect only when a connection is in use. |
| Ops/runtime | `compatible-change` | Adds bounded metrics and runtime acceptance queries. |
| Rollback | `forward-only` | Binding rows can be disabled; table/audit additions remain additive. If the guard read-model is unavailable, lifecycle mutations fail closed instead of silently disconnecting used connections. |

## Residual Risk / Handoff

| Risk | Status | Next action |
|---|---|---|
| Strategy-side binding management UI | Not expanded in Stage 11. | Binding API is the accepted minimal configuration surface; future UI can add strategy-side management without changing the guard. |
| Visible usage text assertion | Browser body text did not capture the usage count span during runtime proof. | Low-risk follow-up: make usage count copy easier to assert visually; current accepted proof uses API count plus disabled Disconnect title. |
| Exchange execution | Still out of scope. | Any future order/execution/signal contract requires a separate accepted architecture stage. |
