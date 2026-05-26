# Stage 10A: Trading Capability Readiness Model

Дата проверки: 2026-05-26.

Статус: accepted; implementation commit `be359b27` direct-main delivered;
CI/deploy and Mac Studio runtime evidence complete.

Scope: backend/domain/API capability-readiness semantics for Roehub exchange
connections. Stage 10A does not remove the `/settings` permissions selector,
does not auto-validate create/rotate, does not reclassify existing production
records, does not place orders, and does not add exchange-execution behavior.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| Stage prerequisite | Stage 09E must be accepted before 10A starts. | Iteration ledger marks `09E production readiness` as `accepted; direct-main delivered; CI/deploy and Mac Studio runtime evidence complete`; 09E report is accepted. | Accepted. | None. |
| Product intent | `/settings` account exchange connections expose `requested_capability=trading`. | Domain view, internal exchange-control response, apps/api client and public DTO include `requested_capability`. | Accepted locally. | UI still sends legacy `permissions` until Stage 10C. |
| Readiness policy | Only trade-enabled, safe, IP-policy-valid credentials can expose `effective_capability=trading` and `connection_readiness=ready_for_trading`. | Domain truth-table tests cover trade-ready, readonly, unsafe, invalid credentials, missing IP restriction and validation-unavailable cases; deployed public API exposes non-ready active dummy as `needs_action`, not ready. | Accepted. | None for 10A. |
| Readonly handling | Read-only keys must not look like successful Roehub trading connections. | `valid_readonly` and `permission_mismatch` map to `effective_capability=none`, `connection_readiness=rejected`, reason `read_only_not_supported`; legacy `effective_permissions=read` remains visible only as compatibility metadata. | Accepted locally. | Existing active records are not reclassified in 10A; Stage 10D owns controlled backfill. |
| Compatibility | Legacy `permissions`, `requested_permissions`, `exchange_permissions`, `effective_permissions`, `permission_warnings` remain readable but non-authoritative. | DTO keeps old fields and adds `permissions_deprecated=true`. | Compatible additive change. | Old consumers can still misunderstand legacy fields unless they migrate to capability/readiness. |
| Persistence | Prefer existing `permission_summary_json` unless columns are safer. | Runtime DB query for the Stage 10A smoke connection shows `requested_capability`, `effective_capability`, `connection_readiness`, `connection_readiness_reason`, `permissions_deprecated`, `permissions`, and `requested_permissions` sourced from `permission_summary_json`. | Accepted. | Existing older rows without new JSON fields derive readiness at read time until validation/rotate/state transitions update metadata. |
| Metrics | Stage 10 readiness metric exists with bounded, secret-free labels. | Mac Studio `/metrics` exposes `exchange_connection_trading_readiness_total{exchange,result,reason}` and `exchange_control_active 1.0`. | Accepted. | None. |

## Capability/readiness Truth Table

| Validation outcome | Legacy exchange/effective permissions | Effective capability | Connection readiness | Reason |
|---|---|---|---|---|
| `valid_trade_enabled`, exchange `trade`, IP policy OK | `trade` / `trade` | `trading` | `ready_for_trading` | `trading_policy_ok` |
| `valid_readonly` | `read` / `read` | `none` | `rejected` | `read_only_not_supported` |
| requested trade but exchange readonly `permission_mismatch` | `read` / `read` | `none` | `rejected` | `read_only_not_supported` |
| withdrawal or transfer permission present | `withdraw_or_transfer` / `none` | `none` | `rejected` | `unsafe_permissions` |
| invalid credentials | `unknown` / `none` | `none` | `rejected` | `invalid_credentials` |
| missing mainnet IP restriction | exchange-observed value / `none` | `none` | `needs_action` | `ip_restriction_required` |
| validation unavailable or skipped | `unknown` / `none` | `none` | `needs_action` | `validation_required` |
| disabled connection | unchanged compatibility metadata | `none` | `disconnected` | `user_disconnected` |
| archived connection | unchanged compatibility metadata | `none` | `archived` | `archived` |

## Implementation Evidence

| Surface | Change | Evidence |
|---|---|---|
| Domain/read model | `ExchangeConnectionView` now exposes `requested_capability`, `effective_capability`, `connection_readiness`, `connection_readiness_reason`, and `permissions_deprecated`. | `src/trading/contexts/exchange_control/application/connections.py`; `tests/unit/contexts/exchange_control/test_exchange_connection_readiness.py`. |
| Validation mapping | Domain maps validation result/status to trading readiness without treating legacy permission fields as authoritative. | Truth-table unit test covers ready, readonly, unsafe, invalid, missing-IP and unavailable cases. |
| Persistence | `PostgresExchangeConnectionRepository` writes capability/readiness metadata into `permission_summary_json` on create/rotate/validate/disable/archive paths; older rows fall back to derived read-model values. | `src/trading/contexts/exchange_control/adapters/outbound/postgres_connections.py`; no migration added. |
| Internal API | Exchange-control internal connection response includes capability/readiness fields and emits `exchange_connection_trading_readiness_total` observations on validation. | `src/trading/contexts/exchange_control/adapters/inbound/http/app.py`; runtime unit test checks response shape and metric name. |
| Public account facade | apps/api client, DTO and route mapper expose capability/readiness fields while preserving deprecated permission fields. | `apps/api/exchange_control_client.py`; `apps/api/dto/ui_account.py`; `apps/api/routes/ui_account.py`; API route tests. |

## Runtime Evidence

| Surface | Command | Sanitized result | Verdict |
|---|---|---|---|
| Public account API | `curl -fsS "$ROEHUB_BASE_URL/api/ui/account/exchange-connections?status=active" -H "Cookie: $ROEHUB_SESSION_COOKIE" | jq '.items[] | {connection_id,status,requested_capability,effective_capability,connection_readiness,permissions,requested_permissions}'` | Temporary server-side smoke session created a dummy Binance testnet connection. Sanitized active-list shape: `status=active`, `requested_capability=trading`, `effective_capability=none`, `connection_readiness=needs_action`, `permissions=read`, `requested_permissions=read`, `permissions_deprecated=true`. The row was disabled and archived after evidence. | Pass. |
| Postgres read model | `psql "$ROEHUB_PG_DSN" -c "SELECT connection_id, status, permission_summary_json FROM exchange_connections ORDER BY created_at DESC LIMIT 5;"` | Sanitized targeted row after cleanup: `status=archived`; JSON has `requested_capability=trading`, `effective_capability=none`, `connection_readiness=archived`, `connection_readiness_reason=archived`, `permissions_deprecated=true`, plus legacy `permissions=read`, `requested_permissions=read`. | Pass. |
| Metrics | `curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_connection_trading_readiness_total|exchange_control_active'` | Mac Studio scrape returned `exchange_control_active 1.0` and `exchange_connection_trading_readiness_total{exchange="none",reason="stage_10a_no_readiness_observation",result="needs_action"} 0.0`; labels contain no user, connection or credential identifiers. | Pass. |
| Cleanup | DB check for temporary Stage 10A sessions. | `active_stage10a_sessions=0`; the smoke exchange connection was archived, not physically deleted. | Pass. |

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `compatible-change` | Adds capability/readiness fields and `permissions_deprecated`; preserves existing permission fields and enum values. |
| Internal API | `compatible-change` | Adds fields to local-only exchange-control connection response; client parser has defaults for old payloads. |
| Domain / port | `compatible-change` | Connection view expands additively; product readiness is now explicit and no longer inferred from legacy permissions. |
| Persistence schema | `none` | No migration or column change; metadata is additive inside existing `permission_summary_json`. |
| Persistence semantics | `compatible-change` | New JSON keys are additive; existing rows derive fallback readiness until touched. No physical delete or reclassification. |
| Config / env | `none` | No new env vars, flags or default resolution. |
| Request hash / cache / identity | `none` | No cache key, request hash or persistence identity changes. |
| Metrics / ops | `compatible-change` | Adds bounded `exchange_connection_trading_readiness_total{exchange,result,reason}`; no user, connection, key, credential or raw exchange labels. |
| Browser-visible behavior | `none` for Stage 10A | UI selector and visible CJM are intentionally unchanged until Stage 10C. |
| Trading execution | `none` | No order placement, strategy execution, exchange execution or order ledger code added. |

## Validation

| Gate | Result | Evidence |
|---|---|---|
| Focused pytest | Passed: `46 passed`. | `uv run pytest -q tests/unit/contexts/exchange_control/test_exchange_connection_readiness.py tests/unit/apps/api/test_ui_account_routes.py tests/unit/contexts/exchange_control/test_exchange_control_runtime.py`. |
| Required pytest | Passed: `72 passed`. | `uv run pytest -q tests/unit/contexts/exchange_control tests/unit/apps/api/test_ui_account_routes.py tests/unit/apps/migrations`. |
| Required ruff | Passed. | `uv run ruff check src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api tests/unit/apps/migrations`. |
| Required pyright | Passed: `0 errors`. | `uv run pyright src/trading/contexts/exchange_control apps/api tests/unit/contexts/exchange_control tests/unit/apps/api`. |
| Docs index | Passed. | `python -m tools.docs.generate_docs_index --check`; docs index regenerated with `python -m tools.docs.generate_docs_index`. |
| No-order grep | Passed. | `rg -n "place_order|create_order|order placement|exchange-execution|exchange_execution|submit_order|cancel_order|orders" src apps tests docs/architecture/identity/exchange-connections-stage-reports/10a-trading-capability-readiness-model.md`; matches were only Stage 10A non-goal text and unrelated existing backtest tests. |
| Runtime acceptance | Passed. | Public API, Postgres and metrics calls above ran after direct-main deploy of `be359b27`; the Mac Studio public `roehub.com` hairpin was unavailable, so the authenticated public API curl was executed from the local workstation using a temporary server-side smoke session. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before implementation. |
| Commit / push | `be359b27 Add exchange connection trading readiness`; pushed `4b365567..be359b27` to `origin/main`. | Pass; no stage branch or PR. |
| CI / deploy | CI `26474077369` success; Deploy Backend `26474170574` success; Publish App Image `26474170518` success; Deploy Web `26474170572` and downstream `26474181930` success. | Pass. |
| Post-deploy runtime | Public API/DB/metrics evidence above. | Pass. |

## Residual Risk And Stage 10B Handoff

| Risk / handoff | Owner stage | Required next action |
|---|---|---|
| Create/rotate still persist active rows before external validation. | 10B | Auto-validation must make non-ready credentials not active and not limit-consuming. |
| Existing active non-trading rows are not reclassified. | 10D | Controlled dry-run then execution must move non-ready active rows out of Active through supported lifecycle paths. |
| UI still contains permissions selector and old wording. | 10C | Remove user-facing read/trade selector after 10B prevents non-ready active records. |
| Trade-ready runtime proof may need approved env-backed credentials. | 10E | If credentials are unavailable, mark trade-ready success proof partial/blocked rather than accepted. |
