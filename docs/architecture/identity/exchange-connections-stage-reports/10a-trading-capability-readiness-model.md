# Stage 10A: Trading Capability Readiness Model

Дата проверки: 2026-05-26.

Статус: implementation complete locally; direct-main delivery and Mac Studio
runtime evidence pending.

Scope: backend/domain/API capability-readiness semantics for Roehub exchange
connections. Stage 10A does not remove the `/settings` permissions selector,
does not auto-validate create/rotate, does not reclassify existing production
records, does not place orders, and does not add exchange-execution behavior.

## Verdict

| Area | Expected result | Observed evidence | Verdict | Residual risk |
|---|---|---|---|---|
| Stage prerequisite | Stage 09E must be accepted before 10A starts. | Iteration ledger marks `09E production readiness` as `accepted; direct-main delivered; CI/deploy and Mac Studio runtime evidence complete`; 09E report is accepted. | Accepted. | None. |
| Product intent | `/settings` account exchange connections expose `requested_capability=trading`. | Domain view, internal exchange-control response, apps/api client and public DTO include `requested_capability`. | Accepted locally. | UI still sends legacy `permissions` until Stage 10C. |
| Readiness policy | Only trade-enabled, safe, IP-policy-valid credentials can expose `effective_capability=trading` and `connection_readiness=ready_for_trading`. | Domain truth-table tests cover trade-ready, readonly, unsafe, invalid credentials, missing IP restriction and validation-unavailable cases. | Accepted locally. | Runtime acceptance pending until deploy. |
| Readonly handling | Read-only keys must not look like successful Roehub trading connections. | `valid_readonly` and `permission_mismatch` map to `effective_capability=none`, `connection_readiness=rejected`, reason `read_only_not_supported`; legacy `effective_permissions=read` remains visible only as compatibility metadata. | Accepted locally. | Existing active records are not reclassified in 10A; Stage 10D owns controlled backfill. |
| Compatibility | Legacy `permissions`, `requested_permissions`, `exchange_permissions`, `effective_permissions`, `permission_warnings` remain readable but non-authoritative. | DTO keeps old fields and adds `permissions_deprecated=true`. | Compatible additive change. | Old consumers can still misunderstand legacy fields unless they migrate to capability/readiness. |
| Persistence | Prefer existing `permission_summary_json` unless columns are safer. | Readiness metadata is stored additively in `permission_summary_json`; no migration added. Domain also derives safe defaults for older rows. | Accepted locally. | Existing rows without new JSON fields derive readiness at read time until validation/rotate/state transitions update metadata. |
| Metrics | Stage 10 readiness metric exists with bounded, secret-free labels. | `exchange_connection_trading_readiness_total{exchange,result,reason}` added and seeded with a zero-value label set. | Accepted locally. | Runtime scrape pending until deploy. |

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
| Public account API | `curl -fsS "$ROEHUB_BASE_URL/api/ui/account/exchange-connections?status=active" -H "Cookie: $ROEHUB_SESSION_COOKIE" | jq '.items[] | {connection_id,status,requested_capability,effective_capability,connection_readiness,permissions,requested_permissions}'` | Pending direct-main deploy. | Pending. |
| Postgres read model | `psql "$ROEHUB_PG_DSN" -c "SELECT connection_id, status, permission_summary_json FROM exchange_connections ORDER BY created_at DESC LIMIT 5;"` | Pending direct-main deploy. | Pending. |
| Metrics | `curl -fsS http://127.0.0.1:9205/metrics | rg 'exchange_connection_trading_readiness_total|exchange_control_active'` | Pending direct-main deploy. | Pending. |

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
| Runtime acceptance | Pending. | API/DB/metrics calls run after direct-main deploy because acceptance requires the real runtime surface. |

## Direct-Main Delivery

| Item | Evidence | Result |
|---|---|---|
| Branch | `git branch --show-current`. | `main`. |
| Fast-forward | `git pull --ff-only origin main`. | Already up to date before implementation. |
| Commit / push | Pending. | Must be direct `main`; no stage branch or PR. |
| CI / deploy | Pending. | Must be watched after push. |
| Post-deploy runtime | Pending. | Required API/DB/metrics calls must be recorded above before final acceptance. |

## Residual Risk And Stage 10B Handoff

| Risk / handoff | Owner stage | Required next action |
|---|---|---|
| Create/rotate still persist active rows before external validation. | 10B | Auto-validation must make non-ready credentials not active and not limit-consuming. |
| Existing active non-trading rows are not reclassified. | 10D | Controlled dry-run then execution must move non-ready active rows out of Active through supported lifecycle paths. |
| UI still contains permissions selector and old wording. | 10C | Remove user-facing read/trade selector after 10B prevents non-ready active records. |
| Trade-ready runtime proof may need approved env-backed credentials. | 10E | If credentials are unavailable, mark trade-ready success proof partial/blocked rather than accepted. |
