# Stage 03: Live Strategy Profile

Stage 03 adds `LiveStrategyProfile` for strategy launch mode, exchange binding, sizing, per-run limits, readiness status/reason, and fail-closed live-mode recent-auth behavior.

Date: 2026-05-30.

Status: local validation passed; direct-main publish/deploy pending.

## Scope

Included:

- `LiveStrategyProfile` domain entity and Strategy application service;
- additive profile repository port plus in-memory and Postgres adapters;
- additive Alembic storage `strategy_live_profiles`;
- Strategy API endpoints for profile create/read/update/readiness;
- non-secret exchange-control readiness checker for live mode;
- fail-closed `live` readiness when recent auth or eligible exchange connection is absent;
- `/strategies` dashboard live profile panel;
- bounded metric `live_strategy_profile_readiness_total`;
- focused tests for API, dashboard, migration and web asset contracts.

Out of scope:

- no mainnet order submit;
- no testnet order submit;
- no exchange credential decryption;
- no Redis execution stream;
- no paper order/fill/accounting ledger;
- no position ownership or capital reservation.

## Prerequisite

| Requirement | Evidence | Verdict |
|---|---|---|
| Stage `02` accepted before Stage `03`. | Ledger row for `02` is `accepted`; evidence log records post-deploy API/SQL/browser proof. | Pass. |
| Work on `main`, no stage branch or PR. | `git status --short --branch` returned `## main...origin/main` before implementation. | Pass. |
| Runtime acceptance is not tests-only. | Local HTTP and browser boundary proof is recorded below. Production SQL/deploy evidence is pending direct-main delivery. | Pending. |

## Files Changed

Code:

- `apps/api/dto/ui_strategies_dashboard.py`
- `apps/api/monitoring.py`
- `apps/api/routes/strategies.py`
- `apps/api/wiring/modules/strategy.py`
- `apps/api/wiring/modules/ui_strategies_dashboard.py`
- `src/trading/contexts/strategy/domain/entities/live_strategy_profile.py`
- `src/trading/contexts/strategy/application/ports/exchange_connection_readiness.py`
- `src/trading/contexts/strategy/application/ports/repositories/live_strategy_profile_repository.py`
- `src/trading/contexts/strategy/application/use_cases/live_strategy_profiles.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/in_memory/live_strategy_profile_repository.py`
- `src/trading/contexts/strategy/adapters/outbound/persistence/postgres/live_strategy_profile_repository.py`
- strategy package `__init__.py` export files touched for the new entity, port, use-case and adapters.

Schema:

- `alembic/versions/20260530_0017_strategy_live_profiles_v1.py`

UI:

- `apps/web/templates/pages/strategies.html`
- `apps/web/dist/js/pages/strategies.js`
- `apps/web/dist/css/pages/strategies.css`
- `apps/web/locales/en.json`
- `apps/web/locales/ru.json`

Tests:

- `tests/unit/apps/api/test_strategies_routes.py`
- `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py`
- `tests/unit/apps/migrations/test_strategy_live_profiles_sql.py`
- `tests/unit/apps/web/test_app_routes.py`

Docs:

- `docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/03-live-strategy-profile.md`
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md`

## Implementation

The application service is `LiveStrategyProfileService`.

The service:

- requires owned, non-deleted strategy visibility before profile operations;
- creates a default `monitor_only` profile with `monitor_only_no_exchange_submit`;
- persists mode, optional `exchange_connection_id`, sizing method/value, max position notional, max orders per run, max notional per run, readiness status and readiness reason;
- treats `monitor_only` and `paper` as ready, non-money-moving modes in this stage;
- treats `live` as blocked unless recent auth is confirmed and exchange-control reports the connection as active, trading-capable and `ready_for_trading`;
- records strategy events without secrets or raw provider payloads.

The API endpoints are:

```http
POST /strategies/{strategy_id}/live-profile
GET /strategies/{strategy_id}/live-profile
PUT /strategies/{strategy_id}/live-profile
GET /strategies/{strategy_id}/live-profile/readiness
```

The browser-facing same-origin paths are the same under `/api`.

## Local Evidence

| Surface | Evidence | Result |
|---|---|---|
| Focused API/dashboard/migration/web tests | `uv run pytest -q tests/unit/apps/api/test_strategies_routes.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py tests/unit/apps/migrations/test_strategy_live_profiles_sql.py tests/unit/apps/web/test_app_routes.py` | `39 passed, 3 warnings`. |
| Broad required unit slice | `uv run pytest -q tests/unit/contexts/strategy tests/unit/apps` | `337 passed, 3 warnings`. |
| Focused lint | `uv run ruff check src/trading/contexts/strategy apps/api tests/unit/apps/api/test_strategies_routes.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py tests/unit/apps/migrations/test_strategy_live_profiles_sql.py` | Passed. |
| Focused type checking | `uv run pyright src/trading/contexts/strategy apps/api tests/unit/apps/api/test_strategies_routes.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py tests/unit/apps/migrations/test_strategy_live_profiles_sql.py` | `0 errors, 0 warnings, 0 informations`. |
| Repository lint | `uv run ruff check .` | Passed. |
| Repository type checking | `uv run pyright` | `0 errors, 0 warnings, 0 informations`. |
| Repository tests | `uv run pytest -q -ra` | `1006 passed, 3 warnings`. |
| Docs index | `python -m tools.docs.generate_docs_index --check` | Passed. |
| Whitespace | `git diff --check` | Passed. |

The warnings are existing `httpx` per-request cookie deprecation warnings in web route tests.

## Runtime Evidence

Local HTTP boundary probe used a temporary Uvicorn server with Strategy API routes, in-memory repositories and a deterministic non-secret exchange readiness checker:

| Runtime boundary | Evidence | Result |
|---|---|---|
| Strategy create | `POST /strategies` | `201`. |
| Default profile create | `POST /strategies/{strategy_id}/live-profile` | `mode=monitor_only`, `readiness_status=ready`, `readiness_reason=monitor_only_no_exchange_submit`. |
| Paper update | `PUT /strategies/{strategy_id}/live-profile` with `mode=paper` | `readiness_status=ready`, `readiness_reason=paper_no_exchange_submit`. |
| Live without recent auth | `PUT /strategies/{strategy_id}/live-profile` with `mode=live` and connection id | persisted fail-closed profile with `readiness_status=blocked`, `readiness_reason=recent_auth_required`. |
| Live with recent auth and eligible connection | Same `PUT` after recent-auth dependency flipped to fresh | `readiness_status=ready`, `readiness_reason=live_ready_recent_auth_and_connection`. |
| Readiness refresh | `GET /strategies/{strategy_id}/live-profile/readiness` | returned persisted live ready state. |
| Metrics | `GET /metrics` on local probe app | `live_strategy_profile_readiness_total` present with bounded `reason="recent_auth_required"` label. |
| Redaction | Response text scan for `api_secret`, `Authorization`, `Bearer` | none present. |

Browser proof used Playwright CLI against a local web/API Uvicorn pair with one seeded strategy and a blocked live profile:

- URL: `/strategies?strategy_id=<synthetic_strategy_id>`;
- desktop viewport showed profile panel title `Live profile`, `mode=live`, readiness `blocked: recent_auth_required`, reason `recent_auth_required`;
- mobile viewport showed the same profile state and no horizontal document overflow;
- `GET /api/ui/strategies/dashboard?...` returned `200`;
- console errors: none;
- screenshots:
  - `output/playwright/stage03-strategies-live-profile-desktop.png`
  - `output/playwright/stage03-strategies-live-profile-mobile.png`

Production SQL/API/browser proof is pending direct-main publish/deploy.

## Error Behavior

| Case | Code/state | Expected behavior |
|---|---|---|
| Missing profile | default profile | `GET`/`POST` creates inert `monitor_only` profile. |
| `paper` profile | ready | no exchange submit, no credential access. |
| `live` without recent auth | `blocked/recent_auth_required` | fail closed and persist reason. |
| `live` without connection | `blocked/exchange_connection_required` | fail closed and persist reason. |
| `live` with exchange-control unavailable | `blocked/exchange_connection_checker_unavailable` or `blocked/exchange_control_unavailable` | fail closed and persist reason. |
| Strategy missing/deleted/not owned | existing Strategy errors | no profile row is created for unauthorized user. |

## Runtime Config

No new environment variables, YAML files, feature flags or kill switches were added.

Fail-closed defaults:

- if Strategy API is disabled, the profile routes are absent with the Strategy router;
- if `STRATEGY_PG_DSN` is absent in non-fail-fast environments, local development uses in-memory storage only;
- if exchange-control is not configured, `live` mode readiness blocks with an exchange-checker reason.

## Monitoring

`live_strategy_profile_readiness_total` was added with bounded labels:

- `status`: `ready` or `blocked`;
- `reason`: stable readiness reason, truncated to 80 characters.

No user id, strategy id, profile id, exchange connection id, credential, cookie or raw request payload is used as a metric label.

## Logging And Redaction

No secrets, cookies, Authorization headers, API keys, private keys, passphrases, ciphertext, signed exchange payloads, raw exchange provider responses or raw idempotency keys were intentionally printed or committed.

The profile API stores only stable `exchange_connection_id`, readiness reason and non-secret sizing/limit values.

## Contract Impact

| Dimension | Classification | Reason |
|---|---|---|
| Public API / DTO | `compatible-change` | Adds live-profile endpoints and dashboard DTO field. |
| Persistence | `compatible-change` | Adds `strategy_live_profiles`; no destructive migration. |
| Redis | `none` | No stream, consumer group, dispatch, or Redis config change. |
| Config | `none` | No new env/YAML/default. |
| Runtime / ops | `compatible-change` | Adds bounded Prometheus counter; no process/service change. |
| UI / browser | `compatible-change` | Adds live profile panel to existing `/strategies` detail dashboard. |
| Exchange/provider side effects | `none` | Uses non-secret exchange-control readiness projection only; no decrypt or submit. |
| Docs | `compatible-change` | Adds Stage 03 report and updates Strategy API doc and ledger. |

## Rollback

Safe rollback path before later execution stages depend on this table:

- revert the code/UI/docs commit;
- run Alembic downgrade from `20260530_0017` if profile rows are not needed;
- otherwise leave `strategy_live_profiles` as inert audit/config data and remove/disable the profile routes with the Strategy router.

## Publish And Deploy

Pending direct-main commit, push, CI, Mac Studio deploy and production post-deploy smoke.

## Next-Stage Handoff

- `LiveStrategyProfile` is now the Strategy-owned config boundary for mode/sizing/limits/readiness.
- `monitor_only` is the safe default.
- `paper` is ready but non-money-moving in this stage.
- `live` is allowed to persist only as ready when recent auth and eligible exchange connection are proven; otherwise it remains blocked with a stable reason.
- No Redis execution dispatch, order submit, exchange decrypt, paper accounting, position ownership or capital reservation exists yet.
