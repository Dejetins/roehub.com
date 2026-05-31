# Stage 07: Exchange Account Projection And Config Guard

Stage 07 introduces the local, read-only account projection model and a
verify-only exchange configuration guard. The implementation is additive and
does not submit orders, modify exchange settings, decrypt credentials in
`apps/api`, or read exchange state from a strategy hot path.

Date: 2026-05-31.

Status: blocked for acceptance. Local implementation, schema, metrics, tests and
browser proof are complete; required real read-only exchange/testnet sync,
runtime DB snapshot evidence and deployed metrics evidence are not yet proven.

## Scope

Included:

- new `src/trading/contexts/live_execution` bounded context;
- account/balance/position/open-order/filter domain snapshots;
- account projection repository port plus in-memory and Postgres adapters;
- account projection sync use case through a read-only client port;
- verify-only config guard for instrument filters/account mode requirements;
- freshness states `fresh`, `stale`, `degraded`, and `config_mismatch`;
- additive Alembic tables for account, balance, position, open-order, filter and
  config guard evidence;
- bounded API metrics helpers:
  `exchange_account_state_sync_total`, `exchange_config_guard_total`, and
  `exchange_account_projection_staleness_seconds`;
- `/strategies` dashboard account readiness DTO and UI panel;
- focused service, migration and dashboard tests.

Out of scope in this partial stage result:

- no mainnet order submission;
- no exchange setting auto-config;
- no browser/API exposure of balances as a portfolio dashboard;
- no raw exchange payload, signed payload, Authorization header, API key, secret,
  passphrase or ciphertext persistence in the new tables;
- no Strategy/ML/browser/API direct credential access;
- no Redis execution stream or order intent.

## Prerequisite

| Requirement | Evidence | Verdict |
|---|---|---|
| Stage `06` accepted before Stage `07`. | Ledger status says Stage `06` accepted with direct-main delivery and production runtime evidence complete. | Pass. |
| Work on `main`, no stage branch or PR. | Local checkout is on `main`; no branch or PR was created. | Pass. |
| Runtime acceptance is not tests-only. | Required real exchange/testnet read-only sync and deployed DB/metrics proof are not available in this turn. | Blocked. |

## Files Changed

Code:

- `src/trading/contexts/live_execution/domain/account_state.py`
- `src/trading/contexts/live_execution/application/ports/account_projection_repository.py`
- `src/trading/contexts/live_execution/application/ports/clock.py`
- `src/trading/contexts/live_execution/application/ports/exchange_account_client.py`
- `src/trading/contexts/live_execution/application/use_cases/account_projection.py`
- `src/trading/contexts/live_execution/adapters/outbound/persistence/in_memory/account_projection_repository.py`
- `src/trading/contexts/live_execution/adapters/outbound/persistence/postgres/account_projection_repository.py`
- `src/trading/contexts/live_execution/adapters/outbound/time/system_live_execution_clock.py`
- package export files under `src/trading/contexts/live_execution/.../__init__.py`
- `apps/api/dto/ui_strategies_dashboard.py`
- `apps/api/monitoring.py`
- `apps/api/wiring/modules/ui_strategies_dashboard.py`

Schema:

- `alembic/versions/20260531_0020_exchange_account_projection_config_guard_v1.py`

UI:

- `apps/web/templates/pages/strategies.html`
- `apps/web/dist/js/pages/strategies.js`
- `apps/web/dist/css/pages/strategies.css`
- `apps/web/locales/en.json`
- `apps/web/locales/ru.json`

Tests:

- `tests/unit/contexts/live_execution/test_account_projection_service.py`
- `tests/unit/apps/migrations/test_exchange_account_projection_config_guard_sql.py`
- `tests/unit/apps/api/test_ui_strategy_dashboard_routes.py`

Docs:

- this stage report;
- `docs/architecture/live_execution/live-execution-universal-order-gateway-v1-stage-reports/live-execution-universal-order-gateway-v1-iteration-ledger.md`

## Implementation

`ExchangeAccountProjectionService` is the application boundary. It can sync a
connection only through an injected `ExchangeAccountReadOnlyClient` port, records
the resulting local projection, and writes verify-only guard results for
provided instrument requirements. The risk/readiness path reads only the local
repository and never calls an exchange adapter.

Readiness states:

- `fresh`: latest local projection is within freshness policy and config guard
  is verified;
- `stale`: latest local projection is older than the freshness threshold;
- `degraded`: projection, repository, connection or guard evidence is missing or
  degraded;
- `config_mismatch`: latest projection is fresh but verify-only guard found a
  mismatch such as missing filters or min-notional below requirement.

The `/strategies` dashboard now receives `exchange_account_readiness` and renders
the account readiness status, connection, instrument, age, config guard state,
reason and checked timestamp. It does not render balances, raw positions, raw
orders or raw provider payloads.

## Local Evidence

Quality gates:

| Gate | Command | Result |
|---|---|---|
| Focused ruff | `uv run ruff check src/trading/contexts/live_execution apps/api/monitoring.py apps/api/dto/ui_strategies_dashboard.py apps/api/wiring/modules/ui_strategies_dashboard.py tests/unit/contexts/live_execution/test_account_projection_service.py tests/unit/apps/migrations/test_exchange_account_projection_config_guard_sql.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | Passed. |
| Focused tests | `uv run pytest -q tests/unit/contexts/live_execution/test_account_projection_service.py tests/unit/apps/migrations/test_exchange_account_projection_config_guard_sql.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | `7 passed`. |
| Focused pyright | `uv run pyright src/trading/contexts/live_execution apps/api/monitoring.py apps/api/dto/ui_strategies_dashboard.py apps/api/wiring/modules/ui_strategies_dashboard.py tests/unit/contexts/live_execution/test_account_projection_service.py tests/unit/apps/migrations/test_exchange_account_projection_config_guard_sql.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py` | `0 errors`. |
| Required ruff | `uv run ruff check src/trading/contexts/live_execution src/trading/contexts/identity apps tests` | Passed. |
| Required pyright | `uv run pyright src/trading/contexts/live_execution src/trading/contexts/identity apps tests` | `0 errors`. |
| Required tests | `uv run pytest -q tests/unit/contexts/live_execution tests/unit/contexts/identity tests/unit/apps` | `293 passed, 3 warnings`. |
| Docs index | `python -m tools.docs.generate_docs_index --check` | Passed. |
| Whitespace | `git diff --check` | Passed before docs update; rerun required before delivery. |

Browser evidence:

- local secret-safe mock target served the real modified `/strategies` templates
  and assets with a deterministic dashboard API;
- Playwright checked desktop `1440x1000` and mobile `390x844`;
- states proved visible: `fresh`, `stale`, `degraded`, `config_mismatch`;
- console errors: `0`; failed dashboard request count: `0`;
- DOM secret scan for `api_secret`, `authorization`, `passphrase`, `signature`,
  `x-mbx-apikey`, and `x-bapi-sign`: `false`;
- screenshots:
  - `output/playwright/stage07-account-desktop-fresh.png`
  - `output/playwright/stage07-account-desktop-stale.png`
  - `output/playwright/stage07-account-desktop-degraded.png`
  - `output/playwright/stage07-account-desktop-config_mismatch.png`
  - `output/playwright/stage07-account-mobile-fresh.png`
  - `output/playwright/stage07-account-mobile-stale.png`
  - `output/playwright/stage07-account-mobile-degraded.png`
  - `output/playwright/stage07-account-mobile-config_mismatch.png`

## Runtime Evidence

Blocked for acceptance:

- no real read-only exchange/testnet account sync was executed;
- no production or sandbox DB query proved inserted
  `exchange_account_snapshots`, balances, positions, open orders, filters and
  config guard rows;
- no deployed `/metrics` or Monit evidence proved
  `exchange_account_state_sync_total`, `exchange_config_guard_total`, or
  staleness gauge behavior;
- no direct-main commit, push, CI, deploy or post-deploy Mac Studio smoke was
  performed because the stage is not accepted.

The local implementation keeps the sync boundary explicit through
`ExchangeAccountReadOnlyClient`, so the missing runtime part can be completed by
adding or wiring an approved exchange-control scoped read-only account adapter
without changing Strategy or browser boundaries.

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | compatible-change | Adds optional dashboard response field `exchange_account_readiness`; existing fields remain. |
| Ports | compatible-change | Adds `ExchangeAccountReadOnlyClient` and `ExchangeAccountProjectionRepository`. |
| DTO schema | compatible-change | Dashboard DTO gains additive account readiness panel. |
| Persistence | compatible-change | Additive snapshot/config guard tables only; no destructive migration. |
| Redis | none | No Redis streams or consumers added. |
| Config | none | No runtime env/config default changed. |
| Runtime/Ops | compatible-change | Adds metrics helpers; no supervised process yet. |
| UI/browser | compatible-change | Adds account readiness panel on `/strategies`; no portfolio balance dashboard. |
| External side effects | none | No exchange write, no order submit, no auto-config. |
| Logs/redaction | compatible-change | New fields store reason codes/source hash/counts only; no secrets/signed payload columns. |

## Rollback

- disable the dashboard account readiness source by not wiring
  `account_projection_service`;
- leave additive tables as inert evidence data or drop
  `20260531_0020` before it is applied to a shared environment;
- metrics are additive and can remain registered without producers.

## Next-Stage Handoff

Stage `07` cannot be marked accepted until a real approved read-only account
sync path is wired and proven with runtime DB, metrics and browser evidence.
Stage `08` remains blocked until `fresh` and `config_mismatch` account projection
states are accepted from a real boundary, not only local tests or mock browser
data.
