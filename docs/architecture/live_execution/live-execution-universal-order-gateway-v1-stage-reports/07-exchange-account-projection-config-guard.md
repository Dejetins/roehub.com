# Stage 07: Exchange Account Projection And Config Guard

Stage 07 introduces the local, read-only account projection model and a
verify-only exchange configuration guard. The implementation is additive and
does not submit orders, modify exchange settings, decrypt credentials in
`apps/api`, or read exchange state from a strategy hot path.

Date: 2026-05-31.

Status: accepted. Local implementation, schema, metrics, tests, browser proof,
direct-main delivery, Mac Studio runtime sync, DB snapshots and deployed metrics
evidence are complete.

Clarification: Stage `07` uses read-only exchange operations through an approved
trade-ready connection. Roehub still does not support read-only exchange API keys
as active/usable live-trading connections; read-only keys are accepted only as
negative evidence and remain `read_only_not_supported` / not ready for trading.

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
| Runtime acceptance is not tests-only. | Mac Studio production proof used the smoke account's Bybit mainnet trade-ready connection for safe account-state reads, DB snapshot checks, API readiness checks and metrics scrapes. | Pass. |

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
connection only through an injected `ExchangeAccountStateReader` port, records
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

Accepted runtime proof:

| Surface | Evidence | Result |
|---|---|---|
| Runtime config | `ROEHUB_EXCHANGE_ACCOUNT_STATE_SYNC_ENABLED=1` was enabled in the Mac Studio production env, then launchd services were reloaded. OpenBao recovery still returned `exchange_control_encrypt=ok` and `apps_api_decrypt_denied=403`. | Pass. |
| Trade-ready account sync | `scripts/live_execution/sync_exchange_account_projection.py --owner-user-id <smoke-user> --exchange-connection-id 8e3999ba-c35d-4bcc-8253-a12b1d458114 --instrument-key bybit:spot:BTCUSDT --min-notional 0` called exchange-control `POST /internal/v1/exchange-connections/{connection_id}/account-state` and returned `status=fresh`, `reason=account_state_read_ok`, `balance_count=9`, `position_count=0`, `open_order_count=0`, `filter_count=1`, 64-char `source_hash`. | Pass. |
| Verify-only config mismatch | Same sync path with `--min-notional 999999999` produced a fresh account snapshot plus config guard `mismatch` with `min_notional_below_requirement`; no exchange setting write or order submit occurred. | Pass. |
| Read-only key rejection | The smoke read-only Bybit key remains `valid_readonly`, `connection_readiness=rejected`, `connection_readiness_reason=read_only_not_supported`, `effective_capability=none`. Account-state sync against that connection exited `1` with exchange-control `422 read_only_not_supported`. | Pass. |
| Dashboard API readiness | Authenticated production `GET /ui/strategies/dashboard` for the smoke strategy returned `fresh/account_projection_fresh`, `config_mismatch/min_notional_below_requirement`, `degraded/account_projection_missing`, and after the freshness window `stale/account_projection_stale`. | Pass. |
| DB snapshot redaction | Latest projection row recorded `bybit|spot|mainnet|unified|fresh|account_state_read_ok|source_hash length 64|balances 9|positions 0|orders 0|filters 1|metadata source present`; child rows showed `balances=9`, `positions=0`, `orders=0`, `filters=1`, `instrument=bybit:spot:BTCUSDT`, `tick=0.1`, `step=0.000001`, `min_notional=5`. No API key, secret, authorization, signature, passphrase, signed payload or ciphertext columns are introduced. | Pass. |
| Metrics | exchange-control `/metrics` emitted `exchange_account_state_read_total{exchange="bybit",reason="account_state_read_ok",result="fresh"}` and `exchange_account_state_read_total{reason="read_only_not_supported",result="rejected"}`. API `/metrics` emitted account readiness counters for `fresh`, `config_mismatch`, `degraded`, `stale`, config guard counters for `verified`, `mismatch`, `degraded`, and staleness gauge `130`. | Pass. |
| Mac Studio smoke | `bash scripts/macos/smoke_prod.sh` passed after reload; core services were running and API health returned `{"status":"ok"}`. | Pass. |

Public web image note: the first implementation commit contained the `/strategies`
panel but its CI failed on a test import-order issue. The later fix commit passed
CI but did not touch `apps/web`, so the normal image route did not rebuild the
public web container with the Stage `07` template. The acceptance path therefore
requires a manual `Publish App Image` workflow dispatch from the accepted `main`
commit and the subsequent `Deploy Web` run before public `/strategies` browser
proof is considered final.

## Contract Impact

| Surface | Impact | Notes |
|---|---|---|
| Public/API | compatible-change | Adds optional dashboard response field `exchange_account_readiness`; existing fields remain. |
| Ports | compatible-change | Adds `ExchangeAccountReadOnlyClient` and `ExchangeAccountProjectionRepository`. |
| DTO schema | compatible-change | Dashboard DTO gains additive account readiness panel. |
| Persistence | compatible-change | Additive snapshot/config guard tables only; no destructive migration. |
| Redis | none | No Redis streams or consumers added. |
| Config | compatible-change | Adds fail-closed `ROEHUB_EXCHANGE_ACCOUNT_STATE_SYNC_ENABLED`; production enables it explicitly for the diagnostic sync path. |
| Runtime/Ops | compatible-change | Adds exchange-control internal account-state read boundary, diagnostic sync script, metrics and deployment sync for `scripts/live_execution/`; no new supervised worker yet. |
| UI/browser | compatible-change | Adds account readiness panel on `/strategies`; no portfolio balance dashboard. |
| External side effects | none | No exchange write, no order submit, no auto-config. |
| Logs/redaction | compatible-change | New fields store reason codes/source hash/counts only; no secrets/signed payload columns. |

## Rollback

- set `ROEHUB_EXCHANGE_ACCOUNT_STATE_SYNC_ENABLED=0` or unset it to fail closed
  at the exchange-control account-state boundary;
- disable the dashboard account readiness source by not wiring
  `account_projection_service`;
- leave additive tables as inert evidence data or drop
  `20260531_0020` before it is applied to a shared environment;
- metrics are additive and can remain registered without producers.

## Next-Stage Handoff

Stage `08` may rely on the accepted local projection tables and freshness/config
guard reason codes. It must not call exchange-control or a native exchange
adapter from the strategy/run hot path; it should read local projection state
only.

Future stages must preserve the read-only-key clarification: safe account-state
reads are allowed only through the exchange-control boundary and an active
trade-ready connection. A read-only key is a rejected/not-ready connection and
must not satisfy live, risk, execution, ownership, capital or adapter readiness.
Stage `11` risk checks can use `fresh`, `stale`, `degraded`, and
`config_mismatch` as accepted account/config inputs; Stage `14` may add native
testnet submit adapters only after the same config guard is checked and still
must not add auto-config behavior.
