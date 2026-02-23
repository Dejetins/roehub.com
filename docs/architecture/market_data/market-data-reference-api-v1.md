# Market Data -- Reference API v1 (markets + instruments)

This document fixes WEB-EPIC-03 architecture: auth-only JSON API endpoints for UI dropdown/search,
backed by ClickHouse reference tables `market_data.ref_market` and `market_data.ref_instruments`.

## Goal

- Let Web UI load enabled markets and search enabled/tradable instruments without direct ClickHouse access.
- Keep responses deterministic (stable ordering) and storage reads explicit (no ORM, no hidden FINAL).
- Reuse existing identity auth (HttpOnly JWT cookie) via API `current_user_dependency`.

## Context

- Reference data source-of-truth lives in ClickHouse (DDL):
  - `migrations/clickhouse/market_data_ddl.sql`
  - tables:
    - `market_data.ref_market` (ReplacingMergeTree(updated_at), key `market_id`)
    - `market_data.ref_instruments` (ReplacingMergeTree(updated_at), key `(market_id, symbol)`)

- UI runs same-origin behind gateway (WEB-EPIC-02): browser calls JSON API via `/api/*`,
  but API router paths remain without `/api` prefix.

Docs:
- `docs/architecture/roadmap/milestone-6-epics-v1.md` (WEB-EPIC-03)
- `docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md` (UI uses these endpoints)
- `docs/architecture/apps/gateway/nginx-gateway-same-origin-ui-api-v1.md` (same-origin `/api/*` contract)
- `docs/architecture/market_data/market-data-reference-data-sync-v2.md` (how ref tables are populated)
- `docs/architecture/shared-kernel-primitives.md` (MarketId/Symbol/InstrumentId semantics)

## Scope

### 1) Application ports + use-cases (market_data context)

Add read-only ports and use-cases for reference lookup:

- `ListEnabledMarketsUseCase`:
  - reads latest-state enabled markets from `ref_market`
  - deterministic ordering: `market_id ASC`

- `SearchEnabledTradableInstrumentsUseCase`:
  - searches latest-state instruments within one `market_id`
  - filter: `status='ENABLED' AND is_tradable=1`
  - prefix search by `symbol` (`q`), deterministic ordering: `symbol ASC`
  - `limit` validated (`default=50`, `max=200`)

### 2) ClickHouse adapters (explicit SQL)

Implement ClickHouse adapters that:

- use explicit SQL and deterministic `ORDER BY`.
- implement latest-state reads without `FINAL` via:
  - `ORDER BY updated_at DESC`
  - `LIMIT 1 BY <key>`

### 3) API router (auth-only)

Add a new router `market-data reference` with endpoints:

- `GET /market-data/markets`
  - auth-only (requires identity principal)
  - returns enabled markets only
  - ordering: `market_id ASC`
  - response shape:
    - `{"items": [{"market_id": 1, "exchange_name": "binance", "market_type": "spot", "market_code": "binance:spot"}, ...]}`

- `GET /market-data/instruments?market_id=&q=&limit=`
  - auth-only
  - query params:
    - `market_id` required (int)
    - `q` optional prefix filter (string; blank treated as "no filter")
    - `limit` optional, default 50, max 200
  - behavior:
    - unknown/disabled `market_id` => `200 {"items": []}`
  - deterministic ordering: `symbol ASC`
  - response shape:
    - `{"items": [{"market_id": 1, "symbol": "BTCUSDT"}, ...]}`

### 4) Unit tests

Add unit tests (no external services):

- deterministic ordering guarantees
- enabled/tradable filters
- `q`/`limit` behavior
- SQL shape assertions for adapters (query contains deterministic clauses)

## Non-goals

- Enrich `ref_instruments` from exchange REST metadata (base/quote/steps/min_notional).
- Public unauth endpoints.
- Caching layer (can be added later if needed).

## Key decisions

### 1) Auth-only via existing identity dependency

Endpoints are protected and require a valid identity cookie.

Reason:
- UI operates only after login; reference data should not be publicly scraped by default.

### 2) Deterministic ordering is part of the contract

- markets: `market_id ASC`
- instruments: `symbol ASC`

Reason:
- UI dropdown/search must be stable and testable.

### 3) Latest-state reads without ClickHouse `FINAL`

We treat reference tables as versioned rows and read the latest version deterministically:

- `ORDER BY updated_at DESC LIMIT 1 BY <key>`

Reason:
- avoids expensive `FINAL` on ReplacingMergeTree.

### 4) Unknown/disabled market_id returns empty list (HTTP 200)

`GET /market-data/instruments` returns `items=[]` for unknown/disabled markets.

Reason:
- simplifies UI: markets are sourced from `/market-data/markets` anyway;
- prevents error popups for edge-cases (stale UI state).

## Contracts & invariants

- `GET /market-data/markets` returns enabled markets only.
- `GET /market-data/instruments` returns enabled+tradable instruments only.
- All outputs are deterministically ordered.
- `q` is prefix search by normalized `symbol` semantics (UI SHOULD send uppercase).
- `limit` default is 50, maximum is 200.

## Related files

- `docs/architecture/roadmap/milestone-6-epics-v1.md` -- WEB-EPIC-03.
- `migrations/clickhouse/market_data_ddl.sql` -- ClickHouse DDL for `ref_market`/`ref_instruments`.

- `src/trading/contexts/market_data/application/dto/reference_data.py` -- existing reference DTOs.
- `src/trading/contexts/market_data/application/ports/stores/*` -- existing ports patterns.
- `src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/gateway.py` -- ClickHouseGateway abstraction.

- `apps/api/main/app.py` -- API composition root (will include new router).
- `apps/api/wiring/modules/*` -- wiring module for new router.
- `apps/api/routes/*` -- new router implementation.

## How to verify

```bash
uv run ruff check .
uv run pyright
uv run pytest -q

python -m tools.docs.generate_docs_index
python -m tools.docs.generate_docs_index --check
```

## Risks / open questions

- Risk: `q` omitted (no filter) may be slow if `ref_instruments` becomes large.
  Mitigation v1: UI uses typeahead and should pass `q` for search.
- Risk: reference tables might be empty if market-data sync has not been run.
  Mitigation: run seed/sync jobs (see `docs/architecture/market_data/market-data-reference-data-sync-v2.md`).
