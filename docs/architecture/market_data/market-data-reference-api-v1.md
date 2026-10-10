# Market Data -- Reference API v1 (markets + instruments)

Validation boundary: a local Docker `runtime smoke` and real browser proof are
required in addition to `pytest`, `ruff`, and `pyright` for browser-visible or
persistence-affecting changes to this contract.

> Статус: дополнен в `roehub-market-data-activation-and-instrument-selection-v1`.
> CSV whitelist больше не является runtime-политикой. Каталог остаётся глобальным
> справочником, а пользовательский выбор хранится и проверяется в PostgreSQL по
> организации. Браузер обращается к этим маршрутам через `/api/market-data/*`;
> пути API-router ниже приведены без proxy-префикса.

This document fixes WEB-EPIC-03 architecture: auth-only JSON API endpoints for UI dropdown/search,
backed by ClickHouse reference tables `market_data.ref_market` and `market_data.ref_instruments`.

## Goal

- Let Web UI load enabled markets and search enabled/tradable instruments without direct ClickHouse access.
- Keep responses deterministic (stable ordering) and storage reads explicit (no ORM, no hidden FINAL).
- Reuse existing identity auth (opaque Roehub session cookie) via API `current_user_dependency`.

## Context

- Reference data source-of-truth lives in ClickHouse (DDL):
  - `migrations/clickhouse/market_data_ddl.sql`
  - tables:
    - `market_data.ref_market` (ReplacingMergeTree(updated_at), key `market_id`)
    - `market_data.ref_instruments` (ReplacingMergeTree(updated_at), key `(market_id, symbol)`)

- UI runs same-origin behind edge/web proxy (WEB-EPIC-02): browser calls JSON API via `/api/*`,
  but API router paths remain without `/api` prefix.

Docs:
- `docs/architecture/roadmap/milestone-6-epics-v1.md` (WEB-EPIC-03)
- `docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md` (UI uses these endpoints)
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

- `GET /market-data/catalog?market_id=&q=&limit=`
  - auth-only and organization-scoped for `selected`, `strategy_pinned` and `effective`
  - returns public catalog rows plus `catalog_state`, `coverage_state`,
    `coverage_percent`, `artifact_state` and actual `artifact_bytes`
  - `coverage_state=unknown` remains explicit until a symbol-specific exchange
    history lower bound is confirmed; the API must never infer `100%`

- `PUT|DELETE /market-data/selections/{market_id}/{symbol}`
  - auth-only, idempotent organization-owned intent with an audit event
  - deletion remains allowed while the instrument is pinned by an active strategy;
    the response reports `strategy_pinned=true` and `effective=true` rather than
    stopping the strategy or deleting market data

### Navigator full-history semantics (2026-10-04)

The candidate `apps/navigator-web` measures history from the instrument's confirmed
`market_data_history_bounds.first_open_at` to the latest closed UTC minute. The
start is discovered once through the existing provider-boundary inbox; an unknown
start is explicit and cannot imply zero or complete coverage. The accepted
`apps/platform-web` client is unchanged.

- `GET /market-data/workspace/instruments?history_scope=full` adds full-history
  coverage, per-instrument `history_start_at` and `history_end_at`. The default
  `history_scope=range` and existing bounded coverage endpoint stay compatible.
- `GET /market-data/workspace/history` validates authenticated catalog access and
  returns exact unique/expected/missing minute counts, bounds and discovery state.
  Actual counts use the canonical minute bitmap, not raw row totals.
  `source_absent_minutes` reports confirmed exchange omissions separately. Expected
  candles exclude those minutes; incomplete coverage never rounds up to 100 percent.
- Work requests and batches accept additive `full_history=true`. Each instrument
  uses its own confirmed start. The requested end remains part of the idempotent
  command; Navigator captures a fresh closed-minute end when the user submits.
  Custom ranges remain supported, but cannot precede a known exchange origin.
- Existing work runners find the first missing interval, skip confirmed history,
  and fetch only absent minute runs. They verify each written window before
  acknowledging progress. A fully consumed valid REST response is authoritative
  about candle availability, including empty windows and trading halts. A returned
  candle missing from canonical storage uses `source_history_incomplete`; three
  retries without confirmed progress end in an explicit failure. Normal
  source/storage backoff, fencing, pause/cancel priority and four-worker admission
  are retained. No candle deletion or checkpoint reset is involved. ClickHouse
  migration `0003` adds the separate source-absence table.
- Streaming selection persists desired collection immediately, including unknown,
  empty or partial history. PUT no longer returns `409 full_history_required` and
  does not synchronously read history storage. Authentication, organization scope,
  same-origin and catalog eligibility remain enforced. `effective` means intent,
  not verified live delivery. Navigator permits disabling during preparation.
- The existing collector discovers history origin and queues the first missing
  interval. Its durable REST queue checkpoints 60-minute windows without waiting
  for the next subscription refresh between windows. Each window rechecks global
  effective selections (including strategy pins); disabling all consumers stops
  preparation after the in-flight window. Receipts for actual received candles
  are still repaired independently. Only verified history admits a new WS stream;
  normal reconnect/tail repair handles minutes closing during the handoff.
- Existing selection audit rows persist requested/disabled actions. Sanitized
  collector transition logs record origin discovery, preparation, history ready
  and disable; durable recovery receipts expose retry time and terminal cause.
  Temporary readiness reads retry with bounded backoff without tearing down
  unaffected active streams. This is not a browser-error telemetry service.
- Existing selected instruments are verified on collector startup. The existing
  durable REST queue fills gaps before WebSocket admission. Unknown origins enter
  the discovery inbox. Pending repairs deduplicate by instrument so a moving
  clock cannot bypass backoff. Persistent source/storage errors block that
  instrument after bounded retries, without stopping siblings. Successfully
  verified exchange omissions do not block admission. Failed admission
  receipts survive restart and expose only a sanitized `history_repair_error`.
  Manual history repair can unblock admission; verified completeness acknowledges
  stale failure receipts before live collection starts. Existing live reconnect
  and tail repair continue after admission.

Navigator collection status also distinguishes the saved intent from the live
collector lease and durable repair receipts. Additive `stream_state` values are
`disabled`, `unknown`, `offline`, `disconnected`, `running`, `recovering`, `retry_wait`
and `blocked`.
The lease proves that the collector process owns ingestion, not an open provider
socket. Migration `0033_market_data_stream_connections_v1.sql` adds ephemeral
per-instrument socket observations. The single collector publishes its current
bounded snapshot every 5 seconds using its lease session; API accepts observations
only from the active lease PID and younger than 20 seconds. Disconnection (including
an initial failed handshake) takes precedence over pending recovery. A missing or
expired observation cannot produce `running`; it produces `unknown` unless a
repair state is known. Worker death still produces `offline`. On disconnect the UI
shows an exchange connection error, without healthy collection or updating history
copy. Current coverage and last stored candle remain separate evidence.
The latest stored candle is read from the existing exact actual-minute bitmap,
restricted to minutes strictly before the requested bound. Empty history returns
null; verified source absences are excluded. This avoids rescanning all canonical
daily parts on each inspector poll. Index verification/repair is the same
prerequisite as full-history coverage; no source-absence marker becomes a candle.
The endpoint shape and exclusive-bound semantics are unchanged (`none`).

Compatibility: persistence is `compatible-change` (additive table); the expanded
closed `stream_state` enum is `breaking-change` for older strict Navigator clients.
Deploy the updated client, migration, API and collector together. Apply the schema
before starting updated API/worker processes; an old collector yields unknown
socket health rather than a false healthy status. No provider endpoints or intent,
repair and candle persistence contracts change. Pending
repairs may expose their safe `next_retry_at`; internal tokens are never returned.
Selected stream views poll every two seconds independently of general page refresh
preferences. A pending candle is shown as history updating, not a fabricated 100%.
Manual fill controls are suppressed while streaming owns recovery, except an
explicitly blocked history repair. The enabled switch uses the existing violet
state token.

History coverage, completed job range and collection intent are distinct:
`effective=true` is a requested/strategy-required collection intent, not proof of
an open WebSocket. A completed job confirms all exchange-available candles in its fixed requested
interval, with source omissions recorded separately; later
closed minutes can make current full-history coverage incomplete when streaming
is disabled. Canonical bitmap aggregates assume the existing append-only candle
lifecycle; administrative deletion requires rebuilding that aggregate index.

### Verified source availability and index repair (2026-10-10)

`source_absent_candles_1m` stores minute keys omitted by a fully consumed valid
REST response. These records contain no prices or volume and never enter candle
exports, charts, backtest inputs or canonical counts. Transport/provider errors,
malformed responses, partial iteration, failed writes and lease loss cannot
acknowledge source absences. Every existing/returned candle must first be found
in canonical storage. Receipts for actually received WebSocket candles continue
to require real candles, even if a REST response is empty.

Verification repairs missing bitmap keys from the bounded canonical window. The
repair adds bitmap/min/max states with a zero raw-row counter, so retries do not
inflate ingestion totals. Candle values are not rewritten. The normal window
limit is 1,440 minutes for manual jobs and 60 for stream preparation; the verifier
rejects ranges larger than seven days. Work-request metadata writes recheck the
same coordinator/worker fence as candle writes. Source omissions are written
only after canonical verification, so interruption before metadata persistence
leaves the interval eligible for another read.

Gap discovery and stream admission use the union of actual candle keys and
confirmed source absences. Expected history is calendar minutes minus source
absences that have no actual candle. A later real candle wins over an old absence
without double counting. A known fully absent range is complete with zero actual
candles; an unknown origin remains unknown. Job `completed_units` counts verified
calendar minutes, whereas `rows_written` counts real writes. A job's fixed range
and the moving current-history range remain distinct.

The catalog shows `Collecting` only for a fresh connected stream with current
coverage, `Catching up`/`Recovering` for incomplete/recovering history, and explicit
retry, stopped, disconnected, blocked or checking states. Saved intent still
controls the switch; it does not display as proof of live collection. Row status
stays on one line in loading, retry and error states. The inspector explains the
excluded source minutes.

Compatibility: persistence and optional API fields are `compatible-change`;
provider endpoints, selection/auth contracts and candle payloads are `none`.
The meaning of full-history completeness is a user-authorized `breaking-change`
from a candle for every calendar minute to every exchange-available candle.
Apply ClickHouse `0003` before updating API, scheduler and collector; update the
Navigator client with them. Existing actual/expected fields retain numeric types.
Rollback preserves canonical data and the additive table, but old readers ignore
source absences and can block again at legitimate trading halts. Rebuilding the
bitmap after administrative deletion remains a separate operator operation.

### Artifact source observations (S01, 2026-10-10)

The consumer-owned `StoredCoordinates` port in Backtest Artifacts adds
`observation_page(after, limit, before)` and a single-coordinate `observation`.
The adapter merges actual-minute bitmap states, then clips keys to the exclusive
cutoff. Non-aligned cutoffs include an existing minute whose opening time is
strictly earlier; publisher calls use the last closed-minute boundary. Empty
bitmaps return null bounds, while epoch minute zero remains valid. Source absence
records never supply first/last artifact candles. The canonical scalar reader
also uses nullable aggregates for its documented empty-result semantics.

Ordered pages contain at most 50 coordinates, including future-only/empty states
so their cursor advances. The existing coordinate-only method remains available.
Normal publisher reconciliation reads the index once per page with bounded query
resources; it does not scan canonical history or perform per-coordinate probes.
REST/WS post-commit hints retain batch coalescing and failure isolation. Optional
hints and the periodic sweep observe the same index; neither repairs drift by
itself. Missing index keys use the bounded verification/repair procedure above;
unavailable queries propagate failure instead of reporting a successful empty
sweep. No canonical fallback, schema migration or absence-policy change is added.

Compatibility: supported internal composition callers retain their constructors
and coordinate-only reads (`compatible-change` under the existing index lifecycle).
Public API, artifact identity, candle payloads, persisted schemas and configuration
are unchanged (`none`). Source obligations, policy dispatch and installation
activation remain independent. Deploy updated reader/adapter/application files
together; rollback restores the prior code without rewriting data or cursors.

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

For every runtime-visible change, run a local Docker `runtime smoke` and real
browser proof: apply the PostgreSQL migration in a clean Compose environment,
open `/settings#market_data`, find `BTCUSDT`, make both mutations through the
UI, and confirm `/api/market-data/*` returns `200`. This does not replace the
future `linux/amd64` matrix.

## Risks / open questions

- Risk: `q` omitted (no filter) may be slow if `ref_instruments` becomes large.
  Mitigation v1: UI uses typeahead and should pass `q` for search.
- Risk: reference tables might be empty if market-data sync has not been run.
  Mitigation: run seed/sync jobs (see `docs/architecture/market_data/market-data-reference-data-sync-v2.md`).

## Artifact policy scheduling clarification

Persisted canonical candles and NPY readiness remain independent observations.
The post-commit signal plus bounded reconciliation discovers only stored coordinates;
reference catalog membership alone does not start downloads or artifact builds.
In artifact policy commands, transitioning on-demand → precompute requests immediate
work. Saving an already-precompute profile, repeated precompute mode or a schedule-only
edit returns `policy_updated` without a new request ID; the next cadence occurrence
uses the current policy. Explicit Update now remains immediate and does not move the
anchor. A retry rechecks the submitting actor's current membership and coordinate
access; a revoked request is not promoted to ready by another shared request's success.


### ClickHouse admission and API compatibility (S02, 2026-10-10)

Reference, catalog, workspace and dashboard ClickHouse reads select the
`interactive` identity in trusted composition. HTTP inputs cannot select a class
or settings profile. Public response schemas, authorization, source absence and
read retry/loading contracts are unchanged. Missing/unreadable class credentials
fail startup; they never fall back to shared credentials in an activated install.
Read failures remain failures, not successful empty catalogs or exchange absence.
Native admission code 202 reuses the existing safe `storage_unavailable` reason
for durable ingestion retries; 241 retains `storage_memory_pressure`. Worker
checkpoints, backoff and pending recovery metadata survive both conditions.
No new public state or UI retry policy is introduced. Server enforcement and
upgrade/rollback are specified in the S02 section of the local recovery runbook.

### Rolling execution and shared REST admission (S03, 2026-10-10)

The request coordinator retains ownership over a rolling pool: a completed slot
can claim the next oldest eligible job while another instrument is still slow.
One daily 1,440-minute window per claim and configured instrument concurrency
(dev: 4) remain unchanged. Orphan recovery happens before dispatch, never beside
live siblings. A durable coordinator token prevents takeover on session loss;
shutdown stops admission and drains owned I/O before ownership release.

Bounds discovery has an independent single-worker coordinator. A temporary
source/quota failure remains queued with a durable retry deadline, rather than
claiming the instrument's history is unavailable. Existing API states, DTOs,
selection/auth scope, source-absence semantics and shared UI-loading contract
remain unchanged; internal retry columns are not a new browser contract.

Scheduler, collector REST children, catchup CLIs and strategy history lookups
share PostgreSQL provider admission under an explicit egress scope. Every HTTP
attempt reserves all weighted quota dimensions; shared cooldown and spent quota
survive process restarts. REST session reuse has an absolute call deadline and
confirmed I/O termination before capacity release. WebSocket traffic itself does
not consume REST permits. Unknown coordination fails closed; external clients
sharing a NAT are outside this contract.

Migration `market-data-provider-admission-0034` and matching configuration must
precede new runtime clients; mixed legacy/new clients do not provide a shared
quota guarantee. Upgrade, rollback and exact-token confirmed-drain recovery are
specified in the S03 section of `docs/runbooks/navigator-local-market-data-recovery.md`.
This describes the source candidate, not installed-runtime acceptance.

### Asynchronous durable collector admission (S04, 2026-10-11)

Collector acceptance follows capacity reservation → batched PG commit → buffer
admission → synchronous canonical/MV write acknowledgement → token-fenced receipt
completion. Pending producers, in-flight batches and retries share bounded row
and accounted-byte budgets. Awaiting capacity is not acceptance. Cancellation
cannot cancel another producer's shared commit or delete an uncertain receipt.
A lost PG reply closes the lease session; restart reconciles stable keys, keeping
failed/rearm semantics and actual-candle-only WS recovery. No exactly-once claim
is made across the two databases.

All collector recovery and health operations use the lifecycle-owned async PG
session. Synchronous administrative/history consumers retain their existing port.
The internal async port is additive; journal schema, public DTOs, authorization,
source-absence semantics and UI loading behavior are unchanged. This source
contract is not an installed-runtime rollout receipt. Drain/rollback and observed
proof boundaries are in the S04 section of the local recovery runbook.
