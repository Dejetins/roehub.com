# Roehub UI functional contract v1

## Status and authority

- Status: `current functional contract`, consolidated on 2026-09-08 under the
  user's instruction to reuse existing and historical functional requirements.
- Scope: self-hosted local platform. Public `roehub.com` retains its separate
  identity, account, release, responsive and deployment specifications.
- Machine companion: [functional registry](roehub-ui-functional-registry-v1.json).
- First execution task: [Backtests client](../../../../.codex/tickets/2026-09-08-roehub-backtests-client.md).
- Local Backtests implementation and T1–T8 proof status:
  [completion report](../../../../.codex/delivery/evidence/ROEHUB-BACKTESTS-CLIENT-2026-09-08.md).
  Its bounded own-resource evidence does not accept target roles, default cutover,
  or the wider platform; registry `requires_ui` / `requires_api` tags remain.
- This document and its registry own current functional coverage and delivery
  classification. The older v1 inventory/screen registry remain baseline evidence;
  their smaller counts do not limit the current product scope.
- Existing [access contract](roehub-local-platform-access-and-route-contract-v1.json)
  and domain/server contracts retain security authority. Additional screen
  records never grant a role a new capability. No security policy or runtime
  route is implemented by this consolidation.

The accepted v23 is the visual guide for all subsequent local UI work. It is
not a source of fictional users, plan tiers, service status, new permissions or
API behavior. Reuse its visual language while adapting composition to each task.
The former staged design program remains retired; this contract selects no
prompt pack, ledger, separate atlas ceremony or mandatory prototype series.

## Accepted Backtests implementation

The current Backtests UI iteration was accepted by the product owner on
2026-09-11. Its accepted composition, chart behavior and retained limitations
are recorded in the [iteration log](backtests-ui-iteration-log.md). This is a
bounded acceptance of the implemented journey, not target-role/default-cutover
acceptance for the entire platform.

## Sources and reconciled decisions

1. [Product requirements](../../platform/roehub-product-transformation-requirements-v1.md)
   own product goals, client stack, chart platform, localization and responsive scope.
2. [Baseline IA](roehub-local-platform-information-architecture-v1.md) and its
   access/route contract own preserved navigation/compatibility and security rules.
3. Functional additions come from the accepted v2 completeness package at Git
   revision `fd453b5a`: screen registry, surface registry and information architecture.
   They add nine records and six journeys; all original identities are retained.
4. The later `48d23ad5` journey graph had `status=review`. Its 61 edges are
   candidate evidence only; this contract does not silently accept them all.
5. [Previous Web behavior contract](web-ui-v2-1-production-contract.md), current
   API/DTO, domain documents and tests supply reusable behavior. Old visual theme
   counts, mobile layouts, OIDC-only wording and deployment instructions do not transfer.

| Topic | Current decision |
|---|---|
| Client architecture | Existing accepted pnpm/React/TypeScript/Vite stack; `apps/platform-web`, separate `apps/site`, bounded Roehub packages; `apps/web` retains session/proxy authority during migration |
| Charts | New charts use ECharts; backend owns metrics, time identity and aggregation |
| Visual/responsive scope | v23 direction; 820/1024/1440, RU/EN, keyboard, zoom, reduced motion; no local phone product or inherited six-theme requirement |
| Strategy editing | Immutable spec changes create/clone a strategy; launch profile configuration is a separate mutable operation |
| Backtest retry | Target operator retry creates a new domain job with the same immutable input and renewed admission; not a generic runtime retry |
| General job retry | Same job, new attempt; retain semantic input identity and apply isolated runtime eligibility; cancellation is not automatically retryable |
| Stored secrets | Never reveal, echo, cache in UI state journals, or include in evidence |
| Current implementation | Source evidence only until the implementing task obtains real browser/API proof |

## Reading delivery status

The four tags are independent and may coexist on one record:

- `implemented` — **реализовано**, but only the explicitly named existing
  behavior in `implemented_scope`; never a claim that the new client is complete.
- `requires_ui` — **требуется UI**: client composition, interactions and states.
- `requires_api` — **требуется API**: new projection/command or integration of an
  already specified capability, lifecycle or validation contract.
- `needs_clarification` — **нужно уточнение**: a specific remaining behavior
  decision, named below; this does not reopen established requirements.

`source_refs` and `remaining_work` explain each classification. Historical and
QA/nonvisual records carry an explicit disposition; they are not extra product
screens. The registry contains 44 records, 43 source surfaces and 18 journeys.

## Product coverage

| Area / route | Required functionality | Remaining boundary |
|---|---|---|
| Login / setup | Local passkey, approved fallback/recovery, safe return, owner/organization bootstrap, selection skip and resumable setup | Target UI; complete setup routing and catalog protocol |
| Shell / dashboard | Organization context, navigation, search, notifications, preferences, session state, summary/readiness and next action | Target UI; target role-filtered projections |
| `/data`, `/data/ingestion` | Catalog/selection, provider segment, history, pinned instruments, coverage, gaps, backfill, freshness and permitted retry | Target UI; snapshot/cursor and ingestion projection/action API |
| `/artifacts`, `/artifacts/{bundle_id}/{version}` | Search/filter, manifest, provenance, compatibility, storage/quota impact and consumer links | Existing ArtifactStore is reusable; browser catalog/detail API and UI required |
| Strategies | Library, builder, clone, archive, immutable spec detail, launch profile, run/stop/restart and risk/unknown state | Target UI and role integration; durable editor draft policy needs clarification |
| Backtests | Configure/preflight, source selection, admission, submit, queue/progress, cancel/delete, results, materialization, trades/CSV and create strategy | Opt-in artifact-mode own-resource client implemented with local browser/API evidence; broader modes/ETA/target roles and their UI remain separate work |
| Models | Model/experiment registry, training, comparison, readiness, promotion and rollback | Dedicated API/UI binding; exact consumer actions need clarification before its task |
| `/live`, `/live/activity`, `/live/executions/{intent_id}` | Paper/live positions, orders/fills, risk, bounded execution history, source-to-fill chain and reconciliation | Reuse execution domain; target projections and action policy integration |
| Hosted manual position flow | Open/close, strategy/connection, size or close scope, risk preflight, recent auth, confirmation, result identity | Universal execution path and target UI; no independent trading engine |
| `/jobs`, `/jobs/{job_id}` | Cross-capability queue/history, attempts, inputs/outputs, measured progress, errors, cancellation/retry eligibility | Runtime exists; browser APIs/UI and explicit per-capability identity mapping |
| `/monitoring`, `/monitoring/{service_id}` | Functional groups, service/datastore detail, freshness, metrics/backlog, dependency impact, jobs, runbook and safe action | Target projections/UI; databases remain services, not separate navigation sections |
| Personal settings | Profile, preferences, notification routes, sessions, audit and recovery | Reuse account/local-auth APIs; target UI |
| Administration | Members, last-owner protection, separate resource budgets, extensions, backup/update/rollback and operation reconciliation | Reuse administration APIs; map target installation authority and remaining projections |
| `/docs`, canonical runbooks | Local version matching, search, visibility, 403/404, incident context and safe login continuation | Product docs UI and explicit framework-doc route migration |

Grouped navigation is Overview (`/dashboard`), Research (`/data`, `/artifacts`,
`/strategies`, `/backtests`, `/models`), Operations (`/live`, `/jobs`, `/monitoring`,
`/connections`), System (`/docs`, settings, administration). Detail routes and
the manual-position overlay do not become additional primary destinations.
The eight restored route patterns are additive targets; no runtime route or
old bookmark is removed by this document.

## Shared behavior contract

### Data and errors

- Distinguish loading, empty, ready, degraded, unavailable, stale and failed.
  Retain last valid data on transient refresh failures and label its age/source.
  Failure of one module does not hide healthy siblings.
- Follow server refresh limits, `retry_after_seconds`, `next_allowed_refresh_at`
  and materialization hints; bounded read retries stop on session/policy errors.
- Display user-actionable localized errors. Map validation paths to fields and
  provide an accessible summary; preserve the raw error code for support without
  displaying private payloads. Follow the [API error envelope](../../api/api-errors-and-422-payload-v1.md),
  allowing documented endpoint-specific responses instead of assuming all errors
  share one shape.
- 401 means re-authenticate with sanitized local `next`; 403 means insufficient
  authority; 404 means absent/not visible according to the endpoint; identity
  outage is not an invalid password. 429 is not permission denial.

### Commands, cancellation and recovery

- Server owns authorization, scope, recent-auth, CSRF/origin, validation, risk
  and audit. UI disabled state cannot substitute for these controls.
- Every command has pending feedback and duplicate-click prevention. Destructive
  or high-risk actions require confirmation stating the affected object.
- Separate canceling a dialog (no request), aborting a browser read, requesting
  job cancellation, and deleting job history. They have different consequences.
- Cancellation remains pending until the server reports a terminal outcome;
  completion can win the race. Keep committed results per domain contract.
- Unknown mutation outcome means the request may already have succeeded.
  Reconcile using the authoritative object/command identity, or replay the same
  request/key only where the endpoint guarantees idempotency. Never manufacture
  a fresh key as an automatic retry. Request IDs are tracing, not deduplication.
  Recovery identity includes organization and actor; same-user re-authentication
  alone does not prove the original organization scope. If safe server-bound
  replay is unavailable, preserve an unresolved state and read-only guidance.
- Treat `202` result materialization separately from job failure. Retry the
  documented read, keeping job and variant identity stable.
- Recovery is domain-specific: regain a session, refresh a stale projection,
  find a submitted job, reconcile execution, retry an eligible attempt, or follow
  an installation recovery operation. Do not implement a universal Retry command.

### Drafts, routes and accessibility

- Validation or transient read failures preserve editable in-memory input.
  The first Backtests task defines its bounded reload/unknown-command behavior;
  durable strategy drafts remain Q-DRAFT rather than an implied product promise.
- Resource identity belongs in canonical URLs. Preserve safe selection across
  refresh/back/forward and invalidate stale requests when the selected object changes.
- New client routes retain SSR fallback until their scoped browser/API proof
  passes. The first ticket does not switch production routing by default.
- Controls have accessible names, focus order and status feedback; dialogs
  support Escape where cancellation is safe, focus containment and focus return.
  Meaning is not carried by color alone; charts expose units/timezone/source and
  a text/table alternative. Verify RU/EN expansion and zoom at the accepted widths.

## Clarifications and delivery dependencies

| ID | Remaining question / work | Resolution owner and deadline |
|---|---|---|
| Q-DRAFT | Strategy draft persistence/lifetime and return after reload or re-auth | Strategy-editor task before durable draft implementation; no blocker to Backtests |
| Q-MODELS | Exact model/experiment comparisons and promotion/rollback consumer action mapping | Models task using existing domain contracts before exposing controls |
| Q-JOB-ADAPTER | Which job/attempt identity and retry endpoint each capability uses | Jobs API task before a generic queue control is exposed |
| API-AUTHZ | Kernel exists; delegation/envelope and product route integration are unfinished | Existing authz tickets; target role acceptance and default client cutover depend on relevant integration |
| API-DATA | Snapshot/cursor catalog and ingestion browser projections | Data API/UI task |
| API-BACKTEST-MODES | `auto`/`direct_db`, equivalence and source-choice truth | Dedicated backend task; artifact-only first delivery does not remove full target |
| API-ETA | Reliable queue/execution ETA with confidence and calibration | Backend evidence task; show unavailable rather than simulated estimates |
| API-JOBS-ARTIFACTS | General runtime/store exist without complete target browser APIs | Separate projection/action contracts, reused by domain journeys |

No unresolved item is silently treated as implemented. A task can proceed on a
bounded available contract; it cannot claim the deferred capability. Public
site account/privacy/download/demo work remains in its separate specification.

## Delivery and compatibility

The user-selected [UI implementation plan](roehub-ui-implementation-plan-v1.md)
sequences this coverage. Its six-stage Backtests pack decomposes the existing
ticket while retaining complete-journey acceptance; later blocks remain a
dependency backlog. Pack entry readiness is tracked separately in its ledger.

Use a ready task for one bounded journey. Specify fields, states, actions,
API bindings, errors/recovery, owned paths and observable acceptance. Reuse
these requirements instead of requesting another whole-product design cycle.

This documentation change is `compatible-change` to functional coverage and
`none` to runtime API, persistence, authorization and deployed routes. Future
framework `/docs` relocation remains `breaking-change`; role integration and
client cutover require task-specific classification. No deployment target is
selected by this contract.

## Verification

Registry IDs, sources, states and journey references are checked during this
consolidation; existing docs tests remain checks of the preserved baseline.
Implementation evidence must subsequently include real API-backed browser
journeys, affected roles, failure/recovery, localization, keyboard/zoom and
820/1024/1440. Static registry validity never proves application completion.
