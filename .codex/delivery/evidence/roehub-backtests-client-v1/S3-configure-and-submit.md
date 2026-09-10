# S3 — Configure, preflight and submit with recovery

Date: 2026-09-08. Execution: one executor, local checkout, S3 only.
Status: implementation and required local checks passed; coordinator final acceptance remains separate.

## Authority and proof boundary

The user selected sequential execution and superseded staged-plan-runner,
PACK-CLAIM, updater, claims, ledger transitions and receipts. Those requirements
were not executed. No Goal, agent, additional task, branch, worktree, stash,
commit, push, merge, deployment, default cutover, AGENTS or global skill change
was made. The old ledger and prompt pack are unchanged. The v23 reference is
unchanged: SHA-256 `3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.

S3 proves the local authenticated configure → preflight → create boundary,
field/error handling, one logical create and bounded unresolved recovery. It
uses real production API/auth/runner composition with disposable PostgreSQL,
ClickHouse and synthetic artifact data. It does not accept lifecycle/cancel,
results, strategy-save, exports, provider behavior, target roles or deployment.
`local_journey_verified=false`; `target_role_cutover_ready=false`.

## Delivered behavior

- `/backtests/new` now renders `builder.tsx`; the library New action is a real
  link. The existing v23-derived palette, panel borders, radius, sidebar and
  compact controls are preserved. The form and preflight review use two panels
  at wide desktop widths and stack at narrower widths. No fictional progress,
  permissions, data source mode, account or results were added.
- `builder-api.ts` provides specific Zod allowlists for defaults, catalog,
  artifact bounds, research requests, preflight and create identity. Generic
  workstation responses, storage paths, provider data and unused artifact
  hashes are excluded from builder state and recovery. There is no direct
  browser connection to internal services or the artifact filesystem.
- Runtime defaults and workstation selections initialize the draft. The
  workstation's proposed 2023 start/1h/first indicator is preserved rather than
  replaced with the fixture's working values; server admission/artifact
  validation remains required. Missing defaults block configuration/submission.
  The existing jobs-projection resolver defect remains untouched and does not
  block the separate healthy catalog/defaults or direct jobs API.
- Exchange and market edits requery workstation with `instrument_exchange` and
  `instrument_market_type`. Query identities and AbortSignals isolate each
  selection. A late old symbol list cannot validate a new market. Existing
  coordinates remain visible when unavailable; the client never silently
  changes market to permit short execution. Submission waits for current
  catalog and bounds. Read errors retain draft values and offer explicit
  cooldown-respecting catalog/bounds refresh.
- `builder-model.ts` validates the entire ticket field matrix. Ordered indicators
  can be added, removed and moved; source-less indicators permit an empty source
  list. IDs lacking the API's required window axis are explicitly unavailable.
  Integer grids are checked against actual materialized values without allocating
  arbitrary user-sized ranges. Unsupported extra parameter axes are not invented.
- Time inputs require valid UTC timestamps ending in Z and nonempty half-open
  `[start,end)` semantics. The selected artifact's maximum end is checked;
  preflight remains authoritative for additional artifact/time constraints.
  No dates are clamped. The hit-times timeframe is shown as source metadata,
  not incorrectly imposed as a restriction on all supported research timeframes.
- TP/SL has separate explicit enable controls, positive percent ranges and
  coverage checks against the runtime grid. Direction uses supported modes and
  market compatibility. Initial cash, rates, mode-specific sizing and top N are
  validated as finite values with the server's documented limits. Fees and
  slippage display percent and convert once to fractions; equity_pct and TP/SL
  remain API percentage values. Inactive risk/sizing fields are omitted from
  submission while retaining editable draft state.
- Funding, profit lock, close-on-end and minimum-trade policy are carried from
  authoritative defaults without unsupported editors. Preflight renders
  normalized policies, cash, sizing, direction, rates, dates, ranking and top N,
  artifact freshness, cost, funding coverage and warnings. Labels and known
  policy/metric/status values are localized in RU/EN. Server issue messages and
  support codes are retained as text, never HTML.
- Body `errors` block creation even on HTTP 200. Errors are mapped from dotted
  paths, including `body.` prefixes and indexed fields, to accessible fields
  and a focusable summary. Result-affecting edits increment a generation:
  edit→revert cannot revive a successful preflight or admit a late in-flight
  response. Label-only edits do not invalidate computation. Label normalization
  matches whitespace normalization and the server's 96 Unicode-character limit.
- Create independently checks admission again. Error handling covers 422,
  403/404, 409 normalized/effective identity conflict, 429, 503, transport and
  invalid responses. It retains input and identity. Admission resource/limit,
  requested value or current usage are specifically allowlisted and displayed.
  Both Retry-After and `error.details.retry_after_seconds` are honored. Definite
  403/404/422/429 rejection permits an explicit return to editable configuration;
  409 and unknown outcomes do not offer key rotation or blind resubmission.
- A synchronous command lock precedes fresh subject verification and prevents
  duplicate clicks. Before POST, one exact allowlisted request and one
  `Idempotency-Key` are recorded. Controls remain frozen during the attempt.
  Only a validated returned job UUID causes detail navigation. Resolved recovery
  payloads are removed; a persisted known UUID is read directly on restoration.
  An absent/hidden known object remains unresolved rather than proving failure.
- `recovery.ts` stores at most one operation record per tab, bounded to 256000
  serialized UTF-16 code units. Allowed fields are operation, exact research body,
  key, creation time, authenticated subject, server-derived organization when
  known, and known result UUID. It stores no credentials, cookies, generic API
  response or durable ordinary draft. Storage failure preserves memory state
  and visibly warns before sending. A newly detected write failure stops POST
  until the user explicitly proceeds with the displayed memory-only limitation.
- The current current-user API cannot establish organization and create does
  not bind a replay to its original organization. Therefore job-create replay
  is disabled in every recovery state. Unknown outcomes show read-only history
  guidance and preserve body/key. Empty history, unknown/expired retention,
  changed/unknown organization, and same-subject login cannot enable replay.
  No client organization override or server authorization change was introduced.
- Logout and subject change clear query/recovery state. Session expiry retains
  the unresolved attempt for same-subject re-entry while removing private UI.
  A protected 401 from any read or imperative preflight/create signals expiry
  immediately after receiving the status, before consuming a potentially broken
  response body. Further private reads/commands stop; identity outage remains
  unavailable rather than logout. Aborted late work cannot repopulate old state.
- Ordinary unsent configuration stays only in memory. Data-router blocking
  covers in-app links and browser back/forward; SSR links and reload have
  discard protection. Confirmed SSR navigation does not prompt twice. Locale
  changes retain their existing SSR endpoint and follow draft discard semantics.

## Validation and evidence

Commands ran from `/Users/daniildegtyarev/Projects/roehub.com`.

| Command / check | Actual result |
| --- | --- |
| `pnpm --filter @roehub/platform-web typecheck` | Passed on final client/tests |
| `pnpm --filter @roehub/platform-web test` | Passed: 104 tests in 4 files; 41 original tests retained, including the intentionally updated New-link assertion |
| `pnpm --filter @roehub/platform-web build` | Passed; final JS `main-DBt7Ww8x.js`, CSS `main-BpfXO_k1.css`; Vite emits its >500 kB chunk-size advisory |
| `ROEHUB_PROOF_STAGE=S3 pnpm --filter @roehub/platform-web test:e2e` | Passed: 5 foundation/library tests, then 6 builder tests, using separate cold fixtures |
| `pnpm --filter @roehub/platform-web test:e2e builder.spec.ts` | Passed: 6 S3 scenarios after subsequent model/error/recovery corrections; 50.3 s |
| `pnpm --filter @roehub/platform-web test:e2e builder.spec.ts --grep 'dirty discard\|real preflight 401'` | Passed: 2 affected scenarios on final build, including added back/forward discard and immediate 401; 42.7 s |
| `pnpm --filter @roehub/platform-web test:e2e builder.spec.ts --grep 'real configure/preflight/create'` | Passed: final-build real create/identity flow and real preflight screenshot checkpoint |
| `.venv/bin/python -m pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_web_v2_1_routes.py` | Passed: 98 tests, 4.96 s; one existing httpx per-request-cookie deprecation warning |
| `python3 -m tools.docs.generate_project_map` | Regenerated required generated outputs |
| `python3 -m tools.docs.generate_project_map --check` | Passed |
| `python3 -m tools.docs.generate_docs_index --check` | Passed; docs index did not need regeneration |
| `git diff --check` | Passed |
| v23 SHA-256 | Matches the preserved S1/S2 value above |

The full S1/S2 browser regression preceded final narrowly scoped admission-body,
validation and immediate-401 improvements. Existing read behavior was additionally
covered by the final unit suite; the final browser reruns targeted changed S3
boundaries rather than claiming every screenshot came from one identical build.
No test failure was treated as passed or silently retried to obtain a green result.
One supplemental screenshot command initially selected zero tests because its
anchored grep matched neither the file-qualified test title nor a test; the
corrected unanchored selection ran the actual test. That zero-test run is not proof.

Browser evidence is relative to this report under `browser/S3/`:

- `real-preflight.png`, `real-preflight-effective.png`, `created.png`, `real-create.json` — actual client create, exact rate conversion,
  one UI POST despite double click; separate direct server identity probes.
- `lost-response-reload.png`, `lost-response.json` — actual accepted create
  whose response is dropped, exact body/key after reload, one persisted job,
  changed sole active organization, empty new-scope history, same-subject
  re-authentication and logout clearing. No UI replay occurs.
- `memory-only-unresolved.png`, `faults.json` — explicitly controlled body-error,
  field 422, stale-preflight, 429 admission, write failure before POST and
  unknown 503 scenarios. These are fault-injection evidence, not real success.
- `en-820.png`, `en-1024.png`, `en-1440.png`, and corresponding `ru-*` files;
  `visual.json` records no horizontal document overflow and zero axe violations
  at all six combinations, plus dirty link/back/forward discard behavior.
- `en-native-zoom-200.png`, `ru-native-zoom-200.png`, their `*-actions.png`
  counterparts and `zoom.json` — native Chromium zoom 2, outer width 1440,
  inner width 720, DPR 2; actions remain reachable and keyboard focusable,
  with zero axe violations. No CSS/pinch zoom is claimed as native zoom.
- `preflight-expired.png`, `preflight-expiry.json` — actual persisted session
  expiration, actual preflight 401, private UI removed, **zero protected requests
  for 31 seconds**, then successful same-subject login with no durable unsent draft.

Screenshots of RU desktop, native zoom actions, recovery and real review were
visually inspected. Source palette/layout are retained, text wraps and primary
controls remain reachable. No speech-output or screen-reader certification is
claimed. S1/S2 regression outputs are isolated in `browser/S3-library-regression/`
and `browser/S3-foundation-regression/`; accepted prior evidence was not overwritten.
Normal create had no JavaScript exceptions. Its only HTTP/console error was
from the deliberate 409 direct API probe. Lost-response abort, controlled fault
statuses and actual session 401 are explicitly expected faults, not normal success.

## Reproduction and real-server guarantees

Use the S1 fixture and its documented Node/pnpm/Python/Docker prerequisites.
No additional dependency was installed. Build before starting Web: the manifest
is read at startup. `e2e/run.mjs` invokes Playwright sequentially with separate
fixture lifetimes for foundation/library and builder; explicit arguments select
one focused invocation. This preserves the library's exact cold-empty assertions
and avoids weakening admission limits or seeding unrelated jobs into that test.

The fixture remains unmodified: Web :18480, API :18481, flag-off SSR :18482,
runner :18483, disposable PostgreSQL/ClickHouse, production artifact precompute
from 4320 synthetic 1m candles over `[2026-03-26T00:00:00Z,2026-03-29T00:00:00Z)`.
The real test explicitly selects `ma.ema`, `15m`, close source, window 10,
spot and long_only. Actual password authentication is used; no fabricated cookie,
auth override or intercepted successful create replaces the real path.

The create test performs two distinct, explicit **direct API identity probes**
after the UI's one POST: same key + changed label returns 200 and the original
UUID; same key + changed top N returns 409. These probes verify server semantics
and are never offered as recovery in the client. `real-create.json` separates
one UI create from those probes. Replay scope remains unsafe despite those
same-organization test observations.

Replay-retention inspection used the actual imported production
`_build_jobs_use_case` AST and `BacktestJobsUseCase.__dataclass_fields__`:
the factory constructs that dataclass without an `idempotency_ttl_seconds`
override; the effective default is **86400 seconds**. The S1 fixture calls this
same factory. This is a local factory/configuration fact, not a browser-provided
retention guarantee or a production-deployment claim. The browser intentionally
has unknown replay scope/retention and never replays.

The changed-organization test changes **disposable data only**. It creates a
replacement owner for the original organization, suspends the test actor's old
membership while preserving job foreign keys, and creates a single active
membership for the same actor in a new organization. Production constraints and
the real sole-active-membership resolver remain enabled. Reads show empty history
in the new scope while the saved unresolved request remains visible. Same-subject
re-auth does not rotate/replay the key. All fixture memberships/jobs/artifacts and
private credentials are removed with fixture teardown.

## Debugging and review

Review mode: executor cold self-review plus coordinator independent changed-boundary
review during implementation. The coordinator's final report acceptance is pending.
All communicated implementation findings were corrected and regression-tested:

1. Coordinate catalogs now depend on exchange/market, preserve invalid selected
   values and ignore late responses from the old market.
2. Imperative 401 handling closes the global private boundary. The final follow-up
   moved handling before body parsing; a rejected 401 body cannot delay expiry.
3. Policies, costs, funding counts/coverage, boolean values and effective settings
   now have localized primary labels and explicit percentage units.
4. Generation invalidation covers edit→revert both after success and during pending
   preflight; metadata-only label edits preserve computation identity.

Additional observed pre-fix failures and their resolution:

- The first data-router root consumed `/backtests` before the inner absolute routes,
  rendering an empty main. The root match now leaves child route resolution intact;
  real entry and all preserved routes passed afterwards.
- The first unit run omitted jest-dom matcher registration. Adding its import
  corrected the test harness; no product assertions were removed.
- Nested label/select text confused exact label targeting. Separate explicit labels
  preserve clean accessible names and field-error descriptions; browser checks pass.
- Dirty-link dialog assertions ran before the asynchronous blocker effect. The test
  now observes the dialog, and additionally verifies actual browser back/forward.
- A controlled preflight test inadvertently depended on free-tier active-job quota
  after two real creates. It now uses an explicitly controlled preflight response;
  independent real preflight/create evidence remains mandatory and passes. Server
  limits and cold-empty library assertions were not relaxed.
- The first fixture membership move was rejected by the real job/member foreign key.
  It rolled back. The final fixture uses suspension/new active membership and a
  replacement owner, preserving FK and last-owner constraints rather than disabling them.
- Self-review corrected materialized-range bounds, omitted inactive risk/sizing
  fields from the exact command, and used Unicode code points for metadata limits.
  Focused tests cover these cases.

No blocking implementation finding remains in the completed local S3 boundary.
Residuals: original organization-bound replay is unavailable by API contract;
workstation jobs projection is unavailable; the bundle has a Vite size advisory;
target authz/cutover, S4/S5 behavior and complete journey proof remain separate.
No runtime performance improvement or regression measurement is claimed.

## Compatibility assessment

Baseline: accepted S1/S2 gated client and unchanged current API/domain/auth behavior.
Candidate: S3 client, bounded per-tab recovery, tests and directly affected docs.

| Surface / consumer | Before → after | Classification / evidence |
| --- | --- | --- |
| Opted-in new route / library New action | Placeholder → real builder using same URL | `compatible-change`; real entry, deep links, locale, back/forward and regression proof |
| Catalog/defaults/preflight/create DTO consumers | New strict client reader/command adapter | `compatible-change` against the inspected current server; real API flow plus field/unit contracts |
| Computation/idempotency identity | No create action → exact request/key lifecycle and read-only unknown recovery | `compatible-change`; actual 201/200/409, double click and accepted-response-loss proof |
| Private per-tab storage | No S2 record → one bounded allowlisted operation record | `compatible-change`; subject/logout clearing, storage failure, expiry/unknown scope and changed-org proof; no server persistence migration |
| Session/error seam | Query-observed 401 → immediate status-based stop for every protected request, plus body retry/admission details | `compatible-change`; final transport regression, final real preflight 401 and original session tests |
| Browser navigation / drafts | Existing URLs → data-router discard protection for unsent form | `compatible-change`; final real link/back/forward/SSR/locale checks |
| Backend domain, DTO producers, authz, SQL schemas, runtime limits and SSR feature default | No product source change | `none`; source/diff inspection; fixture data changes do not alter those contracts |
| Replay across original organization | Existing API still lacks the required binding precondition | `none` to the server contract; replay remains unsupported and the new client uses read-only recovery |
| Rollback | Existing disabled-by-default feature flag restores SSR | `none` to mechanism; real flag-off regression passes; no data migration required |

Existing SSR can read jobs created through existing APIs. Old clients ignore the
new namespaced session record; new readers validate/strip stored data. Disabling
`WEB_BACKTESTS_CLIENT_ENABLED` and restarting Web still restores SSR. No production
rollout or mixed-deployment guarantee is inferred from these local observations.

## Changed paths and preserved work

Created:

- `apps/platform-web/src/builder-api.ts`, `builder-model.ts`, `builder.tsx`,
  `builder-i18n.ts`, `builder.test.tsx`, `recovery.ts`.
- `apps/platform-web/src/test-data/builder-catalog.json` (reduced public fixture
  catalog/defaults; no account, credentials or generic provider response).
- `apps/platform-web/e2e/builder.spec.ts`, `e2e/run.mjs`.
- This report and the S3/browser-regression evidence listed above.

Modified from accepted S1/S2:

- `apps/platform-web/src/api.ts`, `api.test.ts`, `app.tsx`, `i18n.ts`, `main.tsx`,
  `library.tsx`, `library.test.tsx`, `style.css`.
- `apps/platform-web/package.json` — sequential isolated e2e entry only; no
  dependency or lockfile changes.
- `apps/platform-web/e2e/foundation.spec.ts`, `library.spec.ts` — optional S3
  evidence destinations only; existing assertions and original destinations remain.
- `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md`
  — S3 canonical integration section; existing foreign/S1/S2 content preserved.
- Generated `docs/architecture/project-map/PROJECT_MAP.md` and
  `docs/architecture/project-map/project-map.json`; all five generated artifacts
  pass the generator check. The existing component-map changes were preserved.

Deleted: none. Outside expected touch zones: none.
No S3 changes to API/domain/authz/Python product sources, fixture implementation,
Web server integration, shared package source, root package/lockfile, AGENTS,
ticket, functional contract/registry, plan, prompt pack or old ledger. Their
pre-existing dirty state was preserved. No staging or Git cleanup occurred.
Final cleanup confirmed that `.local_artifacts/backtests-client` was absent
and no Docker containers were running. Ignored build/test outputs remain local
transient artifacts. Accepted S1/S2
reports and browser evidence remain untouched.

## S4 handoff

The next prompt `04-execution-and-cancel.md` and its declared inputs were inspected.
The ticket, functional contract, immutable v23, package scripts and this producer
report exist. S4 can use actual returned job UUIDs with `library-api.ts` / `readJob`,
without replaying create. `api.ts` provides one-request transport, protected-401
stop and allowlisted retry/admission errors. `builder.tsx` owns only configure,
preflight and submit; execution/cancel remains in the detail boundary. The fixture
and runner are reproducible with the S1 setup; fresh independent fixture runs are
required when tests assume empty history or isolated admission counts.

All server authorization, original-org replay and unavailable workstation-projection
limitations remain. S4 was inspected, not executed. The next safe action is the
coordinator's independent acceptance of this S3 report and a separate sequential
S4 assignment. No next-stage authority was invented or persisted in a ledger.

## Coordinator acceptance

Accepted on 2026-09-08 after independent review of the builder/request/recovery,
session and error paths, regression tests, this report, real-create/lost-response/
fault/expiry observations and rendered RU desktop evidence. All coordinator
findings (dependent catalogs, imperative/broken-body 401, localized effective
policies and generation invalidation) were resolved. No blocking S3 finding
remains. Executor verification is reused; coordinator `git diff --check` passed.
Original-organization replay remains unsupported and safe recovery remains
read-only. This accepts only the local S3 boundary; S4–S6 and T1–T8 remain open.
