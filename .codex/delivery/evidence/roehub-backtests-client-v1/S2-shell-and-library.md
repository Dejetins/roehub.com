# S2 — v23 shell and API-backed Backtests library

Date: 2026-09-08. Execution: one executor, local checkout, S2 only.
Status: implementation and required local checks passed; independent final acceptance remains with the coordinator.

## Authority and scope

The user's sequential workflow supersedes staged-plan-runner, PACK-CLAIM,
claims, ledger transitions, updater and receipt requirements. The old ledger,
prompts and protected v23 are unchanged. No Goal, task, agent, branch, worktree,
stash, commit, push, merge, deployment, default cutover or backend/auth change
was created. No additional dependency was installed.

This stage proves real API-backed library and selected-job reads plus the
rendered v23-derived shell. It does not accept configure/submit/recovery,
execution controls, results, exports or the complete Backtests journey.
`local_journey_verified=false`; `target_role_cutover_ready=false`.

## Delivered behavior and implementation decisions

- `src/app.tsx` provides the narrow v23-style sidebar, header, locale links,
  skip link and private session boundary. `src/style.css` reuses the v23
  palette, rounded panels, compact controls and selected-row accent. The
  library/detail composition adapts to two stacked panels where space requires
  it; it does not reproduce the specimen's fictional results, service badges,
  account or permissions. No unused package/component catalogue was created.
- `src/library-api.ts` binds same-origin direct jobs/status and workstation
  reads with Zod allowlists. Artifact metadata, organization IDs, execution
  payloads and other unused response data are excluded from the library cache.
  The read DTO uses `request.risk_mode`, not the submission body's nested risk.
  Its `refresh_status=poll|terminal` is a polling hint, not a freshness verdict.
- `src/library.tsx` renders API-backed records, record labels, real state,
  created time, instrument, timeframe and selected job inputs. UTC is explicit.
  Status reads use the exact validated UUID independently of list membership.
  Unknown/invalid IDs show a clear error; invalid UUIDs are not requested.
- The URL preserves supported `state`, `risk_mode`, `limit` (1–250), opaque
  cursor, and safely encoded detail `variant`. Unknown/invalid query values
  are removed using replace navigation. Filter edits reset cursor; browser
  back/forward/refresh preserve valid filters, page and selection. A detail
  remains readable while the list's state filter excludes it. Variant identity
  is preserved for later stages; no variant result is claimed.
- Risk filtering is performed by the current server **within each page**.
  An empty risk-filtered page can therefore retain its next cursor. The UI
  explains that behavior and never treats a missing page entry as proof of a
  failed submission. Counts are explicitly page counts, not global totals.
- Loading, genuinely empty history, filtered empty page, unavailable, forbidden,
  absent/hidden object, validation, transport, invalid response, 429 and stale
  refresh states are distinct. Transient refresh failures retain last valid
  records and label them stale; 403/404 hide the corresponding cached display.
  A 60-second snapshot age is a client presentation threshold, not a backend
  SLA or inferred worker liveness. No generated progress or ETA is shown.
- Refresh is manual and respects both response deadlines/delays and HTTP
  Retry-After. Automatic stale-cache remount refresh is disabled on library,
  workstation and status reads. Query/mutation retry remains false.
- The existing workstation `backtest_jobs` projection remains actually
  unavailable. A disclosure explains the limitation and disables search,
  instrument and date inputs. The healthy direct list remains usable. No
  full-dataset frontend filtering or backend resolver/auth wiring repair occurs.
  If a future backend makes the projection available, these extended controls
  still require explicit binding work; this stage does not claim that future path.
- Session is rechecked every 30 seconds and when returning to the window.
  Any observed protected 401 stops the private UI and all further protected
  reads; subject change or identity failure also cancels/removes private queries.
  Late cancelled responses cannot restore old data. Sign-out immediately clears
  query state and uses the existing SSR logout. S1 pagehide/bfcache protection
  remains. No recovery/session storage is introduced in S2.
- Overview, Strategies and Settings retain real SSR destinations. The Data
  item is explicitly unavailable because the actual Web has no `/data` route.
  New configuration is disabled with an explanation; `/backtests/new` is a
  truthful route placeholder. Job execution/results/export controls are not
  represented as working actions.
- RU/EN covers the new controls/states. The filter dialog uses native modal
  inertness, Escape and explicit Tab/Shift+Tab cycling with trigger focus return.
  A scrollable table region has a unique accessible name. Route changes focus
  the reading heading; filter changes retain control focus. No motion was added.

## Validation and evidence

All commands ran from `/Users/daniildegtyarev/Projects/roehub.com`.

| Command / check | Actual result |
| --- | --- |
| `pnpm --filter @roehub/platform-web typecheck` | Passed |
| `pnpm --filter @roehub/platform-web test` | Passed: 41 tests, 3 files; 20 library tests plus the 21 accepted foundation tests |
| `pnpm --filter @roehub/platform-web build` | Passed; final JS `main-VVBAeDfM.js`, CSS `main-BmdGJi8Q.css`, Vite manifest |
| `pnpm --filter @roehub/platform-web test:e2e` | Passed: 5 real Chromium scenarios, 48.8 s including cold fixture setup/cleanup |
| `python3 -m tools.docs.generate_project_map` | Regenerated required map outputs after new source/test files |
| `python3 -m tools.docs.generate_project_map --check` | Passed |
| `python3 -m tools.docs.generate_docs_index --check` | Passed; no docs-index rewrite needed |
| `git diff --check` | Passed |
| `shasum -a 256 .../2026-08-03-linear-black-workbench-v23.html` | Preserved `3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4` |
| Cleanup observation | Private `.local_artifacts/backtests-client` absent; no Docker containers left running |

No Python product code or Python tests changed, so S1's 59 Python Web tests were
not rerun or newly claimed. Real Web/auth/API integration and feature-off SSR
were rerun in Chromium. No remote CI, provider, production or target-role proof
was attempted.

Browser evidence is under `browser/S2/` next to this report:

- `empty.png`, `projection-unavailable.png`;
- `en-820.png`, `en-1024.png`, `en-1440.png`, `ru-820.png`, `ru-1024.png`, `ru-1440.png`;
- `en-native-zoom-200.png`, `ru-native-zoom-200.png`,
  `en-native-zoom-200-dialog.png`, `ru-native-zoom-200-dialog.png`;
- `expired.png`, `observations.json`, `zoom-observations.json`, `expiry-observations.json`.

Screenshots were visually inspected against the actual rendered v23 reference.
The shared visual language is retained; result-panel fidelity is intentionally
outside this stage. There is no page-wide horizontal overflow at 820/1024/1440.
Native Chromium zoom is set by an ephemeral local test extension using
`chrome.tabs.setZoom(2)` and verified with `getZoom`: outer width 1440, inner
width 720, DPR 2, scroll width 720 in RU/EN. Native screenshots use CDP capture
without a clip because Playwright's full-page capture clipped the zoomed output.
The complete dialog fits horizontally and its actions are visible in the final
images. The temporary extension/profile are deleted after the test.

Axe reported zero violations for the library/detail at all six locale/width
combinations, for the native-zoom page and dialog, and for filter dialog keyboard
proof. Real keyboard checks cover opening via Enter, both focus-cycle boundaries,
Escape and focus return. No assistive-technology speech-output claim is made.

Normal library observations contain zero JavaScript/console errors and zero
HTTP errors; one `/api/backtests/jobs` `net::ERR_ABORTED` is recorded from an
obsolete navigation read, as intended. Other transport failures are asserted
absent. Real expiry records the jobs 401 separately, removal of private content
and **zero protected requests during the following 31 seconds**, exceeding the
identity refresh interval. Controlled unit tests additionally cover subject
change, in-flight cancellation, late responses, 403/404/422/503, 429 cooldown,
old snapshots, cache stripping, sanitized routes and empty-page cursors.

The two foundation scenarios rerun locale/auth entry, cache/assets, independent
SSR navigation, existing logout, identity outage, expired-session continuation
and flag-off rollback. Their new artifacts are in `browser/S2-foundation-regression/`
(`foundation-observations.json`, `foundation-820.png`, `foundation-1024.png`,
`foundation-1440.png`, `foundation-ru.png`, `ssr-rollback.png`), preserving the
accepted S1 evidence. Existing unrelated SSR projection failures and missing
Settings assets remain recorded separately as in S1; they are not claimed fixed.

## Real-data reproduction and boundaries

Use S1's existing Node/pnpm/Python/Docker prerequisites and fixture recipe. No
fixture implementation change is needed. The complete `test:e2e` script starts
fresh local Web :18480, API :18481, flag-off SSR :18482, runner :18483, disposable
PostgreSQL/ClickHouse and the production-computed artifact data from 4320
synthetic 1m candles. Password login uses the actual seeded local account; no
cookie fabrication, auth override or intercepted API success is used.

`e2e/library.spec.ts` first proves the empty list, then performs actual preflight
and two single POST creates as **disposable test setup**, with separate random
idempotency keys and explicit fixture-only labels. It records only safe job IDs,
creation status/state and observations, never credentials or raw response bodies.
The final run created jobs `1d17d7b4-d91a-4eff-be89-038abefed7ad` and
`3b818fed-8ee7-4169-b124-66fb2b42cfff`, both 201/queued at creation. Their actual
subsequent states were read; runner lifecycle/results acceptance remains S4/S5.
These jobs were removed with the disposable databases at fixture shutdown.

For focused reproduction, run the entire `library.spec.ts` file against a cold
fixture: its later zoom/expiry scenarios reuse its two jobs. Do not run a
seed-dependent individual test against an empty or unrelated reused fixture.
S3 tests that create additional jobs need their own fresh run/data isolation;
S2's cold-empty assertion should not be weakened to accommodate foreign test data.

## Debugging and review

Cold self-review covered route targets, read DTO semantics, private query lifetime,
refresh hints, filter/cursor identity, stale-data errors, keyboard and screenshots.
The coordinator independently identified the nonexistent `/data` destination;
it was changed to unavailable and included in the SSR navigation smoke. The
coordinator retains the requested independent changed-boundary/report review;
this report does not claim that final review has already accepted S2.

Observed pre-fix failures and narrow corrections:

1. Real jobs were rejected by the initial read schema. A diagnostic returned only
   Zod issue path/code/message: `request.risk` was missing. The production
   `_request_summary` confirmed the read field `risk_mode`; the adapter and tests
   now use it. Final real list/status checks pass without broad schema relaxation.
2. Self-review of `_refresh_status` confirmed `poll|terminal`, correcting the
   initial erroneous fresh/stale assumption; a fresh terminal-read regression
   assertion now fails under that old interpretation.
3. Native dialog Shift+Tab left the expected focus cycle; explicit boundary
   cycling fixes it. Axe found duplicate Jobs landmarks; the table now has a
   distinct label. Final keyboard/axe runs pass.
4. Foundation's new-route heading assertion needed level 1 after adding a
   context heading. Its assertions/evidence target the current S2 surface.
5. Native zoom capture initially clipped screenshots despite valid layout;
   unclipped CDP capture and dialog geometry checks corrected the evidence.
6. An expiry probe initially attempted manual refresh during the terminal job's
   30-second cooldown, then used an overly strict label locator. The final probe
   changes a real state filter through the combobox, observes the direct jobs
   401 and checks the following 31-second quiet interval. Product refresh limits
   were preserved.

## Compatibility assessment

Baseline: accepted S1 gated client and unchanged local server DTO/SSR behavior.
Candidate: S2 client, tests and narrow canonical documentation update.

| Surface / consumer | Before → after | Classification / evidence |
| --- | --- | --- |
| Opted-in Backtests browser routes | Foundation placeholders → list/read shell with same route identities | `compatible-change`; real entry/refresh/back/forward and encoded variant proof |
| Direct jobs/status and workstation DTO consumers | New allowlisted client reader; server unchanged | `compatible-change` for new reader against current server; actual API plus DTO source inspected |
| Filter/query behavior | No client library → supported sanitized state/risk/limit/cursor | `compatible-change`; actual server pagination including empty risk page |
| Session/private cache | Initial subject gate → periodic recheck, global 401 stop, cancellation/clear | `compatible-change`; unit subject/late-response tests and real 401 quiet interval |
| Existing unrelated Web/locale/logout destinations | SSR paths preserved; unavailable Data has no false link | `none` for server routes; existing destinations exercised in browser |
| Default feature config, server authorization, domain ports, persistence and replay identity | No source change | `none`; S2 writes are frontend/docs/evidence only |
| Rollback | Existing flag off/restart restores SSR | `none` to mechanism; real browser regression passed |

Rollback still sets `WEB_BACKTESTS_CLIENT_ENABLED=false` and restarts Web, with
no data migration. A build requires a Web restart because S1 loads the manifest
at startup. No production transition or deployment is implied.

## Changed paths and preserved foreign work

Created in S2:

- `apps/platform-web/src/library-api.ts`
- `apps/platform-web/src/library.tsx`
- `apps/platform-web/src/library.test.tsx`
- `apps/platform-web/e2e/library.spec.ts`
- This report and the exact browser evidence files listed above.

Modified from accepted S1:

- `apps/platform-web/src/app.tsx`, `src/i18n.ts`, `src/style.css`
- `apps/platform-web/e2e/foundation.spec.ts` (S2 evidence destination, level-1
  new-page heading, nonexistent-Data-link regression)
- `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md`
  (only the added S2 subsection; existing content preserved)
- Generated `docs/architecture/project-map/PROJECT_MAP.md` and `project-map.json`
  (new S2 paths/metadata; the generator reported these two stale outputs).

Deleted: none. Outside expected touch zones: none.
No S2 changes to package manifests/lockfile, `api.ts`, `query-client.ts`,
`main.tsx`, shared contracts, Web/Python fixture or backend sources.
Ignored build/test artifacts remain ordinary local outputs.

All preexisting AGENTS/ticket/plan/prompt-pack/ledger, Web integration, S1 client,
architecture-index/functional-contract, project-map/component-map and planning
changes were preserved. Accepted S1 report and original browser files remain
unchanged. No broad staging or cleanup occurred.

## S3 preflight / next safe action

The S3 prompt `03-configure-and-submit.md` and its declared producer inputs were
inspected. The ticket, functional contract, v23, package scripts and this S2
report are present. The same real fixture still supports catalog/defaults,
artifact date bounds, preflight and create, now exercised by S2 test setup.

S3 can integrate the builder into `BacktestsWorkspace`'s `mode="new"` context,
replace the disabled New control, reuse the same-origin transport/error seam,
locale resources, session cancellation boundary and route identity. Job/status
reads intentionally expose only library fields; create/recovery needs its own
allowlisted response/request contracts, not a widened generic cache.

The S1 backend workstation resolver dependency remains unresolved and read-only.
Do not pretend extended search/date/instrument filtering is available. Current
current-user still has no server-bound original organization replay precondition;
S3 must retain the ticket's unresolved/read-only recovery rule where replay cannot
be made safe. No new product-owner decision was needed for S2. S3 was inspected,
not executed. The next safe action is coordinator review/acceptance and its
separate sequential S3 assignment.

## Coordinator acceptance

Accepted on 2026-09-08 after independent reading of the changed client/session/list
code, API DTO semantics, tests, this report and browser observations. The coordinator
visually inspected RU desktop and corrected native-zoom dialog evidence. The missing
Data route finding was resolved; no remaining blocking S2 finding was identified.
Validation above is reused from the executor; coordinator `git diff --check` passed.
Workstation projection and unrelated SSR limitations remain explicit. This accepts
only S2's local list/read boundary, not T1–T8 or target-role cutover.
