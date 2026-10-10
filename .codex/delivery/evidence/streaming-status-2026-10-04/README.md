# Streaming intent, runtime health and switch correction — 2026-10-04

## Confirmed causes

1. The enabled switch selector was less specific than its base background rule.
   The aria-checked state and thumb position changed, but the gray background won.
2. ACTUSDT selection was saved, but no normal collector process was running.
   The preceding latency probe had stopped its owned test collector normally.
   UI `effective` represented intent and was incorrectly presented as Enabled.
   Missing candles were real; they were not a percentage arithmetic error.
3. General page refresh settings controlled stream reads, and manual fill remained
   available even while streaming was expected to maintain history.

Preflight read: only ACTUSDT selected, empty recovery journal, no collector process.
No user job, candle, selection or checkpoint was manually rewritten.

## Fix and runtime

- The checked switch now uses the same scoped selector specificity as its base
  rule plus aria-checked, restoring the existing `--violet` token (#9a78ff).
- Additive `stream_state` in catalog/collection distinguishes saved intent from
  collector lease ownership and durable repair states: disabled, unknown, offline,
  running, recovering, retry_wait, blocked. A running lease is explicitly not
  proof of an open provider socket; coverage/last candle remain separate evidence.
- The read-only PostgreSQL status projection returns sanitized receipt state and
  retry times for the requested instrument; no token or task key is exposed.
- Selected stream views poll every two seconds. Pending history is shown as
  updating without fabricating 100%; exact counts remain available.
- Manual fill and catalog download selection are hidden/disabled while streaming
  owns maintenance. Explicitly blocked history repair remains actionable.
- Started the normal collector from persisted selections, not the fixed test
  reader. It repaired the missing tail and established the ACT WebSocket.
  Local `start_market_data_ws.py` helper records PID; preview restart invokes the
  helper and preserves an existing collector. Collector PID 64660 is left running.
  No OS service installation or production deployment.
- Redis is unavailable in this local preview; its existing test configuration
  boundary remains disabled for the local collector. This proves canonical
  ClickHouse collection, not Redis/strategy delivery. No system resources changed.

## Verification

- Focused API/wiring tests: 10 passed.
- Full Navigator suite: 28 files, 291 tests passed.
- TypeScript typecheck and candidate build passed; existing large-chunk warning.
- Focused Ruff and production Pyright passed; git diff --check passed.
- Real PostgreSQL isolated-schema test
  `test_collector_status_contains_only_safe_repair_fields`: 1 passed, 5 deselected.
  Other lease ownership tests were deliberately not run beside the live user
  collector. Earlier offline state and current real live lease were inspected.
- Browser IAB: aria-checked=true, computed background rgb(154,120,255), equal to
  --violet; no Fill missing history button. Selected row says Collecting.
- Without reloading, coverage advanced from 997045/997045 to 997046/997046, both
  100%, and last candle advanced from 19:24 to 19:25 UTC. Final screenshot shows
  997047/997047 through 19:27 UTC, zero missing, last open 19:26 UTC.
- Runtime snapshot 19:27:41 UTC: ws_connected=1, insert_rows_total=4, WS/INSERT/REST
  error counters zero, pending recovery empty, ACT selection preserved.
- Browser error log was empty in the inspected interval. Screenshot visually
  inspected. Collector remained running across final API/web restarts.

Commands: `.venv/bin/pytest -q tests/unit/apps/api/test_market_data_catalog.py
 tests/unit/apps/api/test_market_data_reference_wiring_module.py`; Navigator
 `pnpm --dir apps/navigator-web typecheck`, `test`, `build`; opt-in real test through
 existing credentials in process memory. No credentials or raw provider data saved.

Compatibility: additive status fields, no schema migration; Navigator presentation
and polling change; accepted platform client untouched. Review mode: focused
self-review. Offline/retry/blocked rendering covered by unit tests; no deliberate
live collector outage was introduced. Current live proof is ACTUSDT across several
closes, not a long-duration service availability guarantee.

## Compact Data controls follow-up

Replaced the CSS track/thumb with installed Lucide ToggleLeft/ToggleRight icons;
checked state uses the existing violet token. Reused `work-icon` (28px button,
16px icon), `library-pagination`, and `navigator-page-arrows` for journal actions
and catalog pagination. Arrow buttons now sit together without an intervening
page number. Existing command eligibility, ARIA state and handlers are unchanged.

Validation: Navigator typecheck, 42 data-page Vitest tests and production build
passed. Build still reports its existing >500 kB chunk advisory. Focused self-review
and scoped `git diff --check` passed. IAB at native 1414x959: measured all four
buttons at 28x28 with 16x16 icons; paged 1–50 -> 51–100 -> 1–50 of 514; refreshed,
closed and reopened journal; inspected checked ACTUSDT and disabled/unchecked
ADAUSDT switches. Browser error log empty. Screenshots: `controls-on.png` and
`controls-off.png`. Returned to original ACTUSDT URL. Collector PID 64660 preserved;
no streaming commands issued during visual QA. Compatibility: compatible-change
for Navigator presentation; API/persistence/worker impact none. No publication.

## iOS-style switch and explicit log visibility labels

Supersedes the outlined Lucide switch from the preceding follow-up. User selected
an iOS-style filled track with an inset white circular thumb. Track 40x24px, thumb
20px, uniform 2px inset, 16px travel; enabled violet and disabled/off graphite.
Shared motion duration/reduced-motion tokens and explicit keyboard focus outline
are retained. Journal header trigger now says Show log / Hide log (Russian:
Показать журнал / Скрыть журнал), using the same button shape and handlers.

Checks: typecheck, 42 data-page tests (including updated journal visibility label
assertions), build and scoped diff check passed. Focused self-review. IAB native
1414x959 verified opening/closing and corresponding labels, checked ACTUSDT switch
geometry and unchecked ADAUSDT switch geometry; no stream command issued. Original
ACTUSDT URL with hidden journal restored. Evidence: ios-switch-log-open.png,
ios-switch-off.png, ios-switch-log-hidden.png. Compatible presentation change;
API/storage/worker impact none. Existing collector preserved by preview restart.

## Quiet background refresh

The catalog header Refresh button was driven by query.isFetching, so every
streaming poll changed disabled/opacity. It now uses manualRefreshing only.
Manual refresh waits for catalog and inspector reads, including replacement of a
paginated snapshot, and clears pending in finally. Background polling remains
unchanged. No shared RefreshButton component or other clients changed.

Validation: typecheck, all 44 data-page tests, build and scoped diff check passed.
New regression cases hold background/manual GETs and verify steady background
button state, pending manual state, success/error recovery and snapshot reset.
IAB: 40 DOM samples over approximately 8 seconds all showed enabled and opacity=1;
manual click completed and button was enabled afterward. No browser errors.
Screenshot refresh-steady.png records final state, not temporal proof of no flicker.
Focused self-review; compatible Navigator presentation change; no API, schema or
worker changes. Preview updated, existing collector preserved. No publication.

## Exchange logos and aligned history actions

Replaced sidebar monograms with locally bundled official Binance website icon and
Bybit press-header SVG. Asset provenance is recorded in
apps/navigator-web/src/assets/exchanges/README.md. The SVG was inspected: static
paths only, no scripts or external references. Empty image alt text avoids
repeating the adjacent exchange label.

History coverage and job rows now share a fixed 88px action rail with flexible
status text; the history cell has a 290px minimum width. Download controls no
longer move horizontally with label length or coverage bar presence. Existing
command handlers, eligibility and row selection guards are unchanged.

Validation: typecheck, 44 data-page tests, build and scoped diff check passed.
Focused self-review. IAB at 1414x959: both images decoded successfully; all 50
first-page download buttons had x=853.6328125 across unknown/partial history.
Verified 0GUSDT Spot download action opens its inspector and Bybit navigation
selects the correct exchange. No download was submitted. Returned to /data.
No browser error logs. Screenshot: exchange-logos-aligned-actions.png. Live proof
covers unknown/partial rows, not a newly created active job. Shared layout rules
also apply to active/failed job actions; existing behavior tests remain green.
Compatibility: compatible presentation change, no API/schema/worker changes.

Bybit wordmark follow-up: removed adjacent duplicate text, displayed the single
wordmark at 52px width, and supplied the link's accessible name through aria-label.
Typecheck, 44 data-page tests, build and scoped diff check passed. IAB confirmed
one loaded image, no visible text duplicate, accessible name Bybit; inspected
bybit-single-logo.png. Focused self-review; presentation-only compatible change.

## Platform catalog filter menus

Replaced the three native selects (Market, History, Stream) with the existing
ChartDisplayMenu used by Strategies. A local DataFilter adapter maps URL keys to
its single-choice options; shared component unchanged. Scoped CSS retains 28px
catalog controls and platform popup surfaces. Journal selects are outside this
request. Existing filter/reset/pagination URL semantics are preserved.

Typecheck, 45 data-page tests, build and scoped diff check passed. Regression
coverage exercises keyboard radio selection, URL values, Escape focus return,
mouse selection and outside dismissal. IAB at 1414x959 verified Market -> Spot,
History Enter/ArrowDown -> Downloaded/Escape, Stream -> Enabled, Reset and outside
click dismissal. Screenshot custom-catalog-filters.png inspected. No console
errors. Default /data filters restored. Focused self-review; compatible UI change,
no API/schema/worker impact. Screen-reader runtime not tested; semantic roles and
keyboard behavior checked. No publication; collector preserved during web restart.

## Immediate Pause/Resume controls (2026-10-05)

Shared row/inspector command state now exposes the opposite control immediately
on user intent. One request stays in flight; subsequent explicit pause/resume
choices coalesce to the latest intent and use the acknowledged control_version.
Unknown command outcomes stop the chain and require reconciliation. Dependent
GET invalidations no longer hold the command fence.

API can_resume includes pause_requested. Repository CAS reverses that state to
running without releasing worker_token or resetting progress; paused resume
retains the existing queue/cooldown behavior. The reversal atomically records a
bounded resumed event. Existing worker fencing, organization and origin checks
remain unchanged. No schema migration or worker restart.

Validation: 46 data-page tests, 24 focused API/runner unit tests, 21 real PostgreSQL
tests passed. One unrelated ClickHouse fault test skipped because its explicit
disposable target was not configured. Typecheck, Vite build, focused Ruff and
scoped diff check passed. Build retains the large-chunk warning. Held-response UI
test proves immediate controls and coalescing across both surfaces; real PG tests
exercise progress/token preservation, stale and foreign commands, worker release
and cancellation. Independent review by review_immediate_resume found the missing
resumed event; fixed and re-reviewed with no further findings.

Local API/web restarted (79246/79250), collector 64660 preserved. IAB smoke:
existing user job 17946439-61bf-4027-8005-3b357dbfce62 remains paused at 20,160 /
543,531, Resume enabled, no console errors. No user job command was sent.
Screenshot immediate-resume.png. Browser smoke does not itself prove the timing
race; that proof comes from controlled UI and real PostgreSQL tests above.
Compatibility: compatible-change to allowed resume transition and UI behavior;
API and client updated together locally. No production/publication claim.

## Neutral catalog selection (2026-10-05)

Removed persistent selected-row fill in the instrument catalog at user request.
Hover and keyboard focus styling, current-link semantics and inspector selection
remain unchanged. Scoped one-line CSS change; compatible presentation change.
Build and scoped diff check passed (existing chunk-size warning). IAB verified
selected 0GUSDT Futures row background transparent and box-shadow none after
reload. Screenshot: catalog-selection-neutral.png. Focused self-review; no job
commands sent, collector preserved. No backend or platform-web changes.

## Correction: whole-row instrument selection (2026-10-05)

The neutral-selection change above misinterpreted the request and is superseded.
Restored selected-row fill. Replaced the separate symbol link with passive text;
the whole row owns selection, keyboard focus and Enter/Space activation. Nested
controls retain their actions, and retained/unauthorized rows cannot activate.
Current instrument semantics move to the row. Symbol no longer offers independent
link navigation, intentionally matching the requested row interaction.

Typecheck, 47 data-page tests, build and scoped diff check passed. IAB verified
1000000MOGUSDT symbol click selects the full row and updates inspector; Futures
cell click selects 1000000BOBUSDT; Enter selects MOG again. Restored BOB via mouse:
row fill rgb(25,31,37), no mouse focus outline, symbol is SPAN and focus target TR.
Screenshot whole-row-selection.png. Focused self-review; compatible interaction
change, no backend/API changes. No download commands. Collector preserved.

## One catalog history indicator (2026-10-05)

HistoryCell now renders one coverage percentage/bar sourced exclusively from
full-history candle coverage. Current task status and shared controls occupy the
same cell/rail for active and failed jobs. Task progress and ETA stay in inspector
and journal; neither is substituted for full-history coverage in the catalog.
Uses fresh shared job state with catalog summary fallback. Unknown bounds remain
explicit, and terminal task completion cannot manufacture 100% history coverage.

Typecheck, 47 data-page tests, build and scoped diff check passed; build retains
its existing chunk warning. Updated regression uses differing coverage/task
percentages, asserts one catalog bar and independent inspector task progress.
IAB observed real running 0GUSDT: Downloading + Full history 37.88%, exactly one
progress element, Pause/Cancel enabled. Screenshot single-history-progress.png.
No job commands sent. Focused self-review; compatible UI change only. Collector
preserved during local web restart; no API, storage or platform-web changes.

## Coverage meter alignment and restored ETA (2026-10-05)

Grouped full-history percentage and its bar into one non-wrapping inline flex
unit. Restored the shared task ETA below it, with a task-completion tooltip;
there is still only one percentage/bar in the catalog. ETA does not estimate a
custom task's unrequested historical range. Typecheck, all 47 page tests, build
and scoped diff check passed. IAB verified text/bar share center y=342 and ETA is
present (Estimating after page reload); numerical ETA remains covered by the
shared polling test. Screenshot history-meter-eta.png. Focused self-review,
compatible presentation change. No worker or API edits, no job commands.

## Inline ETA, compact stream and no horizontal overflow (2026-10-05)

Expanded history column, reduced Stream to 86px with On/Off values; full collector
state remains in tooltip/accessibility name and inspector. Percentage, coverage
bar and ETA are one non-wrapping flex unit. Removed fixed table/history minimum
widths. Found the immediate overflow source: Stream header exceeded its 72px
column; corrected width accommodates header without clipping or hiding overflow.

Typecheck and 47 page tests passed for markup/stream changes; final CSS build and
scoped diff check passed. IAB at 1414x959: catalog clientWidth=scrollWidth=788,
history column 423.73px, Stream 86px; percent/bar/ETA centers all y=342. Off visible;
ETA estimating after reload. Screenshot history-inline-eta-no-overflow.png.
Focused self-review; compatible UI-only change. No job commands or API changes.
