# Navigator design and browser QA

final result: passed

Scope: selected concept 1, amended by the user to remove the global header, use an icon rail, and enforce identical chart/table tracks on all three pages. This is local review evidence, not user visual acceptance or production readiness.

## Visual truth and comparison

Source directory: `/Users/daniildegtyarev/.codex/generated_images/01a0dd8a-6c08-7e71-9437-022105eb084a/`.
- Overview: `exec-12a54b1e-e83d-4f09-afb6-a2bd91090ba6.png`
- Backtests: `exec-0b9182af-45c2-4709-a0c1-98913ea109d8.png`
- Strategy: `exec-33f6e2b7-06c2-4540-9c8e-a8bcab3714c5.png`

Implementation evidence: `../../.codex/delivery/evidence/roehub-navigator-2026-09-26/`:
`overview-desktop.jpg`, `backtests-desktop.jpg`, `strategy-desktop.jpg`, respective mobile captures, `geometry.json`.

Source images 1504×1046; implementation desktop images 1476×959 at CSS viewport 1476×959 (one screenshot pixel per CSS pixel). Mobile 390×844. Native JPEG browser screenshots were used without image editing (file dimensions verified). Sources and implementations were viewed together in one six-image comparison input. This is a composition comparison, not a pixel difference: the user explicitly changed the shell and required uniform tracks where the generated concepts differed. Same product datasets were selected (Binance Live portfolio, historical EMA50 backtest, running MA20/50 strategy). Generated chart samples and some invented concept figures are not data authority; retained fixture values and domain semantics are authoritative.

## Geometry

All desktop pages: chart panel x302 y196 w912 h430, actual chart canvas container x315 y301 w886 h292; bottom panel x302 y636 w1162 h300. Shared plot grid: left72/right18/top18/bottom62. Shared tabs: 28px pill controls, same padding/gaps and neutral selected surface. Library224px, inspector240px, global rail56px. These measured rectangles verify alignment independently of screenshots.

## Iterations and findings

1. P1: inherited strategy library width overlapped the chart; P1: inherited job-information absolute positioning covered the workspace. Fixed candidate-specific selectors and put job inputs/actions into their own inspector DOM region.
2. P2: closed save/export disclosure and job state retained absolute baseline positioning. Fixed specificity; measured final disclosure x1241/y431 within inspector, state x1385/y60 within object heading.
3. P2: fullscreen inherited viewport width in addition to inset, overflowing right edge. Fixed width constraints; measured final x12..1464 in 1476 viewport, no page overflow. Inspector position details were compacted so lifecycle actions fit.
4. Post-fix desktop captures show all three panels aligned, no overlapping regions, and readable retained actions. Last Overview pass moved valuation/period explanations into the inspector and guarded zero-equity exposure display.

No remaining actionable P0/P1/P2 finding within the selected local variation scope. Focused control geometry was read directly from DOM; full-resolution source controls and final screenshot controls were compared in the paired inputs. No asset-detail crop was needed: product imagery is absent, and charts/icons remain native rendered components.

## Required fidelity surfaces

- Typography: retained platform font stack and compact scale; 12px shared tabs, 20px KPIs, existing 10–12px dense table hierarchy. Generated image typography is not substituted for the accepted application font. Long names wrap in libraries and truncate only in constrained object headings.
- Layout rhythm: one shared grid, equal plot and lower-pane areas; user-requested header removal/icon rail intentionally differs from source. Panels retain existing neutral cards and rounded borders. Mobile stacks the same regions with scrollable tables/libraries, not hidden functionality.
- Tokens: existing dark canvas/panel/line palette; violet chart accent, neutral KPI cards and selected pills, retained semantic green/red. No purple first-KPI highlight.
- Image quality: no rasterized UI, generated decorative assets or chart screenshots embedded in the product. Existing Lucide icons and live ECharts used; plots remain interactive.
- Content: preserved actual fixture labels/counts and existing guarded controls. Synthetic Overview is explicitly marked. Extra existing content is mapped into lower tabs/disclosures instead of being discarded to mimic invented mock copy.

## Browser verification

In-app browser, real local candidate on20120 and unchanged reference on20110.
- Historical EMA50/45 selection, 109/114 trades; lower Trades shows six rows while Equity stays mounted.
- Price chart interval1h loads720 candles, returns to Equity; ranking/metrics/trades/monthly sections retained.
- Strategy chart Equity and lower Events independently selected; unread count clears, five event rows visible. Fullscreen/Escape and inert outside content verified; stopped strategy retains unavailable state.
- New-backtest form opens with all Market/Signal/Trade/Risk/Ranking groups; submission not performed.
- Overview Binance selection, monthly returns, custom date controls and portfolio-editor dialog verified; no portfolio save needed.
- All three pages at390px: document scrollWidth390, no horizontal page overflow. Desktop viewport reset afterward.
- Candidate console error log empty in inspected session. Both `/login` HTTP200; both tabs marked to remain open.

## Commands and limits

From candidate directory: `node_modules/.bin/tsc --noEmit` passed; `node_modules/.bin/vite build` passed. Build retains the inherited large-bundle warning (~1.33MB uncompressed); splitting is outside this layout variation.
`node_modules/.bin/vitest run src/report-ui.test.tsx src/query-client.test.ts src/read-snapshot.test.tsx src/overview-model.test.ts src/strategy-operations.test.ts src/chart-period.test.ts`: 6files,32tests passed, including new chart/lower-table independence test. Fullscreen test now targets the shared table sibling for inert assertions. An initial old-selector failure was corrected; final run passed.

Commands share disposable API data between sites. Trading, deletion, save-strategy and submit were not confirmed; no mutation/provider proof claimed. No production release performed. Self-review mode: local cold review, compatible experimental UI change; no API/persistence change. User review of the separate variant is the next decision.

## 2026-09-27 — shared lower table refinement

User selected the current Backtests table as the alignment reference. Added an optional active-tab tools slot to `NavigatorTable`, outside the tablist, and moved Strategy direction/statistics and Events filters into that header. Contributions count and context now occupy the same header; its duplicate title row was removed. Shared CSS normalizes margins, typography, 44px rows, neutral P&L cell backgrounds and numeric alignment while retaining domain-specific columns and drilldowns.

Browser evidence at 1476×959: Backtests, Strategy Trades and Overview Contributions all have header y=637/h=49, column header y=686/h=32, first row y=718/h=44. Strategy direction menu opened and Events switched to its own tools. At 390×844 the header wraps within its panel and document scrollWidth equals viewport width (390). No captured browser console errors. Screenshots: `.codex/delivery/evidence/roehub-navigator-2026-09-26/strategy-table-sep27.jpg` and `overview-table-sep27.jpg`.

Validation: `node_modules/.bin/tsc --noEmit`, `node_modules/.bin/vitest run src/report-ui.test.tsx` (15 passed), `node_modules/.bin/vite build` passed. Existing large-bundle warning remains. Candidate 20120 restarted; baseline 20110 unchanged. API/persistence/cache compatibility impact: none. Browser-visible layout: compatible-change. Proof is local only.

## 2026-09-28 — fullscreen restoration

Removed the unsolicited job projection notice (and its unused request); search controls remain disabled as before. Fullscreen state now uses the synchronous transition path in both ExpandableOverview and Overview, avoiding the document View Transition snapshot morph competing with ECharts resize. No remount, data or chart-selection changes. Browser cycles verified for Backtests, Strategies and Overview: same ECharts instance ID and exact chart rectangle before/after (x302/y196/w912/h430); Backtests lower table remained x302/y636/w1162/h300. Tested Escape and collapse button. Screenshot: `fullscreen-fixed-sep28.jpg`. A temporal video trace was not captured; source confirms removal of the fullscreen snapshot animation, and regression test verifies startViewTransition is not called.

Typecheck/build passed; report-ui + motion: 19 tests passed. Focused library test updated for removed copy and the pre-existing limit=10 route. Existing build chunk warning unchanged. Candidate 20120 only; baseline untouched. Compatibility: API/cache/persistence none; presentation compatible-change.

## 2026-09-29 — default backtest selection

Bare `/backtests` now replaces its route with the first job in the fresh server-ordered history (created_at DESC, job_id DESC). Explicit job links and an active builder are preserved; no redirect occurs on errors or empty results. Selection does not use a retained snapshot from another filter. Initial reads show loading rather than the select-job placeholder. Changes: library.tsx and focused library tests. TypeScript and build passed; 2 focused tests passed (20 other library tests not executed). Browser navigation to `/backtests` opened historical job 54429a5c-c5e5-4f24-bb68-75df21f26d7c with populated chart and 109 trades. Evidence: `default-backtest-sep29.jpg`. Existing bundle-size warning unchanged. Candidate only; navigation compatible-change, API/persistence none.

## 2026-09-29 — default strategy selection

Bare `/strategies` now selects the most recently created available strategy matching library filters, replacing the list route. Uses created_at with ID tie-break rather than relying on list display order. Explicit IDs are preserved; pending/error/empty list never supplies an automatic target. Changes: strategies-page.tsx, strategies-api.ts and strategies.test.ts. Typecheck/build passed; all 12 strategy API tests passed, including unordered newest-selection and empty input. Browser opened strategy 877000c0-79f8-47f1-a826-4e221baba760 from `/strategies`; that newest fixture is stopped with no executions, accurately reflected in the panel. Screenshot `default-strategy-sep29.jpg`. Navigation compatible-change; API/persistence none; candidate only. Existing bundle warning unchanged.

## 2026-09-29 — strategy library runtime consistency

Confirmed cause: the opt-in execution demo overlaid only the selected runtime,
leaving its selector row `stopped` on every dashboard. The fixture now updates
its existing authorized selector row, selected summary and totals from the same
demo state. It never adds a strategy absent from the authorized response.
Navigator retains one fresh subject-scoped status snapshot during selection;
60-second expiry and restricted-access clearing remain. All library states show
the same dot as the detail header; running dots inherit green, stopped gray.

Validation: `pytest -q tests/unit/tools/qa/test_strategy_execution_demo.py`
(3 passed; two regression failures observed before fixing), focused Ruff passed;
Navigator `tsc --noEmit`, `vitest run src/strategies.test.ts src/read-snapshot.test.tsx`
(13 passed), and Vite build passed (existing chunk-size warning).
Browser: cold stopped detail, stopped → running → Back, including pending reads:
library consistently Running / Stopped / Stopped; selected headers agree.
No console errors observed. Inspected narrow default and 1476×959 desktop;
viewport override reset. Screenshots: `strategy-status-fixed-sep29.jpg` and
`strategy-status-fixed-desktop-sep29.jpg` in the Navigator evidence directory.
No trading commands exercised. Focused self-review completed.

Compatibility: local demo response semantics and library presentation are
`compatible-change`; DTO shapes, persistence, authentication and cache keys are
`none`. Production API unchanged; baseline client source unchanged. The shared
local API fixture corrects demo status data for both previews. API and candidate
preview restarted with retained data; no publication.

## 2026-09-29 — shared explicit date selection

Replaced Overview Custom's immediate hard-coded range/inspector inputs with the
shared `DateRangeControl`; adopted it for Backtests and Strategies chart periods.
The native top-layer popover opens beside its trigger, commits inclusive UTC
calendar days on Apply, and discards drafts on Cancel/Escape/light dismissal.
Labels show the applied date range; presets clear custom selection. Candle
interval stays separate. Backtest slider zoom no longer rewrites period state.
The strategy controls fit the existing desktop toolbar height; mobile reserves
room for wrapping. Baseline platform-web remains unchanged.

Semantics: Overview period calculations and historical tables follow the range;
current capital/positions/funds stay current. Backtest saved metrics and live
strategy accounting are unchanged: their picker is explicitly a chart-period
control, not a new server-backed period-report implementation. See README.
Compatibility: UI compatible-change; API/DTO/persistence/cache identities none.

Checks: Navigator `tsc --noEmit`; `vitest run src/date-range-control.test.tsx
src/chart-period.test.ts src/overview-model.test.ts` (13 passed); Vite build passed
with existing chunk-size warning. Focused self-review. Real browser: Overview
Sep 10–20 changed Bybit P&L from +5,226.04 to +87.24, return to +0.15%, while
current equity stayed 58,726.04 and available funds 7,500.00. Editing/Cancel and
Escape preserve committed values; All resets. Backtests Mar 10–20 restricted
the chart; strategy Sep 28 persisted across Price/Equity. Fullscreen popup and
390×844 viewport tested, popup inside viewport; override reset. No console
errors observed. Screenshot `date-range-overview-sep29.jpg` in Navigator evidence.
The original user tab's browser control timed out; verification used a new tab
in the same in-app browser/session, kept as deliverable. Native screen-reader
announcement and actual provider-backed period calculations were not tested.

### Compact date picker follow-up
Removed the visible UTC row (date arithmetic unchanged). Shared popup width is
300px, padding 10px; inputs/actions use 28px controls and smaller gaps.
TypeScript and Vite build passed; real Overview popup inspected, From/To and
Cancel/Apply visible without UTC. Screenshot: `date-range-compact.jpg`.

## 2026-09-30 — unified upper workspace

Candidate-only changes: Strategies filter icon matches Backtests, inline filter
panel retains existing filters; All/Running/Stopped use the existing state URL
parameter. Overview creation/edit moved to library header and Live/Paper before
the list. Shared title/control tokens and KPI rows documented in README.
TypeScript + Vite passed (existing bundle warning); 19 strategies/model tests
passed. Browser quick filters returned 1 running and 2 stopped, All restored all;
filter panel opened and Escape closed it. At the desktop viewport all three
libraries measured 48px headers, workspace headings 40px, heading font 13px;
all five KPI values on all three pages had y=145. Overview and Strategies checked
at 390×844, filter controls reachable, viewport reset. No console errors observed.
Screenshots: unified-strategies-header.jpg, unified-overview-header.jpg,
unified-backtests-header.jpg. Focused self-review; UI compatible-change,
API/persistence unchanged. No baseline client edits or publication.

## 2026-09-30 — single chart selector row

Removed the extra period row from price views. Strategies use five available
candle intervals (demo: 15m/30m/1h/4h/1d); Backtests use 1m/5m/15m/1h/1d.
Equity/Drawdown expose five period presets (1D/1W/1M/1Y/All); Overview retains
1D/1W/1M/YTD/All. Each row ends with the shared date picker. Display controls
stay on the right. Date-range state and candle aggregation remain independent.
TypeScript, six focused chart-period/date-picker tests and build passed.
Browser at 1476×959: every page panel x302 y196 w912 h430; chart x315 y301
w886 h292. Backtests price and equity both matched. Strategies 30m selection
and switching to Equity verified. Screenshot single-chart-selector-sep30.jpg.
UI compatible-change; no API/persistence changes. Baseline client untouched.

### Display options placement follow-up

Removed selector flex expansion and the display-menu auto margin. The settings
button now follows Date range with an 8px gap. Vite build passed (existing
chunk-size warning). Live Strategies at 1476×959: date control right edge
642.5625px, settings left edge 650.5625px, both y241 and 28px high. Open menu
measured x650.5625 y277 w113.03125 h138 and was fully visible. Screenshot:
`chart-settings-inline-sep30.jpg`. Candidate restarted; focused self-review.

## 2026-09-30 — unified chart chrome

Removed Backtests' inherited 4px report padding and expand-button margin;
matched the selector-row gap to the other pages. Overview expand control now
uses the full panel width. Shared button line height/weight, 50px desktop preset
slots and Date range spacing. Equity/Drawdown are first on every page; removed
percent suffix from the Backtests tab and the hover/zoom instruction paragraph.
Expand/collapse accessible labels now match Overview. Preserved chart semantics,
percentage axis formatting, source data and the accepted baseline client.

TypeScript and Vite passed (existing bundle warning). Report UI: 15 tests passed;
updated labels and an outdated assertion expecting the previously removed second
period row in price view. Focused self-review; UI compatible-change, API and
persistence none. In-app browser desktop 1476×959: all three pages have Equity
x315, Drawdown x374.3047, tabs/expand y209, selectors/date y241, Date range x585;
charts x315 y301 w886 h292. Backtests Drawdown and expand/collapse exercised.
All three checked at 390×844: controls fit and document width remains 390;
viewport reset. No captured console errors. Network/provider behavior untested.
Screenshot: chart-chrome-unified-sep30.jpg. Candidate restarted on 20120.


## 2026-09-30 — active default and filter popup

Cause: bare-route navigation selected only newest creation; status query was
previously disabled until a strategy ID existed. Read the existing authorized
library status projection through the newest strategy dashboard before selecting
newest running, falling back to newest creation. No temporary detail selection.
Explicit IDs stay unchanged. Display order now explicitly sorts creation time
newest first without mutating query data. Native popover uses Backtests menu
styles, remains above scrolling library, and supports nested filters.

TypeScript, 13 strategies tests and build passed (existing chunk-size warning).
Tests cover multiple running choices, stopped/unknown fallback, empty lists and
non-mutating date sorting. Real browser: bare route selected a26b24e6 (Running);
`/strategies?state=stopped` selected newest 877000c0. All-stopped global dataset
was covered by unit test, not by stopping the live demo. Menu opening preserved
row origin y159; nested states were fully visible; Escape and outside dismissal
worked. Desktop 1476×959 and mobile 390×844 verified, viewport reset. No captured
console errors. Screenshot strategy-default-filter-sep30.jpg.
Focused self-review: default selection/UI compatible-change; DTO, authorization,
persistence and commands none. Existing status endpoint/cache key structure and
freshness retained; bare-route adds a read before redirect. Baseline untouched.

### Filter containment follow-up
Popup geometry now follows the Library bounds with 12px horizontal/bottom insets,
including resize/scroll updates. Fields use one column. Nested options expand
inside the popup; constrained height scrolls internally without list reflow.
TypeScript and build passed (existing bundle warning). Browser desktop: Library
x68–292, popup x80–280; nested State options x93–267, fully visible. At 390×844,
Library bottom254 and popup bottom242, with internal scrolling. Screenshot:
strategy-filter-contained-sep30.jpg. Viewport reset. Focused visual self-review.

### 2026-09-30 — table scrolling, actions and filter consistency
Shared contained filter hook, identical Reset controls, single metric separators
and 13px labels. Removed inspector Refresh results and export disclosure; direct
Save strategy plus format/download controls preserve command eligibility.
Windowed Trades reads replace manual pagination, with full-panel table expansion
and tab switching. Failed page retry targets the failed query.

Validation: 87 Python route/cache/export tests; 28 report/strategy tests and one
windowed-scroll test (305 rows, forward/back scroll, variant isolation, bounded
DOM). TypeScript, Ruff and selected Pyright checks passed; production build passed
with the existing bundle-size warning. Final retry fix reran typecheck, scroll test
and build. Browser at 1476x959: table expanded, switched to Metrics, collapsed;
scroll reached trade108 of109. Compact view at scroll4144.5 rendered nine rows,
confirming old rows were removed. Actual browser downloads were parsed: CSV109
data rows; XLSX109 data rows. Backtests popup x80–280 remains inside Library
x68–292. Screenshots: backtests-polished-sep30.png, trades-expanded-sep30.png.

Focused self-review. Compatibility: additive export query options and UI behavior
are compatible-change; authentication, ownership and persistence unchanged.
Proof is local fixture/runtime only, not production or huge-dataset load testing.
Baseline frontend20110 was not changed.

### Export menu and save icon follow-up
Download now opens a compact CSV/XLSX choice; choosing a format downloads it.
Removed the standalone select. Save and Download share 28px icon controls.
Save tooltip and confirmation describe the Strategies destination and Start flow;
command eligibility/confirmation remains unchanged. Typecheck and 15 report UI
 tests passed; build passed with existing chunk warning. Browser: menu CSV choice
reported 118 trades downloaded, menu closed after selection, Escape dismissed it.
Screenshot export-menu-sep30.png. UI compatible-change; API/commands unchanged.

### Chart controls and plot alignment follow-up
Overview legend moved after chart/slider. Backtests Trade exits moved into period
controls directly after Date range; price markers use the same control row.
Date range labels no longer have ellipses. Plot boxes grew from292 to316px while
removing the extra24px row above them. Browser1476x959 measured all three pages:
x315,y277,width886,height316. Overview legend begins y593 below the slider.
Trade exits checkbox checked in browser and markers visibly appeared. Typecheck,
18 date-range/report tests and build passed (existing bundle warning). Final
price-toolbar follow-up passed typecheck/build. Screenshot chart-alignment-sep30.png.
UI compatible-change; API, command and data contracts unchanged. Local proof only.

### 2026-10-03 — accepted exchange-centered Data implementation

Result: passed for the accepted layout adapted to the user's explicit native
Roehub styling requirement. No remaining P0/P1/P2 visual finding in the inspected
states. The original platform baseline remains unchanged.

Source: `.codex/delivery/evidence/roehub-workpages-2026-10-03/data-approved-exchange-concept.png`
(selected third concept, amended to remove duplicate streaming controls and use
compact rows). Compared together with `data-exchanges-final-1487.jpg` at 1487×1058.
Retained: application rail, exchange library, large catalog, right inspector,
passive table stream state, one inspector toggle, adjacent From/To fields.
Intentional overrides: existing 40px heading, native system font/tokens, restrained
shared colors and Lucide icons, real catalog rows and states instead of the
concept's illustrative data/logos. Native row height is 38px; switch is 38×24px.

Real browser on localhost:20120 verified both exchanges, actual catalog refreshes,
search, cross-market rows, coverage sorting, page 2 (51–100 of 1,421), selection and
Back, streaming preference persistence/reversion, command recovery and download
progress/outcomes. 390×844 stacks the panels, keeps document width 390 and limits
horizontal table overflow to its 324px scroll surface; the inspector has no
inherited 420px cap and its whole form is reachable in page flow. Desktop document
width 1487 equals viewport width. Final-document console query had no errors or
warnings. This is not an all-session network trace or collector endurance proof.

Independent review found four behavior issues (cross-market batch retention,
command denial, stale period draft, failed job hidden by empty coverage). They
were fixed and covered by regression tests. Browser testing additionally found
native datetime input/state divergence and a restored requestkey race; input
handling, accepted-period commitment and one shared submission/recovery controller
fixed these. The initial browser request consequently loaded the previous one-day
range; it is recorded in the evidence log rather than represented as a five-minute
success. The final recent five-minute request completed with 5/5 candles. The
January 2024 Bybit request failed honestly; its provider/storage cause is unresolved.

Checks: 248 client tests passed, TypeScript and production build passed; existing
large-bundle warning remains. 12 focused API/repository tests, touched-backend Ruff
and Pyright passed. Screenshots: `data-exchanges-final-1487.jpg` and
`data-exchanges-final-390.jpg` in the evidence directory above. Runtime/fidelity
proof is local only. The broader private-provider Goal remains blocked.


### 2026-10-03 — Data download refinement

Implemented the user's icon action and whole-history defaults. Native download
buttons are 28×28px with accessible instrument/market labels; rows stay 38px. The
existing rail, library, catalog and inspector layout/colors/type are preserved.
Inspector From/To defaults now come from confirmed exchange candle bounds, separate
from bounded table coverage. Manual drafts survive refresh; unavailable discovery
allows manual input, and any denied batch read clears protected data.

Browser proof: 1414×959 document width matches viewport; 398px narrow view retains
stacked panels and usable dates. Bybit ZRXUSDT Spot starts at 2021-10-19 13:40 UTC;
Futures at 2022-03-28 10:51 UTC. Only three earliest Spot minutes were actually
loaded for QA: canonical coverage 3/3, worker succeeded. Multi-year downloads were
not run. Futures metadata had real intermittent timeouts before a later confirmed
bound; no arbitrary fallback was displayed. Final console query was empty.

252 client tests, typecheck and production build passed (existing bundle warning).
Backend tests, six real PostgreSQL tests and focused independent review are recorded
in `.codex/delivery/evidence/ROEHUB-WORKPAGES-2026-10-03.md`. Final screenshots:
`data-full-history-final-1414.jpg`, `data-full-history-398.jpg`,
`data-first-history-three-candles.jpg` in its sibling evidence directory.
No outstanding visual issue in these inspected states; provider endurance and
multi-year ingestion are outside the performed browser proof. Local candidate only.


### 2026-10-03 — Data filter icon alignment

Corrected only the Data catalog toolbar in `data-page.tsx` and
`data-workspace.css`. The search SVG previously started at the input's top edge:
its center was 7px above the 28px field center. The three selects used native
arrows positioned against their right edge. Labels now establish a block positioning
context; search and three decorative Lucide chevrons are 14px, centered vertically,
and inset 8px. Native select semantics, field heights and existing visual tokens remain.

Real browser at localhost:20120/data: all four vertical center offsets are 0px and
edge insets are 8px at both 1414×959 and 398×844. Narrow document width is 398px.
Search BTC plus Futures returned matching futures rows; reset restored defaults.
Typecheck, 18 existing Data tests and production build passed (existing bundle-size
warning). No new tests were added for this cosmetic change. Focused self-review;
no API/persistence compatibility impact. Local preview rebuilt and restarted.
Screenshots: `data-filter-icons-1414.jpg`, `data-filter-icons-detail.jpg` and
`data-filter-icons-398.jpg` in the workpages evidence directory. Temporary viewport
was reset. This entry verifies this toolbar change, not unrelated page functionality.


### 2026-10-03 — Downloads merged into Data Navigator

User-approved replacement: the standalone Downloads screen and duplicate creation
form are removed from the Navigator candidate. `data-downloads.tsx` provides an
optional compact journal below the still-mounted catalogue, with state/type
filters, source labels, status icons, bounded cursor history and a close control.
The instrument inspector now owns job progress, periods, cancellation, retry,
errors and expandable attempt details. Stored coverage and job progress are
labelled separately. Catalog refreshes and batch requests retain details too.
Connections and Monitoring links target Data; `/data/ingestion` only redirects
old bookmarks into the journal, preserving job/filter/recovery identities. A
legacy download range exceeding the coverage-read limit is not reused as a
catalogue coverage range; the requested job period remains visible in its details.

Touched client boundaries: `data-page.tsx`, `data-downloads.tsx`,
`data-workspace.css`, `app.tsx`, Connections/Monitoring links, query recovery budget,
Data and cache tests; removed `ingestion-page.tsx`. New journal reads occur only
while expanded. Commands lock during retained/error/pending states, use existing
server attempt fencing, and never retry automatically. Unknown action recovery is
subject/job scoped, survives inspector reopening for five minutes outside the read
budget, and is cleared with existing session cache cleanup. An older selected job
cannot enable a second download while the catalogue reports a current active job.
A local missing-job 404 preserves the instrument; denied reads/actions clear the
protected views through the existing access-loss path.

Validation: `pnpm --dir apps/navigator-web typecheck`, all 262 tests in 26 files,
production build, and `git diff --check` passed. Build retains the existing bundle
size warning. The 27 Data tests cover legacy redirect/filter preservation, lazy
journal reads and pagination, retained selection locks, cancellation confirmation,
retry attempt identity/double clicks, unknown-result reconciliation after reopening,
access denial, current-vs-historical job eligibility and catalog refresh selection.
Focused self-review found no outstanding issue in the changed scope. No independent
review threshold was crossed; no backend, authentication or persistence contract
was changed.

Real browser proof on localhost:20120: journal open/filter/select, failed-job reason
and retry availability, completed details/attempts, old-link redirect, refresh,
Enter/Escape and focus return, 1414×959 and 398×844 layouts. At both widths the
document matches the viewport; tables scroll within their panels. A new minimal
Bybit Spot ZRXUSDT request `dbe7e0d0-5f3b-4e12-9d1e-7a47b1381601` covered only
2021-10-19 13:42–13:43 UTC, succeeded with 1 candle read and 0 written because it
already existed. It finished before browser cancellation could be exercised.
Cancel/retry/unknown outcomes are verified by deterministic client tests, not by a
new live cancelled/retried job in this amendment. Existing failed jobs were not
mutated. Final current-document console query returned no errors or warnings;
no full network trace or multi-year ingestion test was performed.

Compatibility: API/DTO, backend side effects, auth and persistence `none`;
existing job bookmarks and cross-page links `compatible-change` via redirect;
standalone creation-form workflow `breaking-change`, intentionally retired by the
user and replaced by the existing instrument/batch controls. This assessment is
for the local Navigator candidate, not a default platform-client cutover or a
production delivery claim. Existing unrelated Goal blockers remain unchanged.

Screenshots under `roehub-workpages-2026-10-03/`: `data-journal-1414.jpg` and
`data-journal-398.jpg`. Preview rebuilt/restarted; temporary viewport reset.

### 2026-10-04 — Continuous download progress and ETA

The user authorized the four diagnosed corrections: one shared job-progress
source, quiet same-identity reads, a continuous public download state across worker
yields, and an estimate immediately after the progress bar. Implementation is
limited to the selected Navigator candidate. The ingestion worker, provider calls,
API/DTO, persistence and running user request were not changed by this amendment.

`download-progress.ts` shares one 2-second timer and QueryClient job key across the
catalogue cell, journal row and inspector. Only mounted jobs retain a subscription;
job payloads keep the existing QueryClient budget. Catalogue/inspector periodic
reads pause while the catalogue contains active work; journal history no longer
polls for progress. Commands, explicit refreshes and a terminal job transition
refresh the relevant derived snapshots. History-bound reads retain one-minute
freshness independently of background job reads. A first resolved terminal job
also reconciles a catalogue entry that still reported it as active.

The public state is Downloading for running jobs and for queued jobs with a
started_at or completed work. A genuinely unstarted job remains Queued; cancellation,
failed and terminal states remain explicit. Routine reads do not show Updating or
disable cancellation. Retained identities, failed reads, lost access, commands in
flight and uncertain command outcomes retain their existing safety locks. Inspector
status slots, table columns, percentage and ETA space have stable geometry.
Percentages are consistent across all three surfaces and do not round unfinished
work up to 100%.

ETA uses a rolling 120-second window of observed completed units and elapsed wall
clock, including queue/worker pauses. It warms up for at least 20 seconds, smooths
rate changes and recalculates at most every 10 seconds. No advancement for 60 seconds
shows No progress; read errors show Unavailable; neither invents a countdown. A new
attempt or reset of completed units clears the rate history. ETA is an estimate of
remaining work, not an ingestion-speed improvement or a completion guarantee.

Validation: typecheck and production build passed; all 268 tests in 27 files passed.
After the final completed-bar styling adjustment the 33 focused Data/ETA tests and
typecheck/build passed again. git diff --check passed (new untracked candidate files
are not included by Git's tracked diff check). The existing bundle-size warning
remains. New checks exercise twelve successive polls with row/journal/inspector
mounted, exactly one job request per tick, no repeat catalogue/coverage/bounds/history
reads despite a configured 15-second refresh preference, stable DOM/focus/scroll,
continuous state through running/queued yields, ETA agreement, pending-read command
eligibility, visible errors, explicit read recovery, and a single terminal snapshot
refresh. Rate tests cover warmup, pauses, smoothing, rolling-window adaptation,
stalls, read loss, retry reset and terminal percentage semantics.

Real browser proof: external Brave, existing Binance Spot ZROUSDT full-history
request 85e21d9a-ca42-42b8-91df-8fd1dd88d0cf. Its 1,202,849-minute job advanced from
62,280 to 69,180 and later 80,160 completed units while showing Downloading, matching
percentages/ETA in all three surfaces, and an enabled Cancel download control. No
cancel/retry/new ingestion command was issued. Observed estimates changed from
about 9 hours to 8 hours and later 6 h 45 min as the measured throughput changed.
Native screenshots confirm the desktop layout; the three saved 21:00:24–25 UTC
frames have the same 69,180 count and must not be described as three distinct
progress ticks. Stable nodes/focus/scroll and per-endpoint read counts are proved by
client tests, not a browser DOM/network trace. Mobile emulation was attempted but
was interrupted by native browser-control changes; the saved DevTools screenshot
is desktop, not mobile proof. A new mobile pass and console/network capture remain
unverified for this amendment. Device emulation was confirmed off and DevTools
closed afterward. Only the web preview on 20120 was restarted; ingestion continued.

Evidence: roehub-workpages-2026-10-03/data-progress-brave-0.png (plus -1/-2),
data-progress-brave-desktop.jpg, data-progress-brave-inspector.png, and
data-progress-brave-devtools-desktop.png. The inspector image is a crop of the
native desktop capture, not a separate rendering. Preview assets are
main-DyvD9IVB.css and main-DII_0tV8.js.

Compatibility: browser-visible progress behavior compatible-change; API/DTO,
authentication, persistence, backend side effects and configuration none. Focused
self-review found no remaining issue in these four changes. No independent-review
threshold was crossed. This does not establish a platform-client cutover, production
delivery, or completion of unrelated Goal blockers.


## Download pause/resume and recovery — 2026-10-04

Implemented in the selected Navigator candidate and existing Market Data worker:
checkpointed pause/resume, terminal cancel preserving candles, durable transient
source/storage backoff, lost-worker recovery, explicit wait states, reset ETA epoch,
and independent automatic browser GET reconnection. New controls follow existing
compact button/icon styles. Unchanged manual Retry semantics restart its work
counter; stored candles remain and the fill deduplicates their minute keys.

Verification:
- Full Navigator suite: 272 passed / 27 files. After the final storage-wait copy and
  test type corrections, affected suites: 37 passed / 2 files; typecheck/build pass.
  Build assets: main-BOJ1_pTp.css and main-vKoAnSB4.js; existing bundle-size warning.
- Python: `pytest -q tests/unit/contexts/market_data tests/unit/apps/api/test_market_data_workspace.py
  tests/unit/apps/api/test_market_data_catalog.py tests/unit/apps/migrations/test_storage_lifecycle.py`
  passed 208 tests. Final runner/HTTP/fill subset passed 19 (including one additional
  invalid-JSON regression). Focused Ruff passes; focused Pyright passed with 0 errors / 0 warnings.
- `tests/real_infra/market_data/test_work_requests_postgres.py`: 9 passed with an
  explicitly configured local PostgreSQL DSN, isolated disposable schemas. Covered
  pause during failing I/O, resume cooldown, concurrent controls, stale worker and
  command fencing, cancelled/paused priority, retry schedule and unchanged counters.
- One independent review found two P2s (truncated HTTP response classification,
  stale GET overwriting a command response). Both fixed and regression-tested;
  reviewer confirmed closure. Follow-up review of storage classification found no
  new blocker, noting ambiguous psycopg errors without SQLSTATE.

Browser proof used only Codex IAB, no external browser. Production built frontend,
production work-request router and repository served an isolated PostgreSQL job;
source was an explicitly synthetic bounded source with no candle writes. Verified:
pause at 2,460/20,160, reload keeps Paused and Resume; temporary source failure keeps
12.2% in retry_wait with due time; automatic recovery reached 4,140 and ETA; injected
browser GET503 disables commands and shows No connection at 4,800 while worker keeps
running; reconnect alone showed 6,480 and enabled controls, with no command replay;
Cancel became terminal at 9,120 with its progress preserved. IAB console-error query
returned no entries (not a claim that injected failed network reads did not occur).
Snapshots: data-pause-iab.png, data-retry-wait-iab.png,
data-browser-offline-iab.png, data-cancel-preserved-iab.png in
`.codex/delivery/evidence/roehub-workpages-2026-10-03/`.
This validates real UI/API/PostgreSQL controls, not external-provider fault recovery
or ClickHouse writes. Existing REST fill deduplication is covered by its unit tests.
Desktop 1280x720 checked; mobile/responsive proof remains outside this narrow pass.

Local runtime updated in order: cooperative requests-only shutdown, migration 0025,
new worker/API, new web bundle. Initial user's running job
85e21d9a-ca42-42b8-91df-8fd1dd88d0cf advanced from 172,920 to 173,400 at shutdown and
continued to 180,120/1,202,849 after restart. It then became failed with
source_or_storage_unavailable at 2026-10-03T21:46:43Z. Its state/progress was not reset,
paused, cancelled or retried by this task. A read-only next-window probe through
real fill with NoWrite returned 60 rows and found them already canonical, but this
does not reconstruct the failure: writer, lease-check and checkpoint were not
replayed. Cause remains undetermined; future errors now record phase+class only.
Do not present the user's current download as running or successfully recovered.
An initial API restart omitted its opt-in work-request flag; this was corrected,
the local restart helper now preserves it, and both command routes were verified
present afterward. This temporary read 404 did not stop the separate worker.

Compatibility: additive API fields/endpoints and current Navigator consumer are
compatible-change; migration preserves old rows. Expanded persisted states and the
new Store.defer port require coordinated worker/API upgrade (old-worker coexistence
and rollback are breaking-change). Browser default automatic GET recovery is an
accepted compatible-change with existing denial/unknown-command boundaries.
Org/origin/auth checks, candle identity, minute deduplication and trading behavior:
none. No platform-client cutover, publication, production delivery or completion of
unrelated Goal blockers is claimed. PostgreSQL connection setup errors lacking
SQLSTATE may be retried even when caused by misconfiguration; explicit auth/SQL
error codes and invalid instruments remain terminal. Bybit temporary error mapping
uses its official [error codes](https://bybit-exchange.github.io/docs/v5/error) and
[rate-limit headers](https://bybit-exchange.github.io/docs/v5/rate-limit).
