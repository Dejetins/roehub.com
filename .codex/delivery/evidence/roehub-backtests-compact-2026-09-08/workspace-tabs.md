# Persistent Backtests workspace tabs — 2026-09-08

## Authority and implementation

Owner explicitly accepted and requested the five-point tab workflow recorded in
the current Backtests ticket. This is local implementation, not publication.

- One shared Backtests heading and keyboard-operable History / Settings tabs.
- The accepted form fills the Settings workspace; no centered outer window.
- Both views retain their mounted state. History's route/search/selected job are
  frozen during configuration. Window scroll is remembered per tab; nested
  scroll remains on the retained DOM. Draft and valid preflight survive switches.
- Close returns to the saved History location; Reset form restores server defaults
  and invalidates preflight. Reset is disabled for pending/frozen submissions.
- Successful create refreshes active history reads, opens the new job with retained
  filters, and clears the submitted draft. The existing progress/result flow remains.
- A late result in hidden History cannot navigate away from Settings. Charts defer
  initialization until measurable and resize when History becomes visible.
- Hidden dirty drafts still guard document departure. Unsent drafts are memory-only;
  reload, sign-out or leaving Backtests discards them. Recovery payload/key semantics
  remain separate and are not replaced by draft persistence.

## Changed sources

`apps/platform-web/src/backtests-page.tsx` (new), `app.tsx`, `library.tsx`,
`builder.tsx`, `builder-i18n.ts`, `i18n.ts`, `style.css`, `execution.tsx`,
`results.tsx`, `builder.test.tsx`; current ticket and this receipt. Existing foreign
changes in the shared checkout are not part of this receipt.

## Compatibility

Baseline is the immediately preceding accepted compact form in this checkout.

| Boundary | Change | Classification |
| --- | --- | --- |
| Browser navigation and deep links | Existing list/new/job URLs remain supported; new route is a persistent workspace tab | compatible-change |
| Draft and close | Closing/switching preserves draft instead of discarding; explicit reset remains available | compatible-change; owner-selected semantics |
| History filters, selection, result variant | Saved during tab changes; hidden result defaults wait until History is active | compatible-change |
| Create/read lifecycle | Same one-shot command, key and DTO; successful create additionally refreshes active job lists | compatible-change |
| API, stored request, date boundaries, auth, recovery identity | Unchanged contracts and command gates | none |

No migration, backend configuration change, deployment or production target.
Rollback is a frontend rebuild; existing jobs/recovery records stay readable.

## Verification

- `pnpm --filter @roehub/platform-web test`: **152 passed** (147 existing + 5 new).
  New tests cover persistent draft/preflight, close/reset, keyboard tabs, saved
  filters/selection, retained DOM, browser back/forward, hidden draft departure
  guard, one in-flight create across a tab switch, fresh draft after success, and
  a late hidden result selection that must not steal the active tab.
- `pnpm --filter @roehub/platform-web typecheck`: passed.
- `pnpm --filter @roehub/platform-web build`: passed; existing ~1.11 MB JS chunk
  warning remains. jsdom reports its unimplemented `window.scrollTo`; actual
  browser scroll restoration is not proven by jsdom.
- Existing local service: `GET http://localhost:18480/health/live` returned 200.
  Rebuilt assets copied to the running server's cached manifest aliases without
  restarting the disposable database stack.

Browser mechanic: existing user-owned in-app Browser tab. Initially showed
`Your session has ended`. Opening the refreshed preview
`http://localhost:18480/backtests?asset_version=workspace-tabs-v1` redirected to
Sign in. The owner was asked to sign in; credentials were not read or entered.
**Authenticated browser verification is pending**: desktop/mobile visual review,
actual scroll restoration, live tab/back/forward navigation, real create/progress/
results and console/network checks are not claimed. No new authenticated browser
screenshots or repository Playwright runs were produced for this change.

Readiness: local code/build verified; browser acceptance remains blocked by the
expired selected session. This is not production or deployment evidence.

Documentation checks: docs index check, regenerated project map/check, and
`git diff --check` passed. Generated project-map inventory was refreshed for the
new workspace source and receipt; the accepted v23 specimen was not edited.

## Owner correction: bounded gray Settings panel

The owner rejected the unbounded transparent Settings composition. This correction
supersedes the earlier “fills the Settings workspace” styling: removed the four
Settings overrides that erased the panel surface/border/radius/header treatment
and maximum width. Settings now uses the same `.panel` gray gradient, border and
16px corners as History, with the accepted 90rem (1260px) maximum builder width,
anchored to the workspace's leading edge. Small input widths and tab state are
unchanged. Scope is `style.css` only; browser-visible compatibility is
`compatible-change`, API/state contracts `none`.

`pnpm --filter @roehub/platform-web build` passed; live preview aliases refreshed.
The selected Browser still lists the sign-in page; authenticated visual proof
remains unavailable. No unit tests rerun for this CSS-only restoration.

## Owner correction: platform workspace surface, compact inner form

The owner rejected the previous isolated capped card as inconsistent with History.
Inspected the current History implementation/styles, its saved EN 1440px browser
capture, and the accepted v23 panel/header rules. The mismatch was structural:
History's surface occupies the work area; Settings capped the entire surface and
left canvas around/below it.

Changed `builder.tsx` and `style.css`: the embedded builder now has one shared
`.panel` workspace surface using History's background, border, corners and desktop
minimum height. The 90rem limit applies only to the form inside it. Removed the
nested form/review cards in embedded mode; used the shared header padding and
button sizing, and shared input/background/border tokens. Accepted grid/short
input sizes, dates and state behavior remain intact. This supersedes the previous
90rem cap on the whole Settings panel. No change to History itself.

Typecheck, all 152 frontend tests and Vite build passed. Runtime asset aliases were
updated. In-app Browser still lists Sign in, so the saved History screenshot is
reference evidence only: there is no new authenticated screenshot or visual
acceptance claim for this correction. New preview query: `asset_version=platform-panel-v4`.

## Inline disclosure above Jobs (supersedes the tab layout)

Owner selected the first proposed integration and requested smoothly configurable
animation. `backtests-page.tsx` now places a single disclosure before the existing
History/Jobs panes. The current form appearance remains within the shared gray
panel, without the tab bar or full-height empty surface. Summary includes symbol,
timeframe, calendar period and check/submission status. New backtest opens it;
collapse/Escape preserves form and preflight, removes fields from focus and the
accessibility tree, and returns focus to its trigger. Jobs and selected results
stay visible; ordinary disclosure toggles leave job/filter URL unchanged.
`/backtests/new` still deep-links to the expanded form. Successful create collapses
the editor and opens its job; the next edit starts a fresh draft.

Motion uses interruptible CSS grid-row/opacity/chevron transitions, not timers or
a guessed max-height. Speed is exposed beside the disclosure: 0/180/320/520ms.
Only the allowlisted preference is stored in `roehub.backtests.motion`; drafts
are still memory-only. CSS `--builder-motion-easing` controls the easing curve.
System `prefers-reduced-motion: reduce` forces zero duration. Storage failure
falls back to a working per-visit choice.

Sources: `backtests-page.tsx`, `library.tsx` (New action opens locally), `builder.tsx`
(summary callback and embedded heading), `builder-i18n.ts`, `style.css`,
`builder.test.tsx`. Existing tab regression cases were updated to the approved
interaction; test workstation responses include both catalog and visible library
projections from their shared endpoint.

Compatibility: browser navigation/draft behavior `compatible-change` relative to
the owner-selected interactions and retained deep links. Animation preference is
an additive optional browser-storage `compatible-change`, invalid/unavailable
values fall back to Normal. Research/API/auth/recovery DTO and identity `none`.
No backend changes, migration, publication or production delivery.

Verification: 153 frontend tests passed; typecheck and Vite build passed (existing
large-bundle warning). Tests cover preserved draft/preflight, reset, always-visible
Jobs, inert collapsed fields, Escape/focus, filters/selection/history navigation,
one-shot submission while collapsed, result arrival, speed preference and rapid
state reversals. Browser still exposes the sign-in page; owner was asked to sign
in. Real animated intermediate frames, responsive visual comparison, reduced-motion
runtime and live create/results are pending authenticated Browser access, not
claimed by unit tests. Preview query: `asset_version=inline-disclosure-v1`.

## Disclosure inside Jobs (current placement)

Owner selected integrating the editor under the Jobs header. The page passes one
configuration slot to `BacktestsWorkspace`, rendered inside `.library` before
`.library-body`. The disclosure no longer has its own outer `.panel`; it uses
Jobs' existing gray surface with section separators. Its heading is level 3.
The context/results panel remains a sibling of Jobs and begins alongside it.
For this workspace the default 1.4:1 column split stays stable even when results
are present, instead of shrinking Jobs (and the editor) on result selection.
The existing responsive one-column breakpoint and form container queries remain.
No input widths, animation speeds, persistence, API or command changes.

Sources: `backtests-page.tsx`, `library.tsx`, `style.css`, `builder.test.tsx`.
Compatibility: browser layout `compatible-change`; draft/animation/API/recovery
contracts `none`. Existing deep links and selected-job navigation are retained.

Typecheck, 153 frontend tests and Vite build passed; existing bundle warning
remains. The regression asserts a single disclosure within Jobs after its header,
with the context panel as a sibling, alongside the existing draft/navigation/
submission tests. Runtime aliases updated; new URL query `asset_version=jobs-inline-v1`.
Actual in-app Browser navigation redirects to Sign in; authenticated responsive/
visual verification remains unavailable. No new visual acceptance claim or live
submission. The supplied owner screenshot is the placement baseline.

## Prefilled local synthetic preview

Added explicit localhost-only `preset=synthetic` initialization in the page and
`syntheticResearch` in the builder model. The fixed fixture-compatible three-day
EMA configuration is labelled synthetic and checked once against the server after
catalog/bounds load. This is not a successful-run promise: validation/preflight
errors keep submission disabled; edits require another explicit check. No auto
create or replay. Server-default policies and all ordinary initial values are
retained outside the opt-in preset.

Changes: backtests-page, builder, builder-model, builder-i18n, builder test. New
regression checks exact dates/indicator/timeframe, one auto-preflight, zero creates
and stale-check invalidation after edits. Typecheck/build and all154 tests passed.
Live assets refreshed; new query `preset=synthetic&asset_version=synthetic-preset-v1`.
No authenticated browser launch or result was performed in this amendment.
Compatibility: explicit additive preset/preflight behavior `compatible-change`;
API/identity/auth/recovery/normal defaults `none`. No backend or limit relaxation.


## Expanded synthetic preset — 2026-09-08

User requested more variants and trades while retaining the accepted Jobs layout.
Added `tools/qa/backtests_expand_demo.py`: deterministic oscillating/trending synthetic
OHLC, 43,200 one-minute candles covering 2026-02-27 through 2026-03-29 exclusive.
Production precompute, validation and guarded slot publication passed. Previous active
slot and PostgreSQL/ClickHouse databases were preserved. API and runner reloaded the
local-only `expanded-indicators.yaml` (EMA windows 5..50 step 5); shared test config unchanged.
The explicit localhost synthetic preset now requests those 10 windows and TP/SL 1..2%
step 1 (4 risk cells per indicator). Server chooses each indicator variant's best risk
cell. Normal server-derived UI defaults and accepted layout are unchanged.

Validation:
- `PYTHONPATH=src:. .venv/bin/python -m tools.qa.backtests_expand_demo`: passed.
- `.venv/bin/ruff check tools/qa/backtests_expand_demo.py`: passed.
- `pnpm --filter @roehub/platform-web typecheck`: passed.
- `pnpm --filter @roehub/platform-web test`: 154 passed.
- `pnpm --filter @roehub/platform-web build`: passed; bundle-size warning remains.
- Real authenticated local API: preflight 200, create 201, job succeeded, top 200 with
  10 variants. First variant paginated trades 200, total 182. Summary trade counts in
  `expanded-synthetic-proof.json`; other variants' tables not individually verified.
- Initial trades reads returned 202 while materializing; overly frequent proof polling
  hit 429. A subsequent read returned 200; no product rate-limit changes made.
- In-app Browser opened the new URL and reached login. Authenticated browser flow and
  console/network proof not available in that surface; API proof is not browser proof.

Compatibility assessment (`contract-impact-analysis`): API/schema/auth boundaries none;
local available EMA grid compatible-change (window 10 retained); explicitly opted-in
synthetic preset semantics intentionally changed. Synthetic price baseline replaced,
so new runs are not numerically comparable to the previous monotonically rising demo.
This is disposable local data, not Binance market history or production delivery.

Updated URL: http://localhost:18480/backtests/new?preset=synthetic&asset_version=synthetic-expanded-v2

Local runner follow-up: test runtime intentionally exits after 10 parent task units
(`reached parent max task accounting: 10`). A local supervisor now restarts the worker
on clean exit, preserving databases/config/history. Nonzero exit is not hidden.
Final Web/API health and runner metrics all returned HTTP 200.


## Report workspace and ECharts interaction — 2026-09-09

Accepted option 1 implemented: compact collapsible history, approximately 25/75
history/report desktop allocation; opening the existing inline editor temporarily
restores the wider Jobs column. Draft state and selected report retained.
Moved job identity/timestamps/coordinates and snapshot freshness to closed information
at the bottom. Active/failed execution retains progress/status handling.
Variants now have readable parameter links, numeric summaries and sortable columns.
Report includes Overview (equity + drawdown), Metrics, Trades, and existing monthly/
symbol statistics. Added server-provided averages/exposure/Sharpe-per-trade fields to
the client schema; no invented financial metrics or API changes.

Root cause of missing chart interaction: TooltipComponent was registered but no
`tooltip` option enabled it. Added axis-triggered rich-text tooltip (no HTML payload
rendering), crosshair, slider/inside zoom and connected chart groups per job/variant.
Accessible underlying chart tables remain available. Save/export placed below charts.

Changed source: library.tsx, execution.tsx, results.tsx, results-api.ts,
results-i18n.ts, style.css; report-ui.test.tsx and existing builder/execution tests.
Compatibility: API/persistence/auth none; additive client projection and browser
layout compatible-change. Existing cancellation/recovery/save/delete gates retained.

Validation:
- `pnpm --filter @roehub/platform-web typecheck`: passed.
- `pnpm --filter @roehub/platform-web test`: 156 passed. Two old presentation assertions
  were updated for the accepted header/completion layout; terminal state/race proof retained.
- `pnpm --filter @roehub/platform-web build`: passed (existing bundle-size warning).
- Authenticated in-app browser after user signed in: EMA 35 selection changed URL and
  report to rank 2 / 75 trades; Metrics table displayed additional server values;
  Trades showed paginated actual 182-trade dataset for EMA 5.
- Desktop 1525x968: history/report widths 342/1027 px; hiding history expands report;
  opening configuration changes widths to 799/571 px; collapsing restores analysis.
- At 390x844, document scrollWidth=390, no page-level horizontal overflow on Metrics.
- Chart slider drag changed visible time range; pointer tooltip and crosshair visible
  with timestamp and equity value. Evidence: `report-hover-zoom.png`.
- Browser warning/error logs empty during the exercised flow. Dedicated network trace
  not collected. No production publication or full financial-metric parity claim.

Preview: http://localhost:18480/backtests/64739d71-e1f6-4623-bb59-46f3b7a642af?asset_version=report-workspace-v6


## Local preview restoration — 2026-09-09

User requested service restoration and copying the existing password to clipboard.
Web/API/runner liveness was 200 but authenticated page failed 500: PostgreSQL
connection refused on port 65280. `docker ps -a` showed no containers; original
preview containers were missing. Recreated the two named local databases with
persistent named volumes and restart=unless-stopped, reapplied fixture migrations
and catalog, retained password and filesystem artifacts, restarted API/Web/runner.
No old database history was recovered. Real authenticated preflight/create and runner
recomputed job 465ecc9c-3a8c-4e2d-b1b5-98231d736463: succeeded, 10 variants.
Old job link is replaced, not redirected or falsely reconstructed with its old ID.
Browser verified normal sign-in page at new link. Existing password was copied using
CUA clipboard API, equality verified without logging its value; login proof-owner.


## Annotated report revision — 2026-09-10

Implemented the eight owner annotations in the existing platform report:
- Compact three-row variant comparison with previous/next controls; selection and sorting retained.
- Five highlighted KPI cards, four report tabs; removed single-symbol statistics from navigation.
- One ECharts canvas with Equity / Drawdown selection. UTC date-only axis, crosshair,
  tooltip and slider labels retain unique underlying intraday points.
- Optional green/red trade-exit markers use existing equity point trade_index and
  net_pnl_quote. Tooltip shows date, equity, trade index and net P&L. These are exit
  markers on equity, not invented entry markers on unavailable price candles.
- Drawdown uses red underwater area with a zero ceiling and a visible explanation:
  closed-trade balance peaks only, not open-position / intratrade equity drawdown.
- Monthly matrix has 12 month columns and year rows, percentage and quote P&L at
  one decimal; absent months are dashes. Percentage uses initial_cash_quote plus
  preceding monthly net P&L as the opening balance. The API's sum of per-trade
  return_pct is intentionally not presented as monthly account return. Missing
  initial capital yields an unavailable percentage, not an assumed balance.
- Desktop report flexes to available viewport height; Overview, Metrics, Trades and
  Monthly have no vertical report scrolling for the tested 1525x968 dataset. Trades
  use six-row pages; wide trade columns can scroll horizontally. Mobile uses normal
  document flow. Service actions remain in bottom disclosures; delete gate retained.

Owned paths for this revision: apps/platform-web/src/{results.tsx,results-api.ts,
results-i18n.ts,style.css,execution.tsx,report-ui.test.tsx}, plus this evidence.
Existing foreign workspace changes were preserved. No server/API mutation, database
schema change, publication or installation was performed.

Compatibility: server API, persistence, auth = none. Existing optional response-field
consumption = compatible-change. Browser contract = breaking-change intentionally
requested by owner (remove Symbol statistics; change monthly return semantics and
layout); other report controls remain available.

Validation:
- pnpm --filter @roehub/platform-web typecheck: passed.
- pnpm --filter @roehub/platform-web test: 159 tests passed (7 files). Covers monthly
  opening-balance calculation across years, unavailable baseline, year/month matrix,
  one-decimal output, removed symbol tab, date formatter, marker toggle and chart switch.
- pnpm --filter @roehub/platform-web build: passed; existing >500 kB bundle warning.
- Authenticated CUA browser, real synthetic job 465ecc9c-3a8c-4e2d-b1b5-98231d736463:
  page scrollHeight=968 at 1525x968; result panel clientHeight=scrollHeight=374 for
  Overview, Drawdown, Metrics and Trades. Monthly also fits. Trade pagination moved
  from page 1 to 2 of 182 trades; variant page 2 exposes ranks 4–6, EMA 45 opens rank 4.
- Monthly values observed: Feb 2026 18.8% / 1,877.8; Mar 349.0% / 41,453.7.
- Pointer tooltip observed: Mar 14, 2026, equity 19,098.36, trade #93, P&L 346.36.
  ECharts canvas, markers and crosshair visibly rendered; no timestamps in chart labels.
- Job information opens without page growth and retains the Delete history control.
- Mobile 390x844: document scrollWidth=390; viewport override reset afterwards.
- Browser error/warning logs empty. No dedicated network trace collected.
- Screenshots: report-v7-trade-hover.png, report-v7-drawdown.png, report-v7-monthly.png.

Limits: input dataset is synthetic. Intratrade drawdown is not available in this
series and has not been fabricated. For downsampled series the marker display states
returned/source counts. Full one-screen layout was verified at the owner's desktop
viewport; narrow/short screens retain scrollable accessible content.

Preview: http://localhost:18480/backtests/465ecc9c-3a8c-4e2d-b1b5-98231d736463?asset_version=report-workspace-v7-final

## Candlestick report and ten-row ranking — 2026-09-10

Owner correction: ranking pages now contain ten variants with an internally scrolling,
170px-high table, superseding the three-row pagination decision. Added a third Overview
chart mode, Price & trades, using Apache ECharts CandlestickChart and ScatterChart.
Blue triangles are actual entries and gold inverted triangles actual exits, placed at
server trade timestamps and execution prices. Prices come from the same job-pinned
artifact, not external live market data or an equity reconstruction.

Additive API contract: GET /backtests/jobs/{job_id}/variants/{variant_key}/candles,
max_bars defaults to 6000, valid 100–6000. Response includes job_id, variant_key,
timeframe, source_bars, group_size and candles[{time,open,high,low,close}]. Job ownership
and variant membership are checked before any artifact read. Period is [start,end).
Large datasets aggregate consecutive bars with first open / maximum high / minimum low /
last close; group_size is disclosed. Missing pinned artifacts fail with a typed error,
never fall back to current market prices. No authorization or persistence changes.

Client markers read the existing paginated trades endpoint in bounded 100-row pages,
up to 10,000 trades; truncation is explicitly shown. Existing pending/error/retry and
identity checks retained. Initial candle zoom shows the latest 150 bars, with slider
and Ctrl+wheel zoom. Marker toggle retains the current zoom. Date labels remain UTC
dates without time. Short screens use document flow rather than clipping the chart.

Changed paths: apps/api/routes/backtests.py; backtest application services
v2/{result_candles.py,lazy_trades_detail.py}; use_cases/backtest_jobs.py;
apps/platform-web/src/{price-chart.tsx,results.tsx,results-i18n.ts,style.css,report-ui.test.tsx};
focused tests test_result_candles.py and test_backtests_routes.py; this evidence.
Compatibility: additive API / browser behavior = compatible-change; persistence,
cache identity, authentication = none. Existing foreign changes preserved.

Validation:
- Backend targeted pytest: 84 passed, including access-before-read, invalid variant,
  max_bars validation, end exclusion and OHLC aggregation/extrema.
- Ruff on changed backend and tests: passed.
- Mypy --follow-imports=silent on four changed backend source files: passed.
- Frontend typecheck: passed; Vitest: 161 passed; production build passed with existing
  bundle-size warning.
- API process alone restarted on 18481; existing Web, DB, runner and history retained.
- Authenticated in-app Browser: job 465ecc9c-3a8c-4e2d-b1b5-98231d736463 returns
  2880 native 15m candles (group_size=1); 182 trades loaded for markers.
- At 1525x968 the document remains 968px tall. Ranking has 10 rows, clientHeight=168,
  scrollHeight=306; scrolling reaches scrollTop=138 while page scrollY remains zero.
- Visible price tooltip: Mar 28, 2026; OHLC; Entry #179, long, execution price 62,879.23.
  Evidence: report-v8-candles.png. Browser console checked separately below.
- Dataset remains synthetic, as in the preceding accepted preview.

Preview: http://localhost:18480/backtests/465ecc9c-3a8c-4e2d-b1b5-98231d736463?asset_version=report-workspace-v8-final
- Final v8 browser error/warning logs: empty. Viewport override reset; Price & trades
  left open. Final rendered chart height 266px at the current 1525x968 viewport.

## Real Binance candles and chart timeframes — 2026-09-10

Owner rejected synthetic prices and requested factual candles with timeframe switching.
Downloaded official Binance spot BTCUSDT 1m monthly archives for February/March 2026
from data.binance.vision, verified each SHA-256 against its published CHECKSUM, normalized
post-2025 microsecond timestamps, selected [2026-02-27,2026-03-29). Exactly 43,200 unique,
contiguous minute bars; no missing minutes. Provenance hashes/URLs: real-market-sources.json.
No price interpolation or invented gap filling was used.

Canonical archive OHLCV was passed through the production artifact precompute/validation
and local inactive-slot publisher. Existing synthetic job remains separate and pinned;
it was not relabelled or silently given market prices. Created and completed real-market
job 4299f9ba-c5ad-43e9-b237-59695cf84525, named Binance historical · BTCUSDT · EMA 5–50.
Ten variants. Best EMA 50: 109 trades, total_return_pct=-14.193098247051239. All shown
trade markers therefore belong to the newly recomputed market-data result.

Chart API now accepts optional timeframe: 1m,5m,15m,30m,1h,4h,1d. It reads the pinned 1m
artifact and aggregates OHLC on UTC interval boundaries. Added timeframe-aware client
cache keys and response validation. max_bars upper bound increased to 60,000; default
remains 6,000. Client requests 60,000 to show all 43,200 native 1m bars without implicit
coarsening. Chart timeframe is independent of the strategy's execution timeframe.

Also corrected a separate rendering cause of tiny candle bodies: dataZoom filterMode
was 'none', so off-screen prices influenced the Y range. Price chart now filters the
visible window; actual observed 15m Y range narrowed from roughly 63–78k to 65.5–68k,
showing readable candle bodies. No exchange price movements are smoothed away.

Validation:
- Targeted backend pytest (test_result_candles.py + test_backtests_routes.py): 85 passed.
- Ruff on affected backend, QA helper and candle tests: passed.
- Mypy --follow-imports=silent on four affected backend source files: passed.
- Frontend typecheck, 162 tests and build passed; existing bundle-size warning retained.
- Authenticated API comparison checks every returned OHLC against archive aggregation
  within 0.01 quote units (artifact float32 precision). Counts: 1m 43200; 5m 8640;
  15m 2880; 30m 1440; 1h 720; 4h 180; 1d 30. group_size=1 for every interval. Every
  timestamp delta equals the selected interval. See real-market-api-checks.json.
- Browser verified actual 15m, 1h and 1m selector changes, new candle counts, native
  candlesticks, execution markers and corrected Y scaling. No page growth at 1525x968.
- Evidence screenshots: real-market-1m.png, real-market-1h.png, real-market-15m.png.

Paths: previous candle API/service paths, results candle projection, price-chart.tsx,
results-i18n.ts, style.css, frontend/candle/API tests, tools/qa/backtests_expand_demo.py
(optional supplied canonical rows), local preview artifacts and this evidence.
Compatibility: optional timeframe / expanded input bound = compatible-change. Auth,
persistence schema, existing job identity unchanged. Local artifact pointer advanced
through the existing publisher; no remote publication or production deployment.

Official data format and checksums: https://github.com/binance/binance-public-data
Preview: http://localhost:18480/backtests/4299f9ba-c5ad-43e9-b237-59695cf84525?asset_version=real-market-v9-final


## Expandable Overview and shared motion — 2026-09-10

User-authorized scope: expand the existing Overview to fill the browser viewport while retaining Equity, Drawdown and Price & trades switching; generalize the existing animation speed preference across the client workspace.

Implementation:
- `apps/platform-web/src/expandable-overview.tsx`: reusable in-place expanded surface, mounted chart preservation, modal semantics, inert background siblings, focus containment/restoration, Escape and visible collapse control. Browser-viewport expansion, not operating-system fullscreen.
- `apps/platform-web/src/motion.tsx`: shared duration/easing policy, existing `roehub.backtests.motion` preference retained, interruptible View Transition coordinator, navigation/state adapters and native disclosure/dialog cancellation integration. No new dependencies. Off and OS reduced-motion bypass structural animation; unsupported browsers perform immediate structural updates. CSS disclosure and pane geometry transitions consume the same token.
- `backtests-page.tsx`: one speed selector in the workspace header, removed the builder-local duplicate/control policy. `app.tsx`, `library.tsx`, `execution.tsx`, `results.tsx`: integrate shared motion for client navigation, panel/history changes, report tabs, chart modes and dialogs. `results-i18n.ts` and `style.css`: localized expansion controls, responsive expanded chart and shared CSS motion.
- `builder.test.tsx`, `report-ui.test.tsx`, `motion.test.tsx`: preference/token regression, reduced-motion, interruption/fallback, disclosure content retention, same mounted chart and timeframe across expansion, chart switching, Escape, focus and inert cleanup.

Validation:
- `pnpm --filter @roehub/platform-web typecheck`: passed.
- `pnpm --filter @roehub/platform-web test`: 166 passed, 8 files.
- `pnpm --filter @roehub/platform-web build`: passed; pre-existing large-bundle warning remains.
- Browser: authenticated existing real Binance job 4299f9ba-c5ad-43e9-b237-59695cf84525. All three chart modes exercised in expanded mode, actual 1h candles rendered, Escape restores focus to expand trigger and retains 1h. Desktop 1525x968 and mobile 390x844 inspected; viewport restored. Tab from the last chart control wraps to Equity. Exactly one Animation selector observed; Slow sets shared token to 520ms, Off to 0ms; preference restored to Normal. History hide/show and Metrics/Overview exercised. No console errors/warnings observed in checked browser logs.
- Screenshots: `overview-fullscreen-desktop.png`, `overview-fullscreen-mobile.png`.
- Browser animation timeline instrumentation is unavailable through this browser DOM bridge; verification covers rendered states, real interactions, shared timing tokens and coordinator tests, not a measured frame-rate claim.

Compatibility: browser UI compatible-change; preference key retained; API, financial computations, authorization and persisted backtest data unchanged. Local preview only; no publication. Existing SSR routes outside the platform client are not converted to SPA transitions.

Preview: http://localhost:18480/backtests/4299f9ba-c5ad-43e9-b237-59695cf84525?asset_version=overview-motion-v10-final
Implementation reference consulted: https://developer.mozilla.org/en-US/docs/Web/API/Document/startViewTransition

## Local service recovery — 2026-09-11

Web/API/worker listeners remained active, but the two existing preview database containers were stopped. Started only `roehub-client-pg-2c9c288b` and `roehub-client-ch-2c9c288b` using `docker start`; retained their data. Password login changed from HTTP 500 to 200. Authenticated report page, job API and top variants API returned HTTP 200; the requested real-market job still has 10 variants. No source changes or remote publication.

## Motion polish and icon-only expansion — 2026-09-10

User correction: remove visible Full screen/Exit full screen text and improve result transitions. `expandable-overview.tsx` now renders the icon only, preserving localized title/aria-label. `motion.tsx` separates content and layout transitions under the same saved speed. Content updates are synchronous and use an interruptible local Web Animation (opacity and 3px settle) on the declared motion surface, scoped inside an expanded modal where present. This eliminates document-snapshot input delays observed during rapid chart changes. Geometry changes retain View Transitions, with an opaque surface, sequential chart bitmap fades, no bitmap stretching and no parent-report ghost image. `library.tsx`, `results.tsx`, `price-chart.tsx`, `style.css` integrate these policies; motion/report tests cover the changes.

Validation: typecheck passed; 167 frontend tests passed; build passed with existing bundle-size warning. Real in-app browser, current EMA20 variant: icon-only control, expansion, rapid Drawdown→Price switches, actual candles, Escape and Metrics→Overview verified. No console errors/warnings. Screenshot `motion-v11-fullscreen.png`. Browser 1280x720; no viewport override. CSS treatment was iterated after a captured transition revealed parent snapshot ghosting. No frame-rate claim. API/data/auth unchanged; UI compatible-change. Local preview only; no publication.

### Expansion button density correction

Changed only `apps/platform-web/src/style.css`: desktop expansion control 36→28px, minimum height 28px and radius 8→6px, matching chart switches. Existing mobile 40px touch control retained. Build passed (existing chunk-size warning). Browser computed styles at 1525x968 confirmed 28x28px/radius6; narrow viewport retained40px. Temporary viewport override reset. No behavior/API changes; no additional tests for this CSS-only correction.
