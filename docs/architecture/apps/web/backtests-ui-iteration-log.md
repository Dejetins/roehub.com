# Backtests UI iteration log

## Acceptance — 2026-09-11

The product owner accepted the current result as satisfactory and authorized
publication of the complete Backtests implementation through a technical branch
merged into `main`, followed by deletion of that branch.

Accepted baseline:
- Compact date-only configuration embedded as a disclosure inside Jobs; draft
  preservation, separate reset and retained history context.
- Result-focused workspace with compact clickable variant ranking, summary
  cards, metrics, trades and a year-by-month P&L matrix; technical job metadata
  is collapsed below the report. No redundant symbol-statistics tab.
- Apache ECharts equity, drawdown and candlestick price/trade views. Price bars
  come from the job-pinned artifacts, with UTC 1m/5m/15m/30m/1h/4h/1d rollups.
  The verified local example uses actual Binance archives; earlier synthetic
  examples remain explicitly identified as synthetic.
- Overview can expand to the browser viewport while retaining chart switching,
  zoom and timeframe. Escape and the icon control collapse it. Desktop control
  is 28px, aligned with chart switches; narrow-screen touch size is retained.
- One persisted animation-speed preference. Local content transitions update
  immediately; geometry transitions have an opaque surface and no parent ghost
  image. Off and system reduced-motion remain supported.

This acceptance freezes the current UI iteration, not every future Backtests
requirement. `local_journey_verified=true`; `target_role_cutover_ready=false`.
Target organization/role authorization and default cutover remain separate.
No production runtime is selected by the repository adapter; Git publication
must not be described as production deployment.

## Platform visual baseline — 2026-09-12

The product owner selected this refined Backtests implementation as the visual
baseline for subsequent local-platform UI, replacing the old v23 pilot as the
active design reference. The accepted code is pinned by merge commit
`191a9f8169dab0639d8fb3456eb73b060eb1d2c4` (PR #33). The
[implementation plan, D4](roehub-ui-implementation-plan-v1.md#d4-the-accepted-backtests-implementation-is-the-visual-authority)
records reuse of the current surfaces, control density, hierarchy and shared motion.
The historical pilot and prior evidence remain unchanged. This decision selects
visual authority and further planning; it does not implement the next UI area.

## Iteration history

| Period | Accepted change | Evidence |
| --- | --- | --- |
| 2026-09-08 | Complete local configure/preflight/submit/execution/results journey | [S6 completion](../../../../.codex/delivery/evidence/ROEHUB-BACKTESTS-CLIENT-2026-09-08.md) |
| 2026-09-08–10 | Compact editor, date-only fields, Jobs disclosure and result workspace | [Iteration evidence](../../../../.codex/delivery/evidence/roehub-backtests-compact-2026-09-08/workspace-tabs.md) |
| 2026-09-10 | Actual Binance candles, artifact-pinned rollups and interactive charts | Same evidence, real-market section |
| 2026-09-10 | Expandable Overview, shared motion, icon-only control and density correction | Same evidence, motion sections |
| 2026-09-11 | Product-owner acceptance and authorized merge publication | This entry; [publication record](../../../../.codex/delivery/evidence/BACKTESTS-UI-PUBLICATION-2026-09-11.md) |
| 2026-09-12 | Refined Backtests promoted to the platform visual baseline; next Strategies library/detail iteration planned | Platform visual baseline entry above; [current plan](roehub-ui-implementation-plan-v1.md) |

The dates describe recorded local iterations; individual evidence entries are
historical observations, not claims that every later revision was rechecked by
the same run. The publication record identifies the checks for the shipped code.

## Local correction — 2026-09-24

The owner requested removal of the Backtests workspace title strip and animation
selector. The page now retains only a visually hidden h1 for the skip target.
Shared motion defaults to Normal (320ms), ignores the legacy stored speed, and
continues to honor system reduced motion. This supersedes the selectable-speed
part of the earlier visual acceptance.

Changed `backtests-page.tsx`, `motion.tsx`, their existing builder/motion tests,
and the Strategies browser assertion for the removed selector.
Validation: `pnpm --filter @roehub/platform-web test --run src/motion.test.tsx src/builder.test.tsx`
passed (74 tests); `pnpm --filter @roehub/platform-web typecheck` and
`pnpm --filter @roehub/platform-web build` passed (existing bundle-size warning).
`git diff --check` passed. In-app Browser at localhost:20110/backtests, 1480x969,
confirmed the strip/selector are absent and both seeded jobs remain visible.
No API/data changes or publication. Full e2e, responsive sweep and console/network
inspection were not run for this focused removal.

### 2026-09-24 — Remove duplicate topbar caption
Removed the Backtests text beside the RoeHub brand in `app.tsx`; sidebar and
accessible page heading still identify the workspace. Typecheck/build passed
and in-app Browser at localhost:20110/backtests confirmed the updated header.

### 2026-09-24 — Shared loading indicator
- Added `loading-data.tsx`: a decorative spinning LoaderCircle plus localized
  `Loading data…` / `Загрузка данных…`, exposed as a polite status. Rotation is
  disabled for system reduced motion. No notice border or background.
- Reused it for session, job library/detail, builder reads, projection, results,
  price/trade reads and strategy library/detail/status. Indicators follow active
  fetches; failed/empty/uncertain states retain their own meaning. Accepted
  command text no longer persists after a status refresh. Initial strategy
  loading does not display unavailable execution data or a placeholder title.
- `pnpm --filter @roehub/platform-web test`: 193 passed. Updated two existing
  cancellation tests to await the relevant request instead of ambiguous generic
  loading text. Final typecheck/build and `git diff --check` passed; existing
  build size warning remains.
- In-app Browser at localhost:20110: temporarily delayed the local API, observed
  spinner and English text on monthly statistics, resumed API, and verified
  replacement by loaded statistics. RU and EN initial library/detail loading
  labels also observed; original English locale restored. Screenshot inspected
  at 1480x969. No full responsive/screen-reader/reduced-motion runtime sweep or
  console/network audit. Local synthetic proof only; no publication.

### 2026-09-24 — Remove selected-job header Refresh
Removed the requested duplicate Refresh control from `execution.tsx`; polling
and read-recovery logic are unchanged. `pnpm --filter @roehub/platform-web test
--run src/execution.test.tsx src/library.test.tsx` passed (35 tests); typecheck,
build and `git diff --check` passed, with the existing build size warning.
In-app Browser verified the selected-job header without Refresh on the local
seeded Backtests detail. No publication or API changes.

### 2026-09-24 — Stable uncached report transitions
- Reproduced before the fix with a delayed local API: selecting an uncached job
  collapsed the context panel to a short loading strip and shortened the library.
  Cause confirmed: desktop report dimensions depended on `:has(#results)`, which
  disappears while the keyed JobEntry loads. Click-time transitions also did not
  animate the later asynchronous arrival of results.
- `library.tsx` / `style.css`: report geometry now follows the detail route via
  `backtest-detail-workspace`, independent of data presence. Pending job, summary,
  ranking and variant stages have a centered LoadingData state; new report/variant
  and chart content fades in with the shared duration. Reduced motion disables
  these fades. Query identity, cancellation and cached data boundaries unchanged.
- Changed `execution.tsx`, `results.tsx`, and `price-chart.tsx` to expose readiness
  and avoid placeholder content during initial pending reads.
- Typecheck/build passed (existing bundle warning); frontend tests: 193 passed;
  `git diff --check` passed. In-app Browser regression at 1480x969: delayed uncached
  navigation retained both full-height panels and showed the centered loader;
  rapid return to the cached job remained usable; the loaded target restored its
  own report without collapsing panels. Local API resumed after both bounded
  delay experiments. Before/after screenshots inspected inline. No new automated
  geometry test; responsive, 10%-speed animation and reduced-motion runtime checks
  were not executed. No API changes, production proof or publication.

### 2026-09-24 — Round fullscreen control
Changed `.overview-expand` in `style.css` from 6px radius to 50%, matching the
existing 28px circular icon buttons. Build and `git diff --check` passed; existing
bundle-size warning remains. In-app Browser screenshot at 1480x969 verified the
round Backtests fullscreen button. CSS-only; no new tests or publication.

### 2026-09-24 — Sidebar active state without a border
Removed active sidebar border colors in `style.css`; current-page background,
text and aria-current remain. Build and `git diff --check` passed (existing bundle
warning). In-app Browser screenshot confirmed the active Backtests item without
an outline. CSS-only local change; no new tests or publication.

### 2026-09-26 — Retained data updates across the local platform

User-approved replacement for routine content fades and the September 24 data
entrance animation: keep the last coherent view during reads, update data in mounted
charts, reserve local update status and bound inactive read caching. The normative
rules for current and future pages are in
[the shared data loading contract](roehub-data-loading-contract-v1.md).

### 2026-09-26 — Shared report display periods

Equity, Drawdown and Price & trades now share `1D / 1W / 1M / 3M / 1Y / All`
buttons and one viewport state per job. Presets end at the job's exclusive end;
months/years use UTC calendar subtraction with month-end clamping. Presets longer
than the job are disabled. If job metadata is unavailable, availability falls back
to dated observations. Numeric trade-index series do not offer calendar presets.

Use `chart-period.ts` for the shared viewport semantics in these report views.
A display period changes ECharts zoom only: it does not request/resample report
series, rebase equity or recalculate drawdown. Existing backend sampling remains.
Category boundaries snap to observations and include the preceding observation
at the start to retain the opening realized equity value. Manual zoom stores its
dated bounds and clears the preset highlight; both preset and manual ranges carry
across chart views and candle intervals. `All` restores the full available series.
Candle intervals remain a separate control specific to Price & trades.

Validation: `pnpm typecheck`; 16 focused tests in `report-ui.test.tsx` and
`chart-period.test.ts`; `pnpm build`; `git diff --check` passed. In-app Browser
verified weekly view on all three charts, 1h candles without period reset, manual
slider range transfer, All reset, fullscreen, disabled 3M/1Y and 390px wrapping.
No browser warning/error entries. Local assets updated; no API/database change.

### 2026-09-26 — One switch style across report surfaces

The user's Strategy operations tabs are the accepted control reference. Reuse
`view-switch` in `style.css` for chart views, report tabs and period/interval
buttons: 28px height, 1rem font, pill radius, transparent group and inactive
buttons, neutral #252c34 selected fill, and visible keyboard focus underline.
Do not introduce separate square segments or black framed group backgrounds.
Overview (including mode and period controls), Backtests, Strategy operations
and research tabs share this class; prior conflicting switch rules were removed.
Icon-only Overview controls use the same 28px circular shape as report fullscreen.

Backtests now places period selection in a separate row below graph selection;
candle intervals occupy the trailing side of that row and wrap on narrow screens.
Overview chart selection sits at the leading edge below the Performance heading,
with the circular expand control at the trailing edge. Return metric uses the
same neutral card as the other metrics. Strategy tabs can wrap without colliding
with the expand control. Selection, data loading and chart calculations unchanged.

Proof: typecheck, build and diff check passed; 16 focused report/period tests passed.
Real in-app Browser inspected Backtests and Overview at 1476x959 and Strategy /
Backtests controls at 390px; keyboard graph selection and Overview fullscreen
worked. Browser console error query returned none. Pointer automation did not
reliably activate targets in this session; keyboard interaction was used for
functional proof. No claim of comprehensive pointer or network QA. Cold self-review:
no blocking findings; existing large-bundle warning remains. Local preview only.
Screenshots: `.codex/delivery/evidence/roehub-data-loading-2026-09-26/unified-switches-*.png`.

### 2026-09-26 — Strategy candle interval buttons

Replaced the remaining Strategy operations timeframe dropdown in
`strategy-operations-chart.tsx` with the shared `view-switch` button group.
Available intervals retain the existing source-candle aggregation constraint;
a 15m source exposes 15m, 30m, 1h, 4h and 1d. The chart display menu remains
separate. Toolbar wrapping in `style.css` supports narrow charts.
Typecheck, existing candle aggregation test (1), build and diff check passed.
In-app Browser verified 1h aggregation, 15m restoration and 390px fullscreen
layout; console error query returned none. Screenshots: existing Sep26 evidence
folder, `strategy-timeframe-buttons.png` and `strategy-timeframe-buttons-mobile.png`.
UI compatible change; API and persistence unchanged. Existing bundle warning
remains. Local preview updated; no trading commands or publication.
