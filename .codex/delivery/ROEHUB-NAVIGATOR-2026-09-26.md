# Navigator separate variation — element and delivery map

Status: implemented; local validation passed; awaiting user visual review. Authority: user chose Navigator concept 1, with equal chart/table geometry, icon-only global navigation, visible page titles, no global header, and parallel reference/candidate sites. Local comparison only; no publication.

## Isolation

Reference: apps/platform-web, existing dist and port 20110, untouched.
Candidate: apps/navigator-web, independent source snapshot, dist and port 20120.
Candidate uses the existing local API on 20111 and existing disposable fixtures; datasets are shared, not reseeded. Commands retain existing guards and can affect that shared fixture; QA does not confirm lifecycle/destructive commands.
Allowed writes: candidate directory, tools/qa/navigator_preview.py, this map and candidate evidence. Dependency runtime is reused, not installed.

## Geometry contract

Global rail: 56 px, icons with accessible names. Visible page title above workspace.
At desktop: library 224 px; main chart flexible; inspector 240 px; gaps 10 px.
Object header 40 px; metrics 88 px; chart surface 430 px; bottom table 300 px.
Bottom table spans main and inspector. The same grid tracks, chart controls and table class apply to all three pages. Domain columns and actions differ. Narrow layouts stack the same regions without removing controls.

## Element map

| Region | Overview | Backtests | Strategies |
|---|---|---|---|
| Library | All/exchange/instrument/custom portfolios; create/edit; Live/Paper | Jobs, new/history, builder with draft; state/risk/search/date filters; paging; refresh | Saved strategies; search/market/timeframe/status filters; freshness; refresh |
| Heading | Selected portfolio, demo label | Selected job, execution state | Selected strategy, observed lifecycle state |
| Metrics | Equity, P&L, return, drawdown, available funds (exposure in inspector) | Return, DD, PF, win rate, trade count | Equity, realized/open P&L, fees, drawdown |
| Chart | Equity/Return/DD, range/custom dates, compare, flows, zoom, expand | Equity/DD/Price, periods and candle intervals, markers, zoom, expand, accessible data | Price/Equity/DD, candle intervals, display menu, benchmark, execution selection, zoom, expand |
| Inspector | Scope, valuation, gross/net exposure, calculation notes | Input coordinates/time range/risk, selected source, save/export, refresh, job info/delete disclosure | Current position, entry/mark/quantity/notional/P&L/SL/TP, manual/lifecycle actions, guarded confirmations |
| Bottom table | Contributions, monthly returns, allocation, positions/orders; sort, drilldown, paging | Ranked variants, full metrics, trades, monthly stats; selection, sort, paging | Trades/events, unread count, filters, paging, fill drilldown |
| Secondary | Local portfolio editor; demo scenarios; loading/error/empty | New job form, pending/progress/cancel, errors/recovery | Technical JSON, no data/stopped states, errors/recovery |
| Global | Locale, settings, unavailable Data, sign out, keyboard/skip/focus | Same | Same |

## Proof boundary

Typecheck/build; focused inherited logic checks; browser on all three populated routes, table and chart switches, library selection, empty/stopped/new-form state, fullscreen, mobile overflow. Compare desktop measured chart/table rectangles, screenshots and selected concept plus user amendments. Preserve bounded cache/coherent snapshots. No production/provider or mutation proof claimed.
Compatibility: browser layout compatible-change (separate experimental origin); API/DTO/persistence none; existing source/build none. Candidate browser storage isolated by origin; session cookie may be shared across ports.

## Delivered evidence

Candidate code and shared layout/table: `apps/navigator-web/`. Runtime factory: `tools/qa/navigator_preview.py`. No writes to the reference frontend source/build in this task. Browser checks: all three routes, independent chart/table selection, fullscreen, unavailable strategy, portfolio editor/custom period, and new-backtest form. Both local sites respond HTTP200 on `/login` and remain open.

Measured at1476×959: chart panel912×430, plot container886×292, bottom1162×300 at matching coordinates on all pages. Evidence and geometry are in `.codex/delivery/evidence/roehub-navigator-2026-09-26/`; detailed commands, exceptions and comparison history in `apps/navigator-web/design-qa.md`. Typecheck/build passed; focused tests32/32. No commands confirmed that mutate shared fixture data. Self-review passed for local scope; remaining decision is user visual acceptance.

## Sep27 — library refinement from browser comments

Removed global library-collapse control from all candidate pages. Removed freshness/snapshot text from job list and strategy library. Jobs now use compact64–72px rows with accessible state icons. Fixed request batch10 replaces misleading page-size selector; navigation shows actual returned count and two aligned arrows. Changing an old batch size clears its cursor. Search disclosure moved to header icon and inline panel above rows; existing server-projection limitation remains explicit, fields disabled. Backtest KPI row now always has five ordered slots, including missing profit factor as em dash.

Proof: browser minimal TP/SL job shows5cards; library rows measured71.28/64/64px, arrow y positions identical; no page-header Library button. Search panel opens inline. Screenshot `evidence/roehub-navigator-2026-09-26/compact-library-sep27.jpg`. Focused report UI tests15/15 pass, including missing metric case. Reference20110 untouched.

### 2026-09-27 — lower table header and density

Accepted correction: use Backtests as the table reference. Strategy controls/statistics and Overview context belong in the shared active-tab header, not extra rows above column headings. Implemented reusable `NavigatorTable.tools` slot with common header and row dimensions; preserved tab keyboard behavior and domain schemas. Local browser measurements agree across all three pages (49px header, 32px column header, 44px first row). See `apps/navigator-web/design-qa.md` for commands and screenshots. Candidate-only change, no baseline or backend mutations.

### 2026-09-28 — local runtime recovery

Docker Desktop was stopped and the old `--rm` preview containers were absent after startup. Recreated the local databases in named persistent volumes `roehub-navigator-pg` and `roehub-navigator-ch`; retained artifact directories, login password and original three job/strategy URL identifiers. Recomputed all three jobs. Historical job has 10 variants; best variant retains 109 trades and all saved summary metrics match within 1e-7. Restored strategy fills from the saved execution fixture (8 trades, 16 events); shifted all fixture timestamps by the same offset to today so the chart retains the prior compact research window, with the original state backed up. Current code generates updated canonical strategy name hashes; URL IDs and numerical results remain retained. Local DB backup saved privately under `.local_artifacts/navigator-preview-sep26/backups/`.

API 20111, baseline 20110, Navigator 20120 and worker 20113 started. No product source edits or publication. Recovery scripts are local artifacts only. Verification: saved metric comparison, all 3 jobs succeeded in PostgreSQL; browser strategy screenshot `restored-sep28.jpg`. Local demo proof only.

### 2026-09-29 — restart with reduced Docker memory

Started Docker Desktop; existing API/web/worker processes and persistent databases were reused without reseeding. Changed only Docker Desktop `MemoryMiB` from 16384 to 4096 in the local settings-store, then restarted Docker. All three auto-start containers resumed, including the existing Freqtrade container. `docker info` confirms 4,104,110,080 bytes available in VM (configured 4 GiB minus overhead). `docker stats --no-stream` after browsing: ClickHouse 1.474 GiB, PostgreSQL 55.78 MiB, Freqtrade 530.9 MiB. These are container working-set snapshots, not host process RSS or a peak-load benchmark; no actual-memory reduction claim. All report OOMKilled=false, RestartCount=0. 4 GiB is a practical preview budget with headroom, not a proven absolute minimum for every workload. Heavy concurrent backtests were not exercised.

PostgreSQL read verified 3 jobs and 3 strategies retained. Browser on 20120 loaded historical results/graph with 109 trades; screenshot `started-sep29.jpg`. HTTP login returned 200. No application source or baseline changes. Resource configuration compatibility: compatible-change for verified preview reads; higher-load capacity unverified.
