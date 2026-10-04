# Shared local-platform data loading — 2026-09-26

## Authority and scope

Current user explicitly approved stable data-only updates across all current pages,
a bounded cache, and durable code/documentation rules for future work. Execution is
one local implementation unit; no publication, production deployment or new staged
workflow. The accepted implementation specification is
`docs/architecture/apps/web/roehub-data-loading-contract-v1.md`.

## Delivered

- Shared `read-snapshot.ts`, `loading-data.tsx`, `result-read.ts`, `chart-instance.ts`
  and bounded `query-client.ts`. Content transitions in `motion.tsx` no longer fade
  containers; meaningful structural layout transitions remain.
- Backtests: `results.tsx`, `price-chart.tsx`, `job-selection.tsx`, `library.tsx`,
  `execution.tsx`, `builder.tsx`. Coherent detail + active-panel reads, retained
  variant/job/list snapshots, scoped error/status, stable chart instances and
  preservation of applicable report controls. Command components remain isolated
  at committed entity changes. `delete-history.tsx` and cancel/delete observers
  explicitly retain the prior five-minute marker lifetime.
- Strategies: `strategies-page.tsx`, `strategy-operations.tsx`,
  `strategy-operations-chart.tsx`, `strategy-insights.tsx`. Retained detail/runtime
  bundle; polling leaves navigation operable and disables command eligibility while
  pending. Persistent chart instance and retained research view/pagination.
- Overview: `overview-chart.tsx`, `overview-page.tsx`; shared chart lifecycle, local
  refresh status and no container fade on portfolio/period/view selection. The page
  remains a synthetic local preview, not a new live API integration.
- `style.css` reserves or shares local status space with headings, preserves the
  report flex geometry and removes data-arrival fades. Expanded reports expose a
  local pending status too.
- Current/future requirements linked from `.codex/AGENTS.md`, the functional
  contract and Backtests iteration log. Server-owned initial document navigation,
  login and static Settings are distinguished from asynchronous data refresh.

## Cache contract

Inactive idle reads: at most 64 entries and 16 MiB estimated serialized UTF-16
payload; two-minute inactive GC; 15-second freshness unchanged. Eviction considers
recent observation/update. Active/in-flight data and session/command recovery are
excluded from the inactive read budget. This is not a total browser heap cap.
Response bounds and one retained bundle per mounted consumer bound the current
working set. No persistent result cache or all-variant speculative prefetch added.

## Gates

- `pnpm test`: 207 passed, 18 files (full suite after implementation).
- `pnpm typecheck`: passed.
- `pnpm build`: passed; existing large-bundle warning remains (bundle >500 kB).
- `git diff --check`: passed.
- Focused `src/report-ui.test.tsx`: 12 passed after final fullscreen/cleanup fixes.
- Added regression coverage for delayed coherent commit with a retained canvas,
  obsolete variant response, empty successful candle response, fullscreen during
  pending selection, cache count/size/expiry and observer/recovery protection,
  snapshot access/scope clearing. Existing motion test now forbids content fades.
  Chart mock now models getOption returning null after disposal.

## Browser proof

Mechanic: the user's existing in-app Browser, authenticated local disposable
fixture at `http://localhost:20110`. Desktop 1480×969, narrow 390×844 (plus native
402px panel); temporary viewport reset and English locale restored at completion.
The accepted Binance historical job and all demo data/services were retained.
Only read/navigation/refresh controls exercised; no trading commands sent.

Observed:

- EMA45→EMA40: during pending, old metrics/identity stayed visible; after completion
  new metrics/identity replaced them. ECharts instance `ec_1790439735384` stayed
  identical through the update and rapid EMA25→EMA50 changes.
- Equity→Price retained the old view until candles and markers were ready.
  Candle 15m→1h retained instance `ec_1790440049642`; new caption showed 720 bars.
  Fullscreen preserved the instance; Escape restored trigger focus.
- Trades pagination loaded page 2; switching EMA50→EMA45 retained the Trades tab
  and reset its page to 1, showing 114 trades.
- Switching that job to Demo EMA retained the last displayed EMA45, not its original
  default. Browser Back returned to the historical job.
- Strategies refresh preserved instance `ec_1790440159050`, kept its workspace
  non-inert and allowed the Equity tab. Entity switching showed the explicitly
  labelled previous strategy while preparing the next detail/runtime bundle.
- Overview period and portfolio changes preserved instance `ec_1790440159051`.
- Russian narrow Backtests displayed “Обновление… / Показаны предыдущие данные”;
  full-screen line chart rendered correctly; document scroll width matched 390px.
- A browser-only disposed-chart cleanup error was found during Price→Trades,
  fixed by reading zoom before updates rather than after disposal, and the exact
  flow passed on the rebuilt app. An inert restoration conflict on fullscreen
  during a pending read was also fixed; final browser check reported
  `actionsBlocked: false` and restored fullscreen-trigger focus.
- Final reviewed console entries since 16:29 UTC: no errors or warnings. Earlier
  development error entries were not counted as final-run failures.

Screenshots (same Browser):
`roehub-data-loading-2026-09-26/backtests-desktop.png`,
`strategies-desktop.png`, `overview-desktop.png`, `overview-mobile.png`,
`backtests-mobile-ru.png`.

No full network trace, screen-reader sweep or browser-level reduced-motion override
was run. Reduced-motion priority is covered by the existing unit test; animation
code reads the same preference and initial chart animations are disabled. Browser
proof covers the exercised flows, not every API failure combination. Delayed and
obsolete responses, cache eviction and access clearing have deterministic tests.

## Compatibility and review

Browser update behavior and ephemeral read cache: `compatible-change` for current
local clients, with the user's explicitly accepted motion change. Existing APIs,
DTO validation, persisted data and command side-effect protocols: `none`; no server
or database migration. Client rollback uses the existing server contracts.

Cold self-review found and repaired graph-axis reset, status geometry, failed-202
status handling and fullscreen inert restoration issues. Repository-required
independent read-only review found four issues (empty candles, research remount,
retained job variant and polling inert scope); all were fixed and its targeted
re-review returned no remaining findings. No claim of production readiness or
publication. Residual limitations: large frontend bundle warning; inactive payload
budget is approximate and does not measure total JS heap.

Local preview uses the newly built assets via its existing cached manifest aliases;
API, worker, sessions and databases were not restarted/reset. Next safe action:
user visual acceptance in the existing local Browser.

## Follow-up — segmented candle timeframes

User requested the Backtests timeframe selector as buttons like Overview. Replaced
its native select in `results.tsx` with a labelled seven-button group (`aria-pressed`)
and matched Overview's 24px compact segmented styling in `style.css`. Updated
existing report UI tests to use the buttons; retained loading and chart lifetime
are unchanged. `pnpm typecheck`, 12 report UI tests, build and diff check passed.
Browser verified 1h selection (720 candles) and 15m restoration, fullscreen and
390px wrapping. Evidence: `timeframe-buttons-desktop.png`,
`timeframe-buttons-mobile.png` in the screenshot directory above. Viewport reset;
the user's fullscreen chart remains open. No API or persistence change.

## Follow-up — shared chart display periods

Implemented the accepted period controls for all three Backtests chart views.
Changed `results.tsx`, `price-chart.tsx`, `results-i18n.ts`, `style.css`; added the
shared `chart-period.ts` helper and focused tests. Updated the iteration log with
UTC calendar, boundary snapping and viewport-only semantics. The same job retains
preset/manual bounds between views and candle intervals. API/persistence impact:
`none`; browser behavior: `compatible-change`. No cache expansion or new query.

Checks: `pnpm typecheck` passed; `pnpm --filter @roehub/platform-web test --run
src/report-ui.test.tsx src/chart-period.test.ts` passed (16 tests); `pnpm build`
passed with existing >500kB bundle warning; `git diff --check` passed. An initial
typecheck rejected Testing Library's unsupported `exact` option; removed it and
reran successfully. Browser proof on historical job 54429a5c-c5e5-4f24-bb68-75df21f26d7c:
1W survived Equity -> Drawdown -> Price and 15m -> 1h; manual slider selection
cleared the preset and carried back to Equity; All restored full span. 3M/1Y are
disabled. Fullscreen controls wrap at 390px; viewport override reset. Console
warning/error query returned none. Screenshots in the existing evidence directory:
`period-equity-week.png`, `period-drawdown-week.png`, `period-manual-equity.png`,
`period-price-mobile.png`.

Cold self-review: no remaining blocking findings. Limits: charts retain existing
server sampling and category axes, so viewport edges snap to source observations;
this change does not add more report resolution. Proof is local, not production.
Next action: visual acceptance in the user's existing local report.
