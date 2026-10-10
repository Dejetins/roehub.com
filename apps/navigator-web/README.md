# Navigator comparison variation

Independent local client snapshot of `apps/platform-web`, selected Navigator concept 1
with the user's later geometry corrections. The reference source/build is untouched.

- Reference: http://localhost:20110
- Candidate: http://localhost:20120
- Existing disposable API/fixtures: http://127.0.0.1:20111 (shared)
- Shared geometry: `src/navigator.css`; shared lower pane: `src/navigator-table.tsx`.
- Element map: `../../.codex/delivery/ROEHUB-NAVIGATOR-2026-09-26.md`.
- QA: `design-qa.md` and the map's evidence directory.

This is an isolated review variation, not a second production application or a replacement
of the accepted client. After visual selection, migrate the layout into the maintained
client with its existing component contracts; do not maintain divergent features here.

## Local commands

Reuse the installed dependency runtime (no package installation needed):

```sh
# From repository root, only if the candidate node_modules link is absent:
ln -s ../platform-web/node_modules apps/navigator-web/node_modules
cd apps/navigator-web
node_modules/.bin/tsc --noEmit
node_modules/.bin/vite build
node_modules/.bin/vitest run src/report-ui.test.tsx src/query-client.test.ts src/read-snapshot.test.tsx src/overview-model.test.ts src/strategy-operations.test.ts src/chart-period.test.ts
```

From the repository root, with existing API on 20111:

```sh
PYTHONPATH=src:. ROEHUB_ENV=test WEB_BACKTESTS_CLIENT_ENABLED=true WEB_STRATEGIES_CLIENT_ENABLED=true WEB_API_BASE_URL=http://127.0.0.1:20120 WEB_API_UPSTREAM_URL=http://127.0.0.1:20111 .venv/bin/python -m uvicorn tools.qa.navigator_preview:create_app --factory --host 127.0.0.1 --port 20120 --no-access-log
```

Restart only the candidate web process after rebuilding; its manifest loads at startup.
Never run the fixture supervisor to refresh this client: it reseeds data.
Trading commands retain existing guards; confirming them changes the shared local demo.
Overview is synthetic and origin-local; backtest/strategy fixture identities are retained.

### Date range control

`DateRangeControl` is the shared UTC calendar-date picker for Overview and chart
period controls in Backtests/Strategies. Editing uses a draft; Apply commits once,
Cancel/Escape/outside dismissal discard it. Dates are inclusive and clipped to
available history. Keep candle interval separate. Slider zoom must not change
report dates or the preset label. The popup uses the browser top layer so chart
containers and fullscreen layouts cannot clip it.

Overview applies the period to historical P&L, returns and contribution/activity
views while current capital, positions and available funds remain current.
Backtest/strategy chart periods select historical chart coverage; stored backtest
metrics and live strategy accounting/commands remain unchanged. They are not
recomputed period metrics. Future server-backed period reports require explicit
range-aware data contracts, rather than recalculating from sampled chart points.

### Upper workspace geometry

Navigator uses shared workspace tokens in `navigator.css`: 13px/600 headings,
28px controls, 12px control text and 12px horizontal insets. Library headers use
48px; workspace title strips use 40px. Library mode/state switches follow the
header, before list content. Overview portfolio creation/edit actions belong to
the library header, with Live/Paper below. Strategies quick state filters share
the existing `state` URL parameter with the top-layer filter popup (Backtests
menu styling, no library reflow, outside-click/Escape dismissal).
KPI cards reserve a 26px label row and 24px value row on every screen. Preserve
this shared geometry when adding controls; do not create per-page type scales.

Price charts show candle intervals in the single selector row; performance
charts show period presets there instead. Use five baseline choices plus the
date picker, not stacked period and interval rows. Strategies expose only
intervals supported by their source granularity (up to five). Keep display
options immediately after the date picker with an 8px gap, and preserve the
shared chart rectangle on every page.

Chart chrome uses the same first two views in the same order: Equity, Drawdown,
then the page-specific view. Drawdown units belong to the chart axis, not the tab.
Tabs and expand control share the top axis; selectors start 32px below it.
Desktop preset slots are 50px wide with 4px gaps, keeping Date range aligned.
No instructional hover/zoom paragraph belongs in this toolbar. The plot origin
remains 92px below the inner panel top; chart geometry stays shared.

Bare `/strategies` waits for the authorized dashboard status observation before
selecting the newest created running strategy within the current filters. If no
running strategy is observed, select the newest created matching strategy.
Explicit strategy links remain authoritative. Library rows are always sorted by
`created_at` descending, with a deterministic ID tie-break; active selection does
not reorder them. Missing status is unknown, never an invented running state.
Strategy filter popovers must stay within the Library surface (12px inset),
including nested options. Use one field column and internal overflow when the
library is short; never expand the list or cover adjacent metrics/charts.

The shared Library filter popup is contained within its panel and always includes
a Reset button (disabled with no filters). Lower table panels can expand with all
their tabs. Backtest Trades uses 100-row server reads and renders at most two
pages, with scroll spacers and the bounded shared query cache; it does not mount
while another tab is active. Job/variant changes reset the window.

Backtest inspector actions are direct controls. Complete CSV/XLSX export uses
`trades.csv?all_rows=true&format=csv|xlsx`; legacy requests retain their existing
`max_rows` behavior. Full export validates returned row totals and rejects a
truncated response. XLSX is one workbook, with additional sheets only if Excel's
per-sheet row limit is exceeded. Export still requires the existing owner check
and lazy-cache readiness. Server export memory scales with file size.

## Data artifacts and Storage local proof

The selected candidate has `/data?view=artifacts`, shared candle filters/selection,
catalog-backed profile exclusions, explicit Update now and 1/3/7-day cadence.
`/settings/storage` requires installation-owner authority for privileged paths and
commands. Backtests preflight separates stored source coverage from published NPY
coverage and reports additional on-demand preparation. It never represents queued
history as completed worker execution.

For isolated UI verification, `e2e/integrated_fixture.py` serves the current candidate
on a caller-selected loopback port, using only a task-created PostgreSQL on 25457 and
`.cache/s09-dist`. Build with `node_modules/.bin/vite build --outDir ../../.cache/s09-dist`
from this directory; start from repository root with
`PYTHONPATH=apps/navigator-web/e2e:. .venv/bin/uvicorn integrated_fixture:app --host 127.0.0.1 --port 25459`.
No dependency installation or existing preview restart is needed. Stop this process
before removing its disposable DB. Its storage root is temporary and removed at exit.

This UI fixture uses real artifact routes/PG/storage and native NPY preflight, but
synthetic identity, candle/history responses and queued Backtest history. It does not
run a worker or validate exchange ingestion/session issuance. Actual raw-to-canonical,
managed scheduler and completed child-worker proof is the separate
`tests/real_infra/backtest/test_artifact_integrated.py` disposable test; broader
failure/auth/pin coverage is in the adjacent real-infra suites. Preserve that boundary
when reporting browser results.

Security settings show browser profiles using `/api/ui/account/devices`, rather
than the historical session-record feed. A device row survives relogin and exposes
current-device, measured activity, validity and owner-only revocation. Unidentified
legacy logins form one explicit group. Device metadata is not authentication proof;
see `docs/architecture/identity/local-auth-sessions-recovery-v1.md`. The view keeps
bounded snapshots, disables commands on stale/error data and clears private rows
on read or mutation denial. The Activity tab keeps its existing event feed.
