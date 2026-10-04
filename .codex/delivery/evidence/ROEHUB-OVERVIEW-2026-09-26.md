# Overview iteration 1 — local fixture UI

Status: in progress. Authority: owner request 2026-09-26 to implement the first
Overview iteration with test data and start localhost:20110.

Execution unit: portfolio analytics presentation at /dashboard using accepted
Backtests/Strategies styling. Source: roehub-overview-requirements-v1.md OV-01–12.
Scope: deterministic synthetic portfolios, shared context, summary, ECharts,
contributions, allocation, positions/trades, browser-local custom baskets.
Custom history is a recomputed basket, explicitly labeled. Storage is demo-only,
subject-scoped and separate from financial persistence. No real financial API,
trading commands, role changes or production enablement.

Routing: direct execution from this bounded specification; better-layout,
better-ui, browser-qa-evidence + playwright-cli; contract-impact-analysis.
Allowed paths: new overview client files/tests/styles, narrow app/bootstrap route
integration, local QA fixture adapter and this evidence. Preserve foreign edits.
Proof: typecheck/build, deterministic fixture invariants and local real-browser
RU/EN at 820/1024/1440, keyboard, zoom, reduced motion, storage and context changes.
Screenshots: .codex/delivery/evidence/roehub-overview-2026-09-26/.

Compatibility: optional /dashboard bootstrap route is compatible-change within
matched local client/server build; existing routes unchanged. Default server
routing remains unchanged; only the explicit QA preview selects the new route.
Financial API, authorization and database schemas: none (not modified).
OV-12 domain and integration decisions remain deferred to the real-data iteration.

## Delivery result

Completed first local visual iteration. No publication or production enablement.
The initial in-progress status above records entry; terminal status is
`local-fixture-ui-ready-with-caveats`.

Implemented files:
- `apps/platform-web/src/overview-page.tsx`: shared context, summary, sortable
  strategy contributions, exchange/instrument allocation, positions and paged
  order samples with partial-fill disclosure, custom portfolio editor and states.
- `overview-model.ts`, `overview-chart.tsx`, `overview.css`: deterministic daily
  fixtures, TWR-linked contributions, cash-flow-adjusted drawdown, ECharts modes,
  comparisons, monthly returns, scoped styling and container-responsive layout.
- `overview-model.test.ts`, `overview-routing.test.ts`: fixture reconciliation,
  cash-flow exclusion, mode/period separation and explicit routing opt-in.
- Narrow integrations in `app.tsx`, `main.tsx`, `client-routes.ts` and
  `packages/web-contracts/src/index.ts`; original foreign edits preserved.
- `tools/qa/overview_preview.py`, opt-in in `backtests_client_fixture.py`: local
  authenticated preview routing only. Standard server settings remain unchanged.

The complete page uses the existing dark surfaces, font stack, compact controls,
shared motion coordinator and reduced-motion preference. Default range is YTD.
All data is visibly marked synthetic. Custom definitions are browser-local and
subject-scoped. Current values are independent of historical range selection.
The available-cash fixture assumes the explicit reconciliation cash is available;
this is not a rule for real derivatives account margin.

## Runtime

Fresh isolated disposable stack, because the previous preview was not running:
Web `http://localhost:20110`, API `127.0.0.1:20111`, SSR `20112`, runner `20113`.
Docker Desktop was started. No old fixture state or foreign containers changed.
Start command, from repository root:

```sh
ROEHUB_PROOF_PORT=20110 ROEHUB_PROOF_STATE=.local_artifacts/overview-preview-sep26 ROEHUB_PROOF_STRATEGIES=true ROEHUB_PROOF_OVERVIEW=true .venv/bin/python -m tools.qa.backtests_client_fixture
```

The fixture refuses occupied ports or an existing state directory; do not run a
second copy over the current preview. The existing fixture supervisor owns its
new containers and cleans its own state on shutdown. Test credentials remain
only in its private state directory and are not reproduced in evidence.
During builds the running web process retained its original asset names; local
dist aliases were refreshed to the current build without recreating its database.
A fresh launch after a build uses the new manifest directly.

## Verification

- `pnpm --filter @roehub/platform-web typecheck`: passed after final source edits.
- `pnpm --filter @roehub/platform-web build`: passed. Existing >500kB bundle-size
  warning remains; no bundle optimization claim.
- `pnpm --filter @roehub/platform-web test`: 201 tests passed before the final
  focus/scroll-accessibility refinements.
- `pnpm --filter @roehub/platform-web test --run src/overview-model.test.ts src/overview-routing.test.ts src/app.test.tsx`:
  final focused pass, 11 tests.
- `.venv/bin/ruff check tools/qa/overview_preview.py tools/qa/backtests_client_fixture.py`:
  passed.
- `.venv/bin/pytest -q tests/unit/apps/web/test_web_v2_1_routes.py`: 12 passed;
  one existing httpx cookie-deprecation warning.
- `git diff --check`: passed.
- Real Chromium via `playwright-cli -s=overview`: browser-flows.js passed scopes,
  current/period separation, modes, comparisons, expansion, drilldown, gross/net,
  pagination/fills, basket CRUD and reload, validation, deletion recovery,
  empty/history/partial/stale/error/retry/loading/background refresh.
- browser-layout.js passed RU/EN at 820/1024/1440, 200% CSS zoom, 390px mobile,
  reduced motion and keyboard activation. Tables scroll within their sections.
- browser-accessibility.js final pass: zero axe violations for RU ready/editor/
  validation/fullscreen, EN ready/820, RU final. Verified fullscreen bounds,
  focus containment, Escape and restoration of inert siblings. No physical
  screen-reader test or 10%-speed animation inspection was performed.
- Final console: 0 errors, 0 warnings. Inspected local network responses were
  200/304 and expected locale redirects. Initial obsolete build asset 404s were
  resolved before final testing; no financial-provider requests are claimed.
- Browser defects found and fixed: summary collision under CSS zoom (container
  queries), missing editor autofocus, scroll regions inaccessible by keyboard,
  and shared selector collisions avoided by scoping new styles to Overview.
  Two intermediate test failures were assertions before layout/motion settled;
  checks now wait for browser frames or transition completion, not arbitrary delays.

Evidence scripts and screenshots are in `roehub-overview-2026-09-26/`.
`overview-final.png` is the final inspected RU full-page image.
The preview link was queued in the Codex browser panel. Browser authentication
is still required; no authentication bypass was added.

## Proof boundary and next iteration

This is a visual and interactive local fixture implementation, not complete
OV-01–12 financial integration. Real server-side portfolio storage, access-filtered
aggregation, currency conversion, actual financial events/attribution, and
provider failure/race behavior are not verified or implemented. Fixture order
samples are explicitly not the complete P&L ledger. Margin remains unavailable.
Positions currently show individual constituents; full instrument-position
aggregation and real Strategy detail links remain follow-up work. Comparisons
currently cover all selected strategies in Return mode, not accounts/exchanges.
There is no permission-filtered-data scenario or real save-failure service; local
storage failure is handled. Withdrawals/internal transfers, stopped strategies,
historical allocation revisions and unsupported-data combinations are not a
production calculation proof. Native chart data gaps and account/currency rules
must be specified when real OV-12 sources are selected.

Review mode: bounded self-review plus deterministic and browser checks. No
independent-agent review required: no security/role grant, shared policy,
irreversible migration or publication was changed. Compatibility classification:
`compatible-change` for the matched-build opt-in local route; `none` for existing
financial API, auth policy and database schema. Next safe step: owner visual
feedback on this first iteration, followed by the explicit OV-12 real-data entry
choices before production financial integration.

## Follow-up — restore previous Backtests/Strategies demo

Owner requested loading the existing test data. Located Sep24 seed metadata,
cached canonical variant parameters and execution demo state in
`.local_artifacts/ui-preview-sep24`. SHA-256 of old/current 1m OHLCV matches:
`9feec8828eb426dd9b1bf466b49fe686d28f0fe93d42af2e62e24de0dcc69b10`.
Recreated two backtests through authenticated local preflight/jobs APIs, saved
both selected variants as strategies, and added the existing MA(20,50) operational
fixture. This is reconstruction with new IDs, not recovery of the old database.

- Baseline job: `2c252ee4-aa0a-4f1c-b560-7482d7122cdd`, succeeded.
- TP/SL job: `e9d85b78-58ff-495f-8b0b-2b8e5ac64b8a`, succeeded.
- Operational demo strategy: `a26b24e6-9a49-46ed-bbde-3b5096fa4a7d`.
- Three strategies visible; operational fixture displays 8 trades and 16 events.

Only current disposable runtime data and evidence changed. Original Sep24 files
are preserved. Execution anchor and extra-fill timestamps were shifted together
to the present preview, preserving intervals, prices, quantities, fees and P&L,
so the chart's current window contains the demo executions.

Executed `.venv/bin/python .local_artifacts/overview-preview-sep26/restore_previous_demo.py`:
both jobs succeeded, three strategy records created. Local CLI-browser smoke at
1440x1000 confirmed Backtests chart/result tabs, Strategy chart, 8 trade rows and
paged event rows. Captures: `restored-backtests.png`, `restored-strategies.png`.
Final Strategy console: zero messages; inspected data requests returned 200.
No application code or contracts changed (`none`); no real exchange calls or
new trading commands were submitted. Existing fixture middleware remains the
simulation boundary. Responsive/full regression tests were not repeated for
this data-only change. Credentials/session are unchanged; refresh the local UI.

## Correction — restore the accepted market-history dataset

The previous Sep24 restoration was the wrong dataset for the user's intended
Backtests preview. Its one-trade TP/SL fixture was incorrectly presented as the
previous full test set. This correction supersedes that identification.

Authoritative evidence located in the Sep08 iteration folder records the accepted
Sep10 Binance historical BTCUSDT dataset: 2026-02-27 through 2026-03-29 exclusive,
43,200 minute candles, EMA windows 5–50, ten ranked variants. Re-downloaded the
same two Binance monthly ZIPs, validated official checksums and exact equality
with `real-market-sources.json`; no minute gaps. Used the existing
`backtests_expand_demo.expand_artifacts` path with supplied real candles to build
and validate the inactive artifact slot. Original short-fixture slot preserved.

Local scripts executed under `.local_artifacts/overview-preview-sep26/`:
`download_real.py`, `build_real.py`, `reload_expanded.py`, `run_accepted_real.py`.
All completed successfully. API and worker now use `expanded-indicators.yaml`.
Using the established local preview maintenance procedure, retired the exact
original destructive supervisor and replaced only API/worker, preserving the
Web process, containers, database records, credentials and user sessions.
Current retained process inventory: `retained-processes.json` in that state
folder. The earlier claim of an active cleanup-owning fixture supervisor no
longer describes the runtime: services/containers are now independently retained.

Restored job: `54429a5c-c5e5-4f24-bb68-75df21f26d7c`, succeeded.
Ten ranked variants, trade counts: 109, 114, 118, 150, 142, 131, 187, 227, 280, 381.
Best EMA50: 109 trades, return -14.193098247051239%; every stored summary metric
matches the accepted Sep10 reference within 1e-7. Evidence:
`roehub-overview-2026-09-26/restored-accepted-binance-result.json`.

Verified in the user's selected in-app Browser: ten variants, selected count 109,
and a complete multi-point equity curve. Opened the correct job in the existing
tab, checked fullscreen chart, exited fullscreen and left the correct job open.
No source-code changes, new permission grants or real exchange trading.
Compatibility: none for application contracts; local fixture configuration only.
The two erroneous short-demo jobs were retained as separate entries, not silently
rewritten as market-history jobs. This is exact dataset/metric reconstruction
with a new job ID, not recovery of the deleted original database.
