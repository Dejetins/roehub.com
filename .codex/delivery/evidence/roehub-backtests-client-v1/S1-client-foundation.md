# S1 — Client foundation and reversible Web integration

Date: 2026-09-08. Scope: local checkout, S1 only.
Status: foundation implementation and local integration verified; see validation below.

## Authority and proof boundary

The user selected ordinary sequential execution and explicitly superseded PACK-CLAIM,
ledger transitions, claims, updater, receipts and staged-plan-runner entry checks.
The existing ledger and prompts are unchanged. No Goal, branch, worktree, stash,
commit, push, merge, deployment, default cutover or server authorization change occurred.
S2–S6 were not executed.

The result is a working gated client foundation, not a completed Backtests journey:
`local_journey_verified=false`; `target_role_cutover_ready=false`.
The v23 reference is unchanged (SHA-256
`3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`).
The minimal surface reuses its colors and radius; full workbench fidelity belongs to S2.

## Delivered behavior

- A pnpm workspace contains `@roehub/platform-web` and the sole currently needed
  shared package, `@roehub/web-contracts`. No unused shared package scaffolds.
- The accepted React/TypeScript/Vite/React Router/TanStack Query/Table/i18next/Zod/
  React Hook Form/Lucide/ECharts/Vitest/Testing Library/Playwright/axe stack is pinned
  exactly in package manifests and `pnpm-lock.yaml`. Node 24.18.0 and pnpm 11.13.0
  were used; the root engine supports Node 24.18+ within major 24.
- `WEB_BACKTESTS_CLIENT_ENABLED=false` is the default. `true` switches only
  `/backtests`, `/backtests/new`, `/backtests/{job_id}` after the existing real
  identity lookup succeeds. Other destinations stay SSR. Unfinished job actions
  are not offered. React does not implement roles, domain calculations or grants.
- Web reads the Vite manifest at startup, verifies required asset files, and
  serves only hashed `dist/assets` under `/platform-assets/assets`. It does not
  expose a public SPA index or manifest. A missing enabled build fails startup
  with a specific build/disable instruction; flag-off needs no client build.
- Protected HTML and login redirects use `private, no-store`. Login continuation
  retains the sanitized local path and query, including encoded `variant`.
  The existing locale cookie/endpoint determines RU/EN and keeps the deep link.
- An identity 401 redirects to login. Identity unavailability returns 502 without
  client bootstrap. The client separately distinguishes 401, 403, validation,
  conflict, throttling, transport and unavailable states. Changed subject does
  not reuse the original view. Query state is discarded on pagehide; bfcache
  restoration reloads through the server gate. Sign-out remains the existing
  SSR/local-auth flow. No recovery payload or durable draft storage is introduced.
- The API adapter accepts same-origin `/api/` only, sends session cookies, bounds
  request time to 15 seconds by default (configurable 1–60000 ms), cancels obsolete
  reads, preserves 202 and numeric Retry-After, and performs exactly one request.
  Query and mutation retries are explicitly false. Failed/indeterminate commands
  preserve an unknown outcome; they are never automatically resubmitted.
- 422 uses the actual dotted-string `path`, `code`, `message` contract; raw
  input/context are stripped. Domain error code is retained. Messages are text
  data for later field rendering, never HTML. Current-user provides no organization
  binding: client recovery/replay must not infer one from this seam.

## Reproduction, data and cleanup

Run all commands from `/Users/daniildegtyarev/Projects/roehub.com`.
Prerequisites: Node 24.18+, pnpm 11.13.0, Docker, the existing Python 3.12 `.venv`
with repository dev/data dependencies (`uv sync --locked --all-groups`).

```sh
pnpm install --frozen-lockfile
pnpm --filter @roehub/platform-web typecheck
pnpm --filter @roehub/platform-web test
pnpm --filter @roehub/platform-web build
pnpm --filter @roehub/platform-web exec playwright install chromium
pnpm --filter @roehub/platform-web test:e2e
```

On Linux the browser prerequisite is `playwright install --with-deps chromium`.
The e2e command starts and gracefully stops the complete disposable fixture using
`tools/qa/backtests_client_fixture.py`. It refuses to overwrite an existing state
directory. `playwright.config.ts` waits for Web `/health/ready`, not mere liveness.

For an interactive local investigation:

```sh
.venv/bin/python -m tools.qa.backtests_client_fixture
# In a second terminal, against this explicitly started disposable fixture:
ROEHUB_PROOF_REUSE=true pnpm --filter @roehub/platform-web test:e2e
```

Addresses: Web `http://localhost:18480`, API `http://127.0.0.1:18481`,
flag-off Web `http://localhost:18482`, runner metrics `http://127.0.0.1:18483/metrics`.
Only loopback ports are published. The fixture creates fresh randomly named
PostgreSQL/ClickHouse containers and a seeded disposable local owner account.
The account seed is test setup, not a test of owner bootstrap or target-role policy.
Passwords are generated at runtime; the private file
`.local_artifacts/backtests-client/credentials.json` is mode 0600 inside a mode 0700
directory. It is never printed, included in screenshots, or saved in the report.
Browser tests perform actual password login through Web and production auth;
no dependency override, fabricated auth cookie or intercepted API response is used.

The fixture uses these immutable image references:

- `postgres@sha256:cf78e76683b9ca8c5733cbbdce6c9262b45b6767934dd0a95e671f9a0fc20685`
  (resolved from PostgreSQL 16 Alpine).
- `clickhouse/clickhouse-server@sha256:87e0a5b72f5465b18eacca7c76850e7ff551c9795c50e451f5646299e5e24146`
  (resolved from ClickHouse 25.8 Alpine).

Preparation is verified: existing `run_dev_db_bootstrap` applies identity SQL,
Alembic and auth SQL; remaining greenfield SQL follows the existing migration
manifest. ClickHouse uses `migrations/clickhouse/market_data_ddl.sql`, with one
synthetic `binance:spot:BTCUSDT` catalog entry. No repository migrations are edited.

The source data helper is
`tests/unit/contexts/backtest/application/services/v2/test_artifact_precompute_runner_v2.py`
(`_build_canonical_rows_v2`, `_FakeCanonicalCandleReader`, `_request_v2`);
`artifact_testkit_v2.py` supplies only the initial disposable artifact configuration.
The candle reader provides deterministic test inputs, not provider data.
Actual Numba indicator computation, signal rules, artifact precompute, manifest
validation, PostgreSQL job-blocking check and local filesystem slot publication
use production code. The final fixture is **4320 consecutive 1m candles**, covering
`[2026-03-26T00:00:00Z, 2026-03-29T00:00:00Z)`, with a matching published as-of date.
It materializes `ma.ema` at `15m`, close source, window 10 from
`configs/test/indicators.yaml`; TP/SL fixture levels are 1% and 2%.
The earlier tiny epoch-timestamp artifact helper was replaced and is not the final recipe.

Verified real API prerequisites: session, runtime-defaults, direct empty jobs list,
workstation defaults/catalog, ready artifact-date-bounds, and successful preflight
(one candidate, no errors). The probe uses explicit spot `long_only`, artifact mode,
server execution/ranking defaults and a window obtained from the actual catalog.
The real job runner starts and exposes metrics while idle. **No job was submitted,
no runner lifecycle/result computation was claimed, and no provider was contacted.**
The job-create factory retains the actual configured default replay window of 86400 s
(`BacktestJobsUseCase.idempotency_ttl_seconds`; the factory sets no override).
No replay was attempted. Server-bound organization replay is still unavailable;
S3 must preserve the ticket's read-only unresolved recovery rule.

Ctrl-C/SIGTERM stops this run's child processes, removes its named containers and
removes its private state. Playwright is configured with graceful SIGTERM shutdown.
The final cold e2e run verifies startup and this cleanup. Failure reports in
`.local_artifacts/platform-web-test-results` contain no stored browser session.
An interrupted/force-killed fixture must be inspected before cleanup; do not delete
another run or reuse an unknown state directory.

For a separately configured existing local API, build the client and launch Web
with its existing `WEB_API_BASE_URL` / `WEB_API_UPSTREAM_URL` plus
`WEB_BACKTESTS_CLIENT_ENABLED=true`. The base URL must serve Web's `/api` proxy.
The default build directory is `apps/platform-web/dist`. Restart Web after any
build (including `build:watch`) because its asset manifest is captured at startup.
Rollback: set `WEB_BACKTESTS_CLIENT_ENABLED=false` and restart Web. No data migration,
API change, job deletion or artifact deletion is needed to restore SSR.

## Validation and observed evidence

All commands ran locally; the new GitHub workflow was authored, not executed remotely.

| Check | Result |
| --- | --- |
| `pnpm install --frozen-lockfile` | Passed; installed pinned workspace |
| `pnpm --filter @roehub/platform-web typecheck` | Passed |
| `pnpm --filter @roehub/platform-web test` | Passed: 21 tests in 2 files |
| `pnpm --filter @roehub/platform-web build` | Passed; Vite manifest and hashed CSS/JS |
| `pnpm --filter @roehub/platform-web test:e2e` | Passed: 2 real Chromium scenarios; cold disposable stack startup/cleanup |
| `.venv/bin/python -m pytest -q tests/unit/apps/web/test_web_v2_1_routes.py` | Passed: 10 tests |
| `.venv/bin/python -m pytest -q tests/unit/apps/web` | Passed: 59 tests; 4 existing httpx cookie deprecation warnings |
| `.venv/bin/ruff check apps/web/main/platform_client.py apps/web/main/settings.py apps/web/main/app.py tools/qa/backtests_client_fixture.py tests/unit/apps/web/test_web_v2_1_routes.py` | Passed |
| `.venv/bin/pyright apps/web/main/app.py apps/web/main/settings.py apps/web/main/platform_client.py tools/qa/backtests_client_fixture.py` | Passed: 0 errors |
| `python3 -m tools.docs.generate_docs_index --check` | Passed |
| `python3 -m tools.docs.generate_project_map --check` | Passed |
| `git diff --check` | Passed |

Real Chromium evidence is under `browser/` beside this report:
`foundation-observations.json`, `foundation-820.png`, `foundation-1024.png`,
`foundation-1440.png`, `foundation-ru.png`, `ssr-rollback.png`.
The tests cover anonymous/expired/authenticated entry, encoded deep-link refresh,
RU/EN continuation, protected cache headers, hashed asset loading, private-index
absence, preserved SSR navigation, logout, identity outage without logout,
feature-off SSR, keyboard focus and axe (zero violations on the foundation).
Screenshots were visually inspected. Browser widths 820/1024/1440 are covered;
full v23 fidelity, 200% zoom and complete Backtests actions remain S2–S6 work.

The evidence records console/HTTP/transport observations separately. Foundation
console and HTTP errors must be empty. Existing SSR optional OIDC status and unrelated
strategy/dashboard/account projections are not installed by this focused API fixture;
their 404 responses and console errors are recorded, not treated as working APIs.
Settings also references absent `/assets/css/pages/market-data-settings.css` and
`/assets/js/pages/market-data-settings.js`: these are pre-existing repository asset
omissions, not fixture API omissions. The references are present in HEAD's
`apps/web/templates/pages/settings.html`, while HEAD's asset tree has neither file.
They remain outside this S1 change. SSR navigation is proved; complete SSR Settings
functionality is not claimed.
Navigation can abort outgoing SSR resource requests; only `net::ERR_ABORTED` outside
the foundation phase is allowed, and those events remain in the evidence.

Intermediate failures were corrected: i18next's current option is `initAsync`;
real 422 uses dotted string paths; identity SQL is separate from Alembic;
artifact config needs an explicit indicators-config path; fixture readiness must
include API, and shutdown must be graceful. The first surface lacked SSR's native
view-transition opt-in; it is now set before external CSS loads, with no animation.
The initial preflight probe used window 20 (not present in test catalog) and the
futures-oriented server direction default; the final explicit spot test uses the
catalog window 10 and `long_only`. These were fixture/proof corrections, not server
policy changes or blind retries of mutations.

## Compatibility assessment

Baseline: existing local Web SSR/API contracts in the checkout; candidate: S1 changes.
Direct callers, settings, template routing, current-user API, proxy, Web tests,
Backtests wiring and fixture data/runner boundaries were inspected.

| Surface / consumer | Before → after | Classification |
| --- | --- | --- |
| Config / existing Web launch | No switch → optional strict boolean, default false | `compatible-change` |
| Backtests HTML / opted-in browser | SSR → gated client foundation with preserved URLs/cache/locale | `compatible-change`, opt-in local scope only |
| Default and unrelated HTML / old browser | SSR → same SSR | `none` |
| Login continuation / old and new browser | Path only → sanitized path plus query; private no-store redirect | `compatible-change` |
| Static delivery / new browser | Add hashed client assets; existing `/assets` unchanged | `compatible-change` |
| API DTO, server authz, domain ports, persisted data, jobs/replay identities | No product-source changes | `none` |
| Rollback | Disable setting and restart → old SSR against same API/data | `compatible-change`; real local browser verified |
| Production installation/package deployment | No configured target or installation action | Not evaluated; not claimed |

## Known backend dependency and S2 handoff

`apps/api/wiring/modules/ui_backtests.py` calls `_build_jobs_use_case` without
`organization_scope_resolver`. `apps/api/wiring/modules/backtest.py` returns `None`
when that resolver is absent. Therefore workstation's `backtest_jobs` source is
`unavailable` **even with the verified STRATEGY_PG_DSN**. Its text saying the DSN is
not configured is misleading; this is an existing backend wiring defect.

The direct `/api/backtests/jobs` endpoint is wired with the resolver and is verified
available. S2 may use that endpoint and direct job status for their actual supported
state/risk/cursor contracts, while rendering workstation search/date/instrument
job projection as unavailable. It must not fabricate full-dataset filtering, source
readiness or granted permissions. Repair of the backend wiring needs separately
selected authority; no backend/auth wiring was modified in S1.

The S2 prompt and declared producer inputs were inspected: this report and
`apps/platform-web/package.json` exist, and the workspace scripts/fixture are usable.
Exact seams for S2: `src/app.tsx` route composition, `src/i18n.ts` resources,
`src/query-client.ts`, `src/api.ts`, `packages/web-contracts/src/index.ts`.
S2 owns the real shell/library and may create only the shared packages it needs.
The configured data supports bounded local artifact integration, not broad market
coverage, live data, performance claims or target organization roles.

## Review

Cold self-review plus one independent read-only `production-risk-review` were performed.
No material security defect was found in the gated route/API seam. The reviewer raised
P2 proof gaps in console/HTTP capture and fixture readiness: these were addressed by
explicit observations/source assertions, production-built artifact files and real
preflight. The coordinator also independently identified the 422 format correction
and the workstation resolver defect. No concurrent implementation edits occurred.
The coordinator retains final independent stage-report review.

## Owned paths and preserved foreign work

Created:

- `package.json`, `pnpm-workspace.yaml`, `pnpm-lock.yaml`
- `packages/web-contracts/package.json`, `packages/web-contracts/src/index.ts`
- `apps/platform-web/package.json`, `apps/platform-web/tsconfig.json`,
  `apps/platform-web/vite.config.ts`, `apps/platform-web/vitest.config.ts`,
  `apps/platform-web/playwright.config.ts`
- `apps/platform-web/src/api.ts`, `apps/platform-web/src/api.test.ts`,
  `apps/platform-web/src/app.tsx`, `apps/platform-web/src/app.test.tsx`,
  `apps/platform-web/src/i18n.ts`, `apps/platform-web/src/main.tsx`,
  `apps/platform-web/src/query-client.ts`, `apps/platform-web/src/style.css`
- `apps/platform-web/e2e/foundation.spec.ts`
- `apps/web/main/platform_client.py`, `apps/web/templates/pages/platform_client.html`
- `tools/qa/backtests_client_fixture.py`, `.github/workflows/platform-web.yml`
- This report and the six `browser/` evidence files listed above.

Modified:

- `apps/web/main/app.py`, `apps/web/main/settings.py`
- `tests/unit/apps/web/test_web_v2_1_routes.py`
- `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md`
  (only the added durable S1 integration section; prior foreign changes retained)
- Generated `docs/architecture/project-map/PROJECT_MAP.md`,
  `docs/architecture/project-map/component-map.mmd`,
  `docs/architecture/project-map/project-map.json`, as required by new code paths.

Deleted: none. Outside expected touch zones: none.
Ignored transient outputs: workspace `node_modules`, `apps/platform-web/dist`,
Playwright local result directory; disposable databases/private state are cleaned.

Foreign AGENTS/ticket changes, functional-contract and architecture-index changes,
plan, generated prompt pack, old ledger, planning evidence, and pre-existing project-map
inputs were preserved. No broad staging or Git cleanup was used. The protected v23
was hash-checked and never edited. No receipts or additional workflow-control
artifacts were created.
