# Strategies library/detail implementation — 2026-09-12

## Scope and authority

User-selected standalone implementation prompt, one execution unit. Implemented the
independent default-off Strategies client, complete owned snapshot library/filtering,
immutable detail, optional selected status and Backtests save/open/report-return flow.
No branch, worktree, stash, stage, commit, push, merge, deployment or trading command.
No API/domain/storage changes. Existing SSR management remains reachable.

`local_journey_verified=true`; `target_role_cutover_ready=false`.
`journey.strategies.create_and_control` remains partially migrated. This is local
engineering verification, not new product-owner visual acceptance or production proof.

## Implemented boundaries

- `WEB_STRATEGIES_CLIENT_ENABLED=false` is independent from Backtests. Shared assets
  load if either flag is enabled. Each authenticated route retains its own gate;
  absent opted-in assets fail clearly and HTML remains `private, no-store`.
- Additive `client_routes` bootstrap is presentation-only; an older bootstrap retains
  Backtests-only behavior. Navigation/title/footer/focus follow the active page.
  Repeated `view`/`mode` use the server's last-value semantics.
- `/strategies`, `/strategies/{strategy_id}` and the legacy `strategy_id` query entry
  work. `/strategies/new`, non-default `mode`, and `view=classic` remain SSR, including
  locale/login continuation. Classic visits use document navigation.
- Direct list/detail parse allowlisted fields, bind subject and identity, and consume
  cancellation signals. Complete local filtering uses actual name/instrument/market/
  timeframe values. No missing selected ID is replaced with a first list item.
- Ordered indicator inputs/parameters and signal template are readable as text.
  Technical identity/schema/raw specification are subordinate disclosures.
- Dashboard observation is independent, identity-bound and manual, with no polling or
  per-row queries. It respects refresh hints, Retry-After and 429. Unavailable panels
  never inherit ready/stopped defaults. Raw provider/debug payloads and unrecognized
  reason payloads are excluded from the product cache. Dashboard may itself record a
  compatibility event; it is not reverse-provenance retrieval.
- Save endpoint, idempotency, deduplication and unknown-outcome recovery are unchanged.
  Explicit Open strategy carries only validated `from_job`/`from_variant`. Reload
  treats these as untrusted local navigation context, not provenance. Standalone
  entry has no fabricated report link; missing context is harmless.
- One existing motion preference/coordinator remains. Router DOM integration makes
  rapid filter updates synchronous. Early cross-document transition handlers consume
  expected resize/navigation interruption rejections; ordinary page errors remain
  observable. Long libraries have bounded scrolling.

## A1–A9 evidence

All browser work uses the repository Playwright runner, Chromium, production Web/API/
auth/worker, disposable PostgreSQL/ClickHouse and labelled synthetic candles. Success
responses for owned lists/details and the saved journey are real. Controlled errors
supplement those boundaries. No exchange/provider execution or target-role proof.

| ID | Observed proof | Boundary / limitation |
|---|---|---|
| A1 | `strategies.spec.ts`: real empty list, two created snapshots, search/no matches/reset, market filtering, keyboard row activation, immutable specification, standalone reload, independent detail during list 503, absent ID without substitution; long-library scroll and Back/Forward | Complete direct list; no server search/cursor claims. Timeframe/instrument filter coverage also in `strategies.test.ts` |
| A2 | Python matrix and real browser four Web processes: both off, Backtests only, Strategies only, both on; direct/reload/classic/new/mode/locale entries and gated cross-navigation; repeated-query loop regression; foundation login return/private HTML/assets | Existing authenticated own-resource boundary; route capability grants no authorization |
| A3 | `journey.spec.ts`: real preflight → one UI create → worker completed → one explicit UI save (201) → returned strategy → reload → original job/variant; explicit duplicate API request returns 200 and same strategy; zero run/start commands | Save never starts trading; synthetic candles establish local integration only |
| A4 | Invalid/missing context ignored; standalone detail has no link; valid local context returns to truthful absent job; real saved journey supplemented with source 404 and browser return to same strategy | No persisted reverse provenance. Hard deletion of a saved source is restricted by current provenance FK; see investigation below |
| A5 | Real optional status/freshness; controlled 503, 429, identity mismatch, unavailable/empty panels carrying ready-like fields, blocked runtime and late dashboard/detail reads; immutable detail remains usable | No polling, provider payloads, performance cards or trading-enabled inference |
| A6 | Controlled stale specification after 503, 403/404/invalid response, session-subject change, 401 during refresh, logout/Back, delayed obsolete selection; no cross-strategy flash or retained private screen | Existing session infrastructure reused; no mutation retries introduced |
| A7 | Fresh Strategies and current Backtests images at 820/1024/1440, RU/EN, 200% zoom; real hover/focus/navigation, no document overflow and zero axe violations for tested Strategies states; foundation/journey axe regression | Keyboard/accessibility smoke, not a screen-reader certification; screenshots use disposable data |
| A8 | Slow interrupted collapse/select/disclosure; one Off value across pages; reduced-motion zero duration; responsive controls and no uncaught transition errors; actual Backtests result tabs/charts in journey | Resting-state contrast measured after finite animations finish, without disabling axe rules |
| A9 | Real flag-off SSR reads compare persisted job/spec/trades unchanged, then client restored. Registry scopes updated narrowly; unfinished tags retained | No target-role/default cutover or production claim; no schema migration |

Primary images/observations are under
`roehub-strategies-client-2026-09-12/accepted/strategies/` and
`roehub-strategies-client-2026-09-12/accepted/journey/`.
The directory label identifies the final local verification run only, not owner approval.
Foundation evidence is under `roehub-strategies-client-2026-09-12/final/foundation/`.
Earlier `focused*`, `regression*`, `final` and `complete` captures are retained as
intermediate attempts and are not substitutes for the final passed suite.

## Commands and results

Repository root: `/Users/daniildegtyarev/Projects/roehub.com`.

| Command | Result |
|---|---|
| `pnpm --filter @roehub/platform-web typecheck` | passed |
| `pnpm --filter @roehub/platform-web test` | passed: 9 files, 178 tests |
| `pnpm --filter @roehub/platform-web test strategies.test.ts` | passed: 11 focused tests |
| `pnpm --filter @roehub/platform-web build` | passed; Vite retains the >500 kB chunk warning |
| `.venv/bin/python -m pytest tests/unit/apps/web/test_web_v2_1_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/web/test_security.py` | passed: 55 tests; four existing httpx cookie deprecation warnings |
| `.venv/bin/ruff check apps/web/main/app.py apps/web/main/settings.py apps/web/main/platform_client.py tools/qa/backtests_client_fixture.py tests/unit/apps/web/test_web_v2_1_routes.py` | passed |
| `.venv/bin/pyright apps/web/main/app.py apps/web/main/settings.py apps/web/main/platform_client.py tools/qa/backtests_client_fixture.py tests/unit/apps/web/test_web_v2_1_routes.py` | passed: 0 errors, 0 warnings |
| `pnpm --filter @roehub/platform-web test:e2e foundation.spec.ts` with environment below | passed: 2 tests |
| `pnpm --filter @roehub/platform-web test:e2e strategies.spec.ts` with environment below | passed: 4 tests |
| `pnpm --filter @roehub/platform-web test:e2e journey.spec.ts` with environment below | passed: 1 integrated test |
| `.venv/bin/python -m tools.docs.generate_docs_index --check` | passed: index up to date |
| `git diff --check` | passed; final check repeated after report |

Exact final browser invocations:

```sh
ROEHUB_PROOF_PORT=18580 ROEHUB_PROOF_STATE=.local_artifacts/strategies-accepted ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/accepted pnpm --filter @roehub/platform-web test:e2e strategies.spec.ts
ROEHUB_PROOF_PORT=18680 ROEHUB_PROOF_STATE=.local_artifacts/strategies-journey-accepted ROEHUB_PROOF_STRATEGIES=true ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/accepted pnpm --filter @roehub/platform-web test:e2e journey.spec.ts
ROEHUB_PROOF_PORT=18680 ROEHUB_PROOF_STATE=.local_artifacts/strategies-foundation-final ROEHUB_PROOF_STRATEGIES=true ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/final pnpm --filter @roehub/platform-web test:e2e foundation.spec.ts
```

`e2e/run.mjs` registers Strategies in its default run and enables its fixture flag.
`ROEHUB_PROOF_EVIDENCE` appends the suite name, isolating every new foundation/journey
capture from accepted historical Backtests paths. The obsolete v23 capture is replaced
by `accepted-backtests-reference.png` from the current real client. Historical specimen,
Backtests images and accepted real-price example were not modified.

Ports 18480–18483 belonged to existing processes and were left running. New runs used
18580–18585 and 18680–18685, separate private state and random disposable containers.
The fixture now refuses occupied ports before starting. Fixture teardown removes only
its own processes/containers/state. No review session was replaced. No active preview
is promised after test teardown; the URLs above are local test targets, not a durable
preview or deployment.

## Compatibility assessment

Baseline: existing Backtests client and SSR Strategies in the shared checkout; accepted
visual source `191a9f8169dab0639d8fb3456eb73b060eb1d2c4`. Search covered the selected Web
routes/settings/assets/template, bootstrap parser/contract and direct client consumers,
strategy DTO/dashboard builders, save adapter, fixture and focused tests. Dynamic or
external consumers outside these presentation interfaces were not enumerated.

| Surface / consumer | Change / supported direction | Classification |
|---|---|---|
| Config and HTML presentation | Independent strict default-false flag; both-off SSR unchanged; assets required only if either flag is on | `compatible-change` |
| Bootstrap / current and older server | Optional allowlist; new client with old bootstrap stays Backtests-only. New Strategies must only be enabled with its current built assets | `compatible-change` under coordinated asset/server rollout |
| URL / bookmark / locale/login | Canonical paths and old strategy_id entry retained; classic additive; new/mode remain SSR; repeated params match server | `compatible-change` |
| API, DTO, persistence, command dedupe, server authorization | No producer schemas/endpoints/storage/authz or save-command semantics changed | `none` |
| Cache / client identity | New reads bind subject and strategy, cancel obsolete work, clear through shared session gate | `compatible-change` |
| Rollback | Disable either flag independently and restart Web; persisted strategy/job unchanged in the real fixture | `compatible-change` |

Overall: `compatible-change` within the stated presentation rollout. An old client
bundle is not a supported Strategies implementation; leave Strategies off until the
matching build exists. No organization-role migration or deployment transition is
claimed by the local gate matrix.

## Investigations and residual limitations

- Independent read-only production-risk review found a P2 repeated-query reload loop:
  browser first-value versus server last-value semantics. Fixed, with unit and real
  direct/reload regressions. No further material session/cache/return-context findings.
- Fast reset→market selection reproduced stale `q` resurrection. Merely supplying
  `flushSync` did not work because the provider came from `react-router` without DOM
  flush support. Local installed source confirmed this; `react-router/dom` plus sync
  filter updates passes the unchanged rapid interaction.
- Resize during native document transition produced `Transition was aborted ...
  Viewport size changed`. Early `pagereveal`/`pageswap` promise handling fixes this;
  successful repeated viewport/navigation tests report zero uncaught page errors.
- A journey extension attempted hard deletion after saving and received HTTP 500.
  `strategy_provenance_org_job_fk` references the job with `ON DELETE RESTRICT` in
  `0011_identity_organizations_rbac_audit_v1.sql`. The response alone does not prove
  the exact server exception (private fixture logs were not retained); deletion is
  not fixed or claimed. The existing source-404 recovery is separately demonstrated
  through an explicit controlled error on the real saved journey.
- Earlier test-only failures: getByLabel matching select option text, wrong expected
  invalid-response wording, CSS zero serialized as `0s` rather than `0ms`, and axe
  measuring the .2-opacity intermediate chart fade. Fixed locators/expectations and
  resting-state timing; no accessibility rule or product acceptance was removed.
- Observed HTTP/console errors in the Strategies scenario correspond to intentional
  list 503 and actual absent-resource 404s; foundation separates existing optional SSR
  projection/asset errors. No clean-provider/live readiness inference is made.
- Full CI, production, target roles, provider operations, screen-reader certification,
  create/clone/archive/editor/launch-profile/run/stop/manual execution and general
  Jobs/Data/Models/Artifacts remain outside this unit. No external credentials used.

## Owned paths and preserved foreign work

New feature/test paths:
- `apps/platform-web/src/client-routes.ts`
- `apps/platform-web/src/strategies-api.ts`
- `apps/platform-web/src/strategies-page.tsx`
- `apps/platform-web/src/strategies-i18n.ts`
- `apps/platform-web/src/strategies.test.ts`
- `apps/platform-web/e2e/strategies.spec.ts`
- this report and `roehub-strategies-client-2026-09-12/` images/observations.

Owned edits in expected zones: `app.tsx`, `main.tsx`, `i18n.ts`, `results.tsx`,
`style.css`, Web `app.py`/`settings.py`/`platform_client.py`, `platform_client.html`,
`packages/web-contracts/src/index.ts`, `e2e/run.mjs`, foundation/journey suites,
`playwright.config.ts`, `test_web_v2_1_routes.py`, and the disposable fixture.
Documentation changes are limited to the plan's iteration update, relevant registry
implemented_scope fields, and the Web architecture's new flag/bootstrap/classic entry.
No domain/API implementation paths changed; no outside-expected owned paths.

Foreign changes present before execution: `.codex/AGENTS.md`, architecture README,
Backtests iteration log, functional contract, Web architecture, implementation plan,
and untracked `.codex/agents/generated/roehub-strategies-client-v1/`. They were preserved;
only the explicitly named plan/architecture sections received additive in-scope edits.
No global instructions, completed prompts/ledger or Backtests acceptance were rewritten.

Cold self-review: scope/identity/default gates/rollback and evidence claims checked;
independent security review completed once, finding corrected. Final next action is
user inspection or separately authorized follow-up; no next task or publication starts.

## User-requested UI correction — 2026-09-12

The user rejected the initial screen: three indistinguishable refresh buttons,
raw indicator objects and excessive technical/status copy did not provide a useful
strategy detail. Earlier local test success is not user visual acceptance. This
correction supersedes that presentation; product acceptance remains with the user.

Changed this correction: `apps/platform-web/src/strategies-page.tsx`,
`strategies-i18n.ts`, `style.css`, and `apps/platform-web/e2e/strategies.spec.ts`.
Foreign changes and previous evidence were preserved. No Python, domain API,
persistence, flags or mutation endpoints changed in this correction.

- One page refresh coordinates eligible list/detail/status reads; each read keeps
  its own error, access restriction and retry deadline. A status cooldown does not
  prevent list/detail refresh, and repeated clicks do not bypass the cooldown.
- Instrument, market and timeframe form a compact summary. Indicators retain their
  order; nested parameters render as labelled values, including the actual saved
  variant's `id` field, EMA price source and period. Unknown fields remain readable;
  internal row IDs remain available in subordinate raw specification details.
- State has a short explanation. Compatibility, profile/data details and source
  freshness are subordinate. No performance or trading rules were invented.
- JSON, immutable-snapshot explanations and technical IDs remain disclosed on demand.

Validation of final build:

```sh
pnpm --filter @roehub/platform-web typecheck
pnpm --filter @roehub/platform-web test
pnpm --filter @roehub/platform-web build
ROEHUB_PROOF_PORT=19180 ROEHUB_PROOF_STATE=.local_artifacts/strategies-ui-final ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/ui-final pnpm --filter @roehub/platform-web test:e2e strategies.spec.ts
ROEHUB_PROOF_PORT=19280 ROEHUB_PROOF_STATE=.local_artifacts/strategies-ui-final-journey ROEHUB_PROOF_STRATEGIES=true ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/ui-final pnpm --filter @roehub/platform-web test:e2e journey.spec.ts
.venv/bin/python -m tools.docs.generate_docs_index --check
git diff --check
```

Results: typecheck/build passed; 178 unit tests passed; 4 Strategies browser tests
passed (29.5s); integrated journey passed (28.4s); docs index and whitespace passed.
Vite retains its existing large-chunk warning. Python and foundation checks from the
initial implementation were not repeated for this frontend-only correction.
An intermediate journey run under `ui-fix` was explicitly interrupted after a build
changed its startup asset manifest; it is not counted as passing. Its files remain.

Final evidence: `roehub-strategies-client-2026-09-12/ui-final/strategies/` contains
RU/EN 820/1024/1440 and 200% screenshots, Backtests references, accessibility and
redacted interaction observations. `ui-final/journey/integrated-journey.json` and
`saved-strategy.png` cover the final real configure/save/reload/return/rollback flow.
The saved screenshot catches optional status loading; it proves specification
inspection independently of that read, not a ready status.

A1/A4/A5/A6/A7/A8: four focused browser tests passed, including exactly one refresh,
readable periods, optional-read cooldown without repeated status requests, all
previous error/identity checks, keyboard/filter/scroll, motion and accessibility.
A2/A9: all four gates/classic/new/mode/locale and persisted rollback checks passed.
A3: final real job `2042e0f7-15b6-4579-881f-3f7b55b396fb`, variant
`job_2042e0f796fb__ema_close_w10__risk_none__vh_b39c4279`, strategy
`877f2fe2-d01c-43cf-a80c-5262726e8e7d`; one UI save, duplicate behavior retained,
zero trading starts (see integrated evidence).

Live in-app review uses the existing disposable 18781 API/database with a new owned
Web process on 18880; no existing review fixture was terminated. The preview job
`9de9e617-f14d-4aac-9de0-460d1e5c37a1` was preflighted and created through the real
API, completed by the worker, then saved through the visible Backtests confirmation.
The resulting strategy `3da68f73-2ca2-477e-8d67-b6cf2b437b2b` was opened and reloaded
in the in-app browser with bounded report context. The initial browser date-entry
attempt did not commit its date through the browser mechanic; no job was submitted
from that rejected preflight. The API-created preview uses 2026-03-26–2026-03-29.

Verified preview: http://localhost:18880/strategies/3da68f73-2ca2-477e-8d67-b6cf2b437b2b?from_job=9de9e617-f14d-4aac-9de0-460d1e5c37a1&from_variant=job_9de9e61737a1__ema_close_w10__risk_none__vh_b39c4279

Compatibility: `compatible-change` for browser presentation/refresh interaction;
`none` for API/persistence/auth/config. Rollback: revert only this correction's UI
hunks and rebuild; saved jobs/strategies remain. `local_journey_verified=true`;
`target_role_cutover_ready=false`. Synthetic candles prove the local journey only.

## User-authorized analysis-content continuation — 2026-09-12

Authority: after rejecting both earlier specification-focused presentations, the
user explicitly requested substantive trading/analysis information and instructed
implementation. This expands the earlier read-only slice to research association
and existing trading observations. It does not authorize publication, real trading,
credential changes or migration of mutating controls. No Goal, pack or ledger.

Delivered Overview / Backtest / Execution / Specification views. Overview describes
verified MA-cross evaluator behavior or the saved EMA research signal mapping;
unknown algorithms retain explicit uncertainty. Backtest reads the persisted source
and shows six metrics, test dates, direction/capital/fees/slippage/TP/SL, ECharts
capital/drawdown and paginated trades. The full original report remains reachable.
Execution reads configured profile sizing/limits, paper-account position/equity/P&L/
costs and recent signal observations. Empty/default profiles are not presented as
configured zero limits. Paper accounting is explicitly distinct from exchange P&L.
The source association works on standalone entry and reload without query context.

### Changed boundary and paths

New GET `/api/strategies/{strategy_id}/research-source` returns only strategy ID and
nullable source job/variant. Existing ownership/organization/nondeleted checks run
first; provenance storage filters exact organization + owner + strategy. No source
is 200 with a null pair; unavailable storage is sanitized canonical 503. Source
report reads still enforce Backtests authorization. No schema/index migration,
compatibility-check event, replayed save or provider operation.

Owned additions/changes in this continuation:
- `apps/platform-web/src/strategy-insights{,-api}.tsx` / `.ts` (actual files:
  `strategy-insights.tsx`, `strategy-insights-api.ts`, `strategy-insights.test.ts`),
  Strategies page/API/copy, scoped CSS, and exported existing ECharts `Chart`.
- `apps/platform-web/e2e/strategy-insights.spec.ts`, runner registration and updated
  specification-tab assertions in `strategies.spec.ts`.
- `apps/api/routes/strategies.py`, `apps/api/common/errors.py`,
  `apps/api/wiring/modules/strategy.py`; provenance repository port and both
  Postgres/in-memory adapters under `src/trading/contexts/strategy/`.
- `tests/unit/apps/api/test_strategies_routes.py` and
  `test_strategy_research_source_repository.py`.
- Existing plan, Web architecture, functional registry and this evidence report.

Backend provenance reads are outside the original prompt's expected client-only
zone but necessary for, and covered by, the user's later expanded request. All
foreign modifications identified at the original start remain preserved; nothing
was staged, committed, pushed, branched, stashed or deployed.

Compatibility: `compatible-change` for additive API/port/client content; `none` for
persistence schema, default flags, save semantics, trading commands and credentials.
Rollback: disable the existing Strategies flag or revert this continuation's owned
hunks and rebuild/restart local services. Existing strategies/jobs are preserved.

Independent read-only security review (one reviewer, no recursion) found correct
owner/organization/deleted gating. Two findings were fixed: invalid exception
constructor usage and unhandled storage exceptions. Canonical 503 mapping and
failure-redaction tests now pass. No residual finding from that bounded review.

### Exact final checks

```sh
pnpm --filter @roehub/platform-web typecheck
pnpm --filter @roehub/platform-web test
pnpm --filter @roehub/platform-web build
.venv/bin/python -m pytest tests/unit/apps/api/test_strategies_routes.py tests/unit/apps/api/test_strategy_research_source_repository.py tests/unit/apps/api/test_ui_strategy_dashboard_routes.py -q
ROEHUB_PROOF_PORT=19680 ROEHUB_PROOF_STATE=.local_artifacts/strategy-analysis-final ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/analysis-final pnpm --filter @roehub/platform-web test:e2e strategy-insights.spec.ts
ROEHUB_PROOF_PORT=19780 ROEHUB_PROOF_STATE=.local_artifacts/strategy-analysis-regression ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/analysis-regression pnpm --filter @roehub/platform-web test:e2e strategies.spec.ts
ROEHUB_PROOF_PORT=19880 ROEHUB_PROOF_STATE=.local_artifacts/strategy-analysis-journey ROEHUB_PROOF_STRATEGIES=true ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/analysis-journey pnpm --filter @roehub/platform-web test:e2e journey.spec.ts
.venv/bin/python -m tools.docs.generate_docs_index --check
git diff --check
```

Passed: typecheck/build, 182 unit tests (10 files), 30 focused API/repository tests,
1 analysis browser test (24.1s), 4 Strategies regressions (33.9s), 1 integrated
journey (29.7s), docs index and whitespace. Focused `ruff check` and `pyright` passed
on the seven changed Python files plus the new repository test file (no type errors).
Vite's existing large-chunk warning remains. Earlier Web/foundation checks were not
repeated: Web gate code is unchanged in this continuation; the four-gate regression
and integrated rollback were executed on this build.

Earlier failed attempts are retained, not counted as success: `analysis-a` exposed
an incorrect client requirement for job ID on a variant DTO that only carries the
variant key; fixed with exact variant verification plus server-scoped origin/job
route. `analysis-b` exposed transient ECharts canvas overflow after viewport resize;
fixed with scoped chart geometry containment. `analysis-c` passed before the final
EMA explanation; `analysis-final` verifies the final build.

### Real proof and limitations

`analysis-final/insights/observations.json`: real job
`c46426d6-192b-4293-9746-131f5da94ce8`, variant
`job_c46426d64ce8__ema_close_w10__risk_none__vh_b39c4279`, saved strategy
`40422904-6cf1-49a6-8a6a-c06c645420d2`. Production local auth/API/worker/Postgres;
labelled synthetic candles. The new GET returned the exact persisted association.
Standalone reload, real metrics/conditions/chart/trades, missing profile/account,
wrong-source identity rejection, inaccessible report, RU/EN 820/1024/1440 and 200%
zoom, keyboard tabs and axe (zero violations) passed. Inspection generated zero
mutations and zero page errors. Final screenshots reside under `analysis-final/insights/`.
The existing end-to-end UI save/deduplication/return/rollback is independently green
under `analysis-journey/journey/`.

A1/A2/A4/A5/A6/A7/A8/A9 remain covered by the four Strategies regressions; A3 remains
covered by the integrated save journey. New source scope is additionally covered by
401 browser proof, owner/deleted/missing/error route tests and all-three-identities
SQL/in-memory scope tests. Mocked failures supplement, not replace, the real source,
report and save evidence. Positive exchange execution or paper-account balances were
not manufactured or claimed; the preview has no such observations.

The preserved local review database/API on 18781 remains running. A new owned API
process on 18881 loads the added read using the same disposable fixture environment;
Web18880 now points to it. Existing credentials and sessions were not changed.
Verified preview: http://localhost:18880/strategies/3da68f73-2ca2-477e-8d67-b6cf2b437b2b
The preview source is the prior real local job `9de9e617-f14d-4aac-9de0-460d1e5c37a1`.

Remaining: full history of all tests/clone ancestry, general signal explanations,
live exchange P&L/orders/positions not backed by current projections, migrated editor
and mutating trading controls. Existing classic management remains reachable.
`local_journey_verified=true`; `target_role_cutover_ready=false`. Product visual
acceptance is not inferred from test success and awaits user review.

### Preview recovery — 2026-09-12

The old Web process remained listening, but its disposable Docker databases were no
longer present after Docker stopped. Started Docker Desktop and a separate fixture
with `ROEHUB_PROOF_PORT=19980 ROEHUB_PROOF_STATE=.local_artifacts/strategies-preview-restored ROEHUB_PROOF_STRATEGIES=true .venv/bin/python -m tools.qa.backtests_client_fixture`.
The existing current build was reused (manifest newer than feature source). Preserved
old state files and processes. Created a fresh synthetic-data backtest through real
preflight/job/worker/save APIs: job `ea6b1350-9477-4bf3-8356-090c418b99a2`,
strategy `dcf8c80a-0de2-4f2f-b896-4848e8890c26`. This is newly generated disposable
preview data, not recovery of the deleted database. Verified authenticated rendered
metrics and Backtest tab in the visible in-app browser and retained the tab.
Preview: http://localhost:19980/strategies/dcf8c80a-0de2-4f2f-b896-4848e8890c26
No repository implementation changes or trading commands. Compatibility: none.

### Multi-trade preview data — 2026-09-12

Added opt-in `ROEHUB_PROOF_CANDLES=cycles` to `tools/qa/backtests_client_fixture.py`.
Default historical regression candles are unchanged. New deterministic input has
4320 valid minute candles with slow trends, reversals and a choppy middle regime;
prices are synthetic, not downloaded market history. Production artifact generation,
validation, publication, preflight, worker calculation and strategy save were used.

Live fixture: `ROEHUB_PROOF_PORT=20080 ROEHUB_PROOF_STATE=.local_artifacts/strategies-preview-cycles ROEHUB_PROOF_STRATEGIES=true ROEHUB_PROOF_CANDLES=cycles .venv/bin/python -m tools.qa.backtests_client_fixture`.
Job `0b2628e5-6ff5-4e23-8335-2e6cf418e934`, saved strategy
`54f2d6f4-7cb7-4d7b-9469-437b2bb77362`: 29 trades, win rate 34.48275862068965%,
return -2.3287955348010243%, maximum drawdown 11.57953612066385%.
Verified authenticated in-app browser metrics, equity curve, drawdown switching,
trade rows with positive and negative P&L, and page 2 of the 29-trade table.
Collapsed the library and scrolled the visible browser to the equity chart; kept tab open.
The narrow native browser panel remains responsive; no desktop-width claim is made.

Checks passed: focused `ruff check` and `pyright` on the fixture; pytest
`tests/unit/tools/test_backtests_preview_sessions.py` (3 passed); direct deterministic
OHLC/positive-price/both-direction check and unchanged baseline identity check;
`git diff --check`. No frontend changes, provider operations or trading commands.
Compatibility: compatible-change for opt-in local QA input only. Previous previews
and evidence retained. Preview: http://localhost:20080/strategies/54f2d6f4-7cb7-4d7b-9469-437b2bb77362

### Explicit five-element removal — 2026-09-12

User screenshot comments supersede earlier presentation requirements. Removed trading
logic prose, full-report links, profile/limits section (including Execution duplicate),
classic-management links and persistent strategy-status block. Changes are in
`strategy-insights.tsx` and `strategies-page.tsx`; direct classic routes and bounded
save-origin return navigation remain unchanged. Updated focused browser assertions and
plan correction. Existing 29-trade job, strategy and database retained.

Typecheck, 182 unit tests, production build, docs index and whitespace checks passed.
`strategy-insights.spec.ts` passed (1 test, 22.7s), isolated port 20180/state
`.local_artifacts/strategies-cleanup-proof` and evidence `cleanup-proof`.
In-app browser verified absent selectors and preserved research metrics/chart for
strategy `54f2d6f4-7cb7-4d7b-9469-437b2bb77362`. New standalone Web process on 20090
uses existing API 20081 and current asset manifest; no fixture/database recreation.
Preview: http://localhost:20090/strategies/54f2d6f4-7cb7-4d7b-9469-437b2bb77362

Regression attempts in `cleanup-regression` and `cleanup-regression-final` each passed
3/4 tests. Removing visible status waits exposed unfinished route-handler teardown,
then an intentionally cancelled delayed detail request in the test harness. Teardown
now waits for handlers; delayed route continuation permits cancellation only after
navigation to the explicitly expected second strategy. No application error is hidden.
Compatibility: breaking-change for the explicitly removed UI affordances; API,
storage, authorization and direct classic URL contracts unchanged. User UX acceptance
and target-role cutover are not claimed.
Final `strategies.spec.ts` regression: 4 passed (31.9s), with
`ROEHUB_PROOF_PORT=20480 ROEHUB_PROOF_STATE=.local_artifacts/strategies-cleanup-regression-verified ROEHUB_PROOF_STRATEGIES=true ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/cleanup-regression-verified pnpm --filter @roehub/platform-web test:e2e strategies.spec.ts`.

### List terminology and restrained motion — 2026-09-12

User correction: this is a strategy list, not history. Added RU/EN show/hide-list
labels in `strategies-i18n.ts`, used in `strategies-page.tsx`, and updated browser
assertions. List visibility now updates directly without document snapshots or grid
interpolation. Scoped CSS disables Strategies report snapshot naming. Removed the
outer content animation target so tab changes target the tab panel only. The shared
coordinator supports opt-in quiet opacity (0.85 to 1, half the selected duration,
no translation); Strategies opts in. Existing Backtests defaults and persisted motion
key are unchanged. Off/reduced-motion still produce zero duration.

`typecheck`, `test` (182 passed), `build`, `git diff --check` passed.
Browser suite `strategies.spec.ts`: 4 passed (27.9s), invoked with
`ROEHUB_PROOF_PORT=20580 ROEHUB_PROOF_STATE=.local_artifacts/strategies-motion-proof ROEHUB_PROOF_STRATEGIES=true ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/motion-proof pnpm --filter @roehub/platform-web test:e2e strategies.spec.ts`.
Live Browser exercised repeated list toggles, Backtest/Specification/Overview switches,
slow and Off selections; restored Normal. Tab focus and selected strategy remained
correct. Frame-level 10%-speed inspection unavailable: Browser read-only DOM evaluation
does not expose document.getAnimations. No frame-performance claim is made.
Restarted only the dedicated standalone Web process on the SAME port 20090;
API/database/29-trade strategy retained. Updated page remains open.
Compatibility: compatible-change for opt-in presentation behavior; no API change.

### Animated list correction — 2026-09-12

The user rejected instant list resizing. Replaced it with interruptible live CSS grid
geometry: fixed-width list content clipped inside an animated column, paired with
workspace expansion and gap interpolation. At narrow widths, animate a 1fr/0fr row
instead. The hidden list is inert and aria-hidden while preserving its mounted scroll
state. No screenshot scaling, duplicate document transitions or new preference.
Owned edits: `strategies-page.tsx`, `style.css`, `e2e/strategies.spec.ts`.

Added actual requestAnimationFrame geometry sampling at 1440 and 820 widths, reversing
the transition after 180ms: intermediate geometry observed, final position restored
within 1px, text width stable within 1px, no document overflow, button focus retained.
Evidence: `live-motion/strategies/list-motion-frames.json`. This supersedes the prior
instant-resize implementation; it is not a subjective user acceptance claim.
Typecheck and build passed. `strategies.spec.ts`: 5 passed (27.9s), including new
frame-sampling test, using port 20680, state `.local_artifacts/strategies-live-motion`,
evidence `live-motion` via the existing runner. Live in-app Browser toggles verified;
restarted only standalone Web on the SAME 20090 port, retained data and open tab.
Compatibility: compatible-change (presentation only). No API or trading operations.

### Explicit simulated running-strategy data — 2026-09-12

User requested test data as though a strategy were running. Added local QA module
`tools/qa/strategy_execution_demo.py`, opt-in sidecar
`execution-demo-strategy.txt` inside the private fixture state. The fixture middleware
only overlays an already-successful authenticated dashboard response whose selected
ID matches that file. Output is validated with the production dashboard DTO. This is
synthetic execution observation data, NOT a worker run, persisted fill ledger or proof
of provider execution. No run/profile/order commands were sent.

For current strategy: 13 synthetic signals (6 closed pairs and 1 new entry), position
0.02 BTC at 66000, mark-price assumption 66350, realized gross P&L 35, unrealized P&L 7,
fees 12.76725, equity 10029.23275 from initial 10000. All amounts are USDT except BTC
quantity. UI parses source provenance and prominently labels this as a demo run.
Existing 29 historical backtest trades remain separate and unchanged.

Current Web remains on 20090, new dedicated QA API on 20091 shares original fixture
DB/environment; original supervised API remains untouched. Browser Execution tab shows
account values, all 13 signals and demo label. Tested actual unauthenticated 401,
identity-scoped overlay, signal count, and equity reconciliation through local HTTP.
Typecheck/build, Ruff/Pyright (two changed QA Python files), 182 client unit tests,
3 preview-session tests and whitespace checks passed. Initial direct-API login with
mismatched Origin correctly returned 403; same-origin login passed.
Compatibility: compatible-change, optional display source and local-only fixture data.
No claim of real strategy execution, trading readiness or production delivery.

### Approved operational workspace implementation — 2026-09-12

Authority: the user approved the operational concept (with existing RoeHub styling)
and explicitly requested implementation. This supersedes the research presentation
and previous no-command restriction for local UI implementation/verification. No
exchange execution or publication was authorized or performed.

Implemented: operational header/state/run-stop-restart-delete controls; price chart
with actual projection fill markers and separate signal/manual/SL/TP/trailing causes;
current paper position and observed prices/limits; run-level statistics; expandable
partial fills and trade reasons; events/settings; one shared motion preference.
Backtest content/requests have been removed from this screen. Historical save and
provenance APIs remain intact. List state comes from the observed selector, with
unknown for unobserved rows. Chart zoom survives selected-observation updates.

New direct read projection: PostgreSQL joins are organization/owner/strategy/run scoped,
limited to 5001 rows (5000 visible with partial flag). Position-cycle arithmetic uses
Decimal, separate remaining-position cost basis and full-cycle weighted entry price,
with proportional fee allocation on reversals. Unknown fees/funding withhold net P&L;
known fees remain visible. Initial paper capital comes from the first owned run
accounting record, not an invented constant. Canonical candles are aggregated only
from complete minute buckets. Equity/drawdown are explicitly closed-trade curves.
No real all-run history, live account/FX/funding reconciliation or provider execution
is claimed. The current client covers selected/latest-run data only.

Commands use existing routes and bodyless DELETE 204. New optional `expected_run_id`
on manual requests is checked against the scoped active run before ingress; that
same run object is then used for dedupe/source/intent/accounting. This prevents
same-client-key recovery creating an intent in a replacement run. New client retains
original key, amount, observed reference price and run identity across reload; an
explicit outcome check uses those original values. A read/refresh alone never clears
an unresolved manual command. Known lifecycle/intent outcomes release the lock.
Requests without expected_run_id retain legacy behaviour. Old servers cannot accept
this new guard; deploy the server first if publication is later authorized.

The local simulator is opt-in via the private fixture sidecar, only in the disposable
QA entrypoint. An actual authenticated owned strategy GET precedes simulation. It
provides seven position cycles, multiple reasons and a partial entry. Commands update
only this synthetic local state. Browser proof closes/reopens a position, stops without
closing, resumes and deletes a separate disposable strategy via real DELETE. A lost
response after a simulated fill followed by reload/same-key recovery produces no
second fill. This is NOT a real runner or persisted provider fill ledger.

Owned continuation paths: new `strategy-operations*.{ts,tsx}`, `operation-recovery.ts`,
`strategy-operations.test.ts`, `e2e/strategy-operations.spec.ts`; client Strategies page,
API parser, shell/footer, styles, existing focused suites/runner; API operations DTO,
projection/reader/wiring, manual request guard; new scoped PostgreSQL reader; QA demo
and middleware; focused Python tests; this report and named plan/registry/Web docs.
Outside original library-only touch zone: manual expected-run guard and live-execution
read adapter, justified by the newly authorized operational scope. Existing foreign
`.codex/AGENTS.md`, prompt files and prior documentation edits were preserved. Nothing
staged, committed, pushed, merged, deployed, branched or stashed.

Validation actually executed:

- `pnpm --filter @roehub/platform-web typecheck`: passed.
- `pnpm --filter @roehub/platform-web test`: 185 passed, 11 files.
- `pnpm --filter @roehub/platform-web build`: passed; existing large-chunk warning remains.
- `.venv/bin/python -m pytest tests/unit/apps/api/test_strategy_execution_projection.py tests/unit/apps/api/test_strategies_routes.py tests/unit/apps/web/test_web_v2_1_routes.py tests/unit/apps/web/test_app_routes.py tests/unit/apps/web/test_security.py -q`: 84 passed; four existing httpx cookie deprecation warnings.
- Ruff on changed API DTO/routes/wiring/read adapter/QA/test files: passed.
- Pyright on changed API DTO/routes/wiring/read adapter/QA files: passed.
- Mypy `--follow-imports=silent` on the five new Python projection/DTO/reader/QA modules: passed.
- Real disposable PostgreSQL executed both scoped paper/live SQL branches successfully;
  foreign organization yielded zero rows. This validates SQL/schema/scoping of that
  read, not nonempty production ledger arithmetic; arithmetic has six focused tests.
- `ROEHUB_PROOF_PORT=20880 ROEHUB_PROOF_STATE=.local_artifacts/operations-proof ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/operations-run-guard-proof pnpm --filter @roehub/platform-web test:e2e strategy-operations.spec.ts`: 1 passed (13.4s), commands/partial fills/lost response/reload/run guard/204 deletion/unauthenticated 401, RU/EN 820/1024/1440 screenshots and zero axe violations.
- `ROEHUB_PROOF_PORT=20980 ROEHUB_PROOF_STATE=.local_artifacts/operations-regression ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/operations-regression-v2 pnpm --filter @roehub/platform-web test:e2e strategies.spec.ts`: 5 passed (27.8s), gates/classic/session/errors/late responses/200% zoom and frame-sampled interrupted motion.
- `ROEHUB_PROOF_PORT=21180 ROEHUB_PROOF_STATE=.local_artifacts/operations-save-regression ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/operations-save-regression pnpm --filter @roehub/platform-web test:e2e strategy-insights.spec.ts`: 1 passed (13.1s), real Backtests save, persisted origin API preserved, standalone operational strategy makes no research reads.
- Foundation: 2 passed in the `operations-backtest-regression` fixture/evidence group.
  The combined foundation/journey invocation initially timed out waiting on captured
  animation.finished promises. Kept failure evidence. Changed only the test wait to
  a bounded assertion that all finite animations reach finished; no product animation
  was removed. Exact underlying orphan-promise mechanism remains unconfirmed.
- `ROEHUB_PROOF_PORT=21080 ROEHUB_PROOF_STATE=.local_artifacts/operations-journey-proof ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/operations-journey-proof pnpm --filter @roehub/platform-web test:e2e journey.spec.ts`: 1 passed (22.1s), actual worker/configure/save/flag-off persistence. Historical screenshots unchanged; accepted current reference captured in isolated output.

Earlier failed operations checks found and fixed primary-button contrast and an
unfocusable empty horizontal table. Failure directories remain; successful images
are in `operations-run-guard-proof`, with final zoom-preservation rerun recorded below.

Compatibility: `compatible-change` for existing API consumers (optional operations
DTO and optional manual run guard), `none` for persistence schema and presentation
flag defaults. New-client/old-server manual commands require the new server, as above.
The intentional browser removal of backtest tabs/return links supersedes prior user
acceptance, rather than pretending those UI interactions are unchanged. Rollback:
set Strategies flag false to expose preserved SSR, without recreating strategies/jobs.
No production or target-role claim; `target_role_cutover_ready=false`.

Independent security review (read-only agent) found DELETE204, unknown-outcome replay,
entry averaging and missing initial capital defects; these were fixed. Follow-up found
run-scoped dedupe race; fixed with server expected_run_id and a mismatch409/no-ingress
regression. Residual boundary: simulated commands, bounded selected-run reads, no
provider/cross-run/full-history readiness proof.

Preview remains on http://localhost:20090/strategies/54f2d6f4-7cb7-4d7b-9469-437b2bb77362
with original local data/session retained; only dedicated 20090/20091 processes were
restarted. The supervised original fixture was not terminated.

Final UI rerun: same operations command above with
`ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/operations-final-ui`:
1 passed (12.4s), including retained zoom code, lost-response recovery, real 204 delete,
unauthenticated rejection and six normalized viewport/locale images. In-app Browser
also verified the current original strategy with enabled Stop/Close controls, seven
trades, current position, rendered candles/fill markers and no research tab. Subsequent
small visual adjustment includes SL/TP in the chart extent and removes label outline;
demo candles stop at completed intervals rather than rendering future bars.
Docs index and whitespace checks passed. No stronger production/full-history verdict.

### 2026-09-12 — Strategies chart parity correction

User requested the Backtests Price & trades zoom and marker conventions. Shared
`apps/platform-web/src/price-chart-conventions.ts` now supplies candle lookup,
inside zoom (Ctrl + wheel, filtered visible bars, last 150 bars initially) and
blue entry / gold exit triangles to both charts. Strategies uses category candles
and automatic price scaling; SL/TP annotations no longer force the price extent.
Execution reasons and individual partial fills remain in tooltips. Marker visibility
and per-view zoom survive chart reconstruction. These are execution markers, not
invented unexecuted signals.

Touched this correction: `src/price-chart.tsx`, `src/strategy-operations-chart.tsx`,
new `src/price-chart-conventions.ts`, new `src/price-chart-conventions.test.ts`,
`e2e/strategy-operations.spec.ts` under `apps/platform-web`, and this report.
Existing unrelated changes preserved; no API/domain edits or trading commands on
the user's preview. Compatibility: `compatible-change` (presentation).

Validation: platform-web typecheck and build passed (existing bundle-size warning);
platform-web test: 188 passed in 12 files. Exact browser invocation:
`ROEHUB_PROOF_PORT=20880 ROEHUB_PROOF_STATE=.local_artifacts/operations-chart-proof ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/chart-parity pnpm --filter @roehub/platform-web test:e2e strategy-operations.spec.ts`
passed (1 test, 14.7 seconds total). Includes marker toggle, Ctrl-wheel exercise,
RU/EN 820/1024/1440 layouts, axe smoke and disposable operational regression.
Fresh images are in `chart-parity/`; historical assets untouched. In-app Browser
on the original preview verified slider narrowing from the full series to 14:02+
with automatic price rescaling, marker toggle without range reset, and restored
full range with markers enabled. The current Backtests reference was inspected
in the same browser before changes. `git diff --check` passed. Only dedicated Web
20090 was restarted after build; session/data and API processes retained.
Preview: http://localhost:20090/strategies/54f2d6f4-7cb7-4d7b-9469-437b2bb77362
No production/provider or target-role readiness claim.

### 2026-09-12 — User corrections: levels, controls, compact executions

Implemented independent entry/SL/TP toggles, renamed entry price, grouped Start/Stop
and Delete directly in the header, and removed the Restart UI. Existing server
commands and pending-command recovery remain backward compatible. Compact trades
override the generic first-column 44% rule; dates are YYYY-MM-DD HH:mm UTC. Entry
quantity and actual entry notional have explicit base/quote units (BTC/USDT in
this preview; no false USD conversion). Expanded executions show their individual
price, reference signal/request price and adverse slippage percentage (buy: fill
minus reference; sell: reference minus fill, divided by reference).

Added optional nullable `reference_price` to the execution DTO/parser. The owned
reader selects stored signal price, falling back to stored paper-order reference
for paper/manual orders. Missing references remain unknown. Synthetic fixture
examples explicitly include reference prices. No new storage or provider operation.
Compatibility: compatible-change for existing dashboard consumers (additive optional
field); presentation changes explicitly requested by the user. Rollback: revert
these owned field/UI changes, without data migration. Real exchange execution and
live manual reference capture remain outside this proof; absent prices show a dash.

Changed paths: platform-web strategy-operations.tsx, strategy-operations-chart.tsx,
strategy-operations-api.ts, strategy-operations-i18n.ts, style.css and operational
e2e; apps/api/dto/strategy_operations.py; strategy_execution_projection.py;
Postgres strategy_operation_reader.py; tools/qa/strategy_execution_demo.py; focused
projection tests; this report. Foreign changes preserved, no Git publication.

Checks: platform-web typecheck/build passed; 188 unit tests passed; pytest
`tests/unit/apps/api/test_strategy_execution_projection.py -q`: 7 passed. Ruff on
five changed Python/test files passed after fixing one SQL line-length issue;
mypy on the four changed Python implementation files with --follow-imports=silent
passed. Stored reference columns checked directly against disposable PostgreSQL.
Browser command: same isolated 20880 operations-chart-proof fixture as above, with
ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/compact-controls
and `pnpm --filter @roehub/platform-web test:e2e strategy-operations.spec.ts`:
1 passed (14.3s), independent toggles, no Restart, first-column width, numeric dates,
slippage disclosure, RU/EN responsive/axe and command regression. Final small label
changes (Entry/Exit and fill units) were typechecked/built and inspected in Browser.
In-app browser verified original preview with nonzero demo slippage 0.0185%/0.0122%,
compact table and explicit BTC/USDT. Dedicated 20090/20091 restarted with original
session/data retained. git diff --check passed. No production readiness claim.

### 2026-09-13 — Compact chart controls, status and event attention

User-selected changes: compact Trades/Entry/SL/TP strip with full accessible names;
state badge beside lifecycle actions, green running state in library; one collapsible
filter panel including search/market/timeframe/state; removed Settings tab and Current
run caption. Technical metadata/JSON disclosure remains below the operational panels.
Missing/stale runtime states filter as unknown, with no per-row reads. Unread event
count tracks loaded fills/signals and clears on viewing Events. Read markers are
subject/strategy scoped in the existing private query cache, cleared with that cache;
this is browser-session attention, not server-persisted cross-device read receipts.

Owned files this correction: strategies-page.tsx, strategies-i18n.ts,
strategy-operations.tsx, strategy-operations-chart.tsx, style.css and
strategy-operations.spec.ts under apps/platform-web, plus this report. No backend,
commands, storage schema or publication changes. Compatibility: compatible-change
for presentation/query filters and private ephemeral cache; rollback is code-only.
Foreign changes preserved. Current preview uses dedicated Web :20110 against original
:20101 API, same strategy/session/data; original supervised fixture not stopped.

Validation: 188 client unit tests passed; typecheck/build passed (bundle-size warning).
Initial e2e exposed filter accessible-name lookup; explicit aria-label fixed it.
An immediate 20880 retry was blocked by occupied fixture port. Final exact command:
`ROEHUB_PROOF_PORT=20980 ROEHUB_PROOF_STATE=.local_artifacts/operations-filters-proof ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/status-filters-sep13-final pnpm --filter @roehub/platform-web test:e2e strategy-operations.spec.ts`
passed: 1 test, 14.0s. Includes unread clearing, absent Settings tab, state filtering,
running-row style, independent chart toggles, operational regression and RU/EN
820/1024/1440 axe/overflow checks. In-app Browser observed same changes and Events
read transition on the actual user's synthetic preview. git diff --check passed.
Preview: http://localhost:20110/strategies/c5c83f0e-79c7-464a-948f-e7e2b0bbff66

### 2026-09-13 — Raise work panels and consolidate navigation

Removed visible duplicate Strategies titles from topbar/workspace; retained hidden
h1/skip target for accessibility. Removed the page-local motion preference control,
retaining its existing persisted behavior. Moved list collapse and refresh to compact
icon controls in the strategy header (also available for empty/error detail). Removed
reserved top toolbar space. Chart selectors and Trades/Events now share one row above
the chart. Existing chart and table selection semantics remain independent.

Touched: app.tsx, strategies-page.tsx, strategy-operations.tsx, style.css and this report.
Compatibility: compatible-change, requested presentation only. No APIs/storage or
commands changed. Rollback code-only; foreign edits preserved. Current preview still
http://localhost:20110/strategies/c5c83f0e-79c7-464a-948f-e7e2b0bbff66 with same data.
Typecheck/build passed; 188 tests passed. Browser regression:
`ROEHUB_PROOF_PORT=21080 ROEHUB_PROOF_STATE=.local_artifacts/operations-compact-header-proof ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/compact-header-sep13 pnpm --filter @roehub/platform-web test:e2e strategy-operations.spec.ts`
1 passed (15.1s), including six responsive locale/axe captures. In-app Browser screenshot
verified panels directly below topbar, header icons and combined navigation row. Final
small missing-detail toolbar fallback typechecked and rebuilt. git diff --check passed.

### 2026-09-13 — Right library and mutually exclusive workspace tabs

Moved strategy workspace before library in DOM and grid: main panel left, library
right; narrow layouts follow the same reading order. Updated collapse geometry and
sidebar icon. Replaced the independent selectors with one keyboard-operable five-tab
list. Price/Equity/Drawdown show charts, Trades/Events replace the chart area with
full-width tables. Hidden chart preserves its component state; only one panel is
exposed at a time. Header commands and technical disclosure remain available.

Owned changes: strategies-page.tsx, strategy-operations.tsx, style.css,
strategy-operations.spec.ts and this report. Compatibility: compatible-change,
explicitly requested presentation behavior; no API or mutation changes. Rollback
code-only; foreign work preserved. Typecheck/build and git diff --check passed.
Exact isolated browser proof:
`ROEHUB_PROOF_PORT=21180 ROEHUB_PROOF_STATE=.local_artifacts/operations-unified-tabs-proof ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/unified-tabs-sep13 pnpm --filter @roehub/platform-web test:e2e strategy-operations.spec.ts`
1 passed (13.8s), including mutually exclusive chart/table assertions, existing
operational flow, RU/EN responsive screenshots and axe checks. In-app Browser
confirmed right-hand library and Trades replacing chart on the existing :20110
preview; same strategy/session/data retained. No provider or production claim.

### 2026-09-13 — Position-adjacent lifecycle controls and workspace height

Moved Start/Stop/Delete beneath Open/Close position in chart views, with a footer
continuation on Trades/Events so management remains reachable there. Header order
is state, Refresh, library toggle at the right edge. Desktop workspace/chart height
uses available viewport space; narrow layout sizing remains unchanged. Owned paths:
strategy-operations.tsx, strategies-page.tsx, style.css and this report. Presentation
compatible-change; command eligibility/recovery unchanged, no actual preview trades
or publication. Typecheck/build passed. Isolated browser command using
ROEHUB_PROOF_PORT=21280, ROEHUB_PROOF_STATE=.local_artifacts/operations-actions-proof,
ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/actions-height-sep13
and `pnpm --filter @roehub/platform-web test:e2e strategy-operations.spec.ts` passed
(1 test, 12.8s), including command regression and six locale/width axe/overflow checks.
In-app browser verified controls with loaded original preview data. Final height
adjusted down 80px after visual inspection and rebuilt; :20110 restarted with same
API/data/session. git diff --check passed. No provider/production readiness claim.

### 2026-09-13 — Stable height across all five tabs

Added a shared bounded operations-stage containing the active chart/table and its
controls. Charts fill remaining space; long Events/Trades scroll internally with
overscroll containment. Narrow stacked chart/position content also scrolls inside
the common viewport. Technical disclosure stays outside at a stable offset.
Owned changes: strategy-operations.tsx, style.css, strategy-operations.spec.ts and
this report. Presentation compatible-change; no data/API/commands changed.
Typecheck/build passed; git diff --check passed. Isolated browser invocation:
`ROEHUB_PROOF_PORT=21380 ROEHUB_PROOF_STATE=.local_artifacts/operations-height-proof ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/equal-height-sep13 pnpm --filter @roehub/platform-web test:e2e strategy-operations.spec.ts`
passed (1 test, 13.9s). New regression compares document height, window scroll and
technical disclosure top across all five tabs; all identical. Existing responsive
RU/EN axe/command regression also passed. In-app Browser verified bounded Events
on original :20110 preview with original session/data retained. No publication.

### 2026-09-13 — shared operational expansion and price-only lifecycle
- Current user correction: Start/Stop/Delete belongs only to Price & executions; every operational view shares Backtests-style expansion.
- Updated strategy-operations.tsx, expandable-overview.tsx, style.css and strategy-operations.spec.ts. Reused mounted expansion, background inertness, Escape and focus restoration; nested native command dialogs retain their own keyboard handling.
- Normal stage height remains identical across all five views. Expanded tables and their containers grow naturally; only the outer fullscreen surface scrolls vertically when needed.
- Compatibility: compatible-change (presentation); API, persistence and command eligibility unchanged.
- Verification: pnpm --filter @roehub/platform-web typecheck and build passed. Focused e2e at ROEHUB_PROOF_PORT=21580, ROEHUB_PROOF_STATE=.local_artifacts/operations-expand-proof2 passed (1 test, 15.3s total). Includes all-five-view normal geometry, expansion/collapse focus, absent lifecycle controls outside price, no nested table vertical overflow, existing demo commands and RU/EN responsive axe checks. Screenshots: roehub-strategies-client-2026-09-12/expand-sep13.
- First regression run exposed a 10px scroll shift, corrected by removing extra toolbar spacing; rerun passed. An intermediate fixture attempt could not bind occupied port 21480; final proof used 21580.
- Updated dedicated local Web20110 and inspected expanded Events in the user's in-app browser. Local synthetic-demo proof only; no provider or production changes.

### 2026-09-13 — aligned event/trade tables
- User correction implemented in strategy-operations.tsx, strategy-operations-i18n.ts and style.css: one chronological event table (signals and fills retained), no duplicate empty header or in-journal demo caption, type/reason filters, shared fixed-height filter row and identical table origins. Added 12px above tabs while preserving total normal stage footprint.
- Compatibility: compatible-change, local presentation/filtering only; no API or persistence change.
- typecheck and build passed. Focused strategy-operations.spec.ts passed (1 test, 17.3s total), port21680, state .local_artifacts/operations-events-proof. Added exact X/Y table-origin comparisons in normal/fullscreen modes, filter intersections (6 exits, 1 manual exit, 14 restored events), single-header/caption assertions. Existing five-view geometry, command, responsive and axe checks remain green.
- Screenshots in roehub-strategies-client-2026-09-12/events-sep13. Dedicated Web20110 restarted and Events visually checked in the user browser. Synthetic local proof only.

### 2026-09-13 — synchronized operational motion
- strategy-operations.tsx now places the shared quiet motion target on the stable stage for every tab, replacing the table-only target. strategy-operations-chart.tsx draws in a layout effect so canvas and selected tab commit before paint. Tab color transitions use the same half-duration and easing as the quiet content fade, respecting existing reduced-motion/off preferences.
- Current position and its controls render only on Price & executions; Equity/Drawdown use the full chart width. Styles scoped to operations.
- Compatibility: compatible-change presentation only. No API/command contract changes.
- typecheck/build passed. strategy-operations.spec.ts passed (1 test, 17.5s total; proof port21780; .local_artifacts/operations-motion-proof), including added position absence assertions, existing all-tab stable geometry, fullscreen and filter checks, commands and responsive axe checks. Evidence: roehub-strategies-client-2026-09-12/motion-sep13.
- Dedicated Web20110 refreshed; capital tab inspected in the in-app browser without position controls. Local synthetic-data evidence only.

### 2026-09-13 — benchmark, pagination and smoother transitions
- Added operations-benchmark.ts and tests; Equity overlays optional dashed Buy & Hold against the strategy area series. Benchmark uses the first available equity sample's capital and the latest candle at/before that sample, bounded to the available equity interval. The UI explicitly says available-curve start and excludes fees; missing starting price yields no invented comparison.
- Journal defaults to five rows, supports 5/10/25/50 and previous/next pages; filters reset the page. Counts moved below the table, repeated visible Events label removed. Tab/page changes reset internal table scroll.
- Smooth stage fade and tab colors use the full existing motion duration (normal320ms), preserving reduced-motion behavior.
- Changed strategy-operations.tsx, strategy-operations-chart.tsx, strategy-operations-i18n.ts, style.css, motion.tsx and focused e2e. Compatibility: compatible-change presentation; API/persistence unchanged.
- typecheck/build passed; operations-benchmark.test.ts: 2 passed. Final e2e at port21980/.local_artifacts/operations-pagination-proof2 passed (1 test,17.4s), including pagination/filtering, table origins and no default vertical overflow. Initial run found an ambiguous page-size accessible label; explicit aria-label fixed it before the successful run.
- Evidence: roehub-strategies-client-2026-09-12/pagination-sep13. Dedicated Web20110 updated; Buy & Hold lines and journal paging inspected in the in-app browser. Synthetic local evidence only.

### 2026-09-13 — static table controls and compact pagination
- Both operational tables now share TablePagination (5/10/25/50), aligned fixed-width counters and a common fixed footer. In normal mode only the table body container scrolls; filters/statistics and pagination remain outside it. Fullscreen natural-height tables preserved.
- Reduced selector height to28px with pill radius; fullscreen control is circular. Removed requested Buy & Hold caption.
- Changed strategy-operations.tsx, strategy-operations-chart.tsx, style.css and focused e2e. Compatibility: compatible-change presentation only.
- typecheck/build and git diff --check passed. Final e2e port22280/.local_artifacts/operations-table-proof3 passed (1 test,16.1s), including 10-row no-overflow assertion and fixed filter/footer coordinates after scrolling. An intermediate run failed on canvas after a concurrent rebuild changed asset hashes; final run used the completed stable build.
- Evidence: roehub-strategies-client-2026-09-12/table-sep13. Web20110 updated and the 10-row journal visually inspected. Local synthetic proof only.

### 2026-09-13 — stable badge, chart dates and control polish
- Reserved a fixed badge slot on the Events tab; count caps visually at99+ while accessible text retains full count. Removed chart footer legend. Axis/tooltip dates use YYYY-MM-DD HH:mm UTC instead of localized month names.
- Native selects retain semantics with a consistently centered SVG chevron and10px right inset. Command dialog uses20px padding,16px content spacing and8px action gaps.
- Files: strategy-operations.tsx, strategy-operations-chart.tsx, style.css, strategy-operations.spec.ts. Compatibility: compatible-change presentation, no API/persistence impact.
- typecheck/build passed; focused e2e port22380/.local_artifacts/operations-polish-proof passed (1 test,17.6s), including Events width invariant across unread/read transition and existing layout/command/accessibility checks. Screenshots: roehub-strategies-client-2026-09-12/polish-sep13.
- Web20110 refreshed; modal visually inspected and cancelled without sending command in user's browser. Local synthetic proof only.

### 2026-09-13 — candle timeframe and floating unread badge
- Added compact chart timeframe select modeled on Backtests. Operations dashboard currently returns source-interval candles, so supported multiples are aggregated locally in UTC OHLC buckets; finer intervals are not fabricated. Current15m strategy offers15m/30m/1h/4h/1d. Per-timeframe zoom retained and execution markers use selected interval bounds.
- Restored ordinary Events button padding and moved count above its right edge. Pointer-focused table selects suppress the extra outline; keyboard focus remains visible.
- Changed strategy-operations.tsx, strategy-operations-chart.tsx, style.css, new operations-timeframe.ts/test and focused e2e. Compatibility: compatible-change presentation only; no server schema, command or persistence changes.
- typecheck/build passed; OHLC rollup unit test passed. e2e port22480/.local_artifacts/operations-timeframe-proof passed (1 test,18.2s), including1h selection/fullscreen retention/15m return and existing geometry/pagination/commands/axe proof.
- Evidence: roehub-strategies-client-2026-09-12/timeframe-sep13. Dedicated Web20110 updated;1h candles and floating14 badge visually checked in user browser. Local synthetic proof only.

### 2026-09-13 — date-only axes and chart control focus
- Moved timeframe after the chart toggles. Bottom labels on price/equity/drawdown now show YYYY-MM-DD; hover details retain precise time. Pointer focus on timeframe and fullscreen suppresses the extra outline while keyboard focus remains available.
- Changed apps/platform-web/src/strategy-operations-chart.tsx, strategy-operations.tsx and style.css. Compatibility: compatible-change presentation only; no API or persistence impact.
- pnpm --filter @roehub/platform-web typecheck and build passed (existing bundle-size warning). Focused test:e2e strategy-operations.spec.ts passed, 1 test,18.5s total, port22580, state .local_artifacts/operations-axis-proof. Evidence: roehub-strategies-client-2026-09-12/axis-sep13.
- Updated Web20110. In-app browser visually confirmed timeframe after toggles, dates without time in normal/fullscreen views and no extra fullscreen outline after click. Read-only computed-style inspection timed out; visual verification used screenshots instead. Returned preview to normal view. Local synthetic proof only.

### 2026-09-13 — timeframe chevron and technical disclosure
- style.css: timeframe now shares the table selectors' SVG chevron, centered with10px edge inset and30px reserved text padding.
- motion.tsx: operations technical details animate measured height on opening/closing using shared motion duration/easing (normal320ms), support reversal, retain native semantics and honor reduced motion. Other disclosures retain their existing behavior.
- Compatibility: compatible-change presentation only. No API/persistence changes.
- pnpm --filter @roehub/platform-web typecheck, test motion.test.tsx (4 passed), build passed. Focused test:e2e strategy-operations.spec.ts passed (1 test,18.2s), port22680, evidence disclosure-sep13. Existing bundle-size warning only.
- Updated Web20110; in-app browser visually checked chevron and exercised disclosure opening/closing. Browser read-only animation introspection was unavailable (getAnimations not exposed); no slow-motion verification claim. Local synthetic proof only.

### 2026-09-16 — icon-only chart display menu
- Accepted user reference: one icon trigger and popup with independent light checkmarks. Replaced four inline controls with ChartDisplayMenu; timeframe remains adjacent. No visible trigger caption, accessible name and tooltip retained. Full labels inside popup; multi-selection stays open; outside click, focus departure and Escape dismiss; arrows/Home/End navigate native checkboxes.
- Files: apps/platform-web/src/chart-display-menu.tsx, strategy-operations-chart.tsx, style.css; e2e/strategy-operations.spec.ts updated for new controls and Escape/focus checks.
- Compatibility: compatible-change UI presentation and interaction, no API/persistence changes.
- typecheck/build passed (existing bundle-size warning); focused strategy-operations.spec.ts passed (1 test,29.1s total). Screenshot inspected: roehub-strategies-client-2026-09-12/menu-sep16/chart-display-menu.png. Local synthetic proof only.
- Local preview restarted at20110 with new disposable state .local_artifacts/strategies-preview-sep16 and seeded demo strategy. No production publication.

### 2026-09-16 — benchmark toggle, toolbar alignment and metadata footer
- Buy & Hold is now an aria-pressed button with show/hide eye icon and active styling. Added12px below overview toolbar and removed mini-toolbar padding/border to align left edges.
- Moved instrument/environment and freshness metadata to right-aligned footer alongside technical details; narrow viewport stacks footer content.
- Changed strategy-operations-chart.tsx, strategy-operations.tsx, style.css and focused e2e. Compatible-change UI only.
- typecheck/build passed; focused e2e passed including new benchmark pressed-state checks and existing geometry/responsive/axe coverage. Screenshot footer-sep16/equity-controls-footer.png inspected. Existing bundle-size warning only.
- Refreshed cached build asset URLs in local dist for the running20110 preview, preserving supervised demo service/database lifetime. Local synthetic proof only; no publication.

### 2026-09-16 — sidebar-style tab indicator and fullscreen focus
- Scoped style.css changes: operational tabs use transparent borders with a violet curved left edge and existing surface tint for active state. Removed full focus outline from fullscreen control, including programmatic focus return; keyboard focus uses icon/background emphasis. Tabs use underline for keyboard focus.
- Compatible-change presentation only. Build and git diff --check passed; focused strategy-operations.spec.ts passed (1 test,18.4s), including fullscreen focus return, keyboard and axe checks. Screenshot tabs-sep16/equity-controls-footer.png inspected. Existing bundle-size warning only.
- Local20110 cached asset URLs refreshed without restarting demo data. Local synthetic proof only.

### 2026-09-16 — visible active tab surface
- style.css: strengthened active operational tab fill to #252c34 against the panel background, retaining the accepted curved violet left border. Selected hover keeps the same fill.
- Build and git diff --check passed; focused browser regression passed (1 test,18.2s). Inspected active-sep16/equity-controls-footer.png. Refreshed local20110 asset aliases; presentation-only compatible change, no publication.

### 2026-09-16 — fill-only active tabs
- User superseded the left-edge indicator: style.css now keeps transparent borders on all operational tabs and only the active surface fill. Other controls unchanged.
- Build and git diff --check passed; focused browser regression passed (1 test,18.7s). Inspected fill-sep16/equity-controls-footer.png. Local20110 asset aliases refreshed. Compatible presentation change, local synthetic proof only.

### 2026-09-16 — full-row options and matching timeframe popup
- chart-display-menu.tsx: full-width button rows own pointer activation and checkbox/radio semantics, avoiding label blur/default-action ordering. Text and empty row space activate options. Multi-select stays open; single timeframe selection closes and returns focus. Shared popup style, arrows/Home/End and Escape retained.
- strategy-operations-chart.tsx replaces native timeframe select with shared popup. style.css updates row/checkmark styles. Existing timeframe aggregation unchanged.
- Focused e2e now clicks row whitespace and text, checks resulting states, selects1h/15m and verifies fullscreen retention. Passed (1 test,18.5s); typecheck/build and git diff --check passed. Inspected popup-sep16/chart-display-menu.png. Existing bundle-size warning only.
- Refreshed local20110 assets. Compatible-change UI, no API/data changes. Local synthetic proof only.

### 2026-09-16 — compact chart popups
- style.css: shared display/timeframe popups use content-based width instead of220px;4px outer padding,8px row inset/gap and32px row minimum height. Full-row activation preserved.
- Updated e2e whitespace click to follow actual row width. Build and git diff --check passed; focused browser regression passed (1 test,18.6s). Inspected compact-popup-sep16/chart-display-menu.png. Refreshed local20110 assets; compatible presentation change, local synthetic proof only.

### 2026-09-16 — Direction popup
- Replaced the trade direction native select with the existing compact ChartDisplayMenu radio popup in `apps/platform-web/src/strategy-operations.tsx`; scoped trigger styling and visible toolbar overflow in `apps/platform-web/src/style.css` prevent popup clipping.
- `pnpm --filter @roehub/platform-web build` and `pnpm --filter @roehub/platform-web typecheck` passed (existing bundle-size warning).
- Live local browser at 1480x968: visually inspected content-width popup with three full-row options. Selecting Short closed the menu and filtered demo rows from 7 to 0; selecting Long / Short restored 7 rows. Refreshed preview assets with a revision query to avoid stale cached assets. No production delivery claimed.

### 2026-09-16 — Event filters and drawdown alignment
- Event type and reason now reuse the content-width radio popup in `strategy-operations.tsx`; updated existing E2E interactions to match the new controls.
- `strategy-operations-chart.tsx` reserves the same 28px toolbar row for drawdown, with a scoped spacer in `style.css`. Shared ECharts grid offsets remain unchanged.
- Live browser at 1480x968: equity and drawdown chart rectangles both x=152.59, y=279.80, width=1010.41, height=538; price shares x/y/height and remains narrower for its position sidebar. Visually checked both event menus and drawdown. Exit + Manual returned one matching SELL event; reset both filters afterward.
- Build and typecheck passed; existing bundle-size warning. `ROEHUB_PROOF_PORT=23380 ROEHUB_PROOF_STATE=.local_artifacts/operations-event-menus-proof ROEHUB_PROOF_EVIDENCE=.codex/delivery/evidence/roehub-strategies-client-2026-09-12/event-menus-sep16 pnpm --filter @roehub/platform-web test:e2e strategy-operations.spec.ts`: 1 passed (19.3s). Local proof only.

### 2026-09-16 — Event multiselect and shared table headers
- Added explicit multiple mode to `chart-display-menu.tsx`; event filters in `strategy-operations.tsx` use checkbox groups that stay open, independently selected arrays, OR within each group and AND between groups. All toggles select/clear a group; trigger shows selected count. Single-choice timeframe/direction remain unchanged.
- Unified top-level trades/events header height, padding, line-height and font size in `style.css`. Updated existing E2E filter interactions.
- Live browser: selected Entry + Exit and Manual + Take profit together; observed matching BUY/SELL rows for both reasons, popup remained open. Reset filters to all. Both headers measured 30.5px including collapsed border, padding 6px 8px, line-height 18px.
- Typecheck and build passed (existing bundle-size warning). Operations E2E passed: 1 test, 19.4s; proof directory `roehub-strategies-client-2026-09-12/multiselect-sep16`. Local preview refreshed, no publication.

### 2026-09-16 — Shared control row axis
- Scoped `style.css` change: all operations chart toolbars, including drawdown spacer, use the table filter inset (12px) and 30px row height.
- Build passed with existing bundle-size warning. Live 1480x968 browser measurements: first controls on price, equity, trades and events all x=164.59375, y=264.796875, height=28px. Refreshed local preview with asset revision query. No production publication.

### 2026-09-16 — Library filter menus
- Replaced native market/timeframe/state selects in `strategies-page.tsx` with shared compact checkbox popups. Multiple values use comma-separated existing query keys; previous singleton URLs still match. Empty parameter means all; `none` means empty selection. Compatibility: compatible-change for existing single-value filter URLs; no API changes.
- Added scoped library trigger/popup CSS, opening above the trigger to avoid clipping by the library scroll container. Existing operations E2E now exercises checkbox state selection.
- Build/typecheck passed; operations E2E 1 passed (19.5s), before final popup-position-only CSS fix. Final CSS rebuilt and inspected in live 1480x968 browser: all seven state rows visible, Running + Stopped selectable together, market and timeframe option menus present. Reset to all after verification. Local preview only.

### 2026-09-17 — Backtest library icon actions
- `apps/platform-web/src/library.tsx`: New backtest, Refresh and Filters now have icon-only content, localized titles and retained accessible names. Refresh gets an opt-in iconOnly prop, preserving text in other uses and disabled/cooldown behavior.
- `style.css`: scoped 28x28 circular library action controls.
- Build and typecheck passed (existing bundle-size warning). Live /backtests browser screenshot checked; all three actions measured 28x28, empty visible text, correct Russian title and aria-label. Local preview refreshed, no publication.

### 2026-09-17 — New backtest placement
- Moved New backtest link in `library.tsx` beside Jobs and count, outside right-aligned actions. Scoped CSS gives the link the same default/hover background and pointer cursor as the adjacent icon buttons.
- Build/typecheck passed (existing bundle warning). Live browser screenshot verified plus next to heading, refresh/filter right aligned. All three computed default backgrounds match rgb(11,14,17). Existing click handler and href preserved. Local preview updated only.

### 2026-09-17 — Backtest anchored filters
- Replaced modal filter dialog in library.tsx with an anchored, nonmodal compact popup. State/risk use the shared selection component; query filtering and page-size validation retained. Outside click, focus departure and Escape close the popup, without focus trapping. Removed modal-only controls.
- Scoped style.css removes trigger focus outline/box shadow; expanded/keyboard focus use icon color and background instead.
- Build/typecheck passed (existing bundle warning). Live browser visually checked popup under trigger with no backdrop; selected No TP/SL and restored All risk modes, closed with Escape. Trigger reported expanded=false, outline=none; no open dialog. Local preview only.

### 2026-09-17 — Backtest compact pagination
- Removed projection Refresh control from library.tsx. Moved page-size selection from filters popup into right-aligned compact pagination with round icon controls and current returned-item count. Retained cursor semantics: double-left returns to first page, right follows next_cursor; no invented total/page count.
- style.css reuses compact 28px controls and select chevron. Build/typecheck passed (existing bundle warning). Live browser verified projection has zero buttons, compact pagination renders, selection 25 updates URL to /backtests?limit=25; restored 50. Actual cursor traversal untested because fixture list is empty.

### 2026-09-17 — Consistent control focus
- Shared style.css rule replaces violet focus outlines/box shadows for platform buttons, native selects, button links and disclosure summaries with a filled focus-visible background. Native selects can match focus-visible on mouse activation, so suppression covers focus directly. Text inputs remain unchanged; forced-colors mode retains system Highlight outline.
- Build passed with existing bundle warning. Live browser: clicked page-size select then Escape, verified focused=true, outline-style none, box-shadow none. Keyboard Tab verified focus-visible=true and rgb(48,57,68) background. Filter trigger verified no outline/shadow after activation; closed afterward. Local preview refreshed.

### 2026-09-17 — Jobs history/new tabs
- `library.tsx` now presents History and New backtest tab panels when embedded configuration is available. Existing list, filters, pagination and projection remain in History. Builder contents remain in New; plus opens that tab. Panels stay mounted to preserve draft and history state.
- `backtests-page.tsx` replaces disclosure toggle with tab-controlled form visibility; existing submit routing and draft lifecycle retained. Focus restoration targets history/new tabs. `style.css` uses strategy-style filled selected tabs without borders.
- Build and typecheck passed (existing bundle-size warning). Live 1480x968 browser: inspected both tabs, keyboard ArrowRight selected/focused New, entered a temporary draft label, switched History/New and confirmed value retained, then cleared test label and returned to History. No job submitted and production delivery not claimed.

### 2026-09-17 — New/history order and compact name
- library.tsx: New tab precedes History; keyboard Home/End/arrows follow new order. Refresh/filter actions render only in History; switching to New resets open filter state.
- builder.tsx/style.css: removed draft notice, visually hid associated optional-name label, added localized Name placeholder hidden on focus. Metadata row uses 8px vertical padding; name field spans two grid columns with one-column fallback.
- Build/typecheck passed (existing bundle warning). Live 1480x968 browser: metadata row 50px, name input 412.82px wide, exact right edge 673.414px matches market-type input; placeholder becomes transparent on focus. New has no filter controls; History shows refresh and filters. Form content otherwise retained. Local only.

### 2026-09-17 — Remove remaining violet focus outlines
- Removed base violet focus outline and popup-specific violet outline from style.css. Platform focus suppression now covers all focused elements; inputs, textareas, links and editable content receive the shared filled focus-visible treatment. System forced-colors focus retained.
- Build passed (existing bundle warning). Live browser clicked name input: outline none, box-shadow none, neutral filled background rgb(48,57,68); screenshot inspected. Local preview refreshed.

### 2026-09-17 — Compact Backtests market and risk fields
- Moved start/end dates into the market grid after timeframe; removed the visible UTC helper. Capital now uses the same 7rem numeric width as commission.
- TP/SL enabled ranges use one row per side (checkbox, start, end, step), retaining the existing 16px grid gap and supported-level hints; narrower containers wrap the checkbox above the three inputs.
- Changed sources: apps/platform-web/src/builder.tsx and apps/platform-web/src/style.css. No request/persistence/calculation changes.
- Passed: pnpm --filter @roehub/platform-web typecheck; pnpm --filter @roehub/platform-web build (existing bundle-size warning).
- Real in-app browser proof at 1480x968: dates and timeframe y=445.04px; start x=475 matches market type; end x=689.41 matches symbol. Capital and commission both 98px. All three TP inputs y=1109.45px; expanded TP/SL visually fit compact rows. At 1280px the risk controls wrap without overlap. No backtest was submitted.

### 2026-09-17 — Direct risk switch
- Replaced the risk-mode select with a labeled checkbox using switch semantics in builder.tsx and scoped styling in style.css. On/off maps to the existing tp_sl_grid/none values; nested risk values are preserved.
- Typecheck, production build and git diff whitespace checks passed. Real browser: enabling reveals both TP/SL controls; disabling removes them; screenshots inspected. Updated preview left open. No submission performed. Existing bundle-size warning remains.

### 2026-09-17 — Align builder captions to control text
- Scoped style.css caption insets across builder fields: 11px for text/number/source controls, 9px for dates, 15px for native selects (including the observed native text inset). Hidden accessible name label remains unchanged.
- Production build and git diff --check passed; existing bundle warning remains. In-app browser at 1480x968 verified captions and control text alignment; updated page left open. CSS-only, no data/API changes.
