# Backtests platform client — complete local journey acceptance

Date: 2026-09-08. Scope: S6 in the existing local checkout.
Status: complete local T1–T8 verification passed; independent coordinator acceptance is separate.

`local_journey_verified=true`; `target_role_cutover_ready=false`.
All required local T1–T8 proof and checks passed. Target authorization integration
is a separate open dependency; no full-platform, default-cutover or delivery claim
is made. Local readiness: ready with the explicitly retained scope limitations.

## Authority and implementation boundary

The user authorized ordinary sequential S1–S6 implementation, one executor at a
time, with the existing coordinator as independent reviewer. For this run the
user explicitly superseded PACK-CLAIM, staged-plan-runner, updater, locking,
claims, receipts and ledger management. No such mechanism was used or created.
The historical ledger, accepted prompts and immutable v23 are preserved.

S6 verifies the complete artifact-mode Backtests journey on existing authenticated
own-resource APIs. It adds focused acceptance tests, repairs a result-read cooldown
edge and makes chart units explicit. Backend domain, API producers, authorization,
migrations, limits and fixture composition are unchanged. No dependency was installed.
No Goal, subagent, new task, branch, worktree, stash, Git publication, deployment,
default-route cutover, server-auth change, AGENTS or global-skill edit occurred.

Producer reports read and retained:
[S1](roehub-backtests-client-v1/S1-client-foundation.md),
[S2](roehub-backtests-client-v1/S2-shell-and-library.md),
[S3](roehub-backtests-client-v1/S3-configure-and-submit.md),
[S4](roehub-backtests-client-v1/S4-execution-and-cancel.md),
[S5](roehub-backtests-client-v1/S5-results-and-strategy.md), including their final
coordinator acceptance sections. Their source/contract analysis remains relevant;
S6 reruns all existing browser scenarios against the integrated current client.
No historical intermediate JSON is substituted for the final S6 evidence.

## Local setup and actual proof

Working directory: `/Users/daniildegtyarev/Projects/roehub.com`.

`tools/qa/backtests_client_fixture.py` creates disposable PostgreSQL and ClickHouse,
production API/auth, Web, the real worker/lazy materializer, and production-built
artifacts from 4320 explicitly synthetic one-minute candles. The interval is
`[2026-03-26T00:00:00Z,2026-03-29T00:00:00Z)`. The client selects spot BTCUSDT,
`15m`, `ma.ema`, close, materialized window 10, and `long_only`; remaining settings
come from the actual catalog/defaults. No provider is contacted and no financial
result, job state, cookie, role, grant or successful API response is fabricated.

Web is `http://localhost:18480`, API `http://127.0.0.1:18481`, flag-off Web
`http://localhost:18482`, and worker metrics `http://127.0.0.1:18483/metrics`.
Both Web processes use the same API and databases. Authentication is actual
password login with a fresh disposable owner. The seeded owner is fixture setup,
not target-role-policy proof. Generated credentials remain only in the mode 0600
`.local_artifacts/backtests-client/credentials.json` inside its mode 0700 directory;
no credential, cookie, DSN, browser storage dump or raw provider response enters
this report, screenshots or observation JSON.

`e2e/run.mjs` starts each suite group with a cold fixture, preserving cold-empty
library and admission assertions. The fifth group is the new integrated journey.
Builds happen before Web startup because its asset manifest is loaded at startup.
The fixture scheduler is paused/resumed only by its recorded PID when observing
queued state or queued cancellation; no job state is seeded or edited. A read-only
PostgreSQL observer additionally captures the short real running interval without
bypassing browser/API refresh hints.

The integrated test uses the real form, successful preflight, one UI create despite
a double click, job progress and actual terminal result, variant URL, equity and
drawdown, nonempty monthly/symbol/trade tables, CSV, fresh source-validated readiness
and one bodyless saved-strategy POST. `not_launchable` is a truthful LIVE evaluator
restriction; the existing research-save use case returns 201. No run/start is sent.

Final integrated job: `d3326416-f8fd-40a7-abc3-3d453cd5485e`;
saved strategy: `61c106f3-3cce-4a47-bbb0-d8f166d513a5`.
The integrated browser read queued0/0 then succeeded1/1; its short running phase
was captured by the read-only DB observer. The independent real execution
regression also observed running0/1 in the browser, then succeeded1/1.
Final identities and observation:
[integrated-journey.json](roehub-backtests-client-v1/browser/S6/integrated-journey.json).
The final real result CSV has **one row, not truncated**. Larger/truncated/empty or
failed results are controlled cases, not claimed real fixture outcomes. Production
lazy materialization202, subsequent200, save/provenance dedupe, terminal deletion
and cancel are also exercised in the full result/execution regression.

Flag-off proof visits the existing SSR Backtests deep link with the same job and
variant, compares persisted job identity/state/request/terminal data and the saved
strategy response through its same-origin proxy, and reads the existing trade.
Generated freshness timestamps are intentionally excluded from the persisted-job
comparison. The SSR strategy page renders the actual selected saved row. Returning
to flag-on restores the same client result. No migration, recreation or deletion
is used to obtain rollback. This proves presentation rollback and domain-object
preservation; it does not promise full old-SSR workstation functionality where
its pre-existing jobs projection is unavailable.

## T1–T8 evidence matrix

All paths below are relative to this report. Controlled failures supplement, and
never replace, the real API/worker operations. The final full command's outcome is
recorded in Validation before local acceptance is asserted.

| Criterion | Actual checks and evidence | Boundary |
| --- | --- | --- |
| T1 — v23, RU/EN, widths, zoom, accessibility | [Reference](roehub-backtests-client-v1/browser/S6/v23-reference.png), [real result](roehub-backtests-client-v1/browser/S6/integrated-result.png); library [observations](roehub-backtests-client-v1/browser/S6-library-regression/observations.json), [native zoom](roehub-backtests-client-v1/browser/S6-library-regression/zoom-observations.json); builder [visual](roehub-backtests-client-v1/browser/S6-builder-regression/visual.json), [zoom](roehub-backtests-client-v1/browser/S6-builder-regression/zoom.json); execution [visual](roehub-backtests-client-v1/browser/S6-execution-regression/controlled-visual.json), [zoom](roehub-backtests-client-v1/browser/S6-execution-regression/zoom.json); results [visual](roehub-backtests-client-v1/browser/S6-results-regression/visual.json), [zoom](roehub-backtests-client-v1/browser/S6-results-regression/zoom.json) | All four workflows at 820/1024/1440 and native Chromium 200%, RU/EN, no document overflow; axe, keyboard tabs/dialogs/focus; real server chart table plus explicit quote-currency/percent and UTC/trade-index descriptions. Synthetic controlled long-lived results are labelled. No speech-output or pixel-identical clone claim. |
| T2 — real complete journey | [Integrated observation](roehub-backtests-client-v1/browser/S6/integrated-journey.json), [preflight](roehub-backtests-client-v1/browser/S6/integrated-preflight.png), [queued](roehub-backtests-client-v1/browser/S6/integrated-queued.png), [real lifecycle](roehub-backtests-client-v1/browser/S6-execution-regression/real-lifecycle.json), [real results](roehub-backtests-client-v1/browser/S6-results-regression/real-results.json) | Actual configure/preflight/create, browser progress, read-only DB running observation, terminal result/variant/trades/CSV/save; no replacement response. |
| T3 — fields, validation and warnings | [Real create/rates](roehub-backtests-client-v1/browser/S6-builder-regression/real-create.json), [controlled field/admission faults](roehub-backtests-client-v1/browser/S6-builder-regression/faults.json), `apps/platform-web/src/builder.test.tsx`, focused Python API tests | UTC half-open ranges, actual catalog/grid/direction, once-only percent conversion, admission/body errors, stale preflight including edit/revert, authoritative funding policy/warnings and retained draft. Real success and injected rejection distinguished. |
| T4 — command identity and recovery | [Accepted create with lost response](roehub-backtests-client-v1/browser/S6-builder-regression/lost-response.json), [identity probes](roehub-backtests-client-v1/browser/S6-builder-regression/real-create.json), [real saved-strategy dedupe/lost reply](roehub-backtests-client-v1/browser/S6-results-regression/real-results.json), [save recovery](roehub-backtests-client-v1/browser/S6-results-regression/controlled-save.json) | One UI job create; actual same-key label200 and changed-computation409 probes; exact body/key retained after response loss/reload; actual same-subject sole-org change plus empty history remains unresolved. Expired/unknown retention/scope guards and storage failure are tested. Save uses separate source/provenance/no-job-TTL contract; no UI replay or fresh automatic key. |
| T5 — cancel, race and delete | [Real lifecycle/cancel](roehub-backtests-client-v1/browser/S6-execution-regression/real-lifecycle.json), [controlled terminal race](roehub-backtests-client-v1/browser/S6-execution-regression/controlled-race.json), [real active-delete409/terminal204](roehub-backtests-client-v1/browser/S6-results-regression/real-results.json), [ambiguous delete](roehub-backtests-client-v1/browser/S6-results-regression/controlled-delete.json) | Actual queued-to-terminal cancel and terminal history deletion; controlled pending/late cancellation cannot replace succeeded; unknown command stays read-only, conflict uses authoritative state,404 is absence/invisibility rather than proof of mutation failure. |
| T6 — result/read failures and concurrency | [Materialization](roehub-backtests-client-v1/browser/S6-results-regression/controlled-materialization.json), [export](roehub-backtests-client-v1/browser/S6-results-regression/controlled-export.json), [60s job cooldown](roehub-backtests-client-v1/browser/S6-execution-regression/controlled-cooldown.json), [result cooldown](roehub-backtests-client-v1/browser/S6-results-regression/controlled-unhinted-cooldown.json), [readiness freshness](roehub-backtests-client-v1/browser/S6-results-regression/controlled-readiness-freshness.json); `results.spec.ts` and `execution.spec.ts` | 202/empty/degraded/failed,429 with/no/short hint and tab remount, stale measurement/source, readiness future/stale/source-hash change, transport/manual recovery, rapid variant selection and late response;202→401/403/404 stops reads and delete403 closes selected private results. |
| T7 — navigation, locale and private lifetime | [Foundation](roehub-backtests-client-v1/browser/S6-foundation-regression/foundation-observations.json), [real library expiry](roehub-backtests-client-v1/browser/S6-library-regression/expiry-observations.json), [real preflight401](roehub-backtests-client-v1/browser/S6-builder-regression/preflight-expiry.json), [real cancel-gate expiry](roehub-backtests-client-v1/browser/S6-execution-regression/session-expiry.json), [save logout cleanup](roehub-backtests-client-v1/browser/S6-results-regression/controlled-save.json) | Safe encoded login continuation, deep-link reload/back/forward, RU/EN and dirty discard; logout/subject change clears private queries/recovery; real expired sessions produce 31 s of zero protected requests. Broken401 body is separately tested; identity outage is unavailable, not false logout. |
| T8 — SSR rollback, existing contracts, cutover separation | [Created-job rollback](roehub-backtests-client-v1/browser/S6/flag-off-created-job.png), [saved SSR strategy](roehub-backtests-client-v1/browser/S6/flag-off-saved-strategy.png), [domain comparisons](roehub-backtests-client-v1/browser/S6/integrated-journey.json), [unrelated SSR navigation](roehub-backtests-client-v1/browser/S6-foundation-regression/foundation-observations.json), 98 focused Python tests | Flag-off preserves actual domain objects without migration; protected/no-store routes, API/schema compatibility and default false retained. Target Backtests authz ticket remains draft, its delegation/envelope prerequisites ready but not accepted. No role/cutover/deployment proof is inferred. |

## S6 findings and fixes

1. **Result429 manual cooldown bypass — fixed.** Coordinator review found that
   `resultDelay` imposed `max(2, hint ?? 5)` seconds but manual refresh used
   `hint ?? 0`. Root cause: separate deadlines disagreed; after the next UI-clock
   tick a hint-less error enabled the button during the automatic five-second
   window. A targeted real-browser controlled429 test reproduced enabled state
   after 1.5 s; [red observation](roehub-backtests-client-v1/browser/S6/429-before-fix.json)
   is retained. `useResultRead` now uses the scheduler deadline for rate-limited
   errors. Restricted errors keep Infinity and other transport failures retain
   their own explicit recovery behavior. The regression checks no hint and
   hint 1, delayed button state, tab remount and actual intervals before a second
   GET. An initial test checked only immediate disabled state and therefore
   missed the clock-tick edge; it was strengthened rather than treated as proof.
2. **Chart units — fixed.** The equity chart/table exposed values without an
   explicit currency unit. A visible RU/EN description now states quote currency
   for equity, percent for drawdown, and UTC time/trade index for the horizontal
   axis. Both the chart image role and accessible table region reference it.
   The production series source confirms `equity_after` and `exit_timestamp`/index;
   no points, units or financial metrics are transformed in the browser.
3. **Evidence wording — corrected.** The foundation observation's old boilerplate
   said strategy/dashboard projections were absent. S5 installed production
   strategy CRUD/dashboard composition. It now distinguishes that working SSR
   path from optional OIDC and unrelated overview/account omissions and the
   pre-existing Settings asset defect. No SSR/backend repair was made.

## Validation

| Command | Outcome |
| --- | --- |
| `pnpm --filter @roehub/platform-web typecheck` | Passed on final runtime/test sources |
| `pnpm --filter @roehub/platform-web test` | Passed: 145 tests / 6 files after the runtime fixes |
| `pnpm --filter @roehub/platform-web build` | Passed: `main-DDcgBVmC.js`, `main-TfDMVJxy.css`; Vite >500kB advisory remains |
| `ROEHUB_PROOF_STAGE=S6 pnpm --filter @roehub/platform-web test:e2e` | Passed: 35/35 Chromium scenarios in cold groups 5+6+9+14+1; foundation/library 46.8s, builder 50.5s, execution 2.7min, results 1.5min, integrated journey 21.1s |
| `ROEHUB_PROOF_STAGE=S6 pnpm --filter @roehub/platform-web test:e2e results.spec.ts --grep 'controlled result429 without'` | Pre-fix failed with enabled manual button; post-fix passed:1 test,14.5s including cold fixture |
| `.venv/bin/python -m pytest -q tests/unit/apps/api/test_backtests_routes.py tests/unit/apps/api/test_ui_backtests_routes.py tests/unit/apps/web/test_web_v2_1_routes.py` | Passed: 98 tests, 4.81 s; existing httpx cookie deprecation warning. Python/API/Web/fixture inputs unchanged afterwards; this current evidence is reused |
| `python3 -m tools.docs.generate_docs_index --check` | Passed; existing README output required no S6 regeneration |
| `python3 -m tools.docs.generate_project_map --check` | Initial check identified new test inventory; regenerated with `python3 -m tools.docs.generate_project_map`; final check passed for all 5 artifacts |
| `git diff --check` | Passed after final documentation and source changes |

Before the cooldown finding, the first full command passed 34 browser scenarios
(5+6+9+13+1). That run is not the final-fixed-build claim. The final command passed
all groups, including the new no-hint/short-hint regression. Earlier single
integrated runs also passed, but only final S6 artifacts support the final claim.
No runtime source is changed or rebuilt while its Web fixture is alive.

## Console, visual inspection and unavailable boundaries

Normal integrated workflow assertions permit only the exact optional login
`/api/auth/oidc/status`404; no unexpected HTTP/console error or page exception
is allowed. Navigation-aborted obsolete requests are recorded separately.
The real result regression separately labels the intentionally dropped save
response and actual just-deleted-history404. Controlled fault statuses are
intentional and not normal-success evidence.

The foundation visits unrelated SSR destinations: `/api/ui/dashboard/summary`
and `/api/ui/account/{profile,preferences,notifications,limits,integrations,
notifications/scoped,exchange-connections,sessions,audit-events}` return 404 in
the minimal fixture. Settings references absent
`/assets/css/pages/market-data-settings.css` and
`/assets/js/pages/market-data-settings.js`, an existing repository omission
established in S1. Their console/network errors remain explicit. Only navigation,
not complete Settings/Overview functionality, is accepted by this ticket.

Visual review compares the rendered immutable v23 with actual client shell,
list, builder and result screenshots. Its dark palette, bordered rounded panels,
violet selection, narrow sidebar and compact controls are retained; the layout
adapts to the actual supported fields/states. No fictional specimen service
status, plan/user, ETA, percentage performance or provider data is copied.
Native zoom records actual `getZoom=2`, outerWidth 1440, innerWidth 720, DPR 2;
viewport and axe evidence cover RU/EN at 820/1024/1440. No phone-product or
assistive-technology speech-output claim is made.

Known limitations remain: workstation backtest_jobs lacks existing
organization_scope_resolver wiring; direct jobs APIs work, while extended
search/instrument/date filtering is honestly unavailable. Original organization
binding and a server-proven client replay precondition are absent. Both create
and saved-strategy replay remain disabled; history absence never proves failure.
Job-create retention is the actual unchanged factory default 86400 s from S3;
saved-strategy provenance is a separate no-job-TTL contract. Server-backed replay,
full auto/direct_db, measured ETA, target operator/viewer/delegation, general Jobs,
Artifacts, ingestion and execution detail remain unclaimed. Vite's size advisory
is not a measured performance verdict.

## Compatibility, review and scope updates

S6 baseline is accepted S5 in the shared checkout. New candidate changes only
result manual-refresh timing, descriptive chart semantics, proof and scoped docs.
Source inspection covers the result scheduler, API error parsing/result delay,
server result-series meaning, feature defaults and actual authz dependency status.
Existing APIs/data/session ownership are retained; external/dynamic consumers or
mixed deployed versions are not inventoried or inferred.

| Surface / direction | Classification | Evidence |
| --- | --- | --- |
| Result429 manual reads / current opted-in browser | `compatible-change` | Manual and automatic reads now obey the same already-required bounded cooldown; no command retries introduced; red/green browser proof |
| Chart/table semantics / current opted-in browser | `compatible-change` | Additive localized unit descriptions preserve names, data, query/variant identity and keyboard behavior |
| API/DTO producers, persistence, authorization, limits, default feature config | `none` for S6 | Source unchanged from baseline;98 focused Python tests and real integration |
| New client writes → old SSR reads | `compatible-change` for the combined S1–S6 delivery | Actual same-job/strategy/trade reads at flag-off and restored client, no migration; unavailable legacy workstation projection remains explicit |
| Ticket and functional coverage | `compatible-change` | Only evidence-backed local implemented_scope changes; wider requires_ui/requires_api and target requirements retained |

Cold self-review verdict: no remaining actionable S6 blocker found. It checks
the exact changed paths, timers and manual bypasses,
command/source/privacy semantics, test isolation, actual observations, chart
units, immutable evidence and local/cutover separation. Independent review is the
existing coordinator; its final acceptance is separate from executor validation.
No extra review agent or task is created.

## Owned paths, preservation and final boundary

Created: `apps/platform-web/e2e/journey.spec.ts`, this report, and browser evidence
under `roehub-backtests-client-v1/browser/S6/` and
`S6-{foundation,library,builder,execution,results}-regression/`.

Modified:

- `apps/platform-web/src/results.tsx`
- `apps/platform-web/src/results-i18n.ts`
- `apps/platform-web/e2e/run.mjs`
- `apps/platform-web/e2e/foundation.spec.ts`
- `apps/platform-web/e2e/library.spec.ts`
- `apps/platform-web/e2e/builder.spec.ts`
- `apps/platform-web/e2e/execution.spec.ts`
- `apps/platform-web/e2e/results.spec.ts`
- `.codex/tickets/2026-09-08-roehub-backtests-client.md`
- `docs/architecture/apps/web/roehub-ui-functional-contract-v1.md`
- `docs/architecture/apps/web/roehub-ui-functional-registry-v1.json`
- `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md`
- `docs/architecture/project-map/PROJECT_MAP.md`
- `docs/architecture/project-map/project-map.json`

No deletion or out-of-scope change. Narrow fixture-backed tests are authorized
proof tooling; the production fixture implementation itself is unchanged.

The initial filesystem hash inventory records 3145 existing files. It distinguishes
S6 modifications from pre-existing dirty AGENTS, Web routing/settings/templates,
root workspace/lock/CI, accepted S1–S5 source/evidence and docs/plan/pack changes.
No staging, commit or cleanup of foreign work is performed. Protected v23 SHA-256:
`3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.
Final hash comparison found exactly 14 expected modified paths and 97 allowed
new files (the integrated test, this report and 95 browser evidence files).
All 46 local report links resolve. It confirmed all 237 pre-existing instruction/pack/producer
evidence files unchanged, including the old ledger/prompts and all accepted
S1–S5 reports/screenshots/observations. No outside-expected path changed.
The private fixture directory is absent and no roehub-client containers remain.
The registry audit passed: 44 unique screen IDs, 18 unchanged journeys, only four
local scope records updated, and all target states/actions and delivery tags
preserved. Canonical report links resolve locally. Generated changes are limited
to `docs/architecture/project-map/PROJECT_MAP.md` and `project-map.json`; existing
foreign README and component-map changes were preserved.

S6 ends here. The broader UI backlog, target-role cutover and publication require
their own selected work; no next stage or task is launched.

## Final coordinator acceptance

Accepted after all six sequential executors completed and S6 became idle. The
coordinator independently reviewed the T1–T8 matrix, actual final observations,
runtime/result/recovery and cooldown changes, scoped ticket/registry updates,
current rendered result and SSR evidence, and final gate outcomes. The no-hint
429 finding is closed by the shared deadline and measured browser regression.
The final integrated job is `d3326416-f8fd-40a7-abc3-3d453cd5485e` and strategy
`61c106f3-3cce-4a47-bbb0-d8f166d513a5`: one create, one save, no run/start;
the real worker passed queued/running/succeeded, while final browser polling
observed queued and succeeded. Flag-off comparisons confirm unchanged job,
strategy and trades, and restored client identity. The v23 hash was independently
rechecked and matches the accepted baseline. No blocking local-scope finding
remains. S1–S6 and the ticket's bounded local T1–T8 acceptance are complete:
`local_journey_verified=true`, `target_role_cutover_ready=false`. The documented
server replay/filter/target-role and unrelated SSR limitations remain explicit;
this acceptance grants no publication, default cutover or broader backlog work.
