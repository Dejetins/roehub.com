# Roehub UI implementation plan v1

Continue the local platform UI as complete, API-backed journeys using the refined,
accepted Backtests implementation as the visual baseline. Preserve the full functional
scope while migrating one bounded journey at a time.

## Authority, status and execution artifacts

- Revision: 2026-09-12, under the user's instruction to update this plan before
  writing detailed prompts. The 2026-09-08 decomposition remains its historical base.
  Backtests was implemented, refined, accepted and merged; the next selected planning
  slice is Strategies library/detail. Later priorities remain a dependency backlog.
- The subsequent user request on 2026-09-12 authorizes implementation-prompt authoring
  for the selected Strategies iteration, with a pack only if multiple prompts are
  needed. One cohesive standalone prompt covers it; implementation, branch,
  publication and runtime delivery remain separate instructions.
- The current [functional contract](roehub-ui-functional-contract-v1.md) and
  [registry](roehub-ui-functional-registry-v1.json) own scope and implementation tags.
  This plan owns sequencing and acceptance boundaries; it does not duplicate status.
- The [Backtests ticket](../../../../.codex/tickets/2026-09-08-roehub-backtests-client.md)
  is `implemented`, with `user_acceptance_status=accepted`. Its
  `local_journey_verified=true` and `target_role_cutover_ready=false` are separate
  boundaries. The [iteration log](backtests-ui-iteration-log.md) records the later
  accepted UI refinements; the original S1–S6 specification is not the latest visual target.
- The retired UI design workflow and historical prompt packs remain historical.
  This is the newly selected bounded implementation workflow, not a global requirement
  for every future UI task. Ordinary bounded tickets may still execute directly.

The completed Backtests work retains these historical execution references:

| Field | Repository path |
|---|---|
| `plan_doc` | `docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md` |
| `prompt_pack_dir` | `.codex/agents/generated/roehub-backtests-client-v1/` |
| `stage_ledger` | [.codex/agents/generated/roehub-backtests-client-v1/stage-ledger.md](../../../../.codex/agents/generated/roehub-backtests-client-v1/stage-ledger.md) |

For that run, the user explicitly selected sequential execution and waived PACK-CLAIM,
exclusive-updater and machine-receipt requirements, leaving the formal ledger unchanged.
Its pending/disallowed entries therefore do not contradict the completion reports and
are not current blockers. Preserve the old prompts and ledger as execution history;
do not reactivate S1–S6 or assign later blocks to them. Future detailed prompts must
use a focused source for the next slice and must not inherit this retired entry gate.

Current authored execution instruction:
[Strategies library/detail](../../../../.codex/agents/generated/roehub-strategies-client-v1/implement-strategies-library-detail.md).
It is standalone because the existing foundation and APIs support one bounded
read-oriented journey with integrated acceptance; no inter-stage producer or ledger
is necessary. Its internal steps are not separate execution stages.

## Sources and current baseline

Read the sources relevant to the selected slice. Material baseline sources are:

- [Product requirements](../../platform/roehub-product-transformation-requirements-v1.md),
  [Web architecture](roehub-ui-design-and-delivery-architecture-v1.md),
  [IA](roehub-local-platform-information-architecture-v1.md) and
  [access contract](roehub-local-platform-access-and-route-contract-v1.json).
- `apps/web/main/app.py` and `apps/web/main/api_client.py`: current SSR,
  authenticated HTML entry and same-origin proxy. Current code is evidence;
  accepted server/security contracts remain normative.
- `apps/api/routes/{backtests,ui_backtests}.py` and corresponding DTOs: existing
  artifact-mode commands, reads and result materialization reused by Backtests.
- [Functional contract delivery dependencies](roehub-ui-functional-contract-v1.md#clarifications-and-delivery-dependencies):
  remaining projections, retry adapters, modes, ETA, draft/model decisions and authz.
- [Backtests completion](../../../../.codex/delivery/evidence/ROEHUB-BACKTESTS-CLIENT-2026-09-08.md),
  [accepted iteration log](backtests-ui-iteration-log.md) and
  [publication record](../../../../.codex/delivery/evidence/BACKTESTS-UI-PUBLICATION-2026-09-11.md).
  The accepted code was merged through PR #33 at
  `191a9f8169dab0639d8fb3456eb73b060eb1d2c4`. This is the reproducible source baseline,
  not a claim that a local preview or production installation is currently running.
- `apps/platform-web/src/{style.css,motion.tsx,backtests-page.tsx,library.tsx,builder.tsx,results.tsx,expandable-overview.tsx}`:
  implemented visual and interaction patterns to reuse. Existing shared web contracts
  live in `packages/web-contracts`; other package extraction requires a demonstrated need.
- `apps/api/routes/strategies.py`, `apps/api/routes/ui_strategies_dashboard.py`,
  `apps/api/dto/ui_strategies_dashboard.py` and `apps/web/templates/pages/strategies.html`:
  reusable strategy reads, dashboard projection and existing SSR behavior.
- [Backtests authz ticket](../../../../.codex/tickets/2026-07-20-roehub-authz-backtests.md):
  recheck its actual dependencies at target-role integration/cutover. Acceptance of the
  capability kernel or the local client does not complete those dependencies.

`apps/platform-web` now contains the accepted Backtests client. General platform UI
migration is still incomplete. The old v23 specimen is preserved as historical evidence,
not as the design specification for subsequent implementation (see D4).
The source registry covers 44 records, 43 source surfaces and 18 journeys, including
nonvisual, QA and historical dispositions. No registry record is dropped by this plan.
Current-source observations do not constitute new browser, API or deployment proof.

## Delivery decisions

### D1. A small shared client foundation followed by complete journeys

Keep the accepted React/TypeScript/Vite client and existing `@roehub/web-contracts`.
Feature composition belongs to `apps/platform-web`. Additional shared packages, when
needed, may contain reusable tokens, UI, charts or localization, not domain policy.
Dependencies flow from feature composition to shared code and typed same-origin adapters.
`apps/web` retains session/proxy responsibilities; API/domain services own permission,
validation, admission, job state, strategy identity and calculated metrics.

The viable sequencing choices were one large UI migration or bounded journeys.
The selected journey approach provides useful integration feedback and a reversible
cutover. Its cost is temporary coexistence with SSR and explicit navigation across
both clients. Extract only the shared patterns needed by the next journey, keeping
Backtests visually stable. Do not build a general component catalogue or workflow
framework as prerequisite work. Do not add services or change domain storage under
a UI-only slice.

### D2. Local integration first, route cutover separately

Retain the existing disabled-by-default Backtests setting and existing
URL identities, authenticated entry, safe login continuation and proxy behavior.
In-progress screens do not expose unwired actions. Unrelated destinations remain SSR.
No dual domain writes or schema migration is planned for the next read-oriented slice.
Existing APIs create the same jobs/strategies. Turning the setting off restores SSR
presentation; it does not undo jobs or strategies already created. Never promise rollback of those
side effects through a UI setting.

For Strategies, add independent `WEB_STRATEGIES_CLIENT_ENABLED=false`, preserving
the Backtests flag's meaning. Load shared assets when either is enabled; each route
still uses its own flag. Bootstrap route availability is a presentation allowlist,
not authorization. Preserve explicit SSR continuation via `view=classic` on
Strategies list/detail; `/strategies/new` and out-of-scope `mode` entry remain SSR.
Client navigation must honor these server gates. Verify all four flag combinations,
safe login/locale continuation and flag-off domain preservation.

Backtests local own-resource integration can be accepted before target role integration.
Default-route cutover requires relevant authz dependencies, permission tests and real
browser proof. It is a later scoped task, not an implicit seventh pack stage. Deployment
also needs an explicitly selected current runbook/target; none is configured here.

### D3. Error, cancellation and recovery are part of each action

The Backtests ticket's field/API/state/error/recovery sections are binding in S3–S5.
Keep command identity and unknown outcome distinct from transport status. No automatic
mutation retry, fabricated progress, client role override or generic retry adapter.
Job-create replay requires verified retention and original server-bound organization;
otherwise show truthful unresolved/read-only guidance. Saved-strategy dedupe has its
own source/provenance contract, and saving never starts trading. Per-tab recovery is
allowlisted and cleared on logout/subject change; ordinary drafts remain in memory.
Raw credentials, cookies, provider payloads and generic API responses are not evidence.

### D4. The accepted Backtests implementation is the visual authority

The user's 2026-09-12 decision promotes the refined Backtests page accepted on
2026-09-11 to the visual baseline for all subsequent local-platform UI work. It
supersedes the old v23 pilot wherever visual instructions differ. Do not reconstruct
v23 first or combine its old spacing/components with the current implementation.

Reuse the current shell, graphite panel surfaces and borders, violet accents, type
hierarchy, compact tables, control/icon sizing, selected states and metric cards.
Preserve the accepted balance of useful density and separation between sections.
Keep short numeric inputs narrow and align fields on shared axes. Large forms belong
inside the relevant workspace; expanding fields to fill empty space is not a solution.
Secondary technical information remains in subordinate disclosures.

Reuse the single motion-speed preference and transition coordinator in `motion.tsx`
for navigation, tabs, disclosures and panel expansion. Keep the existing preference
compatible, honor Off/reduced motion, and handle interrupted transitions without
ghost surfaces, delayed clicks, lost focus or resetting selection. Do not introduce
per-screen animation settings or unrelated effects. New charts use Apache ECharts
with the accepted interaction, units, date formatting and expansion patterns.

The functional needs of a different page may change its composition; they do not
justify a different visual language. Use the actual current CSS/components first,
extracting reusable parts only when both consumers need them. Compare changed screens
with Backtests at matching viewport, locale and state; changes to shared primitives
also need a Backtests regression check. Do not force every full editor into one viewport;
keep the primary inspection workspace compact and use bounded scrolling for long lists.

Implement settled compositions directly. Prototype only a named unresolved arrangement
or interaction, without a mandatory atlas, theme series or repeated baseline approval.
Preserve 820/1024/1440 widths, RU/EN, 200% zoom, keyboard access and reduced motion.
Phone UI and public-site redesign remain outside this local-platform plan.

Historical specimen to preserve unchanged:
`.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html`,
SHA-256 `3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.
Neither that specimen nor the new visual baseline grants API/role/runtime capability.

### D5. Keep dependencies explicit without expanding the first journey

Artifact-mode Backtests can use existing source/default APIs without a finished general
Artifacts, Data or Jobs UI. Domain progress/cancel in Backtests does not complete generic
Jobs. Saving a strategy may navigate to the existing SSR detail; it does not require the
new strategy editor. Shared links must use real, available destinations or state their
unavailability, never fake successful navigation. Later API changes require their own
bounded contract/task before a dependent UI action is exposed.

## Backtests stages and acceptance mapping

This section describes the completed first journey, not the next execution queue.
The S6 completion report establishes local acceptance; the later iteration log records
the refined visual result. Retain stage names below as historical identifiers.
The exact stage contracts, entry inputs, touch zones, checks and report paths are in
the ledger and matching prompts. `T1`–`T8` below refer to the numbered acceptance
criteria in the Backtests ticket. A producer report records exact new file paths and
reproducible commands for its consumer; planned paths are not treated as missing inputs.

| Stage | Outcome | Dependency | Ticket coverage / required boundary proof |
|---|---|---|---|
| S1 — client foundation | Workspace/build/scripts, gated route integration, API/session adapter seam, local proof setup | Existing server contracts | T7/T8 foundation: real browser entry and proxy, feature off restores SSR; fixture/runtime recipe verified for subsequent API proof |
| S2 — v23 shell and library | Reusable minimum UI, locale, accessible navigation, real list/filter/detail entry states | S1 | T1/T7, list/status API: real data, loading/empty/error, deep-link reload, browser visual/focus evidence |
| S3 — configure and submit | Catalog-driven form, preflight, admission, one logical submit, reload/unknown-outcome recovery | S2 | T2/T3/T4/T7: real preflight/create; meaningful error and duplicate/lost-response tests; no later stage owns missing recovery |
| S4 — execution and cancel | Real queue/progress/freshness, terminal outcomes, cancellation and reconciliation | S3 | T2/T5/T6/T7: observed local job lifecycle/cancel plus controlled race/error cases |
| S5 — results and saved strategy | Variants, chart/stat/trade views, CSV, materialization, safe strategy save, history delete | S4 | T2/T4/T5/T6/T7: real result/export/save boundary, deletion eligibility, bounded series and late-response protection |
| S6 — full journey acceptance | Integrated acceptance of all ticket requirements and focused repairs | S5 | T1–T8 together: real local configure-to-result flow, RU/EN, widths/zoom/accessibility, errors/recovery, rollback; distinguish local acceptance from cutover |

No stage can pass solely from screenshots for API behavior, mocked responses for real
integration, or unit tests for browser behavior. Stage-specific mocks supplement real
proof for hard-to-trigger failures. S1 checks availability of local proof prerequisites
early; a missing fixture/service is a named verification dependency, not postponed as
a surprise at S6. Do not install or use retired remote environments to fill that gap.

## Remaining UI blocks and dependencies

These block IDs are stable planning identities, not tickets marked ready. Split each
block into the smallest API contract/projection task, client journey task and integration
proof needed when it becomes next. A single bounded ready ticket needs no prompt pack.
Do not pre-author implementation prompts for unresolved future APIs.

`F` means the reusable S1/S2 foundation. It is a prerequisite for client integration,
not for independent API specification work. Target role acceptance/cutover of every
protected block also depends on its server authorization integration (`A`). API work
may proceed alongside UI planning; product execution remains separately authorized.

| Block | Deliverable and next decomposition | Hard prerequisites / dependent completion |
|---|---|---|
| UI-BT | Accepted artifact-mode Backtests baseline; subsequent UI refinements, later modes, operator retry and measured ETA each get a separate task | Local journey complete; full roles/cutover depend on Backtests A. No general Jobs/Artifacts UI prerequisite |
| UI-AUTH | New sign-in/recovery and first-launch setup compositions | F; existing local auth contracts; setup catalog/routing contract before resumable setup acceptance |
| UI-HOME | Dashboard summaries, shell search/notifications and cross-area readiness | F; actual role-filtered projections. Cross-area links gain full acceptance as their consumer routes become available |
| UI-DATA | Catalog/selection/coverage slice, then ingestion history/backfill/cancel/retry slice | F; API-DATA snapshot/cursor + ingestion projections/actions; job links need real capability-to-job identity, general Jobs detail needed for the complete cross-area journey |
| UI-ART | Artifact catalogue then manifest/version/provenance/compatibility and consumer navigation | F; API-JOBS-ARTIFACTS browser catalogue/detail projections; Backtests consumer exists after S3, Models consumer after UI-MODELS |
| UI-JOBS | Cross-capability read-only queue/detail, then eligible per-capability cancel/retry controls | F; API-JOBS-ARTIFACTS projections; resolve Q-JOB-ADAPTER before generic commands. Do not unify domain backtest retry with runtime attempt retry |
| UI-CONN | Connection creation/check/recovery/archive and strategy selection | F; existing exchange-connection contracts and target A; no secret readback; complete consumer link depends on available strategy detail |
| UI-STRAT | Next: library/detail and saved-backtest continuation; later immutable create/clone editor, archive and launch-profile/run-control | Existing F and strategy reads/projections; verify source links/readiness in the next slice. Q-DRAFT before durable editor drafts; UI-CONN/readiness and execution contracts before launch controls. Backtest save already exists |
| UI-EXEC | Execution activity/detail and source-to-fill reconciliation, then live workspace/manual position control | F; execution projections/source identity; strategy/connection/risk/authority contracts for commands. UI-STRAT/UI-CONN integration before full manual-control journey; monitoring detail for full diagnostic links |
| UI-MODELS | Registry/detail/comparison, then train/promote/rollback consumers | F; model/experiment projections + Q-MODELS; data/artifact/job identities before exposing dependent actions; UI-ART/UI-JOBS for complete cross-area navigation |
| UI-OPS | Monitoring workspace, service/datastore detail, incident-to-job/runbook links | F; real monitoring/freshness/dependency projections; UI-JOBS/UI-DOCS for complete diagnostic journeys; any operational action needs its own eligible server contract |
| UI-ACCOUNT | Personal profile/preferences/notifications/security; admin members/resources/extensions/maintenance | F; existing account/admin contracts; installation authority and missing projections per task. Destructive/recovery operations require scoped proof; no inferred deployment authority |
| UI-DOCS | Version-matched local documentation, search, runbooks and safe continuation | F; documentation visibility/version contracts; explicitly migrate framework `/docs` before route cutover, preserving compatibility redirects where selected |

Near-term sequence:

1. Preserve the accepted Backtests result and make the next Strategies library/detail
   slice concrete using the existing implementation and contracts, as specified below.
2. Address Connections and strategy launch-profile/run/stop/restart integration in
   bounded follow-up slices. Confirm their server readiness, permissions and command
   recovery before exposing commands; editor/create/clone/archive has its own slice.
3. Prepare Data, Artifacts and general Jobs API/projection/identity gaps before selecting
   their client work. Their readiness is not a prerequisite for the read-only Strategies
   slice. Other blocks remain in the dependency table without invented delivery dates.

This order does not drop any registry requirement or imply full Strategies completion
after library/detail. Do not claim a cross-area journey complete while a required
destination or operation is unavailable.

## Next iteration — Strategies library and detail

Execution update — 2026-09-12: the selected standalone unit implements the opt-in
library/immutable detail, independent optional status and saved-result return context.
The [implementation evidence](../../../../.codex/delivery/evidence/ROEHUB-STRATEGIES-CLIENT-2026-09-12.md)
records local checks, remaining limitations and A1–A9.
`local_journey_verified=true`; `target_role_cutover_ready=false`. This update does not complete
`journey.strategies.create_and_control`, the editor, launch controls or target-role
cutover. The following scope and later dependencies remain in force.

### User correction — strategy analysis content (2026-09-12)

The user rejected the specification-only presentation and explicitly authorized
substantive analysis/trading information. The current continuation extends the
read surface with overview, persisted source-backtest metrics/conditions/equity/
drawdown/trades, and existing profile/paper-account/signal observations. It adds
`GET /api/strategies/{strategy_id}/research-source`: ownership and organization
checks precede a scoped lookup in existing provenance storage. No schema migration,
save replay or compatibility-check side effect is needed. Query return context
remains navigation only and is never used to attribute performance.

This supersedes the earlier decision below to defer a reverse-origin read. It does
not create a complete history of all backtests or clone ancestry. Mutating launch
controls, exchange execution, editor and target-role rollout remain deferred.
Unknown values stay unknown; historical simulation, paper accounting and live
execution observations must remain distinct. See the implementation report's
analysis-content continuation for exact proof and remaining limits.

User presentation correction, 2026-09-12: remove the trading-logic prose, full-report
link, profile/limits section, classic-management links and persistent strategy-status
section from the client screen. Keep research metrics, conditions, charts, trades and
immutable specification. Direct classic routes remain available; this correction
changes presentation and supersedes earlier requirements to show those blocks.

### Superseding user correction — operational Strategies (2026-09-12)

The user subsequently approved the operational concept and explicitly authorized
implementation. This supersedes the research-centred presentation above: Strategies
contains no backtest metrics, report links or backtest tabs. The existing Backtests
save/API provenance behaviour remains separate and preserved.

The local client now exposes state/actions, executed-fill price markers, current
position, run statistics, trade/fill details with separate entry/exit causes, and an
event journal. Lifecycle/manual commands use existing endpoints; unknown manual
outcomes retain their key and original amount across reload, with intentional
same-key outcome recovery. Only the selected observation is periodically read,
respecting refresh hints and terminal access failures. Real read adapters are
organization/owner/strategy/run scoped. Local command proof uses an explicitly
labelled opt-in simulator, not a real trading worker or exchange.

Current implementation is bounded to the selected/latest run and at most 5000 fills;
all-run accounting, provider reconciliation/quote conversion, live equity curves,
and target-role rollout are not proved by this iteration. Missing funding conversion
withholds net P&L. Equity/drawdown curves are labelled closed-trade curves. Existing
flags/classic rollback persist. `journey.strategies.create_and_control` remains partial;
`target_role_cutover_ready=false`. See the operational continuation in the same report.

### Outcome and scope

Continue the existing Backtests result → save strategy → inspect saved strategy flow
inside the current platform visual language. Preserve `/strategies` and
`/strategies/{strategy_id}` identities, safe login continuation and browser history.
The new primary surface is a compact library with a selected-strategy workspace,
adapting the accepted Backtests library/detail pattern rather than introducing another
design. The selected strategy remains the main reading area; filters and technical
metadata must not dominate it.

| Area | Planned behavior | Required evidence |
|---|---|---|
| Library | Real strategies, compact rows, search/filter on supported fields, clear selection and an intentional no-selection state; preserve filters and list position across detail navigation | Real list, empty list, no filter matches, refresh failure/stale data and rapid selection; no synthetic operational status |
| Detail | Name, instrument, timeframe and readable immutable indicator/signal specification; subordinate IDs, timestamps and raw/debug data | Real owned strategy and direct-link reload; missing/not-visible and session/permission states |
| Backtest continuation | Saving keeps its current semantics, opens the exact returned strategy, and preserves a safe return to the originating job/variant when identity is verified | Actual saved strategy from a completed job; Back/Forward, reload and return to the selected variant; missing/deleted source handled truthfully |
| Readiness | Show available server compatibility/data/profile signals with their source and freshness; separate research compatibility from readiness to trade | Ready, blocked, unavailable and stale responses; optional projection failure must not hide the immutable specification |
| Navigation and presentation | Existing shell, panel surfaces, density, shared motion and locale behavior; retained SSR access to functionality outside this slice | Matching viewport screenshots, keyboard/focus, RU/EN and reduced-motion checks; unchanged accepted Backtests layout |

New-client create/clone/archive, strategy editing, profile changes, run/stop/restart,
manual execution and live monitoring are outside this first slice. Preserve access to
existing SSR management/editor routes through an explicit continuation until migrated;
do not silently remove existing functions or label unimplemented commands as working.
Opening a strategy or checking compatibility must never launch trading. Durable editor
drafts and Q-DRAFT do not block this read-oriented slice.

### Resolved authoring mapping — 2026-09-12

Source inspection confirms these reusable contracts, not their runtime readiness:

- `GET /api/strategies` returns an owned, deterministically ordered list excluding
  soft-deleted records; `GET /api/strategies/{strategy_id}` returns the immutable
  snapshot. Their response includes identity, name, creation time and spec, not live
  performance or a guaranteed source-backtest link. Do not assume server pagination
  or runtime fields on this list.
- `GET /api/ui/strategies/dashboard` already supplies selected-strategy, selector,
  runtime/readiness, source freshness and refresh-control projections. Its route
  accepts `strategy_id`, `state`, `cursor` and `refresh`, not arbitrary DTO filter
  fields. Use the complete direct list for local search/filter; use selected dashboard
  observations independently so optional-service failure does not hide the spec.
- `check_strategy` in
  `src/trading/contexts/strategy/application/use_cases/compatibility_readiness.py`
  explicitly returns null source job/variant fields and records a check/event. It
  cannot supply reverse provenance and must not be polled to discover an origin.
  The save response already validates strategy ID and source provenance. Carry only
  its bounded job/variant navigation context to the detail URL; after reload, treat
  that context as untrusted navigation input, not persisted provenance. Independent
  strategy entry has no invented source link. No reverse-lookup API is required by
  this read-oriented slice; source history remains a separate future capability.
- Reuse the existing Backtests save/deduplication flow, same-origin/session handling
  and API errors, including its explicit Open-strategy action after saving. The Web
  routes already register `/strategies/new` before detail; preserve that precedence
  and use the explicit authenticated classic presentation from D2 for management.

If a required field, source link or safe SSR continuation cannot be supplied by these
contracts, record the exact gap and choose a bounded compatible integration task before
its dependent UI prompt. Optional runtime data can remain truthfully unavailable;
successful saved-strategy navigation and readable identity/spec are required acceptance.
Do not expand the API merely to reproduce decorative data from a prototype.

### Work breakdown and acceptance

These are planning slices, not new S-stage IDs, ready tickets or an execution ledger:

| Slice | Deliverable | Exit condition |
|---|---|---|
| Contract and route mapping | Exact fields, read/refresh behavior, source identity, gated routes/SSR fallback and isolated fixture for the selected scope | Mandatory-path gaps resolved or explicitly scoped; source map sufficient to author executable prompts without invented APIs |
| Library/detail composition | Minimal reuse/extraction from current Backtests, typed reads, URL/state handling, readable specification and all list/detail states | Focused type/unit/build checks plus real browser proof for list/select/deep link; accepted Backtests primitives retained |
| Saved-result integration and acceptance | Actual Backtests save → strategy detail → return; conditional readiness, errors, interrupted reads and fallback | Real local API/browser journey, correct IDs and unchanged save semantics; shared-style/motion regression checks pass |

For changed behavior, update relevant tests with the implementation. Use disposable
local data and the existing QA setup as a starting point, verifying any new fixture
requirements before implementation. Reuse current proof for unchanged behavior;
unit/mocked cases cover edge failures but do not replace the real integrated journey.
Check 820/1024/1440, RU/EN, 200% zoom, keyboard/focus, rapid navigation, Off/reduced
motion and browser console/network failures for affected surfaces. Run only the gates
required by changed boundaries; broader CI/publication is a separate authorized step.

Slice acceptance updates only library/detail and the demonstrated Backtests continuation
in registry `implemented_scope`. It does not mark `journey.strategies.create_and_control`,
target-role cutover or the whole Strategies dashboard complete. The existing registry
status tags remain unchanged during this planning revision.

## Coverage ownership

The following tables map every registry record and journey exactly once to a primary
delivery owner. Cross-block dependencies above remain binding. Nonvisual, QA and
historical records are explicitly retained without turning them into product pages.

| Primary owner | Registry records |
|---|---|
| UI-AUTH | `screen.auth.login`, `screen.setup.first_launch` |
| UI-HOME | `screen.shell.application`, `screen.dashboard.overview`, `screen.system.error_and_recovery` |
| UI-BT | `screen.backtests.library`, `screen.backtests.builder`, `screen.backtests.detail` |
| UI-DATA | `screen.data.catalog`, `screen.data.ingestion` |
| UI-ART | `screen.artifacts.catalog`, `screen.artifacts.detail` |
| UI-JOBS | `screen.operations.jobs`, `screen.operations.job_detail` |
| UI-CONN | `screen.connections.workspace` |
| UI-STRAT | `screen.strategies.library`, `screen.strategies.editor`, `screen.strategies.detail` |
| UI-EXEC | `screen.live.workspace`, `screen.live.execution_activity`, `screen.live.execution_detail`, `screen.live.manual_position_flow` |
| UI-MODELS | `screen.models.registry`, `screen.models.detail` |
| UI-OPS | `screen.monitoring.workspace`, `screen.monitoring.service_detail`, `contract.diagnostics.health` |
| UI-ACCOUNT | `screen.settings.profile`, `screen.settings.preferences`, `screen.settings.notifications`, `screen.settings.security`, `screen.admin.overview`, `screen.admin.members`, `screen.admin.resources`, `screen.admin.extensions`, `screen.admin.maintenance` |
| UI-DOCS | `screen.docs.index`, `screen.docs.runbook`, `contract.framework.api_docs` |
| F/S1 | `contract.integration.api_proxy`, `contract.delivery.static_assets` |
| A | `contract.authorization.target_matrix` |
| QA only | `screen.qa.plugin_panel_lab` |
| Historical only | `historical.react_prototype` |

F/S1 covers transport/static compatibility; A remains server-owned authorization.
QA only preserves a bounded test surface when relevant; Historical only creates no
implementation task. These dispositions do not change the registry status tags.

| Journey | Primary owner |
|---|---|
| `journey.auth.sign_in_and_continue` | UI-AUTH |
| `journey.setup.first_launch` | UI-AUTH |
| `journey.data.select_instruments` | UI-DATA |
| `journey.connections.manage_exchange` | UI-CONN |
| `journey.backtests.configure_and_run` | UI-BT |
| `journey.backtests.inspect_and_promote` | UI-BT |
| `journey.strategies.create_and_control` | UI-STRAT |
| `journey.live.observe_execution` | UI-EXEC |
| `journey.models.inspect_and_operate` | UI-MODELS |
| `journey.monitor_and_follow_runbook` | UI-OPS |
| `journey.administer_and_recover` | UI-ACCOUNT |
| `journey.local_documentation` | UI-DOCS |
| `journey.data.manage_ingestion_and_coverage` | UI-DATA |
| `journey.artifacts.inspect_and_use` | UI-ART |
| `journey.jobs.inspect_and_operate` | UI-JOBS |
| `journey.live.manual_position_control` | UI-EXEC |
| `journey.live.inspect_and_reconcile` | UI-EXEC |
| `journey.monitoring.diagnose_service_or_datastore` | UI-OPS |

## Compatibility, proof and completion

This documentation revision has `none` impact on runtime APIs, schemas, persistence,
authorization and deployed behavior. The next gated library/detail client is planned
as `compatible-change` under the existing session/SSR constraints; verify that verdict
against the actual diff. Replacing default routes, framework `/docs`, API projections
or target grants requires classification at the corresponding future task; it is not
approved by this plan's existence.

Each block is complete only when its owned requirements and links have observed
boundary evidence, the functional registry's implementation scope is updated from
that evidence, and unmet dependencies remain explicit. Use browser evidence for UI,
real local API/domain integration for effects, and server permission proof for target
roles. Reuse accepted previous proof unless code/inputs changed or a new uncertainty
requires rechecking. No performance, provider or production claims from UI tests.

Backtests already has `local_journey_verified=true`; retain
`target_role_cutover_ready=false` until its separate prerequisites are fulfilled.
This does not complete UI-DATA, UI-ART, UI-JOBS or UI-EXEC or the full UI migration.
Apply the same distinction to subsequent slices. Publication/CI and runtime delivery
remain explicitly selected boundaries; the completed Backtests merge does not authorize
publication of future UI changes.

## Open prerequisites and handoff

| ID | Resolution / bounded assumption | Blocks |
|---|---|---|
| STRATEGIES-MAPPING | Resolved for authoring above: direct list/detail, optional selected dashboard, independent route gate/classic continuation and navigation-only return context; no reverse-provenance API assumed | No remaining owner decision for this slice; actual runtime/fixture binding must still be verified during implementation |
| LOCAL-PROOF | Reuse the established local Backtests QA setup; verify saved-strategy fixtures and required services for the next slice. A stopped localhost preview is an operational condition, not missing implementation | Next real-boundary verification; no retired host or unconfigured deployment target assumption |
| API-AUTHZ | Existing delegation/envelope and Backtests authz tickets; re-read actual status at execution/cutover | Target role acceptance/default cutover, not local own-resource integration |
| API-DATA / API-JOBS-ARTIFACTS / Q-JOB-ADAPTER | Separate scoped API/projection and identity/action mapping tasks from the functional contract | Respective later actions and complete cross-area journeys |
| Q-DRAFT / Q-MODELS | Reuse sources, resolve exact product choice before the affected task; no guessed durable draft or promotion rules | Respective later task, not Backtests |
| API-BACKTEST-MODES / API-ETA | Backend contracts/evidence; preserve artifact-only first scope and unavailable estimate | Broader Backtests functionality, not S1–S6 |

Next handoff is user-selected execution of the linked standalone Strategies prompt.
It carries D4, the resolved mapping, bounded touch zones and A1–A9 acceptance.
Do not reopen Backtests S1–S6, revive PACK-CLAIM, or automatically execute remaining
platform blocks. Existing Backtests prompt contracts and ledger remain unchanged.

Planning self-review: current implementation/acceptance and historical execution state
are separated; all 44 registry records and 18 journeys retain their owners; no API,
role, runtime, or full Strategies completion is inferred. Remaining proof is actual
local fixture/integration; target-role and later API dependencies remain separate.
The linked prompt defines implementation acceptance; authoring does not satisfy it.
