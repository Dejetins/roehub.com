# Roehub UI implementation plan v1

Deliver the local platform UI as complete, API-backed journeys using the accepted
v23 visual direction, starting with Backtests and retaining the full functional scope.

## Authority, status and execution artifacts

- Status: accepted delivery decomposition, 2026-09-08. The user accepted the
  general migration plan plus six-stage Backtests pack and requested their authoring.
  Detailed scheduling below is delegated planning, not proof of implementation.
- This request authorizes plan/prompt authoring only. Execution needs a subsequent
  user instruction selecting a stage; publication and runtime delivery are separate.
- The current [functional contract](roehub-ui-functional-contract-v1.md) and
  [registry](roehub-ui-functional-registry-v1.json) own scope and implementation tags.
  This plan owns sequencing and acceptance boundaries; it does not duplicate status.
- The [Backtests ticket](../../../../.codex/tickets/2026-09-08-roehub-backtests-client.md)
  remains the complete first-journey specification. The pack decomposes its execution,
  not its final acceptance. Its `ready` status is distinct from pack entry readiness.
- The retired UI design workflow and historical prompt packs remain historical.
  This is the newly selected bounded implementation workflow, not a global requirement
  for every future UI task. Ordinary bounded tickets may still execute directly.

Exactly three execution artifacts are linked for this pack:

| Field | Repository path |
|---|---|
| `plan_doc` | `docs/architecture/apps/web/roehub-ui-implementation-plan-v1.md` |
| `prompt_pack_dir` | `.codex/agents/generated/roehub-backtests-client-v1/` |
| `stage_ledger` | [.codex/agents/generated/roehub-backtests-client-v1/stage-ledger.md](../../../../.codex/agents/generated/roehub-backtests-client-v1/stage-ledger.md) |

Execution mode is `manual_sequential`: one selected stage per execution request.
Only the ledger stores mutable execution state, claims, permission and receipts.
No Goal, implementation claim, Git branch or deployment is initialized by authoring.
Later UI blocks are a dependency backlog, not executable stages of the Backtests pack.

## Sources and current baseline

Read sources selectively through each stage prompt. Material baseline sources are:

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
- [Backtests authz ticket](../../../../.codex/tickets/2026-07-20-roehub-authz-backtests.md):
  at authoring, `draft`, dependent on delegation core and browser mutation envelope.
  Capability kernel is accepted; those integration dependencies are not thereby complete.
- v23: `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html`.
  Preserve bytes and path; baseline SHA-256
  `3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.

The planned `apps/platform-web` workspace is not an existing implementation claim.
The source registry covers 44 records, 43 source surfaces and 18 journeys, including
nonvisual, QA and historical dispositions. No registry record is dropped by this plan.
Current-source observations do not constitute new browser, API or deployment proof.

## Delivery decisions

### D1. A small shared client foundation followed by complete journeys

Keep the accepted React/TypeScript/Vite client and bounded `@roehub/*` packages.
Feature composition belongs to `apps/platform-web`; packages contain reusable
tokens, UI, charts, localization and web contracts, not domain policy. Dependencies
flow from feature composition to those packages and to typed same-origin adapters.
`apps/web` retains session/proxy responsibilities; API/domain services own permission,
validation, admission, job state, strategy identity and calculated metrics.

The viable sequencing choices were one large UI migration or bounded journeys.
The selected journey approach provides useful integration feedback and a reversible
cutover. Its cost is temporary coexistence with SSR and explicit navigation across
both clients. Do not build a general component catalogue or workflow framework before
the first journey demonstrates a need. Do not add services or change domain storage.

### D2. Local integration first, route cutover separately

Add the client behind a disabled-by-default local feature setting; retain existing
URL identities, authenticated entry, safe login continuation and proxy behavior.
In-progress screens do not expose unwired actions. Unrelated destinations remain SSR.
No dual domain writes or schema migration is needed for this first slice: existing
APIs create the same jobs/strategies. Turning the setting off restores SSR presentation;
it does not undo jobs or strategies already created. Never promise rollback of those
side effects through a UI setting.

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

### D4. v23 is the visual guide, additional prototypes answer named questions

Implement settled compositions directly. If a specific unresolved arrangement or
interaction matters, record the question and affected acceptance in the current stage
report, resolve it in a small local experiment/prototype, and integrate the decision.
Routine visual choices are delegated; no mandatory whole-product prototype, atlas,
theme series or repeated approval of v23. Use current Product Design routing only if
the actual work requires a faithful prototype or visual exploration. Preserve local
820/1024/1440 widths, RU/EN, 200% zoom, keyboard access and reduced motion. Phone UI
and public-site redesign are outside this local-platform plan.

### D5. Keep dependencies explicit without expanding the first journey

Artifact-mode Backtests can use existing source/default APIs without a finished general
Artifacts, Data or Jobs UI. Domain progress/cancel in Backtests does not complete generic
Jobs. Saving a strategy may navigate to the existing SSR detail; it does not require the
new strategy editor. Shared links must use real, available destinations or state their
unavailability, never fake successful navigation. Later API changes require their own
bounded contract/task before a dependent UI action is exposed.

## Backtests stages and acceptance mapping

All stages are implementation slices. S6 alone checks the complete first journey.
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
| UI-BT | S1–S6 artifact-mode Backtests; later modes, operator retry and measured ETA each get a separate task | Current artifact APIs; full roles/cutover depend on Backtests A. No general Jobs/Artifacts UI prerequisite |
| UI-AUTH | New sign-in/recovery and first-launch setup compositions | F; existing local auth contracts; setup catalog/routing contract before resumable setup acceptance |
| UI-HOME | Dashboard summaries, shell search/notifications and cross-area readiness | F; actual role-filtered projections. Cross-area links gain full acceptance as their consumer routes become available |
| UI-DATA | Catalog/selection/coverage slice, then ingestion history/backfill/cancel/retry slice | F; API-DATA snapshot/cursor + ingestion projections/actions; job links need real capability-to-job identity, general Jobs detail needed for the complete cross-area journey |
| UI-ART | Artifact catalogue then manifest/version/provenance/compatibility and consumer navigation | F; API-JOBS-ARTIFACTS browser catalogue/detail projections; Backtests consumer exists after S3, Models consumer after UI-MODELS |
| UI-JOBS | Cross-capability read-only queue/detail, then eligible per-capability cancel/retry controls | F; API-JOBS-ARTIFACTS projections; resolve Q-JOB-ADAPTER before generic commands. Do not unify domain backtest retry with runtime attempt retry |
| UI-CONN | Connection creation/check/recovery/archive and strategy selection | F; existing exchange-connection contracts and target A; no secret readback; complete consumer link depends on available strategy detail |
| UI-STRAT | Library/detail, immutable create/clone editor, then launch-profile/run-control | F; strategy APIs; Q-DRAFT before durable draft behavior; UI-CONN/readiness and execution contracts before launch controls. Backtest save is already in S5 |
| UI-EXEC | Execution activity/detail and source-to-fill reconciliation, then live workspace/manual position control | F; execution projections/source identity; strategy/connection/risk/authority contracts for commands. UI-STRAT/UI-CONN integration before full manual-control journey; monitoring detail for full diagnostic links |
| UI-MODELS | Registry/detail/comparison, then train/promote/rollback consumers | F; model/experiment projections + Q-MODELS; data/artifact/job identities before exposing dependent actions; UI-ART/UI-JOBS for complete cross-area navigation |
| UI-OPS | Monitoring workspace, service/datastore detail, incident-to-job/runbook links | F; real monitoring/freshness/dependency projections; UI-JOBS/UI-DOCS for complete diagnostic journeys; any operational action needs its own eligible server contract |
| UI-ACCOUNT | Personal profile/preferences/notifications/security; admin members/resources/extensions/maintenance | F; existing account/admin contracts; installation authority and missing projections per task. Destructive/recovery operations require scoped proof; no inferred deployment authority |
| UI-DOCS | Version-matched local documentation, search, runbooks and safe continuation | F; documentation visibility/version contracts; explicitly migrate framework `/docs` before route cutover, preserving compatibility redirects where selected |

Recommended near-term order is Backtests S1–S6, then prepare the Jobs and
Data/Artifacts API dependencies and choose the next ready user journey. This is a
priority recommendation, not a false dependency forcing read-only Artifacts to wait
for every Jobs command. Strategies/Connections and execution projections can be
prepared independently where their own contracts are ready. Do not claim an entire
cross-area journey complete while any required terminal destination is unavailable.

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

Authoring this plan/pack has `none` impact on runtime APIs, schemas, persistence,
authorization and deployed behavior. The planned first client/build/feature setting
is `compatible-change` under the ticket's gate/SSR constraints. Replacing default
routes, framework `/docs`, API projections or target grants requires classification
at the corresponding future task; it is not approved by this plan's existence.

Each block is complete only when its owned requirements and links have observed
boundary evidence, the functional registry's implementation scope is updated from
that evidence, and unmet dependencies remain explicit. Use browser evidence for UI,
real local API/domain integration for effects, and server permission proof for target
roles. Reuse accepted previous proof unless code/inputs changed or a new uncertainty
requires rechecking. No performance, provider or production claims from UI tests.

At S6, write `local_journey_verified` only after all ticket criteria pass; retain
`target_role_cutover_ready` as a separate verdict. The pack does not complete UI-DATA,
UI-ART, UI-JOBS or UI-EXEC and does not claim the full UI migration shipped.
Publication/CI and runtime delivery are later explicitly selected boundaries.

## Open prerequisites and handoff

| ID | Resolution / bounded assumption | Blocks |
|---|---|---|
| PACK-CLAIM | Resolve an existing supported exclusive ledger updater and verify its atomic claim/ownership contract. Read-only search of current repo guidance/tools and installed runner found no supported implementation; manual mode/filesystem writes/validator are insufficient. Do not invent a lock or revive legacy tooling. A separate tooling scope is needed if no current executor can supply it | Initial pack entry/activation only; all six prompts can be authored and structurally validated now |
| LOCAL-PROOF | S1 discovers and verifies current local API/auth/data fixture setup and records a reproducible redacted recipe. No retired host or unconfigured deployment target assumption | Dependent real-boundary acceptance if required runtime is unavailable |
| API-AUTHZ | Existing delegation/envelope and Backtests authz tickets; re-read actual status at execution/cutover | Target role acceptance/default cutover, not local own-resource integration |
| API-DATA / API-JOBS-ARTIFACTS / Q-JOB-ADAPTER | Separate scoped API/projection and identity/action mapping tasks from the functional contract | Respective later actions and complete cross-area journeys |
| Q-DRAFT / Q-MODELS | Reuse sources, resolve exact product choice before the affected task; no guessed durable draft or promotion rules | Respective later task, not Backtests |
| API-BACKTEST-MODES / API-ETA | Backend contracts/evidence; preserve artifact-only first scope and unavailable estimate | Broader Backtests functionality, not S1–S6 |

The initial ledger is an unclaimed draft. Its authoring preflight records PACK-CLAIM;
none of the six rows is executable until the capability is resolved and entry rechecked.
This is a technical capability gap, not a request for renewed visual/product approval.
Once resolved within authorized tooling/authoring scope, enable only dependency-free
S1, validate entry, and hand off to `staged-plan-runner` on an implementation request.
Never edit active stage contracts under the same ID or bypass the ledger to run a prompt.

Future changes update the canonical ticket/functional contract only where behavior
changes, preserve accepted evidence, and follow prompt revision rules for active packs.
Additional future block packs should use their own focused accepted source; do not
mutate this pack's bound plan casually after execution begins.
