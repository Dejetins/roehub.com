# UI functional requirements coverage review — 2026-09-08

## Decision and scope

Verdict: `Ready with changes` for a bounded Backtests implementation;
`Not ready` to claim a complete, current, reconciled whole-product execution
specification. Most behavior already has sources. Missing visual states in v23
do not imply missing functional decisions.

The user's current direction is to reuse functional principles and behavior
from earlier versions and use v23 as the visual guide for subsequent UI work.
This overrides historical restrictions limiting visual reuse to Backtests.
Historical styling, six-theme requirements, mobile layouts and the retired
design workflow are not reactivated by functional reuse.

This report supplements `ROEHUB-UI-NEXT-STEPS-2026-09-08.md`: its first-journey
recommendation remains useful, but its source coverage was incomplete because
it considered only the surviving v1 inventory. No implementation or normative
registry was changed in this review.

Routing: `classification=review`; `execution_unit=functional coverage audit`;
primary skill `architecture-review`; companion `delivery-orchestrator`;
write scope this report; proof boundary static source/history/registry checks.
Baseline `c6f6e93e`; pre-existing `.codex/AGENTS.md` changes preserved.

## Sources and authority

| Source | Evidence and reuse boundary |
|---|---|
| Current product transformation requirements v1 | Accepted product goals, stack, responsive scope, accessibility/localization, data modes, progress, resources, roles and local/public separation |
| Current local screen registry v1 | 35 entries, including nonvisual/history entries; 12 journeys; per-screen required states |
| Current access/route contract v1 | 40 capabilities, 24 surface policies, 29 canonical route records and 7 migration records; target policy distinguished from current enforcement |
| `web-ui-v2-1-production-contract.md` | Historical implementation contract whose server authority, same-origin, secret protection, unknown-result reconciliation and rollback principles are retained by current documents; old visual/auth/runtime specifics need filtering |
| Domain docs, current API/DTO and Web code | Concrete validation, cancellation, materialization, session, lifecycle and error behavior; code is evidence, not a replacement product specification |
| `fd453b5a:docs/architecture/apps/web/*-v2.*` | Previously accepted product-completeness package: 44 screen entries, 43 surfaces, 18 journey identities; now historical after rollback |
| `fd453b5a:.codex/delivery/evidence/roehub-authenticated-platform-ui-design/2026-08-03-g1-atlas-r2-owner-acceptance.json` | Explicit historical acceptance receipt, boundary `source_and_atlas_only`; does not prove UI/API implementation |
| `48d23ad5:docs/architecture/apps/web/roehub-local-platform-journey-registry-v1.json` | 18 journeys and 61 cross-screen transitions, but `status=review`, `authority=proposal_for_product_owner_review`, `owner_decision_ref=null`; useful candidate, not evidence that every transition was accepted |

Historical files were inspected with `git show`, not restored into the working
tree. The next consolidation can reuse functional content with provenance and
current-contract checks without resurrecting the old program or treating all
historical routes and grants as automatically current.

## Functional coverage that must be retained

The historical v2 package retains all twelve current journeys and adds six:

| Functional area | Existing specification coverage | Implementation preparation |
|---|---|---|
| Sign-in and recovery | Passkey, fallback, expired session, safe `next`, identity outage; local-auth doc describes one-time bootstrap/recovery and sessions | Reuse current local auth; reject obsolete v2.1 OIDC-only wording |
| First launch | Owner/organization, passkey, recovery, catalog selection/skip, resumable failure | Backend primitives exist; complete routed setup journey is still target work |
| Data selection | Catalog, selection, coverage, pinned instruments, unavailable artifacts and source freshness | Bind to reference/selection APIs; verify target snapshot/cursor protocol separately |
| Ingestion and coverage — historical addition | `/data/ingestion`, backfill state, partial coverage, gaps, failed operation and safe retry | Carry functional requirement into reconciliation; map supported projections and actions |
| Exchange connections | Create, validate, bind, rotate, recheck, disconnect, archive, recent auth, safe status | Reuse lifecycle contracts; verify target role integration |
| Backtest configuration/run | Defaults, parameter validation, preflight, admission, submit, queue, measured progress, cancellation | Existing artifact path and job APIs support reuse; `auto/direct_db` remain explicit backend work |
| Result inspection/strategy creation | Materialization, variants, equity/drawdown/monthly/trades, CSV, compatibility and separate promotion | Retain immutable identities and idempotency; distinguish saved strategy from live launch |
| Strategy creation/control | Builder, clone, immutable spec, launch profile, run/stop/restart, blocked/unknown runtime | Resolve ambiguous legacy `update/edit` labels into spec clone versus mutable launch-profile configuration |
| Live observation | Paper/live, positions, orders, fills, risk blocks, freshness and safe stop | Existing projections do not by themselves prove all target read models |
| Manual position control — historical addition | Hosted open/close flow, size/close scope, preflight, confirmation, recent auth, durable execution identity | Preserve universal execution/risk path; no blind repeat |
| Execution inspection/reconciliation — historical addition | `/live/activity`, `/live/executions/{intent_id}`, signals/intents/orders/fills/rejections/notifications | Establish bounded read projections and permitted reconciliation commands |
| Models | Registry/detail, training, comparison, readiness, promotion and rollback | Current `/models` compatibility presentation is not the dedicated target workspace |
| Monitoring/runbook | Grouped services, freshness, impact, safe actions and local documentation | Preserve server-filtered diagnostics and incident context |
| Service/datastore diagnosis — historical addition | `/monitoring/{service_id}`, dependencies, backlog, metrics, jobs and runbook | Databases remain service identities, not separate Redis/PostgreSQL product sections |
| General jobs — historical addition | `/jobs`, `/jobs/{job_id}`, attempts, inputs/outputs, failure, cancellation/retry eligibility | Bind to general runtime contracts; do not copy Backtest retry semantics |
| Artifacts — historical addition | `/artifacts`, `/artifacts/{bundle_id}/{version}`, manifest, provenance, compatibility, storage and consumers | ArtifactStore domain exists, but its accepted v1 scope explicitly excludes a storage HTTP API |
| Settings/admin/recovery | Personal settings, notifications, sessions/audit, membership, budgets, extensions, backup/update/rollback | Separate organization rights from installation authority; use operation IDs/status/reconciliation |
| Documentation | Local versioned docs, visibility, 403/404, safe login continuation and compatibility redirects | Canonical product `/docs` requires a separate framework-doc route migration |

Public `roehub.com` retains its separate product requirements and surface
registry. This local-platform review does not certify its account, privacy,
downloads, release, or demo specification as implementation-complete.

## Already specified cross-cutting behavior

These principles should be translated into acceptance cases, not redesigned:

- Loading retains the last valid snapshot when appropriate; one failed panel
  does not hide healthy panels. Empty, unavailable, degraded and stale differ.
- Reads can retry within server limits; honor freshness and rate-limit hints.
- Mutations have pending/disabled state, outcome feedback, server authorization,
  applicable confirmation/recent auth, and redacted audit.
- A timeout after sending a mutation is not proof it failed. Read authoritative
  state or use the endpoint's supported idempotency/reconciliation contract;
  never repeat an unsafe command blindly. `X-Request-Id` is not idempotency.
- Cancellation requests do not prove terminal cancellation. Render the returned
  and subsequently observed state, preserve completed evidence, and respect
  races with completion.
- Backtest result detail may return `202` while materializing. Follow the
  provided retry/status contract without treating this as a failed job.
- Session expiry returns to authentication with a safe destination; 403,
  identity-service outage and missing resources have different meanings.
- Recovery is contextual: regain access, reload data, reconcile a submitted
  operation, or recover a job/installation. It is not one universal Retry button.
- Local UI supports 820/1024/1440, RU/EN, keyboard use, zoom and reduced motion;
  the new visual treatment follows v23 rather than historical theme counts.

Primary sources: current product baseline and IA; v2.1 sections `Контракт
состояний` and `Действия, роли и доверие`; current API errors document;
Backtest API/runtime, local authentication, isolated jobs and strategy contracts.

## Actual gaps and inconsistencies

| Assessment | Observed fact | Implication and smallest action | Verification |
|---|---|---|---|
| Gap / High: active coverage regressed | Current v1 has 35/12 entries/journeys; historical accepted v2 has 44/18 | Consolidate the nine additional records and six journeys as functional scope with provenance; omit retired workflow/style requirements | Reconcile current inventory, screen/access/route registries and task coverage |
| Partial / Medium: transitions are not universally settled | Historical journey graph is `review`; screen lists often name states without event/outcome details | Reuse graph as a candidate and validate only the selected journey's transitions; do not call the entire graph accepted | State/action/endpoint/outcome/next-action matrix for each implementing task |
| Gap / Medium: unsaved draft semantics | Current and historical editor entries contain `draft` and `unsaved_changes`; inspected sources do not specify persistence lifetime, reload or interrupted-auth restoration | Establish a bounded editor policy before its implementation; no global design restart | Navigation, discard, reload and re-auth acceptance cases |
| Risk / High: generic retry conflates contracts | Target Backtest operator retry creates a new job from an immutable input; general isolated runtime retries the same job with a new attempt and forbids retry after cancellation | Name the operation, entity identity and eligibility per domain; never infer a shared retry command | Domain-specific cancellation/completion races and retry identity tests |
| Partial / Medium: old implementation is not complete recovery proof | `api.js` marks mutation timeout/network/5xx as `outcomeUnknown`; Backtest `createJob` creates a fresh key per invocation and catches errors as text, then re-enables controls | Preserve required unknown-outcome behavior in new client; define command identity/reconciliation handling for submit before acceptance | Lost-response-after-server-accept scenario must not create a duplicate job |
| Partial / Medium: edit terminology | Older UI inventory says strategy `update`, while Strategy API defines immutable specs and separate profile updates | Map edit to the actual operation: clone/new spec versus launch-profile settings | Verify endpoint/DTO and returned strategy identity |
| Gap / High: target capability integration | Kernel accepted; delegation/envelope tickets ready; Backtests and other route integrations draft | Reuse these backlog items; require relevant API proof before target-role behavior acceptance | Actor/organization/ownership, recent-auth, origin, replay and audit tests |
| Partial / Medium: API delivery gaps | `direct_db`, calibrated ETA, dedicated Models, full setup, generic Jobs/Artifacts UI projections are not established by screen registries | Mark backend dependencies separately from missing product requirements; inspect each endpoint when scheduling its task | DTO/runtime evidence for each claimed feature; no placeholder presented as working |

The old Backtest code observation is source-level evidence, not a reproduced
runtime defect. No data mutation or fault injection was performed. No assertion
is made that every detailed domain document or every field was exhaustively
reviewed. Registry tests establish structural consistency, not product completeness.

## Recommended next steps

1. **Reconcile the functional baseline once.** Update the current inventory/IA
   companions from the recovered functional scope, with source references and
   a short table of retained, superseded and unresolved decisions. Keep v23 as
   visual guide. Output is a current functional specification; no staged design
   program, prompt pack, role choreography or mandatory atlas is needed.
2. **Prepare the first Backtests execution task from that baseline.** Include
   fields/defaults/validation, actions, states and transitions, endpoint/DTO,
   command identity, rights, error presentation, recovery, deep links and
   acceptance cases. Resolve only gaps that block this journey. Record artifact
   mode as an initial delivery boundary if selected, while retaining all three
   modes as the whole-product target.
3. **Implement foundation and the first real journey.** Use the already accepted
   React/TypeScript client architecture; first visible checkpoint can be
   history → selected job → variant/result. Complete configure/preflight/submit,
   progress/cancel and recovery, with the relevant server-policy dependencies.
   Add missing prototypes only for a concrete layout/interaction uncertainty.
4. **Extend along functional dependencies.** Result → strategy library/editor;
   data selection/ingestion and connections before execution; live/manual
   controls, execution history and monitoring. Schedule common Jobs and
   Artifacts projections where these journeys need them rather than dropping
   them from scope. Models, admin/maintenance and docs follow their contracts;
   first-launch completion remains required before clean-installation acceptance.
5. **Verify and switch incrementally.** Preserve SSR fallback and bookmarks;
   test real API-backed journeys, selected roles, failure/recovery, RU/EN and
   required widths. Publish or deploy only under separately selected authority.

Work that can start without whole-product completion: visual foundations,
read-only client composition and contract-backed prototypes. Work that cannot
be claimed complete early: unimplemented data modes, target role enforcement,
or unknown-result recovery demonstrated only by a mock.

## Checks and review

- `git show` inspected the named historical commits, accepted receipt and
  candidate graph without modifying historical files.
- JSON parsing/count and ID comparison found exactly nine v2-only screen
  entries; existing twelve journey identities are retained plus six additions.
- `.venv/bin/python -B -m pytest -q -p no:cacheprovider
  tests/unit/docs/test_roehub_ui_surface_inventory.py
  tests/unit/docs/test_roehub_local_platform_information_architecture.py`:
  **6 passed**. This proves current registry structure only.
- No browser rerun, backend integration tests, server installation, dependency
  installation or publication: this is a requirements review.
- Runtime/API/persistence impact of this report: `none`. Future route and
  capability adoption needs its own compatibility classification.
- Cold self-review: `Ready with changes`; no independent policy/security design
  review required because no policy, security contract or implementation changed.
- Residual risk: until functional consolidation is selected and completed,
  current registries alone omit historically specified scope; historical graphs
  and labels alone are insufficient to implement all event-level behavior.
