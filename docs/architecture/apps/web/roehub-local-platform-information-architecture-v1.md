# Roehub local-platform information architecture v1

## Status and authority

- Status: `accepted target architecture`.
- Product-owner decision: accepted on `2026-07-20`.
- Design-tool amendment: future Roehub design handoff uses the registered Figma
  workspace from
  [`roehub-figma-design-delivery-standard-v1.md`](../../ui/roehub-figma-design-delivery-standard-v1.md).
  Penpot wording below is retained as `2026-07-20` historical scope evidence.
- Delivery ticket: `ROEHUB-LOCAL-UI-IA-2026-07-20`.
- Scope: the self-hosted local platform only.
- This document converts the accepted surface inventory into a target screen,
  navigation, route, and authorization structure.
- It is not a design-system contract, Penpot artifact, Web implementation, API
  implementation, or evidence that target behavior already exists.
- The public `roehub.com` site remains a separate trust and delivery boundary.

Machine-readable companions:

- [local screen registry](roehub-local-platform-screen-registry-v1.json);
- [access and route contract](roehub-local-platform-access-and-route-contract-v1.json);
- source [surface inventory](roehub-ui-surface-inventory-v1.md) and
  [surface registry](roehub-ui-surface-registry-v1.json).

## Outcome and non-goals

The architecture supplies the final product structure needed before design:

- every inventoried local surface has a target screen or an explicit non-visual
  or historical disposition;
- every canonical journey has an entry point and terminal screen;
- navigation is stable enough to create Penpot boards without inventing routes;
- permissions are expressed as server capabilities rather than hidden UI;
- current enforcement and future policy remain visibly different;
- compatibility URLs have explicit migration behavior.

This task deliberately does not define visual tokens, component anatomy,
typography, chart styling, spacing, motion, or Penpot page IDs. It also does not
resolve implementation work for first launch, data modes, ETA, server roles, or
redirects; it makes those dependencies explicit.

## Observed facts

1. The current Web service is a FastAPI/Jinja application with an authenticated
   page gate and a same-origin `/api/*` proxy.
2. The accepted inventory contains 33 local surface records and 12 local
   journeys. It also records non-product framework routes and one historical
   React prototype.
3. The identity domain already defines organization roles `owner`, `admin`,
   `operator`, `trader`, and `viewer`, plus permissions for organization,
   members, plugins, operations, trading, mainnet approval, and audit.
4. `installation_owner` is separate authority, not another organization role.
5. Administrative APIs use organization permissions and a distinct
   installation-owner check for installation-level operations.
6. Strategy and backtest APIs primarily enforce authenticated user ownership
   and domain/runtime checks. They do not yet implement the complete target role
   matrix.
7. Connection mutations already include same-origin and recent-auth boundaries,
   but the target organization-role split is incomplete.
8. `/models` exists, but reuses the current strategies presentation.
9. `/runbooks/{runbook_id}` exists, while the accepted canonical route is under
   `/docs/operator/runbooks/`.
10. Both the Web and core API FastAPI applications generate documentation. The
    Web application claims `/docs`, while the core API documentation is also
    reachable through the same-origin proxy as `/api/docs`. The target local
    user documentation needs `/docs`, and neither framework surface may remain
    an anonymous alternative.

## Product structure

### Navigation model

The authenticated shell uses a stable grouped navigation rather than one flat
list. At width `1440` it may remain expanded; at `1024` and `820` it may become
compact, but the group and route semantics do not change.

| Group | Canonical destinations | Purpose |
|---|---|---|
| Overview | `/dashboard` | Installation and organization summary, readiness, recent work, and guided next actions. |
| Research | `/data`, `/strategies`, `/backtests`, `/models` | Data selection, strategy creation, experiments, backtests, and models. |
| Operations | `/live`, `/monitoring`, `/connections` | Runtime observation, safe operational action, service health, and exchange connectivity. |
| System | `/docs`, `/settings/*`, `/admin/*` | Local documentation, personal preferences, organization administration, resources, extensions, updates, and recovery. |

The shell also owns organization context, command search, notifications,
personal preferences, locale, theme, and session state. These controls do not
grant capabilities; the server filters every destination and action.

### Route-level screen structure

#### Authentication and first launch

- `/login` is the canonical sign-in and recovery gateway.
- `/setup/{step}` is available only while the installation is uninitialized or
  while an authenticated installation owner resumes an incomplete setup.
- The first-launch flow covers owner creation, initial organization, passkey,
  recovery, catalog snapshot, instrument selection, skip, and resumable error
  states.
- `/register` is not a canonical self-hosted registration surface. It remains a
  state-aware compatibility gateway until migration is complete.

#### Overview and research

- `/dashboard` contains the overview and truthful onboarding/empty state.
- `/data` becomes the canonical market-data catalog and organization selection
  workspace; market data no longer hides inside general settings.
- `/strategies`, `/strategies/new`, and `/strategies/{strategy_id}` separate the
  library, editor, and controlled runtime detail.
- `/backtests`, `/backtests/new`, and `/backtests/{job_id}` separate history,
  configuration/preflight, and measured progress/results.
- `/models` is the canonical model/experiment registry;
  `/models/{model_id}` provides comparison, readiness, promotion, and rollback.

#### Operations

- `/live` is a dedicated target operations workspace. It combines server-owned
  execution projections without taking ownership away from strategy detail or
  monitoring.
- `/monitoring` groups services by platform core, data, computation, trading,
  security, and extensions. A broad unstructured process list is not the main
  presentation.
- `/connections` owns exchange and integration readiness, validation, lifecycle,
  and strategy binding. Stored secret material is never revealed.

#### Settings, administration, and documentation

- Personal settings use `/settings/profile`, `/settings/preferences`,
  `/settings/notifications`, and `/settings/security`.
- Administration uses `/admin`, `/admin/members`, `/admin/resources`,
  `/admin/extensions`, and `/admin/maintenance`.
- `/docs` is reserved for installed, version-matched user documentation.
- `/docs/operator/runbooks/{runbook_id}/` is the canonical role-scoped runbook
  route.
- Web framework OpenAPI/Swagger/ReDoc routes move under `/_internal/web-api/*`;
  core API documentation moves under `/_internal/core-api/*`. Both are disabled
  by default outside an explicitly enabled and authenticated development
  environment. The proxy does not preserve anonymous `/api/docs` aliases.

### Non-product and historical dispositions

- health endpoints, the same-origin proxy, and static asset delivery remain
  non-visual contracts;
- the plugin panel laboratory remains feature-flagged QA-only behavior and is
  absent from product navigation;
- framework API documentation is an internal developer surface;
- `prototypes/roehub-v2/` remains historical evidence and cannot authorize a
  board, route, role, or responsive target.

## Canonical journey rules

The machine registry contains all entry and terminal screen IDs. The following
rules govern their design:

1. **Sign in and continue:** preserve a sanitized `next`; never redirect to an
   external origin; distinguish invalid session from identity-service failure.
2. **First launch:** setup is resumable, selection is skippable, and an empty
   dashboard explains the next safe action.
3. **Select instruments:** show catalog snapshot identity, provider segment,
   coverage, resource impact, and strategy pinning before mutation.
4. **Manage connections:** separate status, binding, credential input, rotation,
   recheck, disconnect, and archive; never display a stored secret.
5. **Configure backtest:** expose `auto`, `direct_db`, and `artifact`, source
   freshness, coverage, preflight, and resource admission.
6. **Inspect and promote:** keep result evidence visible while requiring a
   separate confirmation and server compatibility/risk gate.
7. **Control strategy:** distinguish edit, run, safe stop, restart, and manual
   trading. A visible control is not authorization.
8. **Observe live execution:** show paper/live identity, freshness, risk blocks,
   unknown execution state, safe stop, and reconciliation separately.
9. **Operate models:** distinguish training, readiness, promotion, and rollback;
   operator read access does not imply model mutation.
10. **Monitor and follow a runbook:** preserve the incident context when opening
    permitted local documentation; show `403` rather than a misleading `404`
    when policy requires an explicit restriction state.
11. **Administer and recover:** distinguish organization authority from physical
    installation authority and require recent authentication where specified.
12. **Read local documentation:** remain version matched and locally available;
    unknown content returns `404` and operator-only content is server filtered.

## Authorization architecture

### Role and authority model

The canonical organization roles retain their current persisted names:

| Product label | Server role | Meaning |
|---|---|---|
| Owner | `owner` | Full organization authority, still bounded by installation trust and physical ceilings. |
| Administrator | `admin` | Organization administration and explicitly delegated product capabilities; cannot grant or remove owner authority. |
| Operator | `operator` | Server-filtered operational visibility and an allowlisted safe-action subset without secret or trading escalation. |
| Researcher / trader | `trader` | Own research, strategies, models, and backtests within organization and installation limits. |
| Read only | `viewer` | Server-filtered read projections and permitted completed results only. |

`installation_owner` is an overlay authority for bootstrap, trust, physical
resource ceilings, backup recovery, and update rollback. It does not by itself
grant organization membership, data visibility, trading authority, or secret
access. The initial local owner may hold both authorities, but the server must
check each independently.

### Enforcement rules

1. The API returns only the data allowed for the actor; the client never fetches
   full data and hides it cosmetically.
2. Every mutation checks the selected organization, role/capability, resource
   scope, object ownership where required, and domain invariants.
3. Recent authentication, same-origin, CSRF, idempotency, risk admission, and
   confirmation are additional gates, not replacements for authorization.
4. `operator` receives safe stop, reconciliation, recheck, disconnect, queue
   cancel/retry, and allowlisted operational actions only. It receives no
   credential rotation, secret material, model promotion, manual trade, or
   general strategy/backtest mutation.
5. `trader` mutations remain own-resource scoped unless a separate accepted
   contract defines a shared resource.
6. `viewer` cannot mutate product or organization state.
7. Stored exchange secrets are never revealable by any role. Rotation accepts
   new input after recent authentication and returns redacted status only.
8. Unknown or absent capability results in deny. The UI may explain the denial,
   but cannot override it.

### Delegated capabilities

Delegation is a target server record, not a string interpreted by the client and
not an inference from the current static role permission map.

- only an organization `owner` with recent authentication may grant or revoke;
- the grant names one organization, one active member, one exact delegable
  capability, and an explicit resource scope;
- a grantee cannot delegate further, self-grant, or receive authority exceeding
  the owner's own organization authority;
- `installation_owner`, role management, mainnet approval, physical resources,
  recovery, and secret reveal are never delegable;
- absent, expired, or revoked records deny immediately;
- grant and revoke operations are audited;
- the entire delegation store and enforcement path is
  `target_not_implemented`.

The machine contract lists the exact delegable capability IDs and required grant
fields. Existing `admin` permissions remain valid only for the organization and
plugin operations already implemented; they do not silently satisfy a new
product capability.

### Closed operator safe-action set

`operator` does not receive a generic interpretation of `operations.execute`.
The target server allowlist contains only:

- stop an already running strategy;
- cancel a queued/running backtest;
- retry a failed/cancelled backtest as a new job with the same immutable input
  snapshot and fresh resource admission;
- reconcile execution without creating an order or expanding exposure;
- recheck or disconnect a connection without secret access;
- run read-only diagnostics;
- restart an allowlisted service only when it belongs exclusively to the
  selected organization, the observed state is `stopped`, and the server
  advertises `action_ref=restart_service`.

An organization operator cannot restart a platform-core, data, computation,
security, or other service shared by multiple organizations. Any shared
installation-service restart requires a separate `installation_owner` check,
even when the actor also has the organization role `operator`.

All other actions deny, including strategy start/restart/manual trade, backtest
creation/deletion/promotion, model mutation, credential creation/rotation,
arbitrary service start/stop, installation lifecycle, plugin lifecycle, role
grant, and resource-ceiling changes. The current broad
`operator -> operations.execute` mapping is explicitly insufficient for this
target and must not be reused without the narrower action policy.

### Browser mutation envelope

Every cookie-authenticated browser mutation uses one fail-closed server
envelope:

1. authenticated principal;
2. valid same-origin evidence or CSRF proof, rejected when required evidence is
   absent;
3. selected organization and object scope;
4. role, delegated capability, or installation authority;
5. validated request contract;
6. idempotency for retriable commands, including payload/key conflict rejection;
7. recent authentication for credential, role, mainnet, model promotion,
   physical resource, and recovery actions;
8. audit evidence for security or operational changes.

Current strategy, backtest, and market-data selection mutations do not uniformly
implement this envelope. Those gaps remain `target_not_implemented`; the
architecture does not describe them as current protection.

Installation-wide trust changes use the distinct
`installation.trust.manage` capability. It is non-delegable, requires
`installation_owner`, recent authentication, explicit confirmation, validated
trust policy, no secret echo, and an audit event. This capability is
`target_not_implemented`; organization ownership or administration alone does
not grant it.

The full per-capability grants, denials, conditions, current evidence, and
surface mappings live in the access contract. Entries marked
`target_not_implemented` or describing an incomplete current policy are backlog
requirements, not claims about deployed protection.

### Security-sensitive implementation gaps

| Gap | Current evidence | Required server change before UI claim |
|---|---|---|
| Dashboard role projection | Authenticated read model | Produce role- and scope-filtered projections. |
| Strategy role policy | Primarily user ownership and domain checks | Add role/capability checks for read, manage, run, safe stop, and manual trade. |
| Backtest role policy | User ownership and paid-level checks | Add role/capability checks and an operator queue-safe subset. |
| Operator operations | Current `operations.execute` permission | Narrow actions through a server allowlist; installation operations retain installation-owner checks. |
| Installation trust | No complete target capability contract | Add non-delegable `installation.trust.manage` enforcement with recent authentication, confirmation, validation, redaction, and audit. |
| Connection roles | Ownership, same-origin, recent-auth checks | Add organization role/capability policy; preserve no-secret invariant. |
| Model roles | Dedicated target workspace absent | Add model read/manage/promote capabilities before exposing controls. |
| Runbook visibility | Authenticated file lookup | Add per-document server visibility and explicit `403` behavior. |

## Compatibility and migration

### Route policy

Routes migrate in three phases:

1. **Introduce:** add the canonical route while the current route continues to
   work; record compatibility telemetry only when explicitly allowed locally.
2. **Canonicalize:** internal navigation and generated links use the new route;
   recognized old routes redirect without losing safe resource identity.
3. **Retain redirect:** remove the old renderer only after browser and bookmark
   proof; keep permanent redirects for supported legacy URLs.

Specific decisions:

- `/strategies?mode=rl_ml` becomes a `308` redirect to `/models` only after the
  dedicated models workspace is available.
- `/runbooks/{runbook_id}` becomes a `308` redirect only after versioned local
  documentation and role visibility are implemented.
- `/settings?tab=market-data` maps to `/data`; recognized settings tabs map to
  nested settings routes.
- `/` and `/register` remain state-dependent `302` gateways because installation
  initialization and session state can change their correct target.
- taking `/docs` and proxied `/api/docs` away from FastAPI documentation is a
  `breaking-change` for developer-facing framework URLs. The API schemas are
  unchanged; Web and core API developer routes move to separate authenticated
  `/_internal/*` namespaces and remain disabled by default.

### Rollback

- A new client route may be disabled while the old SSR renderer remains intact.
- Compatibility handlers remain until the corresponding new route passes
  browser, authorization, deep-link, and recovery proof.
- Moving framework documentation is rolled back independently from product
  documentation if developer tooling fails, but anonymous production exposure
  is not restored as a fallback.
- No persisted role or permission is renamed by this architecture task.

## Contract impact

| Surface | Classification | Reason |
|---|---|---|
| Architecture documents and registries | `compatible-change` | Adds target structure without changing runtime. |
| Existing product routes | `none` in this task | No runtime route is changed here. |
| Future nested settings, data, live, models detail, and docs routes | `compatible-change` | Add canonical targets with compatibility handling. |
| Future FastAPI framework-documentation URLs | `breaking-change` | `/docs` is reassigned to product documentation. |
| Persisted organization role names | `none` | Existing `owner/admin/operator/trader/viewer` values are retained. |
| Future authorization behavior | `breaking-change` for previously over-broad access | Server denial may remove actions currently reachable by authenticated users. |
| Public-site identity and routes | `none` | Explicitly outside this contract. |
| Penpot and Web implementation | `none` | No design or runtime write occurs. |

## Design handoff boundary

After user acceptance, the next allowed artifact is a separate design-system
contract. It may consume:

- stable `screen_id` and `surface_id` mappings;
- navigation groups;
- required state names;
- supported widths `820`, `1024`, and `1440`;
- capability-driven visibility and forbidden states.

It must not reinterpret role grants, invent public-site boards, or claim an
unimplemented API. Penpot work remains unauthorized until the design-system
contract is separately accepted and the user supplies the intended Roehub file
IDs.

## Proof boundary

Acceptance of this architecture proves documentation consistency only:

- all 33 accepted local surfaces and 12 journeys are accounted for;
- identifiers, screen routes, capability references, widths, and exclusions are
  deterministic;
- current evidence and target policy are separated;
- the role matrix received an independent security review;
- documentation index, project map, focused tests, local links, and diff checks
  pass.

It does not prove browser behavior, API authorization, redirect behavior,
Penpot completeness, accessibility runtime, persistence, performance, release,
or deployment.

## Sources

- `docs/architecture/platform/roehub-product-transformation-requirements-v1.md`
- `docs/architecture/apps/web/roehub-ui-surface-inventory-v1.md`
- `docs/architecture/apps/web/roehub-ui-surface-registry-v1.json`
- `src/trading/contexts/identity/domain/entities/organization.py`
- `src/trading/contexts/identity/application/use_cases/organizations.py`
- `apps/api/routes/admin.py`
- `apps/api/routes/strategies.py`
- `apps/api/routes/backtests.py`
- `apps/api/routes/ui_account.py`
- `apps/api/routes/market_data_reference.py`
- `apps/web/main/app.py`
