---
artifact_kind: delivery_spec
delivery_contract: global/v1
delivery_schema_version: 1
spec_version: 1.1.0
status: accepted
---

# Roehub Linear-workspace UI transition specification

## Problem and outcome

Roehub's authenticated local platform is currently served through FastAPI,
Jinja/HTMX, and JavaScript islands. Accepted IA and design-system contracts
describe the product, but they predate the decision to move the authenticated
workspace to the shared Linear-like React frontend baseline.

The target is a fast, native-feeling Roehub local platform with measurable
functional, structural, visual-rhythm, and behavioral fidelity to selected
Linear reference surfaces while retaining Roehub trading semantics,
authorization, REST/SSE projections, ECharts rules, deployment topology, and all
backend delivery plans. Fidelity is formal rather than literal: it preserves
useful block roles and relationships without cloning Linear screens.

## Accepted decisions

- Only the authenticated Web application is replatformed. Public `roehub.com`,
  backend services, PostgreSQL/Redis, REST/SSE, jobs, trading and authorization
  plans remain unchanged.
- The frontend baseline is React + TypeScript + Vite + MobX + TanStack Query +
  styled-components + CSS semantic tokens.
- The four shipped base themes are exactly `abyss`, `graphite`, `frost`, and
  `paper`; `graphite` is the UI default.
- Inter Variable is self-hosted and versioned. Roehub brand and financial
  semantic colors remain project-owned.
- Existing SSR routes remain the migration fallback until their React slices
  are accepted in a real browser. SSR and React may coexist behind an explicit
  route boundary; no big-bang rewrite is allowed.
- The first runtime golden slice is the read-only Backtest list/detail journey.
  It must exercise dense tables, filters, charts, freshness/degraded states,
  list-to-detail routing, a resizable right pane, keyboard behavior, four
  themes, and real server projections without enabling a trading mutation.
- The server-authorization stream continues independently and is not blocked or
  rewritten by this UI transition.
- Linear reference surfaces are translated through functional-block
  equivalence. Roehub preserves analogous interaction roles and information
  hierarchy while owning domain meaning, data, routes, permissions, copy,
  visual tokens, and composition.

## Current authority and supersession

The accepted local-platform information architecture, screen registry,
access/route contract, server capabilities, and historical runtime evidence
remain authoritative product inputs.

The accepted `ROEHUB-LOCAL-UI-DESIGN-SYSTEM-CONTRACT-2026-07-20` evidence is
retained. Its six-theme and implementation-independent target is superseded for
future UI implementation by this specification and the shared standard. It is
not relabelled as browser or Penpot proof.

Historical SSR UI and `prototypes/roehub-v2/` remain migration observations, not
the new visual source of truth. No accepted server ticket is superseded.

## Formal reference translation contract

The Linear reference is a functional topology, not a page template. Every
selected Roehub screen must provide an analogue for each applicable reference
block below or record why the block is not meaningful for that Roehub journey.
Blocks may be combined, reordered, resized, or omitted only when the resulting
Roehub flow preserves the user task, hierarchy, state continuity, keyboard
access, and authoritative data ownership.

| Reference role | Required Roehub interpretation |
|---|---|
| Global workspace navigation | Authenticated Roehub navigation, workspace/organization context, route groups, search/commands, and permission-filtered destinations. |
| Page identity and view tabs | Canonical route identity, domain-object title/context, available views, and contextual actions. |
| Primary work surface | The authoritative Roehub table, chart, detail, form, or operational surface for the current task. |
| Properties/summary block | Domain metadata such as status, freshness, ownership, environment, dates, and capability-relevant state when those concepts exist. |
| Resources/related artifacts | Relevant reports, datasets, run artifacts, documentation, or related domain objects. |
| Contextual side panel | Secondary properties, controls, diagnostics, or details that preserve the main task and route context. |
| Progress/milestones/health | Real progress, lifecycle, health, or execution state only when backed by a Roehub domain contract; never decorative fabricated progress. |
| Activity/history | Audit, events, jobs, reconciliations, notifications, or other authoritative domain history. |
| Loading/empty/error/degraded/forbidden states | Truthful Roehub states backed by current server authorization and projection contracts. |

Literal replication is prohibited. Do not copy Linear branding, text, product
taxonomy, proprietary assets, exact pixel coordinates, or unsupported concepts.
Do not add labels, assignees, milestones, progress, activity, or collaboration
objects merely because they appear in the reference. Each Penpot or runtime
surface must include a mapping table with `reference block`, `Roehub function`,
`authoritative source`, `required states`, and `evidence or justified omission`.

## Compatibility

| Surface | Classification | Decision |
|---|---|---|
| Backend/API/persistence/trading | `none` | Existing runtime and authorization contracts remain authoritative. |
| Authenticated frontend | `breaking-change with route fallback` | React application introduced alongside SSR during migration. |
| Theme IDs | `breaking-change before new design implementation` | `slate` and `sand` are removed from the future target. |
| Routes/capabilities | `compatible-change` | Existing canonical route and capability identities are preserved. |
| Public site | `none` | Excluded from this transition. |
| Penpot | `new target required` | Future vNext design is accepted separately. |

## Delivery graph

The executable graph is
`.codex/delivery/graphs/roehub-linear-workspace-ui-transition-v1.json`.
One ready ticket is one execution unit.

1. Complete reference, motion, geometry, keyboard, accessibility, and
   four-theme evidence.
2. Prove React coexistence with the FastAPI/Jinja gateway, MobX/Query state
   boundaries, typed REST/SSE adapters, route fallback, and performance harness.
3. Create and accept Penpot vNext foundations and representative compositions.
4. Implement the React application shell behind the reversible route boundary.
5. Implement the real read-only Backtest list/detail golden slice.

Further route-cluster tickets are derived after the golden slice establishes
measured architecture and browser behavior. Server authorization tickets may
run in parallel on disjoint paths.

## Reference and proof boundary

The shared standard and manifest are under `docs/architecture/ui/`. The supplied
archive and supplemental user captures contain Roehub-named Linear workspace
states and are sufficient for dark shell and functional-block reference, not
for complete visual or motion acceptance. The reference ticket must close or
explicitly waive every missing manifest item.

Penpot proves design intent only. Source tests do not prove browser performance.
The shell and golden slice require real-browser traces on declared local
hardware with client overhead, REST/SSE latency, and render time separated.

## Non-goals

- No Node.js backend, GraphQL, Temporal, proprietary sync framework, Kubernetes,
  Cloudflare, or database migration.
- No change to trading, backtest, authorization, job, or notification semantics.
- No public website or phone-authoring redesign.
- No copying Linear branding, text, source code, private assets, or product
  entities.
- No one-to-one screen cloning or Roehub feature invention solely to reproduce a
  Linear block.
- No deletion of accepted historical evidence.
