---
artifact_kind: delivery_spec
delivery_contract: global/v1
delivery_schema_version: 1
spec_version: 1.0.0
status: accepted
---

# Roehub Linear-workspace UI transition specification

## Problem and outcome

Roehub's authenticated local platform is currently served through FastAPI,
Jinja/HTMX, and JavaScript islands. Accepted IA and design-system contracts
describe the product, but they predate the decision to move the authenticated
workspace to the shared Linear-like React frontend baseline.

The target is a fast, native-feeling Roehub local platform with maximum
measurable visual and behavioral fidelity to selected Linear reference surfaces
while retaining Roehub trading semantics, authorization, REST/SSE projections,
ECharts rules, deployment topology, and all backend delivery plans.

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
archive captures Roehub-named Linear workspace states and is sufficient for the
dark shell, not for complete visual or motion acceptance. The first ticket must
close or explicitly waive every missing manifest item.

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
- No deletion of accepted historical evidence.
