---
document_id: LINEAR-WORKSPACE-UI-TRANSITION-STANDARD-V1
status: accepted
version: 1.0.0
language: en
updated_at: 2026-07-20
---

# Linear workspace UI transition standard v1

## Purpose and authority

This repository-neutral contract defines the shared UI replatforming method for
the authenticated Custometry and Roehub applications. It is accepted product
direction for observable Web presentation and interaction. Each repository owns
its routes, domain composition, authorization, data contracts, copy, branding,
delivery graph, and runtime evidence separately.

This contract does not authorize a backend replatform. Existing Python domain
services, persistence, REST APIs, and SSE projections remain authoritative and
their accepted plans do not change. Node.js, GraphQL, Temporal, a proprietary
sync framework, cloud infrastructure, and public-site redesign are outside this
transition.

"Linear fidelity" means maximum measurable visual and behavioral fidelity on
selected reference surfaces: density, hierarchy, shell geometry, panels,
keyboard behavior, motion, focus, perceived latency, and state continuity. It
does not mean copying Linear trademarks, text, product entities, source code,
private assets, or undocumented authorization behavior.

## Shared frontend baseline

Both authenticated applications use the same baseline:

- React, TypeScript, and Vite;
- MobX for client, workspace, navigation, panel, command, and optimistic UI
  coordination;
- TanStack Query for authoritative REST/SSE server projections, invalidation,
  stale-while-revalidate, and cancellation;
- styled-components for typed component composition, with CSS custom properties
  as the runtime semantic-token and white-label boundary;
- self-hosted, versioned Inter Variable with a system-sans fallback;
- route-backed navigation and detail/focus surfaces with correct browser
  history, deep links, Back, refresh, and Escape behavior;
- project-owned typed API adapters; no UI store may become a second source of
  truth for server authorization, jobs, trading, analytics, or persistence;
- Apache ECharts through each project's existing bounded chart contract;
- Vitest/Testing Library for component behavior and Playwright plus accessibility
  tooling for browser acceptance.

The state boundary is deliberate. MobX owns fast local coordination and may
show reversible optimistic presentation only when the API contract permits it.
TanStack Query owns remote snapshots. A mutation is never reported as durable,
authorized, executed, or successful before authoritative server confirmation.

## Common theme set

Both products ship exactly four base theme IDs, ordered from darkest to
lightest:

1. `abyss` — near-black concentration theme;
2. `graphite` — dark default workspace theme;
3. `frost` — cool soft-light theme;
4. `paper` — bright daylight and export theme.

`graphite` is the authenticated UI default and `paper` is the default static
report/email/XLSX rendering profile unless a project contract explicitly pins
another permitted theme. A `system` preference may resolve to one of the four
IDs but is not a fifth theme. Theme switching is immediate, requires no reload,
and cannot change domain values, permissions, request identity, result hashes,
or cache identity.

Products keep their own brand accents, logos, icon mappings, financial or
analytical semantic colors, and validated white-label overrides. Overrides use
semantic tokens and cannot inject arbitrary CSS, JavaScript, remote fonts, or
untrusted assets. All four themes must satisfy WCAG 2.2 AA for supported text,
focus, controls, status, and chart/table alternatives.

## Observable workspace grammar

### Shell and navigation

- The authenticated application is a stable desktop workspace, not a sequence
  of disconnected full-page documents.
- Primary sidebar, optional secondary pane, top/context bar, content surface,
  and optional detail pane use continuous separators and restrained elevation.
- Expanded navigation shows icon plus label. Collapsed navigation shows icons
  only and retains accessible names and tooltips.
- The expanded sidebar is pointer-resizable through a visible or discoverable
  separator. The initial implementation must validate a default near `240px`,
  a bounded range near `208-320px`, a collapsed rail near `44-48px`, and an
  `8px` effective drag target before freezing tokens.
- Resize is live at 60 fps, uses `col-resize`, never selects page text, persists
  as a presentation preference, and has keyboard-accessible step/reset actions.
- Double-clicking the resize separator may reset the default width. Narrowing
  below the collapse threshold changes to the icon rail without text blur or
  bounce.
- Navigation groups may be user-configurable only inside product permissions;
  hidden navigation never grants or revokes server access.

### Panels and route behavior

- Selecting a dense list row may open a right detail pane while preserving the
  list, scroll position, selection, filters, and URL-addressable entity state.
- Drawers and transient popovers do not become nested modal stacks. A full
  analysis or editing workspace is route-backed.
- Browser Back and Escape return to the exact origin state when safe; refresh
  restores the canonical route and server-authorized view.
- Pane widths may be pointer-resizable when the content benefits from it. Every
  resizable pane has min/max constraints, keyboard controls, persistence, and a
  reset action.

### Density and surfaces

- Base spacing follows a `4px` rhythm; common increments are `4`, `8`, `12`,
  `16`, `20`, and `24px`.
- Default body and dense table typography is approximately `13px`, secondary
  text `12px`, with tabular numerals where value comparison benefits.
- Rows, cards, and panels use subtle borders and state changes instead of large
  shadows or decorative gradients.
- Cards denote independent objects or decisions; ordinary sections are not
  wrapped in cards merely for decoration.
- Hover reveals contextual actions without shifting layout. Focus is visible,
  immediate, and never replaced by hover-only behavior.

### Commands, keyboard, and overlays

- `Cmd/Ctrl+K` opens a global command palette whose results are filtered by
  effective permissions and current workspace.
- Keyboard-first paths cover navigation, search, filter, selection, open/close,
  pane resize, primary actions, tables, and chart-to-table alternatives.
- Menus, popovers, comboboxes, dialogs, and drawers use accessible headless
  semantics and deterministic focus return.
- Destructive or authoritative actions always retain project-specific server
  guards, recent-auth requirements, confirmation, reconciliation, and audit.

## Motion and continuity

The interface must feel like fast native software. Motion confirms spatial
relationships; it never fabricates progress or execution success.

| Interaction | Target duration | Rule |
|---|---:|---|
| Hover/focus/pressed feedback | `0-100ms` | Visible on the next frame; focus semantics are immediate. |
| Popover or menu | `100-140ms` | Fade and minimal scale; no bounce. |
| Tab indicator | `120-160ms` | Previous authorized data remains until replacement is ready. |
| Route content | `120-160ms` | Stable shell; short content fade, no full-screen slide. |
| Modal | `160-200ms` | Fade plus very small scale. |
| Sidebar expand/collapse | `180-220ms` | No overshoot, text blur, or layout flash. |
| Drawer/detail pane | `200-240ms` | Direction reflects pane origin. |
| Safe chart transition | `160-240ms` | Disabled for domain/axis changes, dense/live series, or misleading interpolation. |

Refresh preserves the previous authorized result and shows local freshness and
loading status. Skeletons are primarily for first load. Tables do not animate
rows through space; changed values use a short non-color-only highlight and a
textual status. `prefers-reduced-motion` removes translate, scale, bounce,
shimmer, and continuous chart motion, leaving an instant update or a fade no
longer than `80ms`.

## Responsiveness and initial scope

The first migration targets authenticated desktop applications. Public sites,
marketing pages, and a phone-specific authoring experience are excluded.
Project contracts select their supported desktop widths; the shared reference
capture baseline is `1440x900`, with `1280x800` and a larger desktop check.
Zoom, reflow, keyboard access, and bounded table overflow remain required.

## Perceived-performance contract

Performance is measured on declared local reference hardware and on the lowest
supported client profile. Results report p50, p75, and p95, sample count, cold
versus warm state, data volume, browser build, CPU/RAM, network topology, and
whether the backend result was cached. UI, network, API, and render time are
reported separately.

Initial budgets, to be confirmed by each architecture spike, are:

| Boundary | Target | Hard ceiling |
|---|---:|---:|
| Pointer/key feedback to next paint | p75 `<= 50ms` | p95 `<= 100ms` |
| Warm local navigation to stable shell/content acknowledgement | p75 `<= 100ms` | p95 `<= 200ms` |
| Menu, command palette, or cached pane open | p75 `<= 100ms` | p95 `<= 160ms` |
| REST response end to stable rendered state | p75 `<= 100ms` | p95 `<= 200ms` |
| SSE event receipt to visible state | p75 `<= 100ms` | p95 `<= 200ms` |
| Interaction to request dispatch | p75 `<= 20ms` | p95 `<= 50ms` |
| INP on representative journeys | p75 `<= 100ms` | `<= 200ms` |
| Long main-thread task | none over `50ms` in the accepted steady-state journey | investigated blocker |
| Animation frame cadence | target `60fps` without recurring dropped-frame clusters | no interaction-blocking jank |

For an uncached route, the UI acknowledges the action within `100ms`, preserves
valid prior content or reserved layout, and shows truthful local progress while
waiting for the existing backend. The transition does not set an arbitrary API
latency requirement or conceal a slow server. It makes client overhead and
server latency independently measurable.

## Reference evidence

The machine-readable companion
[`linear-workspace-reference-manifest-v1.json`](linear-workspace-reference-manifest-v1.json)
records the supplied archive hash and observed screenshot states without
committing third-party screenshots. The current archive is sufficient for the
dark shell, navigation, resizable-sidebar intent, list/table, settings, modal,
popover, and right-detail-pane composition.

Before Penpot vNext or runtime foundations are accepted, the reference set must
also contain or explicitly waive:

- all four target themes on representative surfaces;
- command palette open/search/execute states;
- keyboard focus order and visible focus examples;
- sidebar collapse and pointer-resize recordings;
- list-to-detail, drawer, modal, popover, route, and Back/Escape recordings;
- loading, empty, error, stale/degraded, forbidden, and session-expired states;
- exact viewport metadata and sanitized accessibility snapshots;
- measured component geometry and motion timings for selected golden surfaces.

Authentication storage, cookies, tokens, account exports, and raw browser state
must never be committed. Private captures live outside the repository or under
an ignored `.private/` boundary. Repositories commit only sanitized derived
measurements, descriptions, hashes, and acceptance evidence.

## Migration method

Each repository follows the same sequence with its own delivery graph:

1. freeze current UI authority and supersede conflicting future execution;
2. complete and hash the reference pack;
3. run a bounded architecture and compatibility spike;
4. build Penpot vNext foundations and representative compositions;
5. implement the React application shell behind a reversible route boundary;
6. prove one real read-only golden slice against existing APIs;
7. migrate route clusters from read-only to authoring and sensitive/admin flows;
8. cut over only after parity, browser, accessibility, performance, and rollback
   evidence; then remove legacy UI intentionally.

One ready ticket is one execution unit. Accepted historical audits and designs
remain evidence; they are not silently rewritten as the new target. A
superseded ticket is not accepted work. Backend delivery graphs continue in
parallel when their path ownership is disjoint.

## Acceptance boundary

Penpot proves design structure and visual intent only. Source tests prove source
behavior only. A migrated surface is accepted only with real-browser evidence
for the declared viewports, four themes on the representative matrix, keyboard
and focus behavior, reduced motion, localization, required server states,
route/history behavior, real API integration, performance measurements, and a
tested fallback or rollback boundary.
