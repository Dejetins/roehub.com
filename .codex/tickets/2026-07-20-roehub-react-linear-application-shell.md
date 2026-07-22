---
ticket_id: ROEHUB-REACT-LINEAR-APPLICATION-SHELL-2026-07-20
status: draft
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on:
  - ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20
  - ROEHUB-PENPOT-LINEAR-VNEXT-FOUNDATIONS-2026-07-20
evidence: []
---

# Implement the Roehub React Linear-workspace shell

## Outcome

The authenticated local platform has a production-shaped React shell behind a
reversible SSR route boundary with four themes, command/navigation, resizable
panels, system states, localization, accessibility, and measured native-feeling
browser behavior.

## Start probe

Confirm accepted predecessor evidence and the exact Penpot vNext file/revision
before implementation. Stop on identity, source fingerprint, or revision drift.

## Owned paths

- this ticket and terminal evidence;
- `apps/platform-web/**` and its ticket-owned tests;
- explicitly shared `packages/tokens/**`, `packages/ui/**`,
  `packages/localization/**`, and `packages/web-contracts/**`;
- the minimum same-origin `apps/web/**` mount/fallback adapter hunks;
- Node workspace manifests required by the accepted spike.

## Scope

- Implement the stable authenticated shell and honest planned-route surfaces.
- Preserve SSR fallback until browser/performance acceptance.
- Verify four themes, expanded/collapsed/resized sidebar, command palette,
  history, Help, system states, keyboard, focus, reduced motion, localization,
  zoom/reflow, console/network cleanliness, and exact rollback.
- Measure p50/p75/p95 client dispatch, acknowledgement, paint, INP, long tasks,
  and frame cadence on declared hardware.

## Non-goals

- No domain content, trading mutation, backend authorization, persistence,
  public site, release, deployment, or fallback removal.

## Proof boundary

Use `browser-qa-evidence`, measured performance evidence, Playwright, and the
accepted Penpot source. Source tests alone cannot accept this ticket.

## Acceptance

Fresh traces/screenshots, accessibility/theme/viewport matrices, console and
network observations, measured performance distributions, and exercised SSR
rollback are recorded in terminal evidence with explicit exclusions.
