---
ticket_id: ROEHUB-BACKTESTS-LINEAR-GOLDEN-SLICE-2026-07-20
status: draft
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on:
  - ROEHUB-REACT-LINEAR-APPLICATION-SHELL-2026-07-20
  - ROEHUB-AUTHZ-BACKTESTS-2026-07-20
evidence: []
---

# Deliver the read-only Backtest list/detail golden slice

## Outcome

One real read-only Backtest journey proves the new shell against existing
Roehub REST/SSE projections and server authorization: dense list, filters,
charts, freshness/degraded states, route-backed detail, resizable right pane,
keyboard, four themes, and measured end-to-end browser behavior.

## Context and ownership

Read the accepted shell, Figma design, server-authorization Backtests ticket/evidence,
local route/access contract, and real API DTOs. Own only this ticket/evidence,
Backtest presentation/adapters in `apps/platform-web/**`, shared UI packages
when safely separable, and ticket-owned browser/performance tests. Server and
domain code remain read-only.

## Scope

- Render real authorized list/detail projections; preserve list scroll,
  selection, filters, URL identity, Back/Escape, and SSE freshness.
- Provide accessible chart/table alternatives and truthful loading, empty,
  stale, degraded, forbidden, and failed states.
- Measure client-only and end-to-end dispatch, acknowledgement,
  response/SSE-to-paint, INP, long tasks, and frame cadence with p50/p75/p95.

## Non-goals

- No queue, rerun, promote, delete, trading, server-policy, API, persistence,
  public-site, release, or deployment change.

## Proof boundary

Use a real local API, server-authorized persona, deterministic Backtest data,
real browser, and declared hardware. A mock, screenshot, Figma frame, or unit
test cannot accept the golden slice.

## Acceptance

Terminal evidence records runtime/data identity, persona, routes/states/themes,
traces, accessibility, console/network checks, performance distributions,
fallback behavior, and fact-based recommendations for the next route clusters.
