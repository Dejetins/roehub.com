---
ticket_id: ROEHUB-AUTHZ-DASHBOARD-DATA-2026-07-20
status: draft
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-server-authorization-stream-v1.json
depends_on:
  - ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20
  - ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20
evidence:
  - .codex/delivery/evidence/ROEHUB-AUTHZ-DASHBOARD-DATA-2026-07-20.md
---

# Enforce dashboard and market-data policy

## Outcome

Dashboard and market-data APIs return server-filtered organization projections
and enforce catalog-read and instrument-selection capabilities without exposing
cross-organization data.

## Scope

- Apply `dashboard.read`, `data.catalog.read`, and `data.selection.manage`.
- Filter Dashboard/readiness data before DTO construction.
- Require exact delegation for trader instrument-selection mutation.
- Apply the browser-mutation envelope, installation ceilings, strategy-pin
  guards, idempotency, and audit to selection changes.
- Preserve `403` versus non-leaking `404` semantics.
- Write only paths assigned to this ticket by the ticket graph.

## Non-goals

- No personal settings, administration, strategies, backtests, connections,
  monitoring, Web UI, or Penpot.

## Proof boundary

- API tests cover every role, organization isolation, projection filtering,
  delegated selection, pin/resource denial, replay, and audit.
- Current DTO compatibility is classified explicitly.

## Escalation triggers

- Projection code must fetch cross-organization data before filtering.
- A required write falls outside graph-assigned paths.

## Acceptance

- Focused API tests and compact evidence pass; then the ticket may become
  `accepted`.
