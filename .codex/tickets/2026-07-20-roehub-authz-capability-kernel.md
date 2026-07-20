---
ticket_id: ROEHUB-AUTHZ-CAPABILITY-KERNEL-2026-07-20
status: ready
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-server-authorization-stream-v1.json
depends_on: []
evidence:
  - .codex/delivery/evidence/ROEHUB-AUTHZ-CAPABILITY-KERNEL-2026-07-20.md
---

# Implement the server capability-decision kernel

## Outcome

The identity context exposes one default-deny application boundary that decides
an accepted Roehub capability from authenticated actor, selected organization,
persisted organization role, resource scope, ownership, and the separate
`installation_owner` authority. Product routes do not yet migrate in this
ticket.

## Context

- accepted access contract:
  `docs/architecture/apps/web/roehub-local-platform-access-and-route-contract-v1.json`
- `src/trading/contexts/identity/domain/entities/organization.py`
- current identity application ports, repositories, and current-user adapter

## Scope

- Add stable capability identifiers and a deny-reason model without parsing the
  documentation JSON at runtime.
- Add an application service/port for role, organization, ownership, scope, and
  installation-authority decisions.
- Reject client-supplied roles, missing organization context, unknown
  capabilities, inactive membership, and cross-organization resources.
- Keep `installation_owner` independent from organization roles and forbid
  secret reveal for every actor.
- Provide focused unit tests covering all role families, overlay separation,
  own-resource scope, unknown capability, and default deny.

## Non-goals

- No delegation persistence or grant/revoke workflow.
- No browser-mutation envelope and no route migration.
- No UI, Penpot, public site, push, merge, release, or deploy. A local owned
  commit requires explicit authority in the task-launch message.

## Proof boundary

- Unit tests prove deterministic decisions and deny reasons at the application
  boundary.
- Existing organization role values and persisted records remain compatible.
- No API route is claimed protected until its own integration ticket passes.
- Backend focused tests, lint, type checks required by the repository, and
  `git diff --check` pass.

## Escalation triggers

- Implementing the kernel requires renaming persisted roles or changing public
  authentication/session semantics.
- Current identity data cannot determine active organization membership without
  a migration not scoped by this ticket.
- A capability would reveal stored connection secrets.

## Acceptance

- The kernel and tests exist, default deny is demonstrated, and compact evidence
  is recorded.
- A cold self-review finds no route-level protection claim or hidden migration.
- Only then may this ticket become `accepted`, opening the two dependent
  foundation tickets.
