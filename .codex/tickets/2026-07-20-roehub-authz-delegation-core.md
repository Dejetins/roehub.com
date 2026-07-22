---
ticket_id: ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20
status: accepted
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on:
  - ROEHUB-AUTHZ-CAPABILITY-KERNEL-2026-07-20
evidence:
  - .codex/delivery/evidence/ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20.md
---

# Implement delegated-capability persistence and application rules

## Outcome

The server can persist, evaluate, expire, and revoke an exact organization
capability delegation according to the accepted owner-only contract. Browser or
public API endpoints are left to a later integration ticket.

## Scope

- Add the minimum persistence schema, migration, repository port/adapters, and
  application use cases for exact capability grants.
- Enforce active organization membership, owner-only authority, no self-grant,
  no redelegation, non-delegable authorities, expiry, immediate revocation, and
  resource scope.
- Record redacted audit events for grant and revoke.
- Integrate evaluation with the accepted capability kernel.
- Cover PostgreSQL and in-memory parity, migration forward/backward safety, and
  concurrency/idempotency behavior.

## Non-goals

- No browser endpoint, UI, Penpot, role renaming, broad product-route adoption,
  push, merge, release, or deploy. A local owned commit requires explicit
  authority in the task-launch message.

## Proof boundary

- Persistence and application tests prove grant lifecycle, expiry, revocation,
  audit, cross-organization denial, and non-delegable rejection.
- Migration and contract impact are explicit and reversible.
- No current static permission is treated as a delegation record.

## Escalation triggers

- Required storage cannot be added without destructive migration.
- Audit persistence cannot remain redacted.
- The accepted capability kernel is not `accepted` or its API changed.

## Acceptance

- All scoped persistence/application evidence passes and the ticket becomes
  `accepted`; endpoint exposure remains unauthorized.

## Migration bootstrap repair (2026-07-22)

The clean-PostgreSQL CI defect is repaired within this ticket rather than as a
new product boundary. The bootstrap contract is now SQL `0001..0009`, Alembic
`20260711_0043`, SQL `0010..0011`, Alembic `head`, then SQL `0012..0022` in
numeric order. `apps.migrations.main` accepts an explicit Alembic target while
keeping `head` as its default, and CI exercises `apps.migrations.bootstrap_main`
with both DSNs pointed at its one temporary PostgreSQL service.

`0011_identity_organizations_rbac_audit_v1.sql` remains the sole owner of
`identity_organizations` and `identity_memberships`; published Alembic revision
`20260720_0044` remains unchanged and owns only delegated-capability storage.
The compact decision and actual verification results are recorded in
`docs/architecture/identity/identity-migration-channels-delegation-checkpoint-v1.md`
and the linked evidence file.
