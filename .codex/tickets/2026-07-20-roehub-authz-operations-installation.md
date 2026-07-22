---
ticket_id: ROEHUB-AUTHZ-OPERATIONS-INSTALLATION-2026-07-20
status: draft
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on:
  - ROEHUB-AUTHZ-SETTINGS-ADMIN-2026-07-20
  - ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20
  - ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20
evidence:
  - .codex/delivery/evidence/ROEHUB-AUTHZ-OPERATIONS-INSTALLATION-2026-07-20.md
---

# Enforce monitoring safe actions and installation-owner authority

## Outcome

Live-execution, Monitoring, and administration APIs distinguish organization-
owned operational actions from installation-wide trust, resources, recovery,
and shared-service actions, enforcing the accepted closed action set.

## Scope

- Return server-filtered monitoring/service projections grouped by logical
  service area and ownership scope.
- Apply `live.read` and non-escalating `live.reconcile` policy without creating
  orders, expanding exposure, or crossing organization ownership.
- Implement the operator safe-action allowlist for diagnostics, organization-
  owned stopped-service restart, and non-escalating reconciliation. Strategy,
  Backtest, and Connection actions remain owned by their product tickets.
- Deny operator access to shared installation-service restart and arbitrary
  service lifecycle.
- Enforce separate non-delegable `installation_owner` checks for trust policy,
  physical resource ceilings, shared-service actions, recovery, update, and
  rollback.
- Apply recent authentication, explicit confirmation, idempotency, validation,
  no-secret echo, recovery evidence, and audit as required.
- Write only paths assigned to this ticket by the ticket graph. If an
  installation endpoint still lives in `apps/api/routes/admin.py`, stop and ask
  the control chat to serialize or amend ownership after Settings/Admin merges.

## Non-goals

- No deployment target, host SSH, Mac Studio, UI, Penpot, plugin lifecycle,
  release, or deploy.

## Proof boundary

- API/operations tests cover each role, live read/reconciliation, organization-
  owned versus shared service, exact action allowlist, installation overlay
  independence, trust validation, resource ceilings, recovery evidence, replay,
  and audit.
- No test invents a production runtime target.

## Escalation triggers

- Current service inventory cannot identify organization ownership versus
  installation scope.
- An operation requires external runtime or secret access not authorized here.
- Recovery cannot fail closed without a new product decision.

## Acceptance

- The complete operational/installation security matrix passes focused tests
  and independent security review; only then may the ticket become `accepted`.
