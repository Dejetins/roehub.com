---
ticket_id: ROEHUB-AUTHZ-INTEGRATION-PROOF-2026-07-20
status: draft
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-server-authorization-stream-v1.json
depends_on:
  - ROEHUB-AUTHZ-DASHBOARD-DATA-2026-07-20
  - ROEHUB-AUTHZ-SETTINGS-ADMIN-2026-07-20
  - ROEHUB-AUTHZ-STRATEGIES-2026-07-20
  - ROEHUB-AUTHZ-BACKTESTS-2026-07-20
  - ROEHUB-AUTHZ-CONNECTIONS-2026-07-20
  - ROEHUB-AUTHZ-OPERATIONS-INSTALLATION-2026-07-20
evidence:
  - .codex/delivery/evidence/ROEHUB-AUTHZ-INTEGRATION-PROOF-2026-07-20.md
---

# Prove the cross-surface server authorization boundary

## Outcome

One deterministic integration matrix proves the implemented authorization
boundary across Dashboard, Data, Settings, Administration, Strategies,
Backtests, Connections, Live execution, Monitoring, and installation authority
before a new Web UI relies on it.

## Scope

- Build a role/capability/organization/object/action matrix from the accepted
  contract without loading documentation JSON in production.
- Exercise representative read and mutation routes for owner, admin, operator,
  trader, viewer, missing membership, wrong organization, stale authentication,
  revoked delegation, and `installation_owner` without organization membership.
- Prove default deny, field filtering, secret non-disclosure, operator action
  closure, idempotency, `403/404` behavior, recent-auth, and audit redaction.
- Detect any accepted capability or mutation-bearing surface included by this
  stream that lacks an implemented enforcement path; report it as a blocker
  rather than waive it. Preserve the graph's explicit exclusions as
  `target_not_implemented`, not as successful coverage.
- Record any architecture/current-state mismatch in this ticket's evidence.
  Changing accepted architecture documents requires a separate authorized task.

## Non-goals

- No Web UI, browser-visible acceptance, Penpot, model API creation, public site,
  release, or deploy.

## Proof boundary

- Integration and focused regression suites pass for every included surface.
- One independent security reviewer returns no unresolved blocking finding.
- Compatibility and remaining `target_not_implemented` capabilities are listed
  explicitly; absence is not represented as success.

## Escalation triggers

- A predecessor merged without its declared evidence.
- Required proof needs an external runtime or secrets.
- The accepted architecture and implemented contracts materially disagree.

## Acceptance

- The integration matrix, independent review, architecture truth update, and
  compact evidence pass; only then may the stream be considered complete.
