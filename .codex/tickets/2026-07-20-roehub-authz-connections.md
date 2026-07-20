---
ticket_id: ROEHUB-AUTHZ-CONNECTIONS-2026-07-20
status: draft
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-server-authorization-stream-v1.json
depends_on:
  - ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20
  - ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20
evidence:
  - .codex/delivery/evidence/ROEHUB-AUTHZ-CONNECTIONS-2026-07-20.md
---

# Enforce connection status, binding, lifecycle, and secret boundaries

## Outcome

Connection APIs provide role-filtered readiness and lifecycle behavior without
ever returning stored secret material.

## Scope

- Apply accepted capabilities to connection status, strategy binding, create,
  rotate, recheck, disconnect, and archive paths.
- Preserve owner/admin lifecycle authority, trader own binding when delegated,
  and operator recheck/disconnect only.
- Require recent authentication for credential input/rotation and apply the
  browser-mutation envelope and audit.
- Return redacted status only; maintain the invariant that no role can reveal a
  stored secret.

## Non-goals

- No exchange-order submission, strategy UI, new credential provider, Penpot,
  release, or deploy.

## Proof boundary

- API/identity tests cover every role, organization/ownership scope, redaction,
  recent-auth, origin/CSRF, replay, operator subset, and secret non-disclosure.

## Escalation triggers

- Any compatibility path would echo or decrypt a stored secret.
- Existing connection identity cannot be scoped to an organization safely.

## Acceptance

- All connection security tests and compact evidence pass; only then may the
  ticket become `accepted`.
