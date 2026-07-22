---
ticket_id: ROEHUB-AUTHZ-SETTINGS-ADMIN-2026-07-20
status: draft
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on:
  - ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20
  - ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20
evidence:
  - .codex/delivery/evidence/ROEHUB-AUTHZ-SETTINGS-ADMIN-2026-07-20.md
---

# Enforce personal settings and organization administration policy

## Outcome

Personal settings and organization-administration APIs enforce the accepted
personal, organization, membership, delegation, extension, and audit boundaries
with server-filtered projections.

## Scope

- Apply `preferences.manage_personal`, `settings.personal.manage`,
  `settings.organization.manage`, `admin.overview.read`,
  `admin.members.manage`, `admin.plugins.manage`, and
  `audit.organization.read`.
- Expose owner-authorized delegation grant/revoke through recent authentication,
  mutation envelope, last-owner guard, and redacted audit.
- Prevent admin from granting/removing owner and prevent operator/trader/viewer
  organization mutations.
- Preserve plugin permission review without expanding plugin lifecycle authority.
- Write only paths assigned to this ticket by the ticket graph.

## Non-goals

- No installation trust/resources/recovery or operational actions; no Dashboard,
  Strategies, Backtests, Connections, Web UI, or Figma artifact.

## Proof boundary

- API/identity tests cover personal scope, every role, membership changes,
  delegation lifecycle endpoints, extension authorization, audit projection,
  organization isolation, recent-auth, replay, and redaction.

## Escalation triggers

- Existing clients require a field forbidden by the accepted policy.
- A required write falls outside graph-assigned paths or overlaps an active
  ticket.

## Acceptance

- Focused API/identity tests and compact evidence pass; then the ticket may
  become `accepted`.
