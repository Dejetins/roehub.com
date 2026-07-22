---
ticket_id: ROEHUB-AUTHZ-STRATEGIES-2026-07-20
status: draft
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on:
  - ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20
  - ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20
evidence:
  - .codex/delivery/evidence/ROEHUB-AUTHZ-STRATEGIES-2026-07-20.md
---

# Enforce strategy read, manage, run, stop, and manual-trade policy

## Outcome

Strategy APIs enforce server-side role, organization, ownership, delegation,
risk, connection-readiness, and safe-operator rules for read and mutation paths.

## Scope

- Migrate strategy list/detail projections and create/update/delete/run/stop/
  restart/manual-trade paths to accepted capabilities.
- Preserve trader own-resource scope and owner organization scope.
- Permit operator safe stop only for an already running strategy; forbid start,
  restart, edit, delete, and manual trade.
- Apply the browser-mutation envelope, idempotency, recent-auth/risk gates where
  required, and redacted audit events.
- Keep manual mainnet controls fail closed and independently approved.

## Non-goals

- No new strategy UI, model workspace, backtest policy, Penpot, release, or
  deploy.

## Proof boundary

- API/domain tests cover each role, ownership, delegation, organization
  isolation, operator safe stop, mainnet denial, replay, and audit.
- Existing URLs and non-browser API semantics remain compatible unless a
  classified change is explicitly accepted.

## Escalation triggers

- Existing strategy ownership cannot be resolved within an organization.
- A migration would allow operator trading or bypass mainnet safety.

## Acceptance

- All strategy authorization tests and evidence pass; only then may the ticket
  become `accepted`.
