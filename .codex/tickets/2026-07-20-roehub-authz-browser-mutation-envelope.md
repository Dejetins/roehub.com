---
ticket_id: ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20
status: accepted
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on:
  - ROEHUB-AUTHZ-CAPABILITY-KERNEL-2026-07-20
evidence:
  - .codex/delivery/evidence/ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20.md
---

# Implement the shared browser-mutation security envelope

## Outcome

Cookie-authenticated browser mutations can use one fail-closed server boundary
for origin/CSRF proof, capability decision, organization/object scope,
validation, idempotency, recent authentication, and redacted audit hooks.

## Scope

- Build the shared guard and dependency primitives without broad route adoption.
- Preserve existing session and API-client behavior for non-browser clients.
- Define deterministic idempotency-key reuse and payload-conflict behavior.
- Fail closed when required origin, CSRF, organization, capability, object,
  recent-auth, or audit inputs are missing.
- Provide focused tests for origin attacks, replay, cross-organization access,
  stale recent-auth, audit redaction, and unknown capability.

## Non-goals

- No mass route migration; each product route migrates in its own ticket.
- No new UI, Penpot, public site, push, merge, release, or deploy. A local owned
  commit requires explicit authority in the task-launch message.

## Proof boundary

- Component/API tests prove the envelope itself and compatibility with existing
  authentication primitives.
- Tests do not claim Strategies, Backtests, Connections, Settings, or Operations
  are protected until their integration tickets pass.

## Escalation triggers

- The envelope requires changing public authentication/session contracts.
- Existing CSRF or recent-auth semantics conflict with accepted security rules.
- Idempotency requires a persistence contract not safely scoped here.

## Acceptance

- Shared primitives and focused evidence pass with no product-route claim; only
  then may the ticket become `accepted`.
