---
ticket_id: ROEHUB-UI-INSTRUCTIONS-AND-COPY-REVIEW-2026-07-22
status: ready
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on:
  - ROEHUB-UI-DESIGN-ACCEPTANCE-BOUNDARY-REPAIR-2026-07-22
evidence: []
acceptance_authority: product_owner
agent_self_acceptance: prohibited
---

# Review Roehub UI instructions and interface copy

## Outcome

Roehub has one product-owner-approved contract for authenticated-platform UI
instructions, terminology, labels, actions, status language, system-state copy,
and design-review expectations before visual prototyping proceeds in Figma.

## Scope

- Audit active Roehub UI instructions for contradictions, obsolete decisions,
  ambiguous visual authority, and wording that could cause literal copying of
  reference products or reuse of the rejected React-spike visual layer.
- Inventory current authenticated-platform navigation labels, page titles,
  actions, statuses, empty/loading/stale/degraded/forbidden/error messages,
  risk and trading language, help text, and accessibility-facing names.
- Map proposed copy to Roehub routes, capabilities, authoritative projections,
  domain terminology, and truthful system states.
- Define tone, terminology, source-language and localization rules, truncation
  and density constraints, and prohibited placeholder or fabricated content.
- Produce exact recommendations and unresolved choices for product-owner review
  before Figma foundations or application layouts are created.

## Owned paths

- this ticket and its terminal evidence;
- a ticket-owned UI instructions-and-copy architecture contract under
  `docs/architecture/ui/`;
- `.codex/AGENTS.md`, the accepted UI transition specification, Roehub Figma
  standard, and UI migration registry only for approved resulting rules;
- deterministic architecture-index and project-map companions.

Current application templates, scripts, route contracts, screen registries,
reference captures, and the canonical Figma file are read-only context unless
a later explicit amendment changes this ticket.

## Non-goals

- No Figma mutation, component or layout design, React/SSR implementation,
  route/capability change, translation rollout, backend change, or deployment.
- No reuse of the architecture-spike layout, styling, component anatomy, copy,
  or screenshots as design authority.
- No agent-only product decision where terminology, tone, or task priority is
  ambiguous.

## Proof boundary

The evidence identifies every inspected instruction and copy surface, records
the proposed canonical terminology and unresolved choices, maps claims to
authoritative product sources, and distinguishes current facts from proposals.
Repository checks prove document consistency only; they do not approve the
content or any design.

## Acceptance

The product owner explicitly approves the named contract revision and all
recorded decisions or waivers. The agent may not self-accept this ticket. Only
after that approval and a repeated source check may the ticket become
`accepted` and unblock Figma foundations.
