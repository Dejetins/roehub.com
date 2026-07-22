---
ticket_id: ROEHUB-UI-DESIGN-ACCEPTANCE-BOUNDARY-REPAIR-2026-07-22
status: accepted
owner: unassigned
ticket_graph: .codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
depends_on:
  - ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20
evidence:
  - .codex/delivery/evidence/ROEHUB-UI-DESIGN-ACCEPTANCE-BOUNDARY-REPAIR-2026-07-22.md
---

# Repair the UI design acceptance boundary

## Outcome

The accepted React coexistence spike remains technical architecture evidence,
while its visible composition, styling, and copy are explicitly rejected by the
product owner and prohibited as a source for Roehub design or implementation.
Future Figma work uses staged product-owner review, and a separate UI
instructions-and-copy ticket becomes a prerequisite for design work.

## Scope

- Record the product-owner rejection as a later boundary correction without
  rewriting the historical technical observations of the spike.
- Define exactly which spike assets may be reused: route isolation, state and
  transport seams, rollback, tests, measurement harness, and dependency data.
- Prohibit reuse of the spike's layout, styling, themes, component anatomy,
  copy, and screenshots as design direction.
- Add explicit product-owner checkpoints and prohibit agent self-acceptance for
  Figma design work.
- Create a separate draft ticket for reviewing Roehub UI instructions and copy,
  and place it before the Figma foundations ticket in the unified queue.
- Extend repository validation so these boundaries cannot silently regress.

## Owned paths

- this ticket and its terminal evidence;
- the accepted architecture-spike ticket, evidence, and prototype README only
  for the later design-boundary addendum;
- the new UI instructions-and-copy review ticket;
- the unified graph, transition specification, UI migration registry, Roehub
  Figma standard, and repository agent guidance;
- the Figma-foundations and React-shell tickets;
- the delivery-model validator and focused tests;
- deterministic architecture-index and project-map companions.

## Non-goals

- No deletion or redesign of the technical prototype.
- No Figma mutation, UI implementation, product route, backend, authorization,
  persistence, public-site, release, deployment, or Linear operation.
- No review or rewrite of the actual UI instructions and copy inside this
  ticket; that work belongs to the separate follow-up ticket.
- No claim that technical CI, browser, or performance evidence approves visual
  design.

## Proof boundary

Repository validation proves queue membership, dependencies, acceptance
authority, visual-source prohibition, and current status invariants. A cold
review checks that historical technical evidence remains factual and that the
new rules cannot be mistaken for Figma or runtime proof.

## Acceptance

- The technical spike remains `accepted` only for its architecture boundary;
  its visual design status is `rejected_by_product_owner` and its visual source
  role is `prohibited`.
- The separate UI instructions-and-copy review ticket is the first future
  product-design task after this repair.
- The Figma ticket cannot start before that review is accepted and cannot
  become `accepted` without explicit product-owner review of named Figma nodes.
- The React shell must replace, rather than inherit, the spike visual layer.
- Local checks and terminal evidence are recorded. Publication and GitHub
  verification remain a separate authority boundary.
