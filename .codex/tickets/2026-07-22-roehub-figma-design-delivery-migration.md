---
ticket_id: ROEHUB-FIGMA-DESIGN-DELIVERY-MIGRATION-2026-07-22
status: accepted
owner: unassigned
depends_on: []
evidence:
  - .codex/delivery/evidence/ROEHUB-FIGMA-DESIGN-DELIVERY-MIGRATION-2026-07-22.md
---

# Migrate Roehub design delivery from Penpot to Figma

## Outcome

All future Roehub design prototyping, design-system work, visual review, and
design-to-code handoff use the existing Figma project `roehub.com`. The
repository records one canonical authenticated-platform file and routes future
design tickets to Figma without rewriting truthful Penpot history.

## Scope

- Register the Figma project and canonical authenticated-platform file.
- Add a Roehub-specific Figma design-delivery architecture contract.
- Replace active Penpot routing in the unified graph, unfinished tickets,
  transition specification, migration registry, and repository instructions.
- Rename the unstarted Penpot vNext ticket to its Figma equivalent and update
  real dependencies without changing queue priority.
- Extend the delivery-model validator and focused tests so unfinished tickets
  cannot reintroduce Penpot instructions or drift from the canonical Figma
  identity.
- Refresh derived architecture and project-map artifacts.

## Owned paths

- this ticket and terminal evidence;
- `.codex/AGENTS.md`;
- the unified delivery graph and current transition specification;
- unfinished graph tickets whose design-tool language or dependency changes;
- `tools/delivery/validate_roehub_delivery_model.py` and its focused tests;
- Roehub architecture sources directly amended or created by this migration;
- deterministic architecture-index and project-map companions.

## Non-goals

- No screen, component, variable, token, prototype, or Code Connect creation in
  the new Figma file.
- No rewrite of accepted historical evidence or factual statements about past
  Penpot operations.
- No change to Custometry, the shared cross-project transition standard, product
  runtime, routes, backend, authorization, data, release, or deployment.
- No Linear tracker operation, additional repository checkout, branch, or
  worktree.

## Proof boundary

The Figma connector proves account access, project identity, file creation, and
file readability. Repository validation proves current routing, dependencies,
canonical identifiers, the absence of active Penpot instructions in unfinished
tickets, generated documentation freshness, and publication consistency. It
does not prove a design, browser behavior, runtime behavior, or deployment.

## Acceptance

- The project `roehub.com` (`projectId` `629113387`) contains the canonical file
  `Roehub Authenticated Platform UI` (`fileKey` `GBzmB9evtzqnAYNjp9W1sr`).
- The architecture spike remains the only `ready` graph ticket; the renamed
  Figma foundations ticket remains `draft`; no graph ticket is `active`.
- Active Roehub instructions and unfinished tickets route future design work to
  Figma; historical Penpot evidence remains intact.
- Focused tests, repository checks, and cold review are recorded in terminal
  evidence; publication and GitHub Actions results are reported at the
  publication boundary.
