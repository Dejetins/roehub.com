---
ticket_id: ROEHUB-UI-INVENTORY-2026-07-20
status: accepted
owner: unassigned
---

# Inventory Roehub UI surfaces, journeys, states, roles, and public-site content

## Outcome

Roehub has one verified current-state inventory and one proposed target registry
covering the local platform and the public site without changing a design tool
or product code. Every target surface is traceable to an observed route, API,
accepted requirement, historical evidence, or an explicitly labelled gap.

## Context

- Product baseline:
  `docs/architecture/platform/roehub-product-transformation-requirements-v1.md`
- Delivery architecture:
  `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md`
- Public-site proposal:
  `docs/architecture/platform/roehub-public-site-identity-release-and-measurement-v1.md`
- Current local UI: `apps/web/`
- Current API: `apps/api/`
- Historical prototype: Git baseline
  `c6ef2f32464ea681c7582aa8b689aacdc02b5d70`; its live directory has been
  removed.

## Scope

- Inventory public, authentication, protected, dynamic, compatibility,
  runbook, QA-only, and error routes in the current Web service.
- Inventory visible sections, overlays, dialogs, tabs, editors, detail views,
  and system states.
- Map each surface to current API/DTO inputs, roles, mutations, source freshness,
  and known `target_not_implemented` requirements.
- Describe canonical user journeys for first launch, data selection,
  connections, backtests, strategies, live operations, models, monitoring,
  administration, recovery, and local documentation.
- Create a separate public-site sitemap and content inventory covering product,
  documentation, releases, downloads, account, future demo, security,
  community, support, and legal pages.
- Identify route compatibility requirements and unresolved product decisions.
- Produce a compact machine-readable registry suitable for a later UI-program
  atlas and traceability.

## Deliverables

- `docs/architecture/apps/web/roehub-ui-surface-inventory-v1.md`
- `docs/architecture/apps/web/roehub-ui-surface-registry-v1.json`
- `docs/architecture/apps/web/roehub-public-site-surface-registry-v1.json`

The registries must use stable, product-owned `surface_id` values without
embedding temporary design-tool IDs.

## Non-goals

- No design-tool read or write is required for acceptance.
- No visual program, component, token, screen contract, or board.
- No Web, API, persistence, authentication, release, or documentation-site
  implementation.
- No final visual design or component specification.
- No invention of missing API behavior.
- No plan, ledger, prompt pack, Goal, commit, push, release, or deploy.

## Blockers

- None for current-state inventory.
- Target-only public account, release, download, and demo surfaces must be
  distinguished from observed current behavior.

## Repair policy

- Repair only inventory inconsistencies, broken local references, registry
  schema mistakes, or generated documentation drift introduced by this ticket.
- Do not repair product code or historical reports during inventory.
- Escalate when two accepted sources prescribe incompatible user-visible
  behavior and current code cannot resolve the authority.

## Proof boundary

- Every current route is confirmed from current code or generated routing
  evidence.
- Every target surface cites a product requirement or is labelled `proposed`.
- Dynamic routes, roles, states, mutations, destructive actions, and unknown
  behavior are explicit.
- Local platform and public-site registries are separate.
- Historical prototype content is labelled evidence, not authority.
- JSON is schema-valid or validated by a deterministic repository check added
  within the ticket.
- Architecture index, project map, local links, and `git diff --check` pass.

## Escalation triggers

- A route or action has incompatible security semantics across accepted
  sources.
- A missing public-site decision changes registration, distribution, privacy,
  or demo scope.
- The inventory would require a product-code mutation to become truthful.
- Existing foreign changes cannot be separated from ticket-owned paths.

## Acceptance

- Status becomes `accepted` only after the three deliverables exist, validation
  passes, and a cold self-review finds no unclassified surface or hidden product
  decision.
- Acceptance authorizes the next separately selected architecture or UI-program
  execution unit only; it does not authorize implementation work.
