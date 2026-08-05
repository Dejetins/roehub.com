---
ticket_id: ROEHUB-LOCAL-UI-DESIGN-SYSTEM-CONTRACT-2026-07-20
status: ready
owner: unassigned
depends_on:
  - ROEHUB-LOCAL-UI-IA-2026-07-20
evidence:
  - .codex/delivery/evidence/ROEHUB-LOCAL-UI-DESIGN-SYSTEM-CONTRACT-2026-07-20.md
---

# Define the local-platform design-system contract

## Outcome

Roehub has one implementation-independent and Penpot-independent design-system
contract that turns the accepted local screen registry into deterministic
tokens, component families, states, chart primitives, responsive rules, and
design-to-code boundaries. The result is ready for product-owner review before
any Penpot write.

## Context

- `docs/architecture/apps/web/roehub-local-platform-information-architecture-v1.md`
- `docs/architecture/apps/web/roehub-local-platform-screen-registry-v1.json`
- `docs/architecture/apps/web/roehub-local-platform-access-and-route-contract-v1.json`
- `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md`
- `docs/architecture/platform/roehub-product-transformation-requirements-v1.md`
- current `apps/web/` implementation and historical `prototypes/roehub-v2/`

## Scope

- Define base and semantic token namespaces for all six accepted theme IDs:
  `abyss`, `graphite`, `slate`, `frost`, `paper`, and `sand`.
- Define typography, spacing, density, grid, elevation, radius, focus, motion,
  and icon rules for widths `820`, `1024`, and `1440` only.
- Define component families and variants needed by every accepted `screen_id`
  and required state in the local screen registry.
- Define loading, empty, stale, degraded, forbidden, destructive-confirmation,
  recovery, queue, progress, ETA-confidence, and completed-job states.
- Define safe Apache ECharts wrappers, semantic chart specifications, tabular
  alternatives, accessibility descriptions, and renderer boundaries.
- Define localization, keyboard, focus, contrast, reduced-motion, and screen
  reader requirements.
- Define the future Penpot library structure, names, variant conventions,
  publication/versioning boundary, export manifest, and stable mapping from
  design components to planned `@roehub/*` packages.
- Separate shared primitives from local-platform composition. Do not define
  public-site pages or copy.

## Deliverables

- `docs/architecture/apps/web/roehub-local-platform-design-system-contract-v1.md`
- `docs/architecture/apps/web/roehub-local-platform-design-token-contract-v1.json`
- `docs/architecture/apps/web/roehub-local-platform-component-registry-v1.json`
- `tests/unit/docs/test_roehub_local_platform_design_system_contract.py`
- compact evidence at the path declared in front matter

## Non-goals

- No Penpot read or write and no Penpot file creation.
- No product, Web, API, chart-renderer, CSS, package, or build implementation.
- No `roehub.com` information architecture, public-site design, or public-site
  component composition.
- No screenshots, visual acceptance, or claim that a token/component exists at
  runtime.
- No plan, ledger, prompt pack, Goal, commit, push, release, or deploy.

## Proof boundary

- All six theme IDs and all three supported widths are represented exactly.
- Every accepted local `screen_id` and required state maps to one or more
  component families without inventing a new product route or permission.
- Chart contracts prohibit raw JavaScript callbacks, secret-bearing tooltips,
  and unrestricted renderer/plugin options.
- Progress and ETA distinguish queue wait, measured execution, insufficient
  confidence, completion, failure, and cancellation.
- Accessibility requirements are machine-checkable where practical.
- Penpot names and design-to-code identifiers are stable but do not claim a
  live design artifact.
- Focused tests, local links, architecture index, project map, and
  `git diff --check` pass.

## Escalation triggers

- A component or state requires a new route, role, capability, or product
  decision not present in the accepted architecture.
- A theme cannot satisfy the documented contrast/accessibility target.
- A chart requirement would require unrestricted ECharts options or executable
  plugin content.
- Public-site composition cannot be separated from local-platform composition.

## Acceptance

- All deliverables and evidence exist and satisfy the proof boundary.
- A cold self-review finds no Penpot mutation, implementation claim, public-site
  leakage, or hidden product decision.
- The ticket may then become `accepted`; its newly created design-system
  contract deliverables remain `ready_for_product_review` until the product
  owner approves them.
- Ticket acceptance does not authorize Penpot work or implementation.
