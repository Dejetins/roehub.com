---
ticket_id: ROEHUB-LOCAL-UI-IA-2026-07-20
status: accepted
owner: unassigned
evidence:
  - .codex/delivery/evidence/ROEHUB-LOCAL-UI-IA-2026-07-20.md
---

# Define the local-platform information architecture and access boundary

## Outcome

Roehub has one review-ready, implementation-independent target information
architecture for the self-hosted platform. Every inventoried local surface and
journey maps to a canonical screen, system boundary, role/capability rule, or
explicit historical exclusion before any Penpot or Web UI implementation
begins. Product-owner acceptance was recorded separately after ticket
completion on `2026-07-20`.

## Context

- Product baseline:
  `docs/architecture/platform/roehub-product-transformation-requirements-v1.md`
- UI delivery architecture:
  `docs/architecture/apps/web/roehub-ui-design-and-delivery-architecture-v1.md`
- Accepted inventory:
  `docs/architecture/apps/web/roehub-ui-surface-inventory-v1.md`
- Local surface registry:
  `docs/architecture/apps/web/roehub-ui-surface-registry-v1.json`
- Current organization roles and permissions:
  `src/trading/contexts/identity/domain/entities/organization.py`
- Current API evidence:
  `apps/api/routes/admin.py`, `apps/api/routes/strategies.py`,
  `apps/api/routes/backtests.py`, `apps/api/routes/ui_account.py`, and
  `apps/api/routes/market_data_reference.py`

## Scope

- Define the local platform's primary and secondary navigation.
- Convert semantic surfaces into canonical screens, overlays, system states,
  non-visual contracts, and explicit historical exclusions.
- Map every accepted local surface and all 12 journeys without adding public-site
  screens.
- Define the target route namespace and compatibility behavior, including
  `/strategies?mode=rl_ml` and legacy runbooks.
- Define a server-enforced target capability matrix for canonical organization
  roles and the separate `installation_owner` authority.
- Separate current enforcement evidence from `target_not_implemented` policy.
- Define required loading, empty, error, stale, degraded, forbidden, session,
  destructive-confirmation, and recovery states.
- Preserve the local-platform width set `820`, `1024`, and `1440` only.

## Deliverables

- `docs/architecture/apps/web/roehub-local-platform-information-architecture-v1.md`
- `docs/architecture/apps/web/roehub-local-platform-screen-registry-v1.json`
- `docs/architecture/apps/web/roehub-local-platform-access-and-route-contract-v1.json`
- `tests/unit/docs/test_roehub_local_platform_information_architecture.py`
- compact terminal evidence at the path declared in front matter

## Non-goals

- No Penpot read or write.
- No design-system contract, visual styling, tokens, components, or mockups.
- No product, API, persistence, authentication, route, or Web implementation.
- No public-site information architecture beyond preserving the existing trust
  boundary.
- No assertion that target capabilities or redirects are currently enforced.
- No plan, ledger, prompt pack, Goal, commit, push, release, or deploy.

## Blockers

- None. Product requirements already accept the role families, `/models`,
  canonical local runbooks, local widths, and the separation from `roehub.com`.
- Any newly discovered security conflict must stop the task rather than be
  resolved by weakening server enforcement.

## Repair policy

- Repair only inconsistencies in this ticket's deliverables, validation, local
  documentation links, or generated documentation.
- Do not repair product code or retroactively rewrite the accepted inventory.
- Classify every current/target mismatch explicitly.

## Proof boundary

- Every `surface_id` and `journey_id` from the accepted local registry is
  accounted for deterministically.
- Screen and capability identifiers are stable and unique.
- No local screen requires width `390`, and no public-site surface appears.
- Every current mutation-bearing surface has a target server policy or an
  explicit non-authorizing classification.
- Route migrations preserve bookmarks and define authentication, `403`, `404`,
  and redirect semantics.
- Current enforcement claims cite current code; unimplemented policy is labelled
  `target_not_implemented`.
- The capability matrix receives one independent security review.
- Focused tests, documentation index, project map, local links, and
  `git diff --check` pass.

## Escalation triggers

- A required capability would expose secrets or allow a destructive action to
  `operator`, `trader`, or `viewer` contrary to the accepted product baseline.
- A route migration cannot preserve authentication or resource identity.
- A current behavior must be changed in product code to make the architecture
  document truthful.
- Existing foreign changes cannot be separated from ticket-owned paths.

## Acceptance

- The three architecture deliverables and deterministic test exist.
- All proof-boundary checks pass.
- Independent security review has no unresolved blocking finding.
- A cold self-review finds no hidden product decision, public-site leakage, or
  implementation claim.
- Only then may status become `accepted`.
- Acceptance authorizes user review of this architecture. It does not authorize
  a design-system contract, Penpot work, or Web implementation.
