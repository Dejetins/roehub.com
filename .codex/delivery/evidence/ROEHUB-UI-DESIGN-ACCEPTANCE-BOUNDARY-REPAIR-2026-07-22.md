---
evidence_id: ROEHUB-UI-DESIGN-ACCEPTANCE-BOUNDARY-REPAIR-2026-07-22
ticket_id: ROEHUB-UI-DESIGN-ACCEPTANCE-BOUNDARY-REPAIR-2026-07-22
verdict: passed
observed_at: 2026-07-22T16:21:47Z
---

# UI design acceptance boundary repair evidence

## Observed problem

The product owner inspected the visible output of
`ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20` and rejected it as UI
design. The executing agent later confirmed that no product owner or designer
had approved the visible prototype and that `accepted` had been assigned from
technical route, state, transport, performance, test, and CI evidence only.

The spike prompt simultaneously prohibited Figma/design work and required a
visible table, four themes, panel resizing, browser screenshots, and a published
React prototype. That combination produced a design-like artifact without a
design approval boundary.

## Corrected interpretation

- Technical architecture evidence remains factual and accepted.
- Visual design status: `rejected_by_product_owner`.
- Visual source role: `prohibited` (`not_a_design_source`).
- Reusable boundary: route isolation, SSR rollback, MobX/Query authority,
  REST/SSE seams, cancellation, tests, measurement harness, and dependency data.
- Prohibited inheritance: layout, styling, theme values, component anatomy,
  interface copy, fixture presentation, and screenshots.

## Repository changes

- Added the active boundary-repair ticket and this evidence file.
- Added the separate draft UI instructions-and-copy review ticket with
  `acceptance_authority: product_owner` and
  `agent_self_acceptance: prohibited`.
- Inserted both tickets into the unified graph. The boundary repair is
  `accepted`; the instructions-and-copy review is the only `ready` ticket;
  Figma remains `draft` and depends on the accepted review.
- Added later rejection metadata and explanatory addenda to the accepted spike
  ticket, evidence, and prototype README without changing its technical facts.
- Added five mandatory product-owner checkpoints to the Roehub Figma standard
  and Figma ticket. The React-shell ticket now requires complete replacement of
  the spike visual layer from approved Figma nodes.
- Updated the transition specification, UI migration registry, repository
  guidance, validator, regression tests, and deterministic project map.

Exact changed paths:

```text
.codex/AGENTS.md
.codex/delivery/evidence/ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20.md
.codex/delivery/evidence/ROEHUB-UI-DESIGN-ACCEPTANCE-BOUNDARY-REPAIR-2026-07-22.md
.codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json
.codex/delivery/specs/roehub-linear-workspace-ui-transition.md
.codex/tickets/2026-07-20-roehub-figma-linear-vnext-foundations.md
.codex/tickets/2026-07-20-roehub-linear-frontend-architecture-spike.md
.codex/tickets/2026-07-20-roehub-react-linear-application-shell.md
.codex/tickets/2026-07-22-roehub-ui-design-acceptance-boundary-repair.md
.codex/tickets/2026-07-22-roehub-ui-instructions-and-copy-review.md
apps/platform-web/README.md
docs/architecture/project-map/PROJECT_MAP.md
docs/architecture/project-map/project-map.json
docs/architecture/ui/roehub-figma-design-delivery-standard-v1.md
docs/architecture/ui/roehub-linear-ui-migration-registry-v1.json
tests/unit/tools/test_validate_roehub_delivery_model.py
tools/delivery/validate_roehub_delivery_model.py
```

## Verification

- `python -m tools.delivery.validate_roehub_delivery_model` — passed.
- `uv run pytest -q tests/unit/tools/test_validate_roehub_delivery_model.py`
  — `11 passed`.
- Routed `apps-platform` shard — `750` tests, `0` failures, `0` errors,
  `0` skipped.
- Focused Ruff check and format check — passed.
- Focused Pyright — `0 errors`, `0 warnings`, `0 informations`.
- `python .codex/hooks/tests/run_tests.py` — all `11` active regressions passed.
- Architecture index — unchanged and fresh after generation/check under the
  required architecture-index lock.
- Project map — regenerated; all `5` artifacts fresh.
- OSS metadata check, JSON parsing, and `git diff --check` — passed.
- CI routing: `code=true`, `docs=true`, `run_migrations=false`,
  `has_tests=true`, shard `apps-platform`.

Cold self-review found no contradiction between the technical spike verdict and
the later product-owner visual rejection: the historical measurements remain
intact, while all active downstream sources now prohibit visual inheritance.
No Figma, runtime, backend, route, authorization, public-site, or external
tracker boundary was crossed.

## Publication boundary

The product owner separately authorized completion and publication after the
local validation report. This evidence records the pre-push state; the final
handoff records commit identity and GitHub checks. Release and runtime
deployment are not required for this repository-policy change.
