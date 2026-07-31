# Roehub UI agent-governed pilot — preparation review

This review records the preparation boundary before `pilot_brief_approval`; it is not visual or product acceptance.

## Verdict

`Ready for product-owner brief approval`.

The repository and Figma scaffolds are coherent enough to begin visual-direction exploration after
the owner approves or amends the bounded Backtests brief. No library asset, direction, screen, or
runtime implementation is accepted by this result.

## Observed state

- Figma identity: `dejetins@gmail.com`, plan `team::831604964356268687`, `pro`, `Full`.
- `Roehub UI Library` (`rgbNUPCuV7q2pARG4Cml8V`) has seven contract-named empty pages and zero
  scene nodes, variables, styles, or components.
- `Roehub Authenticated Platform UI` (`nzKVsXuCmoTbHJGckHfK3T`) has six contract-named empty pages
  and zero scene nodes, variables, styles, or components.
- Former file `GBzmB9evtzqnAYNjp9W1sr` is historical and excluded from all new task inputs.
- The old copy-review and old Figma-foundations tickets are `superseded`; the new pilot is the only
  selected `active` design ticket.

## Review matrix

| Area | Status | Evidence |
|---|---|---|
| Goal and bounded outcome | OK | Small toolbar, row, dock header, and degraded state only |
| Source-of-truth ownership | OK | Product/runtime contracts → repository design contract → library → manifest → candidate |
| File and mutation boundaries | OK | Exact two file keys, exact pages, one active candidate, historical file forbidden |
| Agent trust boundary | OK | Executor cannot advance candidate/acceptance states; maximum two repair attempts |
| Structural verification | OK | Contract validator, JSON Schema, delivery-model validator, negative tests |
| Visual verification | Partial by design | No visual artifact exists yet; screenshot gate begins after direction generation |
| Product acceptance | Pending | `pilot_brief_approval` is the current explicit owner checkpoint |
| Runtime/API compatibility | N/A | No runtime, API, persistence, route, or browser mutation |
| Rollback | OK | Page scaffold is additive; later candidate writes remain isolated and ID-bounded |

## Risks and mitigations

### Library publication cannot be inferred from Plugin API writes

- Observed fact: local library components are not equivalent to a published and enabled library.
- Risk: a manifest could claim cross-file reuse before the product file can import the keys.
- Mitigation: after `library_slice_review`, the product owner publishes/enables through Figma UI;
  Codex then verifies availability read-only before composition.

### Pilot copy is not the final localization contract

- Observed fact: Roehub requires `ru/en`, while the former copy-review ticket was superseded.
- Risk: short English labels could conceal component failures with longer Russian content.
- Mitigation: English is only the pilot review convention; the library sandbox must include a long
  Russian content-extreme specimen, and final product copy remains a later owner decision.

### Structural gates cannot prove visual quality

- Observed fact: JSON and Plugin API checks do not judge hierarchy, density, or polish.
- Risk: structurally valid output may still look poor.
- Mitigation: every appearance-changing write requires a separate screenshot review; structural
  success cannot override visual failure.

## Validation evidence

- `uv run pytest -q tests/unit/tools/test_validate_roehub_ui_delivery.py tests/unit/tools/test_validate_roehub_delivery_model.py tests/unit/tools/test_generate_docs_index.py tests/unit/tools/test_generate_project_map.py`: `21 passed`.
- `uv run python -m tools.design.validate_roehub_ui_delivery`: passed.
- `uv run python -m tools.delivery.validate_roehub_delivery_model`: passed.
- `uv run python -m tools.docs.generate_docs_index --check`: passed.
- `uv run python -m tools.docs.generate_project_map --check`: passed.
- `git diff --check`: passed.

## Proof boundary

The preparation proves repository consistency, exact live Figma identity, empty page scaffolds, and
the existence of enforceable manifest gates. It does not prove a visual direction, library quality,
cross-file publication, complete product coverage, accessibility implementation, browser behavior,
or runtime readiness.
