---
evidence_id: ROEHUB-FIGMA-DESIGN-DELIVERY-MIGRATION-2026-07-22
ticket_id: ROEHUB-FIGMA-DESIGN-DELIVERY-MIGRATION-2026-07-22
verdict: local_validation_passed
observed_at: 2026-07-22T13:59:01Z
---

# Roehub Figma design-delivery migration evidence

## External design boundary

- Figma account: `dejetins@gmail.com`; plan key `team::831604964356268687`;
  professional plan with a `Full` seat.
- Project: `roehub.com`; `projectId` `629113387`.
- Canonical file: `Roehub Authenticated Platform UI`; `fileKey`
  `GBzmB9evtzqnAYNjp9W1sr`;
  `https://www.figma.com/design/GBzmB9evtzqnAYNjp9W1sr`.
- The project was empty before creation. The created file contains one empty
  page (`Page 1`, node `0:1`) and no design content. This migration does not
  claim a prototype, design system, approved screen, or browser proof.

## Repository migration

- Added `docs/architecture/ui/roehub-figma-design-delivery-standard-v1.md` as
  the Roehub-specific design-tool authority while retaining the shared
  cross-project standard unchanged.
- Registered the exact Figma project, plan, file name, file key, and URL in the
  UI migration registry and repository instructions.
- Renamed the unstarted draft ticket to
  `ROEHUB-FIGMA-LINEAR-VNEXT-FOUNDATIONS-2026-07-20`, updated its proof and
  product-review boundary, and rewired the graph and React-shell dependency.
- Converted every unfinished graph ticket that mentioned Penpot to Figma or a
  tool-neutral design-artifact exclusion. Accepted historical tickets and
  evidence were not rewritten.
- Added an explicit Figma amendment to older accepted architecture contracts
  whose Penpot wording remains historical, and redirected active public-site
  design source references to the Figma standard.
- The graph still contains `15` tickets: `4 accepted`, `1 ready`, `10 draft`,
  and `0 active`. The only `ready` ticket remains
  `ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20`.

## Verification

- `python -m tools.delivery.validate_roehub_delivery_model` — passed.
- `uv run pytest -q tests/unit/tools/test_validate_roehub_delivery_model.py`
  — `7 passed`.
- `uv run pytest -q tests/unit/tools/test_validate_roehub_delivery_model.py tests/unit/docs`
  — `17 passed`.
- `uv run ruff check .` and focused `ruff format --check` — passed.
- Focused `uv run pyright` — `0 errors, 0 warnings, 0 informations`.
- `python .codex/hooks/tests/run_tests.py` — all `11` active regressions passed.
- `python tools/release/oss_metadata.py --check` — passed.
- `python -m tools.docs.generate_docs_index --check` and
  `python -m tools.docs.generate_project_map --check` — passed; the
  architecture index was generated and verified under the required lock.
- CI routing selects `apps-platform`; `web_image_changed=false`.
- `git diff --check` — passed.

## Review and residual risk

- Cold self-review confirmed that the shared standard and Custometry routing are
  unchanged, accepted Penpot evidence remains factual, no unfinished graph
  ticket retains an active Penpot instruction, the old draft Penpot ticket path
  is removed, and all canonical Figma identifiers match the live connector.
- Compatibility: `compatible replacement` for unstarted design work and `none`
  for runtime, API, routes, persistence, authorization, release, and deployment.
- The canonical Figma file is intentionally empty. It proves only a writable,
  readable design boundary; the future Figma foundations ticket must still
  create and obtain product-owner approval for actual design content.
- GitHub Actions and the common `main` publication boundary remain unverified
  until this scoped change is pushed.
