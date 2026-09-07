# UI functional contract and Backtests task delivery — 2026-09-08

## Authorized scope

The user requested consolidation of current/historical functional requirements,
preparation of the first Backtests execution task, and publication through a
technical branch into `main`, followed by branch deletion. No application
implementation or deployment was requested.

Classification: functional specification plus ready local implementation task.
Primary procedure: `delivery-orchestrator`; review: `architecture-review`;
publication: `publish-ci-deploy`. No prompt pack or stage ledger is created.

## Deliverables

- `docs/architecture/apps/web/roehub-ui-functional-contract-v1.md` is the current
  functional coverage/behavior entrypoint.
- `docs/architecture/apps/web/roehub-ui-functional-registry-v1.json` contains
  44 screen/nonvisual/history records, 43 source surface identities and 18
  journeys, with implemented-scope evidence and independent delivery tags.
- `.codex/tickets/2026-09-08-roehub-backtests-client.md` is `ready` for local
  implementation/integration on the existing artifact-backed API. It specifies
  inputs, validation, endpoints, state transitions, cancel/delete, lost-response
  recovery, scoped client introduction and observable acceptance.
- Existing inventory, IA, access/screen/surface companions and product/delivery
  documents point to the current contract. Baseline grants and runtime facts
  are preserved; no historical theme/process requirements are revived.
- The two preceding audit reports and specimen screenshots are retained as
  supporting evidence, with their original observation boundaries.

Technical branch: `codex/ui-functional-contract-backtests`.
Starting revision: `c6f6e93e`, equal to `origin/main` at initial inspection.
The pre-existing `.codex/AGENTS.md` edit is foreign and excluded from publication.
The protected v23 HTML remains unchanged.

## Local validation

- Inline Python registry validation: passed for all 44 historical/current IDs,
  43 source surfaces, 18 unique journeys, unique routes, navigation and journey
  references, status labels, existing capability references, absence of inherited
  six-theme state, existing local/Git source paths and Markdown links.
- `.venv/bin/python -B -m pytest -q -p no:cacheprovider
  tests/unit/docs/test_roehub_ui_surface_inventory.py
  tests/unit/docs/test_roehub_local_platform_information_architecture.py`:
  `6 passed`. These tests verify preserved baseline registries, not new runtime.
- `python3 -B -m tools.docs.generate_docs_index --check`: passed after regeneration.
- `python3 -B -m tools.docs.generate_project_map --check`: passed after regeneration.
- `python3 -B tools/release/oss_metadata.py --check`: passed. The first run
  correctly rejected two new screenshot assets absent from the inventory;
  their own first-party license/hash records were added to
  `tools/release/oss_policy.json`, then all three generated artifacts validated.
- `git diff --check`: passed; new text files are checked through the staged diff.
- Protected v23 SHA-256 remains
  `3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.
- After registering screenshot assets, focused OSS plus both registry suites:
  `.venv/bin/python -B -m pytest -q -p no:cacheprovider
  tests/unit/tools/test_oss_metadata.py
  tests/unit/docs/test_roehub_ui_surface_inventory.py
  tests/unit/docs/test_roehub_local_platform_information_architecture.py`:
  `12 passed`.
- `.venv/bin/ruff check .`: passed; `.venv/bin/pyright`: `0 errors, 0 warnings`.
- CI path classification selects static/metadata checks and `apps-platform` /
  `platform-contracts` suites because screenshot inventory touches release
  policy; migrations are not selected. `web_image_changed=false`.

## Review and publication boundary

Cold self-review completed: current functional scope is separated from existing
implementation, historical graph proposals, future target rights and publication.
Independent review completed under repository release/security review
instructions. Initial verdict `Ready with changes`: two Medium findings required
organization-scoped recovery and separation of job-create request-hash/TTL from
saved-strategy provenance/idempotency. Both were fixed; focused independent
re-review verdict `Ready`, with no remaining findings. The ticket explicitly
retains read-only unresolved recovery when the existing API cannot guarantee
safe server-bound replay. This is a known delivery limit, not a success claim.

No backend/browser runtime, API migration, dependency installation or deployment
was performed. The intended terminal publication state is `shipped-no-runtime`.
GitHub PR/merge/check state and branch deletion are verified separately in the
final user report; this receipt is not a claim that those transitions already
occurred when the document was authored.
