# Backtests Figma pilot cross-chat handoff — cold-head review receipt

## Review identity

- Date: `2026-08-01`.
- Mode: independent read-only subagent.
- Review instructions:
  `/Users/daniildegtyarev/.codex/skills/architecture-review/references/cold-head-plan-prompt-pack-review.md`.
- Review scope:
  - `.codex/agents/generated/roehub-backtests-figma-pilot-v1/task.md`;
  - `docs/architecture/ui/roehub-linear-black-authenticated-workspace-visual-standard-v1.md`;
  - their direct ticket, brief, routing, Figma contract, schema, and accepted-v9 sources.
- Initial verdict: `Block`.
- Files changed by reviewer: none.

## Findings resolved

1. The reusable visual translation is now `proposed_for_library_slice_review`, is absent from the
   repository-wide authoritative source table, and cannot become cross-screen authority without an
   artifact-specific product-owner decision.
2. Font and accent rules preserve the exact accepted-v9 system stack and role-specific violet
   values. Inter substitution or accent normalization requires an explicit review decision.
3. The negative gate now uses separate unknown-component and raw-node fixtures so both rejection
   paths are observable.
4. Audit validation distinguishes the target parent from the created candidate root, requires the
   root to be manifest-owned, and requires exactly one created top-level candidate.
5. `roehub-ui-component-registry-schema-v1.json` and registry validation now cover required pilot
   component identities, bindings, publication state, accepted visual-standard revision, and
   published component keys. Manifest validation enforces exact pilot slots, actions, fields, and
   states.
6. Context is split into a five-file common core and phase bundles of no more than eight files.
7. Skill routing uses only available exact domain skills and adds `backend-quality-gates` for
   validator/test work.
8. Long English, Russian, symbol, degraded-copy, and null-value fixtures are deterministic.
9. The handoff names `/tmp/roehub-architecture-index.lock` and its fail-closed atomic-`mkdir`
   protocol.
10. The curated architecture index identifies the accepted v9 direction and the proposed status of
    the reusable visual translation.

## Local follow-up check

- Status: `completed`.
- `uv run ruff check tools/design/validate_roehub_ui_delivery.py tests/unit/tools/test_validate_roehub_ui_delivery.py`: passed.
- `uv run pyright tools/design/validate_roehub_ui_delivery.py tests/unit/tools/test_validate_roehub_ui_delivery.py`: `0 errors`.
- `uv run pytest -q tests/unit/tools/test_validate_roehub_ui_delivery.py`: `8 passed`.
- Repository contract, architecture-index, project-map, and broader focused gates are recorded in
  the publishing turn report.
- Remaining Blocker/High findings: none.

## Residual risks

- Live Figma identity, page inventory, publication state, component keys, and screenshots remain
  intentionally unverified until the executor performs the named preflight and owner checkpoints.
- `SF Pro Text` availability in the authenticated Figma environment is unknown. The prompt stops
  for an explicit fallback decision rather than silently changing typography.
- This review proves the handoff contract and validator behavior only. It does not accept a library
  slice, product composition, runtime implementation, release, or deployment.

## Follow-up verdict

`Release after fixes` for the standalone cross-chat handoff artifact. Product-owner checkpoints and
manual Figma publication remain mandatory.
