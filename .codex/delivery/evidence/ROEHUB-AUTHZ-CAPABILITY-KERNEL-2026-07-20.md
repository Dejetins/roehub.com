# ROEHUB-AUTHZ-CAPABILITY-KERNEL-2026-07-20 evidence

## Delivered boundary

- Added a default-deny identity application boundary under
  `src/trading/contexts/identity/application/authorization/`.
- Stable `CapabilityId` values cover every capability in the selected delivery
  graph without reading architecture JSON at runtime.
- Decisions resolve the persisted membership role through `OrganizationRepository`;
  a client-supplied role is denied.
- The boundary denies unknown capabilities, missing organization context for
  organization-scoped capabilities, inactive or missing membership,
  cross-organization resources, missing or foreign owned resources, and every
  stored-secret reveal request.
- `installation_owner` is resolved independently through the existing repository
  port and does not use an organization role.

## Checks run

- `uv run pytest -q -ra tests/unit/identity/authorization` — `11 passed`.
- `uv run ruff check src/trading/contexts/identity/application/authorization tests/unit/identity/authorization` — passed.
- `uv run ruff format --check src/trading/contexts/identity/application/authorization tests/unit/identity/authorization` — passed (`5 files already formatted`).
- `uv run pyright src/trading/contexts/identity/application/authorization tests/unit/identity/authorization` — `0 errors, 0 warnings`.
- `git diff --check` — passed; untracked owned files were additionally checked
  with `git diff --no-index --check /dev/null <file>`.

`uv run pyright` was also run before publication. It remained non-zero because
the shared checkout contained ignored `local_artifacts/rl_trading/**` analysis
files plus type errors in a previously accepted, still-unpublished docs test;
no error was reported in this ticket's paths. The docs-test typing was corrected
in a separate publication-preflight commit without changing its behavior.

Publication preflight then ran Pyright against every tracked Python file with
`git ls-files -z '*.py' | xargs -0 uv run pyright` — `0 errors, 0 warnings`.

## Cold self-review

- No API route, browser mutation envelope, session/authentication behavior,
  persistence schema, stored role value, migration, secret storage, or secret
  output was changed.
- No route-level protection is claimed: this ticket only supplies the isolated
  decision kernel for later route-integration tickets.
- Existing persisted role values remain `owner`, `admin`, `operator`, `trader`,
  and `viewer`; the kernel reads those values through the existing membership
  port.

## Verdict

Scoped implementation and acceptance checks pass. The global Pyright baseline
failure above is unrelated pre-existing repository state.
