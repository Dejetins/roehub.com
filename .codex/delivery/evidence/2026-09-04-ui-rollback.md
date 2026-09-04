# UI rollback — 2026-09-04

## Authority and scope

The user authorized restoring the content baseline
`7795e1dd703f4474b634443b89b17ea4803f8621`, discarding the unwanted local UI
changes without backups, retiring the staged design requirements, committing
and publishing the result to `main`, and deleting every other branch.

The rollback is a new commit on the existing `main` history. It removes the
subsequent UI program, its prompts, research and references, v24/v25 specimens
and reviews, and the ignored build/dependency residue of the removed
`apps/platform-web` and `prototypes/roehub-v2` prototypes. The previous audit
report is superseded by this completion record. Unrelated historical backend
work, installed global skills, tags, and current runtime contracts are retained.

The former design stages and their mandatory acceptance/handoff requirements
are removed from repository guidance, product requirements and architecture.
The replacement development workflow remains undecided.

## Preserved pilot and implementation

- Path: `.codex/delivery/evidence/roehub-ui-agent-governed-pilot/specimens/2026-08-03-linear-black-workbench-v23.html`.
- SHA-256: `3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.
- The file is the accepted visual reference and remains byte-identical to the
  selected baseline. It is not proof of implemented APIs or authorization.
- Application code, migrations, tests, installation configuration, release
  tooling and CI configuration remain identical to the selected baseline.

## Verification

Pre-publication verification:

- Protected pilot SHA-256 and byte identity against `7795e1dd`: passed.
- `git diff --quiet 7795e1dd -- apps src tests tools infra configs .github pyproject.toml uv.lock`: passed.
- Search of `docs`, `.codex`, `README.md`, and `AGENTS.md` for the retired
  workflow name and former program-based handoff terminology: no matches.
- `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/docs/test_roehub_ui_surface_inventory.py tests/unit/docs/test_roehub_local_platform_information_architecture.py`: `6 passed`.
- `python3 -B -m tools.docs.generate_docs_index --check`: passed.
- `python3 -B -m tools.docs.generate_project_map --check`: passed after regeneration.
- `python3 -B tools/release/oss_metadata.py --check`: passed, three artifacts.
- `git diff HEAD --check`: passed.
- All 71 tracked deletions exactly match the 71 files added by `c68c6e0b`.
- CI routing: documentation checks only; no application test shard, migration,
  or container-image build is selected by this change.

Cold-head review: completed
Mode: independent subagent
Review scope: repository guidance, architecture/requirements, deletion scope,
protected pilot, implementation baseline, and publication authority.
Review instructions: `architecture-review/references/cold-head-plan-prompt-pack-review.md`
Verdict: Release after fixes
Blockers fixed: none; the sole Low finding identified three generic references
to the retired process, which were replaced with user-authorized task wording.
Local follow-up check: completed
Residual risks: browser/runtime behavior is not revalidated because the pilot
and implementation are unchanged; remote completion is verified separately.

## Publication boundary

At pre-publication inspection, `origin/main` is `c68c6e0b`, is not protected,
and has no open PR. Only `main` exists locally. The sole other remote branch is
`codex/web-execution-telegram-notifications-v1` at `093aa1eb`; it has two commits
outside `main`. The user's instruction explicitly authorizes deleting that
branch without retaining a backup. Tags are outside branch-cleanup scope.

This is the pre-publication receipt. The final task report must verify the
actual main revision, GitHub CI, and the absence of non-main local and remote
branches after publication.

No runtime deployment target is configured or selected. This task authorizes
Git publication, not a deployment to an installation.
