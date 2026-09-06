---
ticket_id: ROEHUB-REPOSITORY-HYGIENE-2026-09-06
status: implemented
---

# Repository hygiene and verification repair

The user authorized fixing the 2026-09-06 repository audit, cleaning obsolete
files, and committing and pushing the owned changes directly to `main`.
This is one bounded maintenance execution unit; no deployment is authorized.

## Scope

- Repair CI test coverage and schema/SDK change routing.
- Correct current documentation, links and generated navigation.
- Remove verified obsolete Web assets, Python definitions, notebook entrypoints,
  Dockerfile and local workspace debris.
- Remove unused direct dependencies while explicitly retaining used dependencies;
  regenerate the lock and release metadata.
- Retire host-specific executable assets after relocating their active consumers.
- Keep historical evidence out of current navigation; preserve migration history,
  legal notices, accepted visual pilot, optional S3 and the accepted authz kernel.
- Preserve all pre-existing foreign changes and exclude them from publication.

## Proof and compatibility

Use focused tests during changes, complete unit tests and CI coverage checks,
Ruff, Pyright, generators, clean wheel installation/import checks, and real
browser smoke against disposable local fixtures. Bind publication evidence to
GitHub checks for the pushed revision. No old host or production data access.

Removal of obsolete direct asset URLs and internal deep-import symbols is an
intentional cleanup; report these separately from unchanged current route,
persistence and identity contracts. Do not claim external consumer discovery.

## Evidence

Audit input: `local_artifacts/repository-hygiene-audit/2026-09-06/audit.md`.
Local logs and ownership baseline:
`local_artifacts/repository-hygiene-cleanup/2026-09-06/`.
Durable outcome: `.codex/delivery/evidence/ROEHUB-REPOSITORY-HYGIENE-2026-09-06.md`.

## Implementation outcome

All F01–F17 findings have a disposition in the durable evidence receipt.
Local implementation checks passed. Publication evidence is bound to the final
commit in the local `publication.json` receipt and GitHub Actions; this ticket
does not claim deployment to an installation.
