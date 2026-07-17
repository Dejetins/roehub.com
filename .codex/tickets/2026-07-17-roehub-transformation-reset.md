---
ticket_id: ROEHUB-RESET-2026-07-17
status: completed
owner: codex
---

# Retire the legacy transformation workflow and preserve product requirements

## Outcome

The uncommitted thirteen-block transformation workflow no longer has execution
authority. Stable product requirements remain in one concise normative
document, existing implementation evidence remains intact, and the repository
is ready for a future independently selected delivery ticket.

## Scope

- Remove the generated transformation prompt packs, plans, ledgers, stage
  reports, and local Penpot-derived manifests.
- Replace the seven broad target documents and the supersession stub with one
  product-requirements baseline.
- Remove or repair references from tracked documentation without restoring
  retired Mac Studio or repository-specific workflow authority.
- Rebuild generated documentation indexes and project-map artifacts.
- Mark the connected Penpot file as archived and non-authoritative.
- Preserve current product code, accepted historical implementation evidence,
  the three commits ahead of `origin/main`, and unrelated workspace changes.
- Publish the verified repository state through the repository publication
  workflow authorized by the user.

## Non-goals

- No Web UI, API, persistence, backtest, deployment, or runtime implementation.
- No revival of a legacy staged workflow.
- No deletion or rewriting of accepted historical implementation reports.
- No destructive whole-tree reset, clean, stash, or checkout.
- No production or installation deployment.

## Proof boundary

- No remaining references to the removed transformation program or documents.
- The requirements baseline grants no authority to stages, ledgers, prompt
  packs, or legacy execution modes.
- Documentation index and project-map checks pass.
- Focused tests for changed repository policy, hooks, CI routing, and generated
  documentation pass.
- `git diff --check` passes.
- Penpot reports an archived cover marker and a final archive version; the MCP
  API does not expose file-name mutation.
- Publication reports the actual PR, merge, `main` synchronization, and branch
  deletion state.

## Escalation triggers

- A required cleanup overlaps inseparable foreign product-code changes.
- Publication requires a runtime mutation or credential not already available
  through the authorized repository workflow.
- A branch is not fully merged into `main` and therefore cannot be deleted
  safely.

## Completion evidence

- Removed all generated prompt directories for the thirteen-block
  transformation program, its program directory, and local design derivatives.
- Replaced the broad target-document family with
  `docs/architecture/platform/roehub-product-transformation-requirements-v1.md`.
- Repository-wide search found no remaining references to the retired
  transformation paths or identifiers.
- Preserved historical self-hosted, identity, market-data, backtest, and Web
  implementation evidence without restoring execution authority.
- Penpot file `20d3f736-cc1b-8043-8008-5622d8ff99af` has the cover marker
  `[ARCHIVE — OLD WORKFLOW]` and saved version
  `archived-old-workflow-2026-07-17`. The available MCP API exposes no file-name
  mutation.
- `python -m tools.docs.generate_docs_index --check`,
  `python -m tools.docs.generate_project_map --check`, runbook generation,
  runtime-input inventory, OSS metadata, `ruff`, hook regressions, and
  `git diff --check` pass.
- Focused changed-boundary tests: `30 passed`; active hook regressions:
  `11 passed`; CI `apps-platform` shard: `735 passed` with four pre-existing
  `httpx` deprecation warnings.
- CI classifies `.codex/hooks/**` as code and runs the dedicated hook regression
  suite in the static job.
- Former Mac Studio runbook paths now contain non-executable tombstones, while
  full historical content lives under `docs/runbooks/legacy/`. The active
  `infra/caddy/Caddyfile.vps` contains no server block or retired upstream; its
  historical content is isolated under `infra/caddy/legacy/`.
- Local full `pyright` is not a valid clean-checkout result because it scans
  ignored `local_artifacts/rl_trading/**`; its 149 errors are outside Git and
  outside this ticket. Pull-request CI remains the publication authority.
- `configs/installation/runtime-input-inventory.json` is intentionally included
  as regenerated drift from the three already committed market-data changes on
  this branch; its `--check` gate passes.

Publication, merge, and branch cleanup are performed separately through
`publish-ci-deploy` under the user's explicit authorization and are reported by
their Git and GitHub evidence.
