# Repository hygiene cleanup — 2026-09-06

Ticket: `ROEHUB-REPOSITORY-HYGIENE-2026-09-06`.
Authority: the user requested fixing the audit, cleanup, commit and push to `main`.
Scope: source hygiene, documentation, dependency and verification repair.
Implementation verdict: local checks passed; publication outcome is recorded
separately against the published revision, avoiding a self-referential receipt.

## Audit dispositions

| Finding | Resolution |
| --- | --- |
| F01 | Expanded CI shards for all current unit modules; regression checks full and changed-file coverage. Moved CLI parsing tests into `tests/unit/apps/cli`. |
| F02 | Schemas and SDK changes run code/contracts; app-image routing follows actual Docker COPY inputs, including schemas and backend sources. Digest-bound excluded outputs do not rebuild the image. |
| F03 | Replaced the obsolete mandatory-Keycloak model with a superseded pointer to local-auth and optional OIDC contracts. |
| F04 | Corrected backtest reset/unimplemented claims, current API/worker references and current-host assumptions. Historical iteration records retain original measurement provenance. Removed the duplicate 1400-line companion in favor of a reference pointer. |
| F05 | Generated links resolve relative to the index and URL-escape spaces. Unavailable benchmark records are identified as historical IDs rather than broken evidence links. |
| F06 | Removed frozen `docs/repository_three.md`; updated references and semantic project map to actual source and installation paths. |
| F07 | Removed dead `Dockerfile.api`; retained the used app Dockerfile. Clean-install verification additionally found missing plugin schemas: now included in wheel and app image, with a real CI wheel-resource check. |
| F08 | Removed 12 unreachable templates and 8 old JS/CSS files, plus tests whose only purpose was preserving these retired paths. Current Web behavior and security tests remain. |
| F09 | Removed 16 unreferenced private functions. Mathematical/regression tests remain and passed. |
| F10 | Removed 21 unreferenced legacy/public definitions and obsolete re-exports. No current internal consumers were found by either review. |
| F11 | Removed 18 unused direct runtime dependencies, declared `requests==2.32.5`, regenerated lock and OSS metadata. No retained package version was changed. |
| F12 | Removed three notebooks importing deleted runtime, the deliberately failing Python mirror and their obsolete README. Current canonical research notebooks remain. |
| F13 | Removed retired macOS/launchd/Monit/VPS bootstrap and monitoring assets, old backend/Web compose files and host-only tests. Relocated seven still-consumed alert-rule files to `infra/monitoring/rules`; unmigrated historical alerts remain explicit in the runbook index. Current generated installation profiles and configuration templates remain. |
| F14 | Current navigation prioritizes current product/runtime sources. Historical stage reports, benchmarks, agent tooling and retired/archival documents are grouped separately; historical evidence is retained. The separately published skill cleanup was adopted as upstream commit `f11417492c4a2c2ac3c14a50cd4bde0eb66e30c7`, not recommitted as owned work. |
| F15 | Removed tracked DS_Store, empty editor settings and obsolete local tool config (a local backup was retained). Removed the orphan frontend-spike node_modules and ignored caches: 344 directories, 189146297 bytes. Kept the active venv, user outputs and evidence. |
| F16 | All hand-authored `apps/web/dist/**` sources are visible to Git; generated root dist remains ignored. |
| F17 | Replaced an unfinished task inside an unclosed code fence with the real API error envelope and deterministic 422 normalization contract. |

## Compatibility and protected content

- Current API/DTO, persisted schema, identity/session and migration contracts: `none`.
- Legacy static URLs and public deep-import symbols: `breaking-change` for any
  unknown consumer relying on those retired surfaces. External consumers were
  not inventoried; no claim of universal backwards compatibility is made.
- Dependency requirements for current runtime consumers: `compatible-change`;
  unused packages are no longer incidentally supplied by the base install.
- Operator entrypoints for retired hosts: `breaking-change`; no current target
  or active runbook selected them. No host was accessed and no data was migrated.
- Optional S3 implementations, capability authorization kernel, schema history,
  current SSR assets, vendor/legal notices and canonical research evidence remain.
- Accepted Workbench pilot SHA-256 remains
  `3ff799ac5a5872662dda8b67fc1bd4db0c7860b7de9d84e6597465209d5dd2a4`.
- The pre-existing `.codex/AGENTS.md` working edit is excluded from this commit.
  Other pre-existing changes were already published independently on upstream
  main; fast-forward adoption preserved all working files.

## Local verification

- `python -m pytest -q tests/unit`: 2410 passed, four existing httpx cookie
  deprecation warnings. After final routing/history additions, their focused
  suite passed 395 tests. GitHub CI verifies the final committed matrix.
- Initial full run: 2406 passed and one stale golden generation-manifest hash.
  Verified that only `generation-manifest.json` changed for three profiles;
  Compose and effective configuration bytes were unchanged. Updated the golden
  and passed the 20 relevant configuration/metadata tests, then the full suite.
- `ruff check .` and `pyright`: passed (zero type errors/warnings).
- `.codex/hooks/tests/run_tests.py`: all active hook regressions passed.
- Docs index, project map, runbook, job, artifact and backup schema generators,
  OSS metadata, runtime topology and input inventory: generated and checked.
- 419 local Markdown links in the changed navigation/current entrypoints resolve.
- `git check-ignore --no-index` permits representative new Web JS/CSS sources
  and ignores root `dist/generated-output.js`.
- `uv lock --check`, wheel build and clean constrained installation passed.
  Imports and HTTP smoke run from installed site-packages, including the API,
  used HTTP providers, backtest services, login page, active JS and plugin schema.
  The first broad clean-wheel probe exposed the missing-schema packaging defect;
  the final focused probe passed after fixing it.
- Browser mechanic: installed `playwright-cli`, isolated session `hygiene`,
  disposable in-memory test clients at `http://127.0.0.1:8765`.
  Login, backtests, strategies and administration rendered at 1440×1000;
  mobile login rendered at 390×844, and language selection changed to Russian.
  No page exceptions or missing assets occurred during the multi-page smoke.
  Initial fixture discovery returned two unimplemented auth-status 404s; these
  were mocked explicitly for the final UI smoke. Locale navigation emitted a
  browser “Transition was skipped” diagnostic while completing successfully;
  this is not hidden as a zero-console-errors claim.
  This proves retained Web asset behavior against fixtures, not real auth,
  exchange, database, trading execution or deployment acceptance.

## Review and publication boundary

Review mode: cold self-review plus one independent production-risk reviewer,
required for this publication boundary by repository instructions.
Independent review found no material runtime regression in the deletions.
Its Russian-history classification, empty details-tag and image schema-routing
findings were fixed and tested before publication.

No current installation/deployment target is configured. No release bundle,
production operation, live provider action or runtime deployment was requested
or performed. The intended terminal state is `shipped-no-runtime` only after
this revision is on `main` and its relevant GitHub Actions are green.

Local logs, ownership baseline, deletion inventories, screenshots, build and
browser fixtures: `local_artifacts/repository-hygiene-cleanup/2026-09-06/`.
Final SHA, run URLs and conclusions are recorded in that directory's
`publication.json` after GitHub completes, and reported to the user.
