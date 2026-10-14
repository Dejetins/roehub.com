# Actual commands, corrections and proof boundary

2026-10-13, S01 executor `s01-20261013-01a136ec-docs`.

- `python .../ledger_update.py --help`: failed, shell `command not found: python`.
  Resolved with `/opt/homebrew/bin/python3`; repository doc generators used the
  existing `.venv/bin/python`, no installation/dependency change.
- Bundled `validate_pack.py --root . --ledger .codex/agents/generated/postgresql-only-storage-v1/iteration-ledger.md --check entry --stage S01`: pass before claim.
- `python3 .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S01 --executor s01-20261013-01a136ec-docs`: atomically `in_progress`, ledger `active`.
- Source discovery: `git ls-files -co --exclude-standard -z`, `git status --short`,
  `git rev-parse HEAD`; targeted `rg` by SQL/ports/callers and current DDL/adapters.
  Baseline HEAD/status/file hashes retained. No env dump or provider payload saved.
- Exploratory `cat .../services/source_reconciliation.py` and
  `.../stores/funding_rate_store.py` failed because guessed paths did not exist;
  resolved via `rg` to `base_source_reconciliation.py` and split funding ports.
  A zsh `migrations/postgres/*repair*` glob had no matches; audit subsequently
  follows actual persisted table references. These failures are not absence proof.
- `python3 .../reports/S01/audit_sources.py`: pass; AST/import + SQL dependency audit,
  evidence snapshot and all dispositions in `audit-final.log`/`audit-summary.json`.
- Initial docs-index/project-map checks after adding the target doc returned
  out-of-date. Ran `.venv/bin/python -m tools.docs.generate_docs_index` and
  `.venv/bin/python -m tools.docs.generate_project_map` against current source;
  generated ownership is separate from preexisting foreign content.
- Ownership check first failed on a frontmatter-bearing document: target block
  had been inserted after opening `---`. Corrected three blocks to follow actual H1.
- Local-link check then caught one excess `..` in target-to-pack links. Corrected
  all such links. Final verifier passes: 34 original document bodies byte-preserved,
  every other baseline file outside the declared 3 generated outputs unchanged,
  all318 Appendix paths classified, 43 introduced local links resolve.
- `python3 .../reports/S01/verify_s01.py`: pass; `ownership-check.log` and
  `ownership-and-contract-check.json` include actual invariants and changed hashes.
- `.venv/bin/python -m tools.docs.generate_docs_index --check`: pass, `docs-index-check.log`.
- `.venv/bin/python -m tools.docs.generate_project_map --check`: pass, `project-map-check.log`.
- `git diff --check -- docs/architecture docs/runbooks`: see `diff-check.log`, exit0.
- Bundled entry recheck with `--executor s01-20261013-01a136ec-docs`: pass, `entry-check.log`.
- One independent `architecture-review`: Ready after R1/R2 corrections; exact findings
  and recheck in `independent-review.md`. No recursive review.

Gate selection follows current `tools/ci/route_changes.py`, `.github/workflows/ci.yml`
and `pyproject.toml`. This stage changes documentation and report audit scripts only.
No agent-policy edits, product Python/TS edits, migration/schema changes or browser
behavior changes: full ruff/pyright/pytest, provider/API/browser, real DB lifecycle,
backup restore and performance tests were not run and are not claimed passed.
Future stage proof is explicitly mapped in the target contract. The updater's nine
fixture tests were reported by the orchestrator before S01; this executor did not
rerun or relabel them as implementation proof. Receipt validation follows `accept`.

Project-map inventory includes report/receipt filenames. A later check failed after
new report evidence paths were added; regeneration after final pretransition files
resolved it. Receipt insertion similarly gets one post-transition generated refresh.
