# S02 commands and corrections

All commands ran in `/Users/daniildegtyarev/Projects/roehub.com`. Python/runtime:
existing `uv run --no-sync`; no dependency installation. Claim command:
`python3 .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S02 --executor s02-20261013-01a136fb-schema`.
The updater ran entry validation under its exclusive lock; later read-only entry
recheck passed (`entry-check.log`).

Final relevant commands (outputs alongside this file):

- `uv run --no-sync pytest -q -ra tests/real_infra/market_data/test_storage_schema_postgres.py`
  with `ROEHUB_S02_DISPOSABLE_DSN` pointing only at the new loopback PG16 container and
  `ROEHUB_S02_DISPOSABLE_CONTAINER=roehub-s02-01a136fb`: 10 passed, `real-pg-verified.log`.
- `uv run --no-sync pytest -q -ra tests/unit/apps/migrations`: 76 passed, `unit-tests.log`.
- `uv run --no-sync ruff check apps/migrations/market_data.py apps/migrations/storage.py tests/real_infra/market_data/test_storage_schema_postgres.py tests/unit/apps/migrations/test_storage_lifecycle.py`: passed, `ruff.log`.
- `uv run --no-sync pyright apps/migrations/market_data.py apps/migrations/storage.py tests/real_infra/market_data/test_storage_schema_postgres.py`: zero errors/warnings, `pyright.log`.
- `uv run --no-sync python -m tools.docs.generate_project_map`, then `--check`: passed.
- `git diff --check -- apps/migrations migrations/postgres tests/unit/apps/migrations/test_storage_lifecycle.py tests/real_infra/market_data/test_storage_schema_postgres.py src/trading/contexts/market_data/adapters/outbound/persistence/postgres/SCHEMA.md`: passed.

Tests create unique databases and remove only their own databases. PG16 container
was explicitly labeled `roehub.disposable=S02`, limited to 2 CPU / 4 GiB / 48 connections.
No secret was configured; trust was confined to the disposable container with a
loopback published port. Full container inspect/environment was not retained.
`disposable-runtime.json` and `postgres-version.txt` bind actual runtime. Final
`docker rm -fv roehub-s02-01a136fb` removed only that container and its own anonymous
volume; see `disposable-cleanup.log`. Existing installation/volumes were untouched.

Retained corrections:

1. First four infrastructure cases passed (`real-pg-attempt1.log`). Expanded run
   had seven passes but restart failed (`real-pg-attempt2.log`): Docker changed its
   dynamically allocated published port from 59118 to 50989. Inspection showed
   running / exit 0 / OOM false. Root-cause hypothesis was stale test endpoint;
   actual `docker port` confirmed it. Recreated only this disposable container with
   fixed loopback port 59118. Eight cases then passed (`real-pg-attempt3.log`).
2. Type checking found unchecked optional cursor rows; added explicit result
   presence and tuple-row handling. An E501 was corrected. Initial type output is
   `pyright-attempt1.log`; final scoped gate passes.
3. Independent review found inherited maintenance membership bypass. Rejected app
   membership in maintenance and both roles' membership in privileged/owner roles;
   real GRANT/REVOKE regression passes. See `independent-review.md`.
4. Expanded readiness helper initially assumed tuple rows under a dict-row status
   connection. Three cases failed (`real-pg-final.log`); the traceback identified
   that direct causal mismatch. Used an explicit tuple-row cursor. Final ten-case
   run passes, including absent partition readiness failure and bootstrap repair.

Applied baseline 0001–0035 files were never modified. New 0036 evolved only against
throwaway test databases before sealing this stage; no deployed/applied installation
migration was edited. Existing checksum values are validated by the real loader and
unit suite. Old-file hashes match the captured dirty-tree baseline.

CI routing was read from pyproject.toml, .github/workflows/ci.yml and
 tools/ci/route_changes.py. No UI, provider, performance or whole-installation claims
are made. Embedded-superuser bootstrap and external-style non-superuser database-owner
bootstrap both exercise the same production PG migration/capability/status path;
full config/Redis/CH profiles and new provisioning wiring remain S06 obligations.
