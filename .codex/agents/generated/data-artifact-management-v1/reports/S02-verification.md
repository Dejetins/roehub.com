# S02 verification evidence

2026-10-08; executor `s02-20261008-01a11c06-durable`; working directory repository root. All results below are observed local execution, not planned checks. Final source bindings are in `S02-source-bindings.json`.

## Isolation and database proof

Created only Docker container `roehub-s02-disposable-01a11c06` with label `roehub.purpose=s02-disposable`, using already-local PostgreSQL image `cf78e76683b9`; host bind `127.0.0.1:25482`. No existing preview/user database, credentials, providers or scheduler were used. Initial network-none container was removed before recreating the owned container with localhost port for psycopg. The test target uses passwordless trust only inside this disposable fixture. Tests create random `artifact_management_test_<uuid>` schemas and drop only their own schemas.

Final focused command:

```sh
ROEHUB_BACKTEST_TEST_DSN='host=127.0.0.1 port=25482 user=postgres dbname=postgres' .venv/bin/pytest -q tests/real_infra/backtest/test_artifact_management.py tests/unit/contexts/backtest_artifacts/test_management_profiles.py
```

**42 passed, exit 0, 4.05s**: 17 real PostgreSQL tests and 25 unit profile tests. Zero skipped. PostgreSQL tests exercise concurrent complete bulk replay, restart response recovery, body conflict, stale and unauthorized bulk all-or-none, roles and tenant-hidden requests/events/commands, installation ownership and CAS/no reseed, concurrent policy CAS, immutable revisions including same-batch conflict, cross-organization claim, durable restart/quarantine, measured progress and retry epoch, stale owner denial, internal dedupe/quota obligation, explicit legacy import replay/conflict, child-command tenant FK, retry budget shrink, revoked coordinate command access, concurrent retry/retry and retry/new claim, database trigger failure after first child rollback, and old replay preserving newer desired projection.

Existing physical ownership regression:

```sh
ROEHUB_BACKTEST_TEST_DSN='host=127.0.0.1 port=25482 user=postgres dbname=postgres' .venv/bin/pytest -q tests/real_infra/backtest/test_artifact_input_lifecycle.py
```

**48 passed, exit 0, 84.84s**. Existing 0028 reader/publisher reservations and filesystem drills remain separate from the new queue's reservation. These tests establish tested fixtures only; the new management layer does not itself write files or validate a current pointer.

## Complete migration path

Created separate empty `roehub_s02_migrations` database in the same owned container. Executed the repository's actual migration orchestrator twice:

```python
from pathlib import Path
import psycopg
from apps.migrations.storage import apply_postgres_migrations
with psycopg.connect('host=127.0.0.1 port=25482 user=postgres dbname=postgres',
                     autocommit=True) as c:
    c.execute('CREATE DATABASE roehub_s02_migrations')
dsn = 'host=127.0.0.1 port=25482 user=postgres dbname=roehub_s02_migrations'
root = Path.cwd()
apply_postgres_migrations(dsn, repo_root=root,
                         manifest_path=root/'migrations/postgres/manifest.json')
apply_postgres_migrations(dsn, repo_root=root,
                         manifest_path=root/'migrations/postgres/manifest.json')
```

**Exit 0**. Both calls run the CI Alembic entrypoint through the storage orchestrator, with actual ordered SQL phases including 0029. Final query found 13 `backtest_artifact_%` tables including the two old 0028 tables; `backtest_artifact_installation_settings` had **zero rows**. No activation or implicit seeding. The real-infra regression separately reapplies 0029 with stored recipe and imported legacy rows and verifies exact preserved values. This is additive migration and restart proof, not a production upgrade claim.

## Python and migration gates

- `.venv/bin/pytest -q tests/unit/contexts/backtest_artifacts tests/unit/apps/migrations`: **102 passed, exit 0, 1.19s** before five additional strict-value/signal-default tests; those five are included in the final 42-test run above.
- Repository-routed `backtest-artifacts` shard: **309 passed, exit 0, 58.91s**. Command used the real current `tools.ci.route_changes.TEST_SHARDS['backtest-artifacts'].target.split()` as arguments to `.venv/bin/pytest -q`. Later changes were limited to the new management models/catalog/import/repository and are covered by the final focused run.
- `.venv/bin/ruff check` on every Python path in `S02-source-bindings.json`: **passed**.
- `.venv/bin/pyright` on the same paths: **0 errors, 0 warnings**. Tool prints an informational newer-version notice; nothing installed.
- `.venv/bin/python .codex/agents/generated/data-artifact-management-v1/test_ledger_update.py -v`: **9 passed, exit 0** on disposable pack fixtures.
- Entry validator `--check entry --stage S02`: **status=pass**, before the atomic claim. Updater claim: **status=ok**, exact executor above.

Initial introduced failures were retained in this account: the first catalog test found four no-parameter indicators incorrectly skipped (36 versus 40); corrected to include empty compute axes and validated natively. Initial lint formatting/import errors corrected on owned files. Expanded pyright found 13 test-constructor/optional-row typing errors; corrected with typed values and explicit row checks. These are resolved, not attributed to unrelated work.

Independent `production-risk-review` found three defects, all fixed with real PostgreSQL regressions: intra-bulk immutable profile conflict, retry after budget shrink, command recovery after coordinate access revocation. Reviewer also requested concurrent retry coverage; added and passed. Final review/disposition is recorded in `S02-review.md`.

## Proof boundary

No browser change or browser pass, no scheduler activation, runtime deployment, public API/CSRF integration, real catalog-coordinate resolver, source ingestion notification, numerical build, due-date/DST scheduler, disk usage inspection or cleanup claim. The current S02 adapters require trusted composition by S03/S05/S06. Source metadata and publication receipt are passed through internal worker seams; later stages must obtain them from canonical source and the actual validated pointer commit. Public HTTP must never expose internal worker/import methods directly. Existing 0028 physical reservations remain mandatory for filesystem mutation. Unknown worker liveness remains quarantined; no timeout reclaims it.

Final pre-transition checks: own S02 entry and projected S03 entry passed through the actual validator `Pack.entry` (S03 projected only; not claimed). All 290 baseline hashes and all 13 final product/test hashes matched; owned tracked `git diff --check` passed. Removed only the owned disposable container and its volumes; the initial network-none container's anonymous volume was identified by Docker mount/unmount events bound to that exact container ID before removal. No disposable database or owned volume is retained.
