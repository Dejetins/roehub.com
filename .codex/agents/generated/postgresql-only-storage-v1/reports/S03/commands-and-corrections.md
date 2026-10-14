# S03 commands and corrections

Working directory `/Users/daniildegtyarev/Projects/roehub.com`. Existing `uv run --no-sync`
runtime used; no dependency installation. `python` was absent; stdlib ledger commands
use `python3`. Initial read-only entry passed; atomic claim:
`python3 .codex/agents/generated/postgresql-only-storage-v1/ledger_update.py claim --stage S03 --executor s03-20261013-01a13706-ingestion`.

Real fixture: `docker run -d --name roehub-s03-01a13706 --label roehub.disposable=S03
--cpus 2 --memory 4g -e POSTGRES_HOST_AUTH_METHOD=trust -p 127.0.0.1:59119:5432
postgres:16 -c max_connections=48 -c shared_buffers=256MB -c work_mem=4MB
-c maintenance_work_mem=64MB`. Only this newly created labeled container was used.
Trust authentication has no secret and is restricted to this disposable loopback
surface. Every suite creates a unique test database; failed setup remnants remain
only inside the container which is removed at completion. Docker restart tests
verify the label before restarting this container. No live installation selected.

Final checks:

- `ROEHUB_S03_DISPOSABLE_DSN='host=127.0.0.1 port=59119 dbname=postgres user=postgres' ROEHUB_MARKET_DATA_TEST_DSN='host=127.0.0.1 port=59119 dbname=postgres user=postgres' ROEHUB_S03_DISPOSABLE_CONTAINER=roehub-s03-01a13706 uv run --no-sync pytest -q -ra tests/real_infra/market_data/test_transactional_ingestion_postgres.py tests/real_infra/market_data/test_stream_recovery_postgres.py`: `real-pg-final.log`.
- `uv run --no-sync pytest -q -ra tests/unit/apps/migrations tests/unit/apps/worker/test_persistent_writer.py tests/unit/contexts/market_data/application/services/test_ws_worker_publishes_redis.py tests/unit/contexts/market_data/application/services/test_insert_buffer.py`: 97 passed, `unit-final.log`.
- `uv run --no-sync pytest -q -ra tests/unit/contexts/market_data/application/services/test_async_recovery_admission.py tests/unit/contexts/market_data/application/services/test_stream_recovery.py tests/unit/contexts/market_data/application/services/test_rest_fill_queue.py tests/unit/contexts/market_data/application/use_cases/test_rest_fill_range_1m.py tests/unit/contexts/market_data/application/use_cases/test_rest_catchup_1m.py tests/unit/contexts/market_data/test_stored_candle_signal.py`: 32 passed, `recovery-unit.log`.
- Scoped `uv run --no-sync ruff check` on all changed Python files: `ruff-final.log`.
- Scoped `uv run --no-sync pyright` on new ingestion modules, async recovery, migration
  storage/market_data, persistent_writer, worker module, stored_candle_signal, raw
  writer port and real-PG test: `pyright-final.log`. No errors/warnings; tool version
  update notice is not a diagnostic or dependency update.
- Scoped `git diff --check`: `diff-check.log`.
- Bundled `validate_pack.py --root "$PWD" --ledger "$PWD/.codex/agents/generated/postgresql-only-storage-v1/iteration-ledger.md" --check entry --stage S03 --executor s03-20261013-01a13706-ingestion`: `entry-check.log`.
- Project-map generator/check, if navigation is regenerated: separate logs; structural
  validation is never substituted for runtime proof.

Retained attempts (all failures are introduced implementation/test defects, not
claimed pre-existing failures):

- attempt1: manifest count needed 28->29 for additive0037; corrected exact count and
  phase whitelist/route. Original0001–0036 hashes unchanged.
- attempts2/3: test CandleMeta positional/required-field construction errors; corrected
  DTO fixture before any test candle writes. Bootstrap itself succeeded.
- attempt4: initial10 real cases passed. attempt5: expanded13 passed.
- attempt6: conservative byte cap fired before8192 row cap; test expected only row-cap
  message. Corrected test to accept either documented bound, keeping both limits.
- attempt7:17 passed. attempt8:31 passed,3 old CH-required cases skipped. attempt9:
  new nonblocking connection/handshake cases20 passed. attempt10:22 passed after
  reference/funding permutations. `real-pg-restart.log`:23 passed including actual
  container restart. Final combined run retained separately.
- Scoped Ruff initially found imports/line lengths/lambda style; corrected only owned
  code/import changes. Pyright initially found optional test cursor rows, incomplete
  FundingInstrument constructor, conninfo union types, explicit tuple-row generic and
  test restart variable initialization. Corrected, retained attempt logs, final passes.
- Independent review findings and follow-up corrections: `independent-review.md`.

Current CI routes read: pyproject.toml, .github/workflows/ci.yml and
 tools/ci/route_changes.py. No browser UI, live provider, full installed service boot,
CH merge parity, TLS/DNS integration, constrained-disk chaos or performance/soak
claim is made. Those boundaries remain S04–S10 obligations as documented in the report.
