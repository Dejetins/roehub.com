# S04 verification record

Commands run from repository root. `python` was unavailable on PATH; `.venv/bin/python`
and `uv run --no-sync` use the existing environment without adding dependencies.
All DSNs here identify explicitly created disposable loopback containers, no secrets.

## Final checks

```sh
ROEHUB_S04_DISPOSABLE_DSN='host=127.0.0.1 port=55895 user=postgres dbname=postgres' \
ROEHUB_S03_DISPOSABLE_DSN='host=127.0.0.1 port=55895 user=postgres dbname=postgres' \
uv run --no-sync pytest -q -ra tests/real_infra/market_data/test_postgres_readers_frontier.py tests/real_infra/market_data/test_transactional_ingestion_postgres.py

ROEHUB_S02_DISPOSABLE_DSN='host=127.0.0.1 port=57710 user=postgres dbname=postgres' \
ROEHUB_S02_DISPOSABLE_CONTAINER=roehub-s04-schema-check \
uv run --no-sync pytest -q -ra tests/real_infra/market_data/test_storage_schema_postgres.py

uv run --no-sync pytest -q tests/unit/contexts/backtest_artifacts tests/unit/apps/migrations tests/unit/contexts/market_data/application/use_cases/test_rest_fill_range_1m.py tests/unit/contexts/market_data/application/use_cases/test_rest_catchup_1m.py

ROEHUB_S04_DISPOSABLE_DSN='host=127.0.0.1 port=55895 user=postgres dbname=postgres' \
uv run --no-sync python .codex/agents/generated/postgresql-only-storage-v1/reports/S04/query_probe.py

uv run --no-sync python -m tools.docs.generate_project_map
uv run --no-sync python -m tools.docs.generate_project_map --check
```

Ruff final scope: PostgreSQL market-data directory, artifact `stored_coordinates.py`
and `automatic_base_repository.py`, artifact `base_source_reconciliation.py` and
`management.py`, `apps/migrations/{market_data,storage}.py`, S04 real tests, schema
real tests and migration lifecycle unit test. Invocation `uv run --no-sync ruff check`
with those explicit paths. Pyright used the same production scope and S04 real test
via `uv run --no-sync pyright`. Scoped `git diff --check -- <owned paths>` passed.
CI routing was read from `.github/workflows/ci.yml`, `pyproject.toml` and
`tools/ci/route_changes.py`; full dirty-tree gates were not claimed.

Entry: `.venv/bin/python /Users/daniildegtyarev/.codex/skills/prompt-manager/scripts/validate_pack.py`
with `--root "$PWD" --ledger "$PWD/.codex/agents/generated/postgresql-only-storage-v1/iteration-ledger.md"`
`--check entry --stage S04 --executor s04-20261014-readers-frontier`.
Original entry before claim passed. A late read-only revalidation omitted executor
and correctly refused claimed-stage resume; corrected invocation passes.

## Retained attempts and attribution

- `real-initial.log`: incomplete CandleMeta fixture; corrected fixture fields.
- Subsequent incremental `real-attempt*.log`: expanded scenarios; successful prior
  subsets are retained without counting repeated runs as independent proof.
- `real-attempt6.log`: broader schema suite lacked its explicit restart target;
  independent fixed-target schema run added.
- `schema-attempt2.log`: ephemeral mapped port changed on actual container restart;
  recreated only the owned disposable target with fixed port57710.
- `schema-attempt3.log`: preexisting test restored0036 checksum from `phases[-1]`,
  now0038, contaminating subsequent bootstrap; exact phase lookup fixed the test.
- `real-attempt7.log`: market session count20 instead of16 while a concurrent query
  probe held four gateways; isolated final run passed45 tests. No cap implementation
  change or ignored assertion; the probe is run separately in final verification.
- Earlier probe attempts: harness syntax, unavailable psutil (replaced with ps,
  no dependency installed), invalid fixture source value, and a previous unreleased
  verifier lease. Corrected source, capture and release lifecycle; final3 passes.
- Earlier Ruff/Pyright outputs expose line-length/type errors corrected before
  final gates. Pyright upgrade notice is informational, no package upgrade performed.

## Boundaries

Only synthetic/disposable PG16 data was used; no live exchange behavior, browser
flow, installation activation or production delivery is inferred. One optional
S03-specific restart test is skipped in the combined suite; actual schema restart
passes separately, and original accepted S03 restart evidence remains unchanged.
Full S09 mixed soak, percentile/long-duration memory results, physical installation
connection cap and activated artifact publication fencing remain later-stage work.
Independent review inspected code and saved evidence, without its own test rerun.
