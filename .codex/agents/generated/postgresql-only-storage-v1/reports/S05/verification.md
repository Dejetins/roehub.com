# S05 verification and reproduction

All execution used the repository virtualenv on the current local host. No real
exchange/provider request, installed environment cutover, or Git publication was
performed. Provider behavior in REST tests is a deterministic test port; database,
Redis, native NPY computation and native exact backtest scoring are real.

## Final commands

```sh
ROEHUB_S04_DISPOSABLE_DSN='host=127.0.0.1 port=57915 user=postgres dbname=postgres' \
ROEHUB_S05_REDIS_PORT=57918 \
ROEHUB_S05_TLS_ROOT=/tmp/roehub-s05-tls/server.crt \
ROEHUB_S05_DISPOSABLE_CONTAINER=roehub-s05-consumers \
.venv/bin/pytest -q tests/real_infra/market_data/test_postgres_consumers.py
```

`consumers-final.log`: 15 passed. Includes the explicitly guarded restart of the
owned S05 container. TLS certificate files were temporary, with localhost SAN;
no private key or authentication secret is retained in evidence.

```sh
.venv/bin/pytest -q \
 tests/unit/contexts/strategy/adapters/test_strategy_live_runner_wiring_module.py \
 tests/unit/contexts/strategy/application/test_closed_candle_tail_provider_contract.py \
 tests/unit/contexts/strategy/application/test_strategy_live_runner.py \
 tests/unit/contexts/market_data/adapters/test_postgres_candle_repair_audit_repository.py \
 tests/unit/infra/test_monitoring_assets.py
```

`repair-consumers-final.log`: 33 passed, including the audit test previously
accidentally deselected by `-k 'not clickhouse'` (32 passed, 1 deselected).

```sh
.venv/bin/pytest -q tests/unit/contexts/market_data/application \
 tests/unit/contexts/backtest_artifacts \
 tests/unit/apps/migrations/test_storage_lifecycle.py \
 tests/unit/contexts/market_data/adapters/test_postgres_dns_lifecycle.py \
 tests/unit/apps/api/test_market_data_reference_wiring_module.py \
 tests/unit/apps/api/test_market_data_reference_routes.py \
 tests/unit/apps/api/test_market_data_catalog.py \
 tests/unit/apps/cli/test_backtest_artifact_publish_cli.py \
 tests/unit/apps/cli/test_backfill_1m_parsing.py \
 tests/unit/apps/cli/commands/test_funding_rate_catchup_cli.py \
 tests/unit/apps/cli/commands/test_rest_catchup_1m_cli.py \
 tests/unit/apps/scheduler/test_backtest_artifact_publisher_app.py \
 tests/unit/apps/scheduler/test_market_data_scheduler_funding.py \
 tests/unit/apps/scheduler/test_market_data_request_concurrency.py \
 tests/unit/apps/worker/test_market_data_io_process.py \
 tests/unit/apps/test_strategy_live_runner_main.py
```

`unit-final2.log`: 248 passed. Includes bounded batching, source revision/recovery,
DNS address refresh/lifecycle, composition and scheduler tests. Five DNS cases in
this run supersede the earlier four-case standalone `dns-lifecycle.log`.

```sh
ROEHUB_S04_DISPOSABLE_DSN='host=127.0.0.1 port=57915 user=postgres dbname=postgres' \
ROEHUB_S03_DISPOSABLE_DSN='host=127.0.0.1 port=57915 user=postgres dbname=postgres' \
.venv/bin/pytest -q tests/real_infra/market_data/test_transactional_ingestion_postgres.py \
 tests/real_infra/market_data/test_postgres_readers_frontier.py -k 'not restart'
```

`ingestion-readers-final.log`: 45 passed, 1 deselected. The S03 restart guard rightly
refused a S05-labeled container in the preceding attempt. S05's dedicated guarded
restart test supplies current restart evidence; prior S03 evidence is immutable.

`schema-access-final.log`: 9 passed, 1 deselected from
`tests/real_infra/market_data/test_storage_schema_postgres.py -k 'not restart'`
using `ROEHUB_S02_DISPOSABLE_DSN` at the same owned target. This exercises real
application-role DDL denial. The earlier S02 test required its own explicitly
named restart target; S05 did not bypass that guard.

`ruff-final.log` and `pyright-final.log` contain complete argument lists and exit
codes for changed Python files (Pyright: production paths only). Ruff passed;
Pyright returned zero errors/warnings. A newer-tool-version notice is not an error.
`project-map-check.log` verifies the owner-generated navigation artifacts.

## Browser proof

Named `playwright-cli -s=s05` Chromium session, accepted `apps/platform-web`,
localhost57917 -> disposable API57916 -> PG57915/s05_browser. Real local identity
bootstrap/passkey flow with a virtual authenticator; no auth-route mocks. API was
configured with `CH_HOST=unreachable.invalid` and `STRATEGY_PG_DSN` pointing to the
fixture. The accepted client was rebuilt with `pnpm --filter @roehub/platform-web
build`; the bundle-size warning is retained, not a runtime failure.

`browser-pg-symbol.png` was visually inspected. `browser-pg-symbol.txt` shows
S05USDT selected in Backtests. `browser-pg-symbol-request.log` records actual200
workstation response with `source=market_data_reference`, `state=ready`, and seeded
BTCUSDT/ETHUSDT/S05USDT options, plus the response to the UI-triggered S05USDT bounds
request. Bounds correctly report unavailable because this browser fixture has no
published artifact. `browser-console-final.log`: zero errors/warnings.

Earlier `strategies-pg.png`/`strategies-browser.txt` and `backtests-pg.png` are
route/session smoke, not storage parity. `browser-reference-api.log` is an
authenticated API read, not by itself a UI journey. Full run/history/progress/event
journal and monitoring journeys remain S08. Candidate navigator-web was neither
promoted nor claimed runtime-tested; API shapes/auth/loading contracts were not
changed. Initial old SSR paths had missing built assets, and an earlier stale
platform bundle rejected the current session shape; rebuilding the accepted client
resolved the observed accepted-route issue without UI code changes.

## Failed attempts and corrections

Earlier real/unit/lint/typecheck logs are retained. Fixture assumptions (positional
adapter args, old source labels, test mock composition, artifact dates/schema and
native input setup) were corrected. Publication race tests exposed and verified
prepublication cleanup and fresh-build retry. A headroom assertion initially counted
one additional browser API pool; the isolated test passed after stopping that owned
process. It is not evidence of an installed physical48-session envelope. DNS now
resolves when a connection is acquired, so the old constructor-time hostname error
expectation was updated to the real operation failure. Final positive evidence is
listed above, not inferred from earlier failures.

## Resource and external-proof limits

Owned PG16: 2CPU/4GiB, max_connections48, shared_buffers256MiB, work_mem4MiB,
maintenance_work_mem64MiB. Owned Redis7.4: 0.5CPU/128MiB. No ClickHouse service was
started for the fixture, but its Python package and operational legacy paths remain
until S06/S07. Actual provider networking, installed embedded/external DNS, all
physical handshakes/control gateway concurrency, full parity and mixed-workload
performance remain S06/S08/S09. No latency improvement or production cap is claimed.
