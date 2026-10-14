"""S02 schema proof, only against an explicitly selected disposable PG16 database."""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from apps.migrations.market_data import provision_market_data_access
from apps.migrations.storage import (
    StorageLifecycleError,
    _postgres_schema_status,
    apply_postgres_migrations,
)

DSN = os.environ.get('ROEHUB_S02_DISPOSABLE_DSN', '')
ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / 'migrations/postgres/manifest.json'
MIGRATION = ROOT / 'migrations/postgres/0036_market_data_storage_v1.sql'
pytestmark = pytest.mark.skipif(not DSN, reason='explicit disposable target required')


def one(cursor):
    row = cursor.fetchone()
    assert row is not None
    return row


def bootstrap(dsn):
    apply_postgres_migrations(dsn, repo_root=ROOT, manifest_path=MANIFEST)


@pytest.fixture(scope='module')
def db():
    # Never touch an existing database: create and clean only this unique test identity.
    assert conninfo_to_dict(DSN).get('host') in ('127.0.0.1', 'localhost')
    name = 's02_' + uuid4().hex
    with psycopg.connect(DSN, autocommit=True) as conn:
        assert 160000 <= conn.info.server_version < 170000
        conn.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    target = make_conninfo(DSN, dbname=name)
    bootstrap(target)
    yield target
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))


def test_clean_repeat_and_status(db):
    bootstrap(db)
    versions = _postgres_schema_status(db, repo_root=ROOT, manifest_path=MANIFEST)
    assert 'market-data-storage-0036' in versions
    with psycopg.connect(db) as conn:
        assert one(conn.execute("SELECT count(*) FROM pg_partitioned_table p JOIN pg_class c "
                            "ON p.partrelid=c.oid JOIN pg_namespace n ON n.oid=c.relnamespace "
                            "WHERE n.nspname='market_data'"))[0] == 6
        conn.execute("UPDATE roehub_storage_migrations SET checksum=repeat('0',64) "
                     "WHERE version='market-data-storage-0036'")
    with pytest.raises(StorageLifecycleError, match='checksum drift'):
        bootstrap(db)
    import json
    checksum = next(
        phase['sha256'] for phase in json.loads(MANIFEST.read_text())['phases']
        if phase['version'] == 'market-data-storage-0036'
    )
    with psycopg.connect(db) as conn:
        conn.execute("UPDATE roehub_storage_migrations SET checksum=%s "
                     "WHERE version='market-data-storage-0036'", (checksum,))


def test_all_provider_fields_and_nullable_types(db):
    import json

    expected = json.loads((ROOT / "tests/fixtures/storage/market-provider-fields.json").read_text())
    with psycopg.connect(db) as conn:
        for name, columns in expected.items():
            actual = {row[0]: row[1:] for row in conn.execute(
                "SELECT column_name,data_type,is_nullable FROM information_schema.columns "
                "WHERE table_schema='market_data' AND table_name=%s", (name,))}
            for column, contract in columns.items():
                assert column in actual, (name, column)
                assert actual[column][1] == ("YES" if contract["nullable"] else "NO")
                if contract["type"] is not None:
                    assert actual[column][0] == contract["type"]


def test_bounded_historical_rollover_concurrent_partition_creation(db):
    def provision():
        with psycopg.connect(db) as conn:
            conn.execute("SET TIME ZONE 'Pacific/Honolulu'")
            return one(conn.execute('SELECT market_data.ensure_partitions(%s,2)',
                                (date(1999, 12, 1),)))[0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: provision(), range(2))) == [12, 12]
    with psycopg.connect(db) as conn:
        for table in ('raw_binance_klines_1m', 'raw_bybit_klines_1m', 'canonical_candles_1m',
                      'raw_binance_funding_rates', 'raw_bybit_funding_rates',
                      'canonical_funding_rates'):
            for month in ('199912', '200001'):
                assert one(conn.execute('SELECT to_regclass(%s)',
                                    (f'market_data.{table}_{month}',)))[0]
        assert not conn.execute("SELECT 1 FROM pg_class c JOIN pg_namespace n "
                                "ON n.oid=c.relnamespace WHERE n.nspname='market_data' "
                                "AND pg_get_expr(relpartbound,c.oid)='DEFAULT'").fetchone()
    for start, count in ((date(2000,1,2),1), (date(2000,1,1),13), (None,1)):
        with psycopg.connect(db) as conn, pytest.raises(psycopg.errors.InvalidParameterValue):
            conn.execute('SELECT market_data.ensure_partitions(%s,%s)', (start,count))


def test_privileges_generations_and_invalidation(db):
    app = 's02_app_' + uuid4().hex
    maint = 's02_maint_' + uuid4().hex
    with psycopg.connect(db) as conn:
        for role in (app, maint):
            conn.execute(sql.SQL('CREATE ROLE {} NOLOGIN').format(sql.Identifier(role)))
        conn.execute(sql.SQL('GRANT {} TO {}').format(
            sql.Identifier(maint), sql.Identifier(app)))
        with pytest.raises(ValueError, match='maintenance'):
            provision_market_data_access(conn, application_role=app, maintenance_role=maint)
        conn.execute(sql.SQL('REVOKE {} FROM {}').format(
            sql.Identifier(maint), sql.Identifier(app)))
        provision_market_data_access(conn, application_role=app, maintenance_role=maint)
        with pytest.raises(ValueError):
            provision_market_data_access(conn, application_role='postgres', maintenance_role=maint)
        conn.execute(sql.SQL('SET LOCAL ROLE {}').format(sql.Identifier(app)))
        gen = one(conn.execute("INSERT INTO market_data.instrument_generations(market_id,symbol) "
                           "VALUES(1,'S02') RETURNING generation"))[0]
        conn.execute("UPDATE market_data.instrument_generations SET content_revision=7, "
                     "funding_revision=3 WHERE symbol='S02'")
        conn.execute("INSERT INTO market_data.candle_coverage_days "
                     "(market_id,symbol,day,generation,policy_version,actual) "
                     "VALUES(1,'S02','2000-01-01',%s,1,set_bit(0::bit(1440),0,1))", (gen,))
    # Reconnect is not a reset; durable revision is separate from unchanged coordinates.
    with psycopg.connect(db) as conn:
        assert conn.execute("SELECT generation,content_revision,funding_revision FROM "
                            "market_data.instrument_generations WHERE symbol='S02'").fetchone() == (
                                gen,7,3)
        assert conn.execute("SELECT actual_count,absent_count FROM "
                            "market_data.candle_coverage_days WHERE symbol='S02'").fetchone()==(1,0)
    for statement in ('CREATE TABLE market_data.forbidden(id int)',
                      "SELECT market_data.ensure_partitions('2001-01-01',1)",
                      'DROP TABLE market_data.ref_market'):
        with psycopg.connect(db) as conn:
            conn.execute(sql.SQL('SET LOCAL ROLE {}').format(sql.Identifier(app)))
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(cast(Any,statement))
    with psycopg.connect(db) as conn:
        conn.execute(sql.SQL('SET LOCAL ROLE {}').format(sql.Identifier(maint)))
        conn.execute("SELECT market_data.ensure_partitions('2001-01-01',1)")
        new = one(conn.execute("SELECT market_data.invalidate_instrument(1,'S02',%s,'rebuild')",
                           (gen,)))[0]
        assert new != gen
    with psycopg.connect(db) as conn:
        assert conn.execute("SELECT generation,content_revision,index_state FROM "
                            "market_data.instrument_generations WHERE symbol='S02'").fetchone()==(
                                new,0,'unknown')
        assert not conn.execute("SELECT 1 FROM market_data.candle_coverage_days "
                                "WHERE symbol='S02'").fetchone()
        with pytest.raises(psycopg.errors.SerializationFailure):
            conn.execute("SELECT market_data.invalidate_instrument(1,'S02',%s,'stale')", (gen,))


def test_killed_migration_rolls_back_and_resumes():
    name = 's02_interrupt_' + uuid4().hex
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
    target = make_conninfo(DSN, dbname=name)
    try:
        with psycopg.connect(target) as conn:
            conn.execute('CREATE TABLE public.backtest_artifact_source_state (id integer)')
        victim = psycopg.connect(target)
        victim.execute(cast(Any, MIGRATION.read_text()), prepare=False)
        with psycopg.connect(target) as conn:
            assert one(conn.execute('SELECT pg_terminate_backend(%s)',
                                (victim.info.backend_pid,)))[0]
        victim.close()
        with psycopg.connect(target) as conn:
            assert one(conn.execute("SELECT to_regnamespace('market_data')"))[0] is None
            conn.execute('DROP TABLE public.backtest_artifact_source_state')
        bootstrap(target)
        bootstrap(target)
    finally:
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))


def test_interrupted_partition_transaction(db):
    victim = psycopg.connect(db)
    victim.execute("SELECT market_data.ensure_partitions('1981-12-01',2)")
    with psycopg.connect(db) as conn:
        conn.execute('SELECT pg_terminate_backend(%s)', (victim.info.backend_pid,))
    victim.close()
    with psycopg.connect(db) as conn:
        assert one(conn.execute("SELECT to_regclass('market_data.canonical_candles_1m_198112')"
                            ))[0] is None
        conn.execute("SELECT market_data.ensure_partitions('1981-12-01',2)")


def test_non_superuser_migration_owner_external_profile():
    from apps.migrations.storage import check_postgres_capabilities
    role = 's02_owner_' + uuid4().hex
    name = 's02_external_' + uuid4().hex
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(sql.SQL('CREATE ROLE {} LOGIN').format(sql.Identifier(role)))
        conn.execute(sql.SQL('CREATE DATABASE {} OWNER {}').format(
            sql.Identifier(name), sql.Identifier(role)))
    target = make_conninfo(DSN, dbname=name, user=role)
    try:
        check_postgres_capabilities(target)
        bootstrap(target)
        bootstrap(target)
        assert 'market-data-storage-0036' in _postgres_schema_status(
            target, repo_root=ROOT, manifest_path=MANIFEST)
        with psycopg.connect(target) as conn:
            assert conn.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user'
                                ).fetchone() == (False,)
    finally:
        with psycopg.connect(DSN, autocommit=True) as conn:
            conn.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))
            conn.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(role)))


def test_real_server_restart_preserves_revision(db):
    import json
    import subprocess
    import time
    container = os.environ.get('ROEHUB_S02_DISPOSABLE_CONTAINER')
    assert container, 'restart proof requires explicitly labeled disposable container'
    info = json.loads(subprocess.check_output(['docker','inspect',container], text=True))[0]
    assert info['Config']['Labels'].get('roehub.disposable') == 'S02'
    with psycopg.connect(db) as conn:
        conn.execute("INSERT INTO market_data.instrument_generations "
                     "(market_id,symbol,content_revision,funding_revision) "
                     "VALUES(2,'RESTART',19,5)")
        before = conn.execute("SELECT generation,content_revision,funding_revision "
                              "FROM market_data.instrument_generations WHERE symbol='RESTART'"
                              ).fetchone()
    subprocess.run(['docker','restart',container], check=True, capture_output=True, timeout=30)
    for _ in range(60):
        try:
            with psycopg.connect(db, connect_timeout=1) as conn:
                after = conn.execute("SELECT generation,content_revision,funding_revision "
                                     "FROM market_data.instrument_generations "
                                     "WHERE symbol='RESTART'").fetchone()
            break
        except psycopg.OperationalError:
            time.sleep(0.25)
    else:
        pytest.fail('disposable PG did not recover')
    assert after == before
    bootstrap(db)


def test_typed_constraints_utc_primary_key_and_artifact_shape(db):
    candle = (
        "INSERT INTO market_data.canonical_candles_1m "
        "(market_id,symbol,instrument_key,ts_open,ts_close,open,high,low,close,"
        "volume_base,source,ingested_at) VALUES "
        "(1,'TYPED','binance:spot:TYPED','2000-01-01T03:00:00+03',"
        "'2000-01-01T00:00:59.999Z',1,2,0.5,1.5,10,'rest','2000-01-02T00:00:00.123Z')"
    )
    with psycopg.connect(db) as conn:
        conn.execute(cast(Any, candle))
        row = conn.execute("SELECT extract(epoch FROM ts_open),trades_count,ingest_id,"
                           "extract(milliseconds FROM ingested_at),tableoid::regclass::text "
                           "FROM market_data.canonical_candles_1m WHERE symbol='TYPED'").fetchone()
        assert row is not None and row[0] == 946684800 and row[1:3] == (None,None)
        assert row[3] == 123 and row[4].endswith('_200001')
        conn.execute("INSERT INTO public.backtest_artifact_source_state "
                     "(exchange,market_type,symbol,desired_target,source_generation,"
                     "content_revision,funding_revision,source_contract_version) "
                     "VALUES('binance','spot','TYPED','{}',gen_random_uuid(),8,3,2)")
        assert conn.execute("SELECT content_revision,funding_revision,source_contract_version "
                            "FROM public.backtest_artifact_source_state WHERE symbol='TYPED'"
                            ).fetchone() == (8,3,2)
        conn.execute("INSERT INTO market_data.raw_bybit_funding_rates "
                     "(market_id,symbol,instrument_key,funding_time,funding_rate,"
                     "funding_interval_minutes,funding_interval_source) "
                     "VALUES(4,'F','bybit:futures:F','2000-01-01',0.0001,240,'metadata')")
        assert conn.execute("SELECT category FROM market_data.raw_bybit_funding_rates "
                            "WHERE symbol='F'").fetchone() == ('linear',)
        conn.execute("INSERT INTO market_data.raw_binance_funding_rates "
                     "(market_id,symbol,instrument_key,funding_time,funding_rate,"
                     "funding_interval_minutes,funding_interval_source) "
                     "VALUES(2,'F','binance:futures:F','2000-01-01',0.0001,60,'metadata')")
        assert conn.execute("SELECT mark_price FROM market_data.raw_binance_funding_rates "
                            "WHERE symbol='F'").fetchone() == (None,)
    with psycopg.connect(db) as conn, pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(cast(Any,candle))
    for query in (
        "UPDATE market_data.canonical_candles_1m SET trades_count=4294967296 WHERE symbol='TYPED'",
        "UPDATE market_data.raw_bybit_funding_rates SET funding_interval_minutes=0",
        "UPDATE market_data.canonical_candles_1m SET ts_open=ts_open+interval '1 second'",
        "INSERT INTO market_data.ref_market(market_id,exchange_name,market_type,market_code) "
        "VALUES(1,'bybit','spot','bybit:spot')",
    ):
        with psycopg.connect(db) as conn, pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(cast(Any,query))


def test_status_requires_partitions_and_bootstrap_repairs_them(db):
    from datetime import UTC, datetime

    name = 'canonical_candles_1m_' + datetime.now(UTC).strftime('%Y%m')
    with psycopg.connect(db) as conn:
        conn.execute(sql.SQL('DROP TABLE market_data.{}').format(sql.Identifier(name)))
    with pytest.raises(StorageLifecycleError, match='required schema tables'):
        _postgres_schema_status(db, repo_root=ROOT, manifest_path=MANIFEST)
    bootstrap(db)
    assert 'market-data-storage-0036' in _postgres_schema_status(
        db, repo_root=ROOT, manifest_path=MANIFEST)
