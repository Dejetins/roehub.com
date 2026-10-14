"""S03 real PG16 transactional boundary; explicitly disposable target only."""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from apps.migrations.storage import apply_postgres_migrations
from trading.contexts.market_data.adapters.outbound.persistence.postgres.funding_writer import (
    PostgresFundingRateWriter,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_common import (
    payload_order,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    ConnectionBudget,
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.reference_writers import (
    PostgresFundingInstrumentStore,
    PostgresInstrumentRefWriter,
    PostgresMarketRefWriter,
)
from trading.contexts.market_data.application.dto import (
    CandleWithMeta,
    FundingInstrument,
    FundingRateRecord,
)
from trading.contexts.market_data.application.dto.reference_data import (
    InstrumentRefEnrichmentUpsert,
    InstrumentRefUpsert,
    RefMarketRow,
)
from trading.contexts.market_data.application.services.ingestion_retry import (
    TemporaryIngestionError,
)
from trading.contexts.market_data.application.services.stored_candle_signal import (
    StoredCandleSignalWriter,
)
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    UtcTimestamp,
)

DSN = os.environ.get("ROEHUB_S03_DISPOSABLE_DSN", "")
ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PG16 required")
BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def db():
    assert conninfo_to_dict(DSN)["host"] == "127.0.0.1"
    name = "s03_" + uuid4().hex
    with psycopg.connect(DSN, autocommit=True) as c:
        assert 160000 <= c.info.server_version < 170000
        c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    target = make_conninfo(DSN, dbname=name)
    apply_postgres_migrations(
        target, repo_root=ROOT, manifest_path=ROOT / "migrations/postgres/manifest.json"
    )
    with psycopg.connect(target) as c:
        c.execute("SELECT market_data.ensure_partitions(%s,1)", (date(2026, 2, 1),))
    yield target
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.fixture
def gateway(db):
    g = IngestionGateway(db)
    yield g
    g.close()


def candle(symbol="TEST", market=1, minute=0, close=11.0, version=0, source="rest"):
    instrument = InstrumentId(MarketId(market), Symbol(symbol))
    start = BASE + timedelta(minutes=minute)
    return CandleWithMeta(
        Candle(
            instrument,
            UtcTimestamp(start),
            UtcTimestamp(start + timedelta(minutes=1)),
            10.0,
            20.0,
            1.0,
            close,
            2.0,
            None,
        ),
        CandleMeta(
            source=source,
            ingested_at=UtcTimestamp(BASE + timedelta(days=1, milliseconds=version)),
            instrument_key=f"{market}:{symbol}",
            ingest_id=uuid4(),
            trades_count=None,
            taker_buy_volume_base=None,
            taker_buy_volume_quote=None,
        ),
    )


def snapshot(db, symbol, market=1):
    with psycopg.connect(db) as c:
        rows = c.execute(
            "SELECT ts_open,close,volume_quote,trades_count,ingested_at "
            "FROM market_data.canonical_candles_1m WHERE symbol=%s AND market_id=%s "
            "ORDER BY ts_open",
            (symbol, market),
        ).fetchall()
        summary = c.execute(
            "SELECT actual_count,first_ts,last_ts,content_revision,index_state "
            "FROM market_data.candle_coverage_summary JOIN "
            "market_data.instrument_generations USING (market_id,symbol,generation) "
            "WHERE symbol=%s AND market_id=%s",
            (symbol, market),
        ).fetchone()
        days = c.execute(
            "SELECT actual_count,absent_count FROM market_data.candle_coverage_days "
            "WHERE symbol=%s AND market_id=%s ORDER BY day",
            (symbol, market),
        ).fetchall()
        return rows, summary, days


def test_four_routes_duplicates_replay_and_nulls(db, gateway):
    w = PostgresRawKlineWriter(gateway)
    batch = [candle("ROUTE", market=m) for m in range(1, 5)]
    w.write_1m(batch * 2)
    w.write_1m(batch)
    for m in range(1, 5):
        rows, summary, days = snapshot(db, "ROUTE", m)
        assert len(rows) == 1 and rows[0][2] == 0
        assert rows[0][3] == (0 if m <= 2 else None)
        assert summary is not None
        assert summary[0] == summary[3] == 1 and summary[4] == "ready"
        assert days == [(1, 0)]
    with psycopg.connect(db) as c:
        assert (
            one(
                c.execute(
                    "SELECT count(*) FROM market_data.raw_binance_klines_1m " "WHERE symbol='ROUTE'"
                )
            )[0]
            == 6
        )


        assert c.execute("""SELECT DISTINCT quote_asset_volume,number_of_trades,
            taker_buy_base_asset_volume,taker_buy_quote_asset_volume
            FROM market_data.raw_binance_klines_1m WHERE symbol='ROUTE'""").fetchall() == [
                (0.0, 0, 0.0, 0.0)]
        assert c.execute("""SELECT DISTINCT interval_min,start_time_ms,turnover
            FROM market_data.raw_bybit_klines_1m WHERE symbol='ROUTE'""").fetchall() == [
                (1, int(BASE.timestamp()*1000), 0.0)]


def test_provider_raw_fields_preserve_values_and_routes(db, gateway):
    from dataclasses import replace

    batch = []
    for market in range(1, 5):
        row = candle("RAWFIELDS", market=market)
        batch.append(replace(row, candle=replace(row.candle, volume_quote=23.5),
            meta=replace(row.meta, trades_count=17, taker_buy_volume_base=1.25,
                         taker_buy_volume_quote=12.75)))
    PostgresRawKlineWriter(gateway).write_1m(batch)
    with psycopg.connect(db) as c:
        assert c.execute("""SELECT market_id,open_time,close_time,volume,quote_asset_volume,
            number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume
            FROM market_data.raw_binance_klines_1m WHERE symbol='RAWFIELDS'
            ORDER BY market_id""").fetchall() == [
                (m, BASE, BASE + timedelta(minutes=1), 2.0, 23.5, 17, 1.25, 12.75)
                for m in (1, 2)]
        assert c.execute("""SELECT market_id,interval_min,start_time_ms,start_time_utc,
            volume,turnover FROM market_data.raw_bybit_klines_1m WHERE symbol='RAWFIELDS'
            ORDER BY market_id""").fetchall() == [
                (m, 1, int(BASE.timestamp()*1000), BASE, 2.0, 23.5) for m in (3, 4)]

def test_equal_stale_metadata_and_lost_hint_correction(db, gateway):
    w = PostgresRawKlineWriter(gateway)
    a, b = candle("TIE", close=11.0), candle("TIE", close=12.0)
    w.write_1m([a, b, a])
    first = snapshot(db, "TIE")
    assert first[1] is not None
    w.write_1m([b, a])
    assert snapshot(db, "TIE") == first
    w.write_1m([candle("TIE", close=1.0, version=-1)])
    assert snapshot(db, "TIE") == first
    w.write_1m([candle("TIE", close=first[0][0][1], version=1, source="ws")])
    assert snapshot(db, "TIE")[1] == first[1]

    def lost_hint(_):
        raise RuntimeError("simulated post-commit hint crash")

    StoredCandleSignalWriter(w, lost_hint).write_1m([candle("TIE", close=13.0, version=2)])
    corrected = snapshot(db, "TIE")
    assert corrected[1] is not None
    assert corrected[1][:3] == first[1][:3]
    assert corrected[1][3] == first[1][3] + 1
    w.write_1m([candle("TIE", close=13.0, version=2)])
    assert snapshot(db, "TIE") == corrected


def test_concurrent_ws_rest_sorted_keys(db):
    def write(reverse):
        g = IngestionGateway(db)
        try:
            rows = [
                candle(s, minute=i, version=i, source="ws" if reverse else "rest")
                for s in ("RACEA", "RACEB")
                for i in range(60)
            ]
            PostgresRawKlineWriter(g).write_1m(rows[::-1] if reverse else rows)
        finally:
            g.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, [False, True, True, False]))
    for s in ("RACEA", "RACEB"):
        rows, summary, days = snapshot(db, s)
        assert summary is not None
        assert len(rows) == summary[0] == 60 and days == [(60, 0)]
        assert summary[3] == 1


def test_rollback_raw_canonical_and_unknown_commit_replay(db, gateway):
    w = PostgresRawKlineWriter(gateway)
    original = w._write

    def fail(conn, batch):
        original(conn, batch)
        raise ValueError("injected before commit")

    w._write = fail
    with pytest.raises(ValueError):
        w.write_1m([candle("ROLLBACK")])
    assert snapshot(db, "ROLLBACK") == ([], None, [])
    with psycopg.connect(db) as c:
        assert (
            one(
                c.execute(
                    "SELECT count(*) FROM market_data.raw_binance_klines_1m "
                    "WHERE symbol='ROLLBACK'"
                )
            )[0]
            == 0
        )
    w._write = original
    # Real durable commit followed by transport receipt loss; caller replays exact batch.
    batch = [candle("UNKNOWN")]
    w.write_1m(batch)
    before = snapshot(db, "UNKNOWN")
    gateway.close()
    g = IngestionGateway(db)
    try:
        PostgresRawKlineWriter(g).write_1m(batch)
    finally:
        g.close()
    assert snapshot(db, "UNKNOWN") == before


def test_late_actual_clears_absence_and_generation_fence(db, gateway):
    w = PostgresRawKlineWriter(gateway)
    w.write_1m([candle("ABSENT")])
    with psycopg.connect(db) as c:
        c.execute(
            "UPDATE market_data.candle_coverage_days SET source_absent="
            "set_bit(source_absent,1,1) WHERE symbol='ABSENT'"
        )
        gen = one(
            c.execute(
                "SELECT generation FROM market_data.instrument_generations " "WHERE symbol='ABSENT'"
            )
        )[0]
        c.execute(
            "INSERT INTO market_data.source_absent_candles_1m VALUES "
            "(%s,%s,%s,%s,%s,1,%s,%s,%s,%s)",
            (
                1,
                "ABSENT",
                int(BASE.timestamp()) // 60 + 1,
                gen,
                BASE,
                uuid4(),
                "rest",
                BASE,
                BASE + timedelta(minutes=2),
            ),
        )
    w.write_1m([candle("ABSENT", minute=1)])
    assert snapshot(db, "ABSENT")[2] == [(2, 0)]
    with psycopg.connect(db) as c:
        assert (
            one(
                c.execute(
                    "SELECT count(*) FROM market_data.source_absent_candles_1m "
                    "WHERE symbol='ABSENT'"
                )
            )[0]
            == 0
        )
        c.execute(
            "SELECT market_data.invalidate_instrument(1,%s,%s,%s)", ("ABSENT", gen, "test rebuild")
        )
    with pytest.raises(RuntimeError, match="requires bounded"):
        w.write_1m([candle("ABSENT", minute=2)])
    assert len(snapshot(db, "ABSENT")[0]) == 2


def test_bounds_pool_global_slots_and_session_reset(db, gateway):
    w = PostgresRawKlineWriter(gateway)
    with pytest.raises(ValueError, match="batch exceeds"):
        w.write_1m(candle("BOUND") for _ in range(8193))
    assert snapshot(db, "BOUND")[0] == []
    assert payload_order([-0.0, float("nan")]) == payload_order([0.0, float("nan")])
    with pytest.raises(ValueError):
        ConnectionBudget(total=47)
    pools = [IngestionGateway(db) for _ in range(4)]
    try:
        for g in pools:
            g.run(lambda c: c.execute("SELECT 1"))
        with pytest.raises(TemporaryIngestionError, match="saturated"):
            gateway.run(lambda c: c.execute("SELECT 1"))
    finally:
        for g in pools:
            g.close()
    gateway.run(lambda c: c.execute("SET LOCAL work_mem='1MB'"))
    assert gateway.run(lambda c: one(c.execute("SHOW work_mem"))[0]) == "4MB"


def test_funding_revision_nulls_versions(db, gateway):
    writer = PostgresFundingRateWriter(gateway)

    def row(m, version=0, rate=0.01):
        return FundingRateRecord(
            InstrumentId(MarketId(m), Symbol("FUND")),
            f"{m}:FUND",
            UtcTimestamp(BASE),
            rate,
            240,
            "provider",
            "rest",
            UtcTimestamp(BASE + timedelta(milliseconds=version)),
            str(uuid4()),
            None,
            None,
        )

    batch = [row(2), row(4)]
    writer.write_funding_rates(batch * 2)
    writer.write_funding_rates(batch)
    with psycopg.connect(db) as c:
        assert c.execute(
            "SELECT funding_revision FROM market_data.instrument_generations "
            "WHERE symbol='FUND' ORDER BY market_id"
        ).fetchall() == [(1,), (1,)]
        assert (
            one(
                c.execute(
                    "SELECT category FROM market_data.raw_bybit_funding_rates "
                    "WHERE symbol='FUND'"
                )
            )[0]
            == "linear"
        )
    writer.write_funding_rates([row(2, 1, 0.02)])
    writer.write_funding_rates([row(2, 0, 0.03)])
    with psycopg.connect(db) as c:
        assert (
            one(
                c.execute(
                    "SELECT funding_rate FROM market_data.canonical_funding_rates "
                    "WHERE symbol='FUND' AND market_id=2"
                )
            )[0]
            == 0.02
        )
        assert (
            one(
                c.execute(
                    "SELECT funding_revision FROM market_data.instrument_generations "
                    "WHERE symbol='FUND' AND market_id=2"
                )
            )[0]
            == 2
        )


def test_reference_partial_null_and_universe(db, gateway):
    market = PostgresMarketRefWriter(gateway)
    market.insert(
        [RefMarketRow(MarketId(1), "binance", "spot", "binance:spot", 1, 2, UtcTimestamp(BASE))]
    )
    assert market.existing_market_ids([MarketId(1), MarketId(2)]) == {1}
    ref = PostgresInstrumentRefWriter(gateway)
    enriched = InstrumentRefEnrichmentUpsert(
        MarketId(1), Symbol("REF"), "ENABLED", 1, "BTC", "USDT", 0.1, 0.1, 1.0, UtcTimestamp(BASE)
    )
    ref.upsert_enrichment([enriched])
    ref.upsert(
        [
            InstrumentRefUpsert(
                MarketId(1), Symbol("REF"), "DISABLED", 0, UtcTimestamp(BASE + timedelta(seconds=1))
            )
        ]
    )
    value = ref.existing_latest_enrichment(market_id=MarketId(1), symbols=[Symbol("REF")])["REF"]
    assert value.base_asset == "BTC" and value.is_tradable == 0
    ref.upsert_enrichment(
        [replace(enriched, base_asset=None, updated_at=UtcTimestamp(BASE + timedelta(seconds=2)))]
    )
    assert (
        ref.existing_latest_enrichment(market_id=MarketId(1), symbols=[Symbol("REF")])[
            "REF"
        ].base_asset
        is None
    )
    store = PostgresFundingInstrumentStore(gateway)
    r = FundingInstrument(
        InstrumentId(MarketId(2), Symbol("UNIVERSE")),
        "2:UNIVERSE",
        "binance",
        "futures",
        "TRADING",
        1,
        "BTC",
        "USDT",
        480,
        "provider",
        0.1,
        -0.1,
        UtcTimestamp(BASE),
    )
    store.upsert_funding_instruments([r])
    empty = replace(
        r,
        base_asset=None,
        funding_interval_minutes=None,
        updated_at=UtcTimestamp(BASE + timedelta(seconds=1)),
    )
    store.upsert_funding_instruments([empty, empty])
    assert store.get_funding_instrument(r.instrument_id) == empty


def test_real_disconnect_retry_and_permission_fail_fast(db, gateway):
    attempts = []

    def operation(c):
        attempts.append(1)
        if len(attempts) == 1:
            with psycopg.connect(db, autocommit=True) as other:
                other.execute("SELECT pg_terminate_backend(%s)", (c.info.backend_pid,))
        return one(c.execute("SELECT 42"))[0]

    assert gateway.run(operation) == 42 and len(attempts) == 2
    attempts.clear()

    def missing(c):
        attempts.append(1)
        c.execute("SELECT * FROM market_data.nonexistent_table")

    with pytest.raises(psycopg.errors.UndefinedTable):
        gateway.run(missing)
    assert len(attempts) == 1


def test_finite_waiters_and_deadline(db):
    entered, release = Event(), Event()
    g = IngestionGateway(db, max_waiters=0)

    def hold(c):
        entered.set()
        assert release.wait(2)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(g.run, hold)
            assert entered.wait(2)
            with pytest.raises(TemporaryIngestionError, match="admission"):
                g.run(lambda c: None)
            release.set()
            future.result()
    finally:
        g.close()
    g = IngestionGateway(db, deadline_s=0.2)
    try:
        with pytest.raises((TemporaryIngestionError, psycopg.errors.QueryCanceled)):
            g.run(lambda c: c.execute("SELECT pg_sleep(2)"))
    finally:
        g.close()


def test_commit_receipt_publication_and_persistent_process(db, gateway):
    import asyncio

    from apps.worker.market_data_ws.wiring.persistent_writer import (
        PersistentRawWriterProcess,
        postgres_writer_child,
    )
    from trading.contexts.market_data.application.services.stored_candle_signal import (
        write_then_publish,
    )

    async def scenario():
        process = PersistentRawWriterProcess(
            environ={"STRATEGY_PG_DSN": db}, child_target=postgres_writer_child
        )
        calls = []

        def publish(row):
            # Independent connection sees canonical + index at publication time.
            assert len(snapshot(db, "PROCESS")[0]) == 1
            calls.append(row)

        try:
            await process.start()
            await write_then_publish([candle("PROCESS")], process.write, publish)
            assert len(calls) == 1
            with pytest.raises(RuntimeError):
                await write_then_publish(
                    [candle("PROCESS", minute=32 * 1440)], process.write, publish
                )
            assert len(calls) == 1
        finally:
            await process.close()
        assert not process.active_pids

    asyncio.run(scenario())


def test_unknown_commit_inside_retry_and_rollback_disconnect(db, gateway, monkeypatch):
    from contextlib import contextmanager

    writer = PostgresRawKlineWriter(gateway)
    real_connect = gateway._connect
    calls = []

    class LostReceipt:
        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, name):
            return getattr(self.conn, name)

        @contextmanager
        def transaction(self):
            with self.conn.transaction():
                yield
            calls.append("committed")
            self.conn.close()
            raise psycopg.OperationalError("injected lost commit response")

    def connect(stop):
        conn = real_connect(stop)
        return conn if calls else LostReceipt(conn)

    monkeypatch.setattr(gateway, "_connect", connect)
    writer.write_1m([candle("RETRYCOMMIT")])
    assert calls == ["committed"]
    assert summary_of(db, "RETRYCOMMIT")[3] == 1
    with psycopg.connect(db) as c:
        assert (
            one(
                c.execute(
                    "SELECT count(*) FROM market_data.raw_binance_klines_1m "
                    "WHERE symbol='RETRYCOMMIT'"
                )
            )[0]
            == 2
        )
    original = writer._write
    attempts = []

    def killed(conn, batch):
        original(conn, batch)
        attempts.append(1)
        if len(attempts) == 1:
            with psycopg.connect(db, autocommit=True) as other:
                other.execute("SELECT pg_terminate_backend(%s)", (conn.info.backend_pid,))
            conn.execute("SELECT 1")

    writer._write = killed
    writer.write_1m([candle("KILLED")])
    assert len(attempts) == 2 and summary_of(db, "KILLED")[3] == 1
    with psycopg.connect(db) as c:
        assert (
            one(
                c.execute(
                    "SELECT count(*) FROM market_data.raw_binance_klines_1m "
                    "WHERE symbol='KILLED'"
                )
            )[0]
            == 1
        )


def test_locks_generation_race_retry_cap_and_non_superuser(db, gateway):
    from apps.migrations.market_data import provision_market_data_access

    writer = PostgresRawKlineWriter(gateway)
    writer.write_1m([candle("LOCKED")])
    with psycopg.connect(db) as lock:
        lock.execute(
            "SELECT * FROM market_data.instrument_generations WHERE symbol='LOCKED' " "FOR UPDATE"
        )
        with pytest.raises(TemporaryIngestionError, match="retry_exhausted"):
            writer.write_1m([candle("LOCKED", minute=1)])
    assert len(snapshot(db, "LOCKED")[0]) == 1
    role, maintenance = "s03_app_" + uuid4().hex, "s03_maint_" + uuid4().hex
    role_password = uuid4().hex
    with psycopg.connect(db) as c:
        c.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
            sql.Identifier(role), sql.Literal(role_password)))
        c.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(maintenance)))
        provision_market_data_access(c, application_role=role, maintenance_role=maintenance)
    g = IngestionGateway(make_conninfo(db, user=role, password=role_password))
    try:
        PostgresRawKlineWriter(g).write_1m([candle("UNPRIVILEGED", market=4)])
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            g.run(lambda c: c.execute("SELECT market_data.ensure_partitions('2000-01-01',1)"))
    finally:
        g.close()
    assert summary_of(db, "UNPRIVILEGED", 4)[0] == 1
    with psycopg.connect(db) as c:
        c.execute(
            sql.SQL("DROP OWNED BY {}, {}").format(
                sql.Identifier(role), sql.Identifier(maintenance)
            )
        )
        c.execute(
            sql.SQL("DROP ROLE {}, {}").format(sql.Identifier(role), sql.Identifier(maintenance))
        )


def test_recovery_cursor_not_completed_before_durable_write(db, gateway):
    import asyncio

    from trading.contexts.market_data.adapters.outbound.persistence.postgres.stream_recovery_store import (  # noqa: E501
        PostgresStreamRecoveryStore,
    )
    from trading.contexts.market_data.application.services.insert_buffer import AsyncRawInsertBuffer

    class Clock:
        def now(self):
            return UtcTimestamp(BASE)

    writer = PostgresRawKlineWriter(gateway)
    store = PostgresStreamRecoveryStore(db)

    async def scenario():
        failed = True

        async def write(rows):
            assert len(store.pending()) == 1
            if failed:
                raise TemporaryIngestionError("injected failure before commit")
            await asyncio.to_thread(writer.write_1m, rows)
            assert len(snapshot(db, "CURSOR")[0]) == 1
            assert len(store.pending()) == 1

        buffer = AsyncRawInsertBuffer(
            writer=writer,
            clock=Clock(),
            flush_interval_ms=100,
            max_buffer_rows=10,
            store=store,
            async_writer=write,
        )
        buffer.submit(candle("CURSOR"))
        await buffer.flush(force=True)
        assert len(store.pending()) == 1 and snapshot(db, "CURSOR")[0] == []
        failed = False
        await buffer.flush(force=True)
        assert store.pending() == []
        await buffer.close()

    asyncio.run(scenario())


def test_async_recovery_session_consumes_reserved_slot(db):
    import asyncio

    from trading.contexts.market_data.adapters.outbound.persistence.postgres.async_stream_recovery_store import (  # noqa: E501
        AsyncPostgresStreamRecoveryStore,
    )

    async def scenario():
        async with AsyncPostgresStreamRecoveryStore(db).lease():
            contender = IngestionGateway(db, workload="recovery")
            try:
                with pytest.raises(TemporaryIngestionError, match="saturated"):
                    await asyncio.to_thread(contender.run, lambda c: c.execute("SELECT 1"))
            finally:
                contender.close()

    asyncio.run(scenario())


def test_fixed_payload_order_and_limits(gateway):
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_common import (  # noqa: E501
        bounded,
    )

    assert payload_order([None, 1, "A", 0.0]).hex() == (
        "01000000016e0000000969000000000000000100000002734100000009660000000000000000"
    )
    with pytest.raises(ValueError, match="8 MiB"):
        bounded([{"x": "x" * (3 * 1024 * 1024)}], lambda r: r)
    ref = PostgresMarketRefWriter(gateway)
    row = RefMarketRow(MarketId(1), "binance", "spot", "binance:spot", 1, 1, UtcTimestamp(BASE))
    with pytest.raises(ValueError, match="4 rows"):
        ref.insert([row] * 5)


def test_generation_invalidation_race(db, gateway):
    writer = PostgresRawKlineWriter(gateway)
    writer.write_1m([candle("GENRACE")])
    entered, release = Event(), Event()

    def invalidate():
        with psycopg.connect(db) as c:
            gen = one(
                c.execute(
                    "SELECT generation FROM market_data.instrument_generations "
                    "WHERE symbol='GENRACE' FOR UPDATE"
                )
            )[0]
            c.execute(
                "SELECT market_data.invalidate_instrument(1,%s,%s,%s)",
                ("GENRACE", gen, "race rebuild"),
            )
            entered.set()
            assert release.wait(2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(invalidate)
        assert entered.wait(2)
        write = pool.submit(writer.write_1m, [candle("GENRACE", minute=1)])
        release.set()
        future.result()
        with pytest.raises(RuntimeError, match="requires bounded"):
            write.result()
    assert len(snapshot(db, "GENRACE")[0]) == 1


def one(cursor):
    row = cursor.fetchone()
    assert row is not None
    return row


def summary_of(db, symbol, market=1):
    result = snapshot(db, symbol, market)[1]
    assert result is not None
    return result


def test_saturated_market_slots_preserve_declared_control_headroom(db):
    from contextlib import ExitStack

    from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (  # noqa: E501
        CLASS_SLOTS,
    )

    pools = []
    try:
        for workload, slots in CLASS_SLOTS.items():
            for _ in slots:
                g = IngestionGateway(db, workload=workload)
                pools.append(g)
                g.run(lambda c: c.execute("SELECT 1"))
        assert len(pools) == 16
        with ExitStack() as stack:
            control = [
                stack.enter_context(
                    psycopg.connect(db, autocommit=True, application_name="s03-control")
                )
                for _ in range(24)
            ]
            for conn in control:
                assert one(conn.execute("SELECT 1")) == (1,)
            for workload in CLASS_SLOTS:
                excess = IngestionGateway(db, workload=workload)
                try:
                    with pytest.raises(TemporaryIngestionError, match="saturated"):
                        excess.run(lambda c: c.execute("SELECT 1"))
                finally:
                    excess.close()
            assert (
                one(
                    control[0].execute(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE application_name LIKE 'roehub-market-%%'"
                    )
                )[0]
                == 16
            )
            assert (
                one(
                    control[0].execute(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE application_name='s03-control'"
                    )
                )[0]
                == 24
            )
            assert one(control[0].execute("SHOW max_connections"))[0] == "48"
            with psycopg.connect(db, autocommit=True) as admin:
                assert one(admin.execute("SELECT 1")) == (1,)
    finally:
        for g in pools:
            g.close()


def test_additive_migration_repeat_and_readiness(db):
    from apps.migrations.storage import _postgres_schema_status

    apply_postgres_migrations(
        db, repo_root=ROOT, manifest_path=ROOT / "migrations/postgres/manifest.json"
    )
    versions = _postgres_schema_status(
        db, repo_root=ROOT, manifest_path=ROOT / "migrations/postgres/manifest.json"
    )
    assert "market-data-reference-0037" in versions
    with psycopg.connect(db) as c:
        assert (
            one(
                c.execute(
                    "SELECT count(*) FROM roehub_storage_migrations "
                    "WHERE version='market-data-reference-0037'"
                )
            )[0]
            == 1
        )
        assert (
            len(
                c.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='market_data' AND table_name='ref_instruments' "
                    "AND column_name IN ('enrichment_updated_at','enrichment_order')"
                ).fetchall()
            )
            == 2
        )


def test_connect_handshake_absolute_deadline_without_background_thread():
    import socket
    import threading
    import time

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    release = Event()

    def blackhole():
        conn, _ = listener.accept()
        try:
            release.wait(2)
        finally:
            conn.close()

    worker = threading.Thread(target=blackhole)
    worker.start()
    g = IngestionGateway(
        make_conninfo(
            host="127.0.0.1",
            port=listener.getsockname()[1],
            user="fixture",
            dbname="fixture",
            sslmode="disable",
        ),
        deadline_s=0.2,
    )
    started = time.monotonic()
    try:
        with pytest.raises(TemporaryIngestionError, match="connect_timeout"):
            g.run(lambda c: c.execute("SELECT 1"))
        assert time.monotonic() - started < 1
    finally:
        g.close()
        release.set()
        listener.close()
        worker.join(2)
    assert not worker.is_alive()
    unresolved = IngestionGateway("host=unresolved.invalid dbname=fixture", deadline_s=0.2)
    try:
        with pytest.raises(TemporaryIngestionError, match="postgres_dns_"):
            unresolved.run(lambda c: c.execute("SELECT 1"))
    finally:
        unresolved.close()


def test_reference_equal_version_permutations_all_null_and_independent_clocks(gateway):
    ref = PostgresInstrumentRefWriter(gateway)
    universe = PostgresFundingInstrumentStore(gateway)
    outputs = []
    for reverse, symbol in ((False, "REFPERMA"), (True, "REFPERMB")):
        a = InstrumentRefEnrichmentUpsert(
            MarketId(1),
            Symbol(symbol),
            "ENABLED",
            1,
            "BTC",
            "USDT",
            0.1,
            0.1,
            1.0,
            UtcTimestamp(BASE),
        )
        b = replace(
            a, base_asset=None, quote_asset=None, price_step=None, qty_step=None, min_notional=None
        )
        for row in [b, a] if reverse else [a, b]:
            ref.upsert_enrichment([row])
        outputs.append(
            ref.existing_latest_enrichment(market_id=MarketId(1), symbols=[Symbol(symbol)])[symbol]
        )
        # A newer status clock must not suppress an independently newer enrichment.
        ref.upsert(
            [
                InstrumentRefUpsert(
                    MarketId(1),
                    Symbol(symbol),
                    "DISABLED",
                    0,
                    UtcTimestamp(BASE + timedelta(seconds=2)),
                )
            ]
        )
        ref.upsert_enrichment(
            [replace(a, base_asset="ETH", updated_at=UtcTimestamp(BASE + timedelta(seconds=1)))]
        )
        value = ref.existing_latest_enrichment(market_id=MarketId(1), symbols=[Symbol(symbol)])[
            symbol
        ]
        assert value.is_tradable == 0 and value.base_asset == "ETH"
        ref.upsert_enrichment([a])
        assert (
            ref.existing_latest_enrichment(market_id=MarketId(1), symbols=[Symbol(symbol)])[symbol]
            == value
        )
    assert outputs[0] == outputs[1]
    values = []
    for reverse, symbol in ((False, "UNIPERMA"), (True, "UNIPERMB")):
        a = FundingInstrument(
            InstrumentId(MarketId(2), Symbol(symbol)),
            f"2:{symbol}",
            "binance",
            "futures",
            "TRADING",
            1,
            None,
            None,
            None,
            None,
            None,
            None,
            UtcTimestamp(BASE),
        )
        b = replace(a, base_asset="BTC", funding_interval_minutes=480, funding_cap=0.1)
        for row in [b, a] if reverse else [a, b]:
            universe.upsert_funding_instruments([row])
        result = universe.get_funding_instrument(a.instrument_id)
        assert result is not None
        values.append((result.base_asset, result.funding_interval_minutes, result.funding_cap))
        # Newer all-NULL record is authoritative, including all optional columns.
        null = replace(a, updated_at=UtcTimestamp(BASE + timedelta(seconds=1)))
        universe.upsert_funding_instruments([null])
        assert universe.get_funding_instrument(a.instrument_id) == null
    assert values[0] == values[1]


def test_funding_provider_only_ties_share_raw_canonical_order(db, gateway):
    writer = PostgresFundingRateWriter(gateway)
    for market, table, field, left, right in (
        (2, "raw_binance_funding_rates", "mark_price", None, 123.0),
        (4, "raw_bybit_funding_rates", "category", "linear", "inverse"),
    ):
        outcomes = []
        for reverse, symbol in ((False, "FUNDPERMA"), (True, "FUNDPERMB")):
            base = FundingRateRecord(
                InstrumentId(MarketId(market), Symbol(symbol)),
                f"{market}:{symbol}",
                UtcTimestamp(BASE),
                0.01,
                480,
                "provider",
                "rest",
                UtcTimestamp(BASE),
            )
            key = "mark_price" if market == 2 else "bybit_category"
            a, b = replace(base, **{key: left}), replace(base, **{key: right})
            for row in [b, a] if reverse else [a, b]:
                writer.write_funding_rates([row])
            writer.write_funding_rates([a, b, a])
            with psycopg.connect(db) as c:
                raw = one(
                    c.execute(
                        sql.SQL(
                            "SELECT {},payload_order FROM market_data.{} "
                            "WHERE market_id=%s AND symbol=%s"
                        ).format(sql.Identifier(field), sql.Identifier(table)),
                        (market, symbol),
                    )
                )
                canonical = one(
                    c.execute(
                        "SELECT payload_order FROM market_data.canonical_funding_rates "
                        "WHERE market_id=%s AND symbol=%s",
                        (market, symbol),
                    )
                )
                assert raw[1] == canonical[0]
                assert (
                    one(
                        c.execute(
                            "SELECT funding_revision FROM market_data.instrument_generations "
                            "WHERE market_id=%s AND symbol=%s",
                            (market, symbol),
                        )
                    )[0]
                    == 1
                )
                outcomes.append(raw[0])
        assert outcomes[0] == outcomes[1]


def test_disposable_server_restart_rolls_back_inflight_and_preserves_revision(db, gateway):
    import subprocess
    import time

    from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (  # noqa: E501
        _canonical,
    )

    container = os.environ.get("ROEHUB_S03_DISPOSABLE_CONTAINER")
    if not container:
        pytest.skip("explicit disposable S03 container identity required")
    label = subprocess.check_output(
        [
            "docker",
            "inspect",
            "--format",
            '{{index .Config.Labels "roehub.disposable"}}',
            container,
        ],
        text=True,
    ).strip()
    assert label == "S03" and container.startswith("roehub-s03-")
    writer = PostgresRawKlineWriter(gateway)
    writer.write_1m([candle("RESTART")])
    baseline = snapshot(db, "RESTART")
    conn = psycopg.connect(db)
    try:
        writer._write(conn, [_canonical(candle("UNCOMMITTED"))])
        assert snapshot(db, "UNCOMMITTED")[0] == []
        subprocess.run(
            ["docker", "restart", "-t", "1", container], check=True, capture_output=True, timeout=20
        )
        with pytest.raises(psycopg.OperationalError):
            conn.commit()
    finally:
        conn.close()
    after = None
    for attempt in range(30):
        try:
            after = snapshot(db, "RESTART")
            break
        except psycopg.OperationalError:
            if attempt == 29:
                raise
            time.sleep(0.1)
    assert after == baseline
    assert snapshot(db, "UNCOMMITTED") == ([], None, [])
    writer.write_1m([candle("RESTART", minute=1)])
    assert summary_of(db, "RESTART")[0] == 2


def test_batch_partition_bounds_preserve_old_corrections_and_replay(db, gateway):
    """Batch pruning must retain both sides of rollover and historical corrections."""
    with psycopg.connect(db) as c:
        c.execute("SELECT market_data.ensure_partitions(%s,1)", (date(2026, 1, 1),))
    writer = PostgresRawKlineWriter(gateway)
    old = candle("S09BOUNDS", minute=-1, close=10.0, version=0)
    new = candle("S09BOUNDS", minute=0, close=11.0, version=0)
    writer.write_1m([new, old, new])
    corrected = candle("S09BOUNDS", minute=-1, close=12.0, version=1)
    writer.write_1m([corrected, new, old])
    before = snapshot(db, "S09BOUNDS")
    assert [row[1] for row in before[0]] == [12.0, 11.0]
    assert before[1] is not None and before[1][0] == 2
    writer.write_1m([old, new, corrected, corrected])
    assert snapshot(db, "S09BOUNDS") == before


def test_reused_candle_stages_are_empty_after_commit_and_rollback(db, gateway):
    """Stable stage identities must never replay rows from an earlier transaction."""
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (  # noqa: E501
        _canonical,
    )

    with psycopg.connect(db, autocommit=True) as conn:
        writer = PostgresRawKlineWriter(gateway)
        with pytest.raises(RuntimeError, match="abort first creation"):
            with conn.transaction():
                writer._write(conn, [_canonical(candle("S09FIRSTABORTED"))])
                raise RuntimeError("abort first creation")
        assert one(conn.execute("SELECT to_regclass('pg_temp.candle_stage')"))[0] is None
        with conn.transaction():
            writer._write(conn, [_canonical(candle("S09STAGES"))])
        names = ("raw_binance_klines_1m_stage", "candle_stage", "candle_winners")
        initial = conn.execute(
            "SELECT relname,oid FROM pg_class WHERE relnamespace=pg_my_temp_schema() "
            "AND relname=ANY(%s) ORDER BY relname", (list(names),)
        ).fetchall()
        assert len(initial) == 3
        with pytest.raises(RuntimeError, match="abort staged write"):
            with conn.transaction():
                writer._write(conn, [_canonical(candle("S09ABORTED", minute=1))])
                raise RuntimeError("abort staged write")
        for name in names:
            assert one(conn.execute(sql.SQL("SELECT count(*) FROM {}").format(
                sql.Identifier(name))))[0] == 0
        with conn.transaction():
            writer._write(conn, [_canonical(candle("S09STAGES", minute=2))])
        assert conn.execute(
            "SELECT relname,oid FROM pg_class WHERE relnamespace=pg_my_temp_schema() "
            "AND relname=ANY(%s) ORDER BY relname", (list(names),)
        ).fetchall() == initial
    assert len(snapshot(db, "S09STAGES")[0]) == 2
    assert snapshot(db, "S09ABORTED")[0] == []
    assert snapshot(db, "S09FIRSTABORTED")[0] == []
    with psycopg.connect(db) as conn:
        assert one(conn.execute(
            "SELECT count(*) FROM market_data.raw_binance_klines_1m "
            "WHERE symbol='S09STAGES'"))[0] == 2


def test_candle_writer_catalog_memory_plateaus_after_warmup(gateway):
    """Repeated writes must not retain new catalog entries for each temp table OID."""
    writer = PostgresRawKlineWriter(gateway)

    def cache_used():
        return gateway.run(lambda conn: one(conn.execute(
            "SELECT used_bytes FROM pg_backend_memory_contexts "
            "WHERE name='CacheMemoryContext'"))[0])

    for i in range(100):
        writer.write_1m([candle("S09CACHE", minute=i)])
    warmed = cache_used()
    for i in range(100, 600):
        writer.write_1m([candle("S09CACHE", minute=i)])
    assert cache_used() - warmed < 256 * 1024
