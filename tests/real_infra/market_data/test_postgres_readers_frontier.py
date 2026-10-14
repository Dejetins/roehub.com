"""S04 PG16 semantics, bounded verification and revision-based reconciliation."""

import os
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from apps.migrations.storage import apply_postgres_migrations
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.stored_coordinates import (  # noqa: E501
    PostgresStoredArtifactCoordinates,
)
from trading.contexts.backtest_artifacts.application.services.base_source_reconciliation import (
    BaseSourceReconciler,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_index_reader import (  # noqa: E501
    PostgresCanonicalCandleIndexReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_reader import (  # noqa: E501
    PostgresCanonicalCandleReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.catalog_coverage_reader import (  # noqa: E501
    PostgresCatalogCoverageReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.coverage_maintenance import (  # noqa: E501
    PostgresCoverageMaintenance,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.funding_rate_store import (
    PostgresFundingRateStore,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_coverage_reader import (  # noqa: E501
    PostgresHistoryCoverageReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_range_verifier import (  # noqa: E501
    PostgresHistoryRangeVerifier,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (
    IngestionGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.instrument_coverage_reader import (  # noqa: E501
    PostgresInstrumentCoverageReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.query_support import (
    CoverageUnavailable,
    minute,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.reference_readers import (
    PostgresBTCUSDTMarketReadinessReader,
    PostgresEnabledInstrumentReader,
    PostgresEnabledMarketReader,
    PostgresEnabledTradableInstrumentSearchReader,
)
from trading.contexts.market_data.application.dto import CandleWithMeta, FundingRateRecord
from trading.contexts.market_data.application.services.source_retry import IncompleteHistoryError
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)

DSN = os.environ.get("ROEHUB_S04_DISPOSABLE_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PG16 required")
BASE = datetime(2026, 2, 1, tzinfo=UTC)


def one(value) -> Any:
    assert value is not None
    return value


def instrument(symbol):
    return InstrumentId(MarketId(1), Symbol(symbol))


def window(a: float = 0, b: float = 10, base=BASE):
    return TimeRange(
        UtcTimestamp(base + timedelta(minutes=a)), UtcTimestamp(base + timedelta(minutes=b))
    )


def candle(symbol, m=0, close=11.0, version=0, base=BASE):
    start = base + timedelta(minutes=m)
    return CandleWithMeta(
        Candle(
            instrument(symbol),
            UtcTimestamp(start),
            UtcTimestamp(start + timedelta(minutes=1)),
            10.0,
            20.0,
            1.0,
            close,
            2.0,
            3.0,
        ),
        CandleMeta(
            source="rest",
            ingested_at=UtcTimestamp(BASE + timedelta(days=2, seconds=version)),
            instrument_key="binance:spot:" + symbol,
            ingest_id=uuid4(),
            trades_count=None,
            taker_buy_volume_base=None,
            taker_buy_volume_quote=None,
        ),
    )


@pytest.fixture(scope="module")
def db():
    name = "s04_" + uuid4().hex
    with psycopg.connect(DSN, autocommit=True) as c:
        assert c.info.host == "127.0.0.1" and 160000 <= c.info.server_version < 170000
        c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    target = make_conninfo(DSN, dbname=name)
    root = Path(__file__).resolve().parents[3]
    apply_postgres_migrations(
        target, repo_root=root, manifest_path=root / "migrations/postgres/manifest.json"
    )
    with psycopg.connect(target) as c:
        for month in (date(1970, 1, 1), date(2026, 2, 1), date(2026, 3, 1)):
            c.execute("SELECT market_data.ensure_partitions(%s,1)", (month,))
    yield target
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.fixture
def gateway(db):
    g = IngestionGateway(db, workload="bulk", deadline_s=30)
    yield g
    g.close()


@pytest.mark.parametrize(
    "cutoff,expected", [(0, None), (0.001, 0), (0.5, 0), (1, 0), (1.1, 1), (2, 1), (3, 2)]
)
def test_exact_before_epoch_zero(gateway, cutoff, expected):
    base = datetime(1970, 1, 1, tzinfo=UTC)
    PostgresRawKlineWriter(gateway).write_1m([candle("EPOCH", m, base=base) for m in (0, 1, 2)])
    before = UtcTimestamp(base + timedelta(minutes=cutoff))
    index = PostgresCanonicalCandleIndexReader(gateway)
    first, last = index.bounds_1m(instrument_id=instrument("EPOCH"), before=before)
    obs = PostgresStoredArtifactCoordinates(gateway).observation(
        instrument=instrument("EPOCH"), before=before
    )
    assert last == (None if expected is None else UtcTimestamp(base + timedelta(minutes=expected)))
    assert (first, last) == (obs.first, obs.last)
    assert obs.actual_count == (0 if expected is None else expected + 1)


def test_stream_dto_arrays_and_index(gateway):
    items = [candle("READ", m) for m in (0, 1, 5, 1440, 1441)]
    PostgresRawKlineWriter(gateway).write_1m(items)
    r = PostgresCanonicalCandleReader(gateway)
    actual = list(r.read_1m(instrument("READ"), window(0, 1442)))
    assert [x.candle for x in actual] == [x.candle for x in items]
    arrays = r.read_1m_arrays(instrument("READ"), window(0, 1442))
    assert arrays.row_count() == 5 and isinstance(arrays.ohlcv_f32, np.memmap)
    assert arrays.ohlcv_f32.tolist() == [[10, 20, 1, 11, 2]] * 5
    assert arrays.ohlcv_f32.dtype == np.dtype("float32")
    expected_open = [int(x.candle.ts_open.value.timestamp() * 1000) for x in items]
    assert arrays.open_time_ms.dtype == np.dtype("int64")
    assert arrays.open_time_ms.tolist() == expected_open
    assert arrays.close_time_ms.dtype == np.dtype("int64")
    assert arrays.close_time_ms.tolist() == [x + 60_000 for x in expected_open]
    index = PostgresCanonicalCandleIndexReader(gateway)
    assert [
        x.count
        for x in index.daily_counts(instrument_id=instrument("READ"), time_range=window(0.5, 1441))
    ] == [2, 1]
    assert (
        len(index.distinct_ts_opens(instrument_id=instrument("READ"), time_range=window(0.5, 1441)))
        == 3
    )
    assert (
        index.max_ts_open_lt(
            instrument_id=instrument("READ"), before=window().end, after=window(6, 10).start
        )
        is None
    )
    assert index.bounds(instrument("READ")) == (window().start, window(0, 1441).end)
    assert [
        x["ts_open"]
        for x in r.latest(instrument_id=instrument("READ"), before=window(0, 1442).end, limit=2)
    ] == [BASE + timedelta(minutes=x) for x in (1440, 1441)]


def test_unknown_not_empty(gateway):
    i = instrument("UNKNOWN")
    with pytest.raises(CoverageUnavailable):
        PostgresCanonicalCandleIndexReader(gateway).bounds(i)
    assert (
        PostgresInstrumentCoverageReader(gateway)
        .read(instrument_id=i, expected_start_at=BASE, expected_end_at=BASE + timedelta(days=1))
        .state
        == "unknown"
    )
    obs = PostgresStoredArtifactCoordinates(gateway).observation(instrument=i, before=window().end)
    assert obs.index_state == "unknown" and obs.actual_count is None


def test_actual_absence_unknown_frontier_and_late(gateway, db):
    i = instrument("GAPS")
    writer = PostgresRawKlineWriter(gateway)
    writer.write_1m([candle("GAPS", 0), candle("GAPS", 5)])
    verifier = PostgresHistoryRangeVerifier(gateway)
    fence = verifier.acquire(instrument_id=i)

    def confirm(a, b, observed=(), accept=True, check=lambda: None):
        verifier.confirm(
            instrument_id=i,
            time_range=window(a, b),
            observed_minutes={minute(BASE) + m for m in observed},
            observed_at=BASE,
            accept_absence=accept,
            check_lease=check,
            fence=fence,
            origin=BASE,
        )

    confirm(5, 10, (5,))
    coverage = PostgresHistoryCoverageReader(gateway)
    assert one(coverage.first_gap(instrument_id=i, time_range=window())).start == window(1, 2).start
    with psycopg.connect(db) as c:
        assert one(
            c.execute(
                "SELECT verified_until FROM market_data.candle_history_verification "
                "WHERE symbol='GAPS'"
            ).fetchone()
        )[0] == BASE + timedelta(minutes=1)
    confirm(0, 5, (0,))
    assert coverage.first_gap(instrument_id=i, time_range=window()) is None
    assert coverage.counts(starts={(1, "GAPS"): BASE}, end_at=window().end.value) == {
        (1, "GAPS"): 2
    }
    assert coverage.source_absent_counts(starts={(1, "GAPS"): BASE}, end_at=window().end.value) == {
        (1, "GAPS"): 8
    }
    assert (
        PostgresInstrumentCoverageReader(gateway)
        .read(instrument_id=i, expected_start_at=BASE, expected_end_at=window().end.value)
        .percent
        == 20
    )
    writer.write_1m([candle("GAPS", 2)])
    assert (
        coverage.source_absent_counts(starts={(1, "GAPS"): BASE}, end_at=window().end.value)[
            (1, "GAPS")
        ]
        == 7
    )
    with psycopg.connect(db) as c:
        assert (
            one(
                c.execute(
                    "SELECT count(*) FROM market_data.candle_history_verification "
                    "WHERE symbol='GAPS'"
                ).fetchone()
            )[0]
            == 0
        )
    with pytest.raises(IncompleteHistoryError):
        confirm(10, 11, (10,))

    def lost():
        raise RuntimeError("lease lost")

    with pytest.raises(RuntimeError):
        confirm(10, 11, check=lost)
    assert (
        one(coverage.first_gap(instrument_id=i, time_range=window(10, 1442))).start
        == window(10, 11).start
    )
    confirm(10, 11, accept=False)
    assert coverage.first_gap(instrument_id=i, time_range=window(10, 11)) is not None


def test_rebuild_delete_and_stale_fences(gateway, db):
    i = instrument("REBUILD")
    PostgresRawKlineWriter(gateway).write_1m([candle("REBUILD", 0), candle("REBUILD", 1440 * 10)])
    v = PostgresHistoryRangeVerifier(gateway)
    fence = v.acquire(instrument_id=i)
    maint = PostgresCoverageMaintenance(gateway)
    generation = maint.begin(
        instrument_id=i,
        expected_generation=fence.generation,
        reason="test-delete",
        delete_range=window(0, 1),
    )
    with pytest.raises(CoverageUnavailable):
        PostgresCanonicalCandleIndexReader(gateway).bounds(i)
    assert maint.step(instrument_id=i, generation=generation)
    assert (
        one(PostgresCanonicalCandleIndexReader(gateway).bounds(i))[0] == window(14400, 14401).start
    )
    with pytest.raises(CoverageUnavailable, match="stale_source_generation"):
        v.confirm(
            instrument_id=i,
            time_range=window(),
            observed_minutes=set(),
            observed_at=BASE,
            accept_absence=True,
            check_lease=lambda: None,
            fence=fence,
        )
    new = v.acquire(instrument_id=i)
    with psycopg.connect(db) as c:
        c.execute(
            "UPDATE market_data.verification_leases SET "
            "expires_at=clock_timestamp()-interval '1s' WHERE symbol='REBUILD'"
        )
    with pytest.raises(CoverageUnavailable, match="stale_verification_lease"):
        v.confirm(
            instrument_id=i,
            time_range=window(),
            observed_minutes=set(),
            observed_at=BASE,
            accept_absence=True,
            check_lease=lambda: None,
            fence=new,
        )


def test_lost_correction_hint_bounded_sweep(gateway):
    i = instrument("REVISION")
    writer = PostgresRawKlineWriter(gateway)
    writer.write_1m([candle("REVISION", 0), candle("REVISION", 10)])
    seen = []

    class Obligations:
        def observe_unavailable(self, **kwargs):
            raise AssertionError("unexpected unavailable observation")

        def observe_source(self, **kw):
            seen.append(kw["source"])

    stored = PostgresStoredArtifactCoordinates(gateway)
    reconciler = BaseSourceReconciler(
        uuid4(), PostgresCanonicalCandleIndexReader(gateway), stored, Obligations()
    )
    now = BASE + timedelta(days=1)
    reconciler.observe(i, now=now)
    before = stored.observation(instrument=i, before=UtcTimestamp(now))
    writer.write_1m([candle("REVISION", 0, close=12, version=1)])  # no hint
    after = stored.observation(instrument=i, before=UtcTimestamp(now))
    assert (before.first, before.last, before.actual_count) == (
        after.first,
        after.last,
        after.actual_count,
    )
    assert isinstance(before.content_revision, int)
    assert after.content_revision == before.content_revision + 1
    reconciler.reconcile_page(after=(1, "REVISIO"), now=now, limit=1)
    assert seen[-1].revision != seen[0].revision
    assert seen[-1].source_generation == before.generation
    writer.write_1m([candle("REVISION", 0, close=12, version=2)])
    assert (
        stored.observation(instrument=i, before=UtcTimestamp(now)).content_revision
        == after.content_revision
    )


def test_reference_search_readiness_catalog(gateway, db):
    with psycopg.connect(db) as c:
        c.execute(
            "INSERT INTO "
            "market_data.ref_market(market_id,exchange_name,market_type,market_code) "
            "VALUES "
            "(1,'binance','spot','binance:spot'),(2,'binance','futures','binance:futures') "
            "ON CONFLICT DO NOTHING"
        )
        c.execute(
            "INSERT INTO "
            "market_data.ref_instruments(market_id,symbol,status,is_tradable) VALUES "
            "(1,'BTCUSDT','ENABLED',1),(1,'BTCDISABLED','ENABLED',0),(2,'BTCUSDT','ENABLED',1) "
            "ON CONFLICT DO NOTHING"
        )
        c.execute("UPDATE market_data.ref_market SET is_enabled=0 WHERE market_id=2")
    assert len(PostgresEnabledMarketReader(gateway).list_enabled_markets()) == 1
    assert PostgresEnabledInstrumentReader(gateway).list_enabled_tradable() == (
        instrument("BTCUSDT"),
    )
    search = PostgresEnabledTradableInstrumentSearchReader(gateway)
    assert search.search_enabled_tradable_by_market(
        market_id=MarketId(1), symbol_prefix="BTC", limit=1
    ) == (instrument("BTCUSDT"),)
    assert not search.search_enabled_tradable_by_market(
        market_id=MarketId(2), symbol_prefix=None, limit=10
    )
    assert not search.search_enabled_tradable_by_market(
        market_id=MarketId(1), symbol_prefix="BTC%", limit=10
    )
    ready = PostgresBTCUSDTMarketReadinessReader(gateway).list_btcusdt_reference_rows()
    assert len(ready) == 2 and any(x.market_enabled for x in ready)
    PostgresRawKlineWriter(gateway).write_1m([candle("CATALOGREAD", m) for m in (0, 1)])
    cat = PostgresCatalogCoverageReader(gateway)
    assert cat.latest(
        market_id=1, symbol="CATALOGREAD", before=window(0, 1.5).end.value
    ) == BASE + timedelta(minutes=1)
    assert (
        cat.counts(market_ids=[1], start_at=BASE, end_at=window(0, 2).end.value)[
            (1, "CATALOGREAD")
        ] == 2
    )


def test_funding_interval_revision_and_nullable_fields(gateway):
    from trading.contexts.market_data.application.dto import FundingInstrument

    i = InstrumentId(MarketId(2), Symbol("FUNDING"))
    store = PostgresFundingRateStore(gateway)
    universe = FundingInstrument(
        instrument_id=i,
        instrument_key="binance:futures:FUNDING",
        exchange="binance",
        market_type="futures",
        status="ENABLED",
        is_tradable=1,
        base_asset=None,
        quote_asset=None,
        funding_interval_minutes=60,
        funding_interval_source="provider",
        funding_cap=None,
        funding_floor=None,
        updated_at=UtcTimestamp(BASE),
    )
    store.upsert_funding_instruments([universe])
    assert store.get_funding_instrument(i) == universe
    assert store.list_tradable_funding_instruments(market_ids=[MarketId(2)]) == (universe,)
    records = [
        FundingRateRecord(
            instrument_id=i,
            instrument_key=universe.instrument_key,
            funding_time=UtcTimestamp(BASE + timedelta(minutes=m)),
            funding_rate=0.01,
            funding_interval_minutes=interval,
            funding_interval_source="provider",
            source="rest",
            ingested_at=UtcTimestamp(BASE + timedelta(days=1)),
            ingest_id=None,
            mark_price=None,
            bybit_category=None,
        )
        for m, interval in ((0, 60), (60, 60), (180, 120))
    ]
    store.write_funding_rates(records)
    result = store.read_funding_coverage(instrument_id=i, time_range=window(0, 181))
    assert result.status == "ready" and len(result.records) == 3
    assert [r.funding_interval_minutes for r in result.records] == [60, 60, 120]
    assert all(r.mark_price is None and r.data_quality == 1 for r in result.records)
    assert store.latest_funding_time(i) == records[-1].funding_time
    obs = PostgresStoredArtifactCoordinates(gateway)
    before = obs.observation(instrument=i, before=window(0, 200).end)
    store.write_funding_rates(
        [replace(records[0], funding_rate=0.02, ingested_at=UtcTimestamp(BASE + timedelta(days=2)))]
    )
    after = obs.observation(instrument=i, before=window(0, 200).end)
    assert isinstance(before.funding_revision, int)
    assert after.funding_revision == before.funding_revision + 1


def test_reader_snapshot_reset_race(gateway, db):
    i = instrument("SNAPSHOT")
    PostgresRawKlineWriter(gateway).write_1m([candle("SNAPSHOT")])
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.query_support import (
        clipped_days,
        require_ready,
    )

    def operation(c):
        gen = require_ready(c, i)[0]
        with psycopg.connect(db) as rival:
            rival.execute(
                "SELECT market_data.invalidate_instrument(1,'SNAPSHOT',%s,'race')", (gen,)
            )
        # The same read-only transaction still has the prior coherent generation.
        assert require_ready(c, i)[0] == gen
        assert list(clipped_days(c, i, minute(BASE), minute(BASE) + 1))[0]["actual"] == "1"

    gateway.run(operation, read_only=True)
    with pytest.raises(CoverageUnavailable):
        PostgresCanonicalCandleIndexReader(gateway).bounds(i)


def test_rebuild_checkpoint_and_writer_race(gateway):
    i = instrument("CHECKPOINT")
    PostgresRawKlineWriter(gateway).write_1m([candle("CHECKPOINT"), candle("CHECKPOINT", 14400)])
    obs = PostgresStoredArtifactCoordinates(gateway).observation(
        instrument=i, before=window(0, 15000).end
    )
    maint = PostgresCoverageMaintenance(gateway)
    gen = maint.begin(instrument_id=i, expected_generation=obs.generation, reason="rebuild")
    assert not maint.step(instrument_id=i, generation=gen, days=3)
    with pytest.raises(Exception, match="rebuild|unknown"):
        PostgresRawKlineWriter(gateway).write_1m([candle("CHECKPOINT", 10)])
    assert not maint.step(instrument_id=i, generation=gen, days=7)
    assert maint.step(instrument_id=i, generation=gen, days=7)
    assert PostgresCanonicalCandleIndexReader(gateway).bounds(i) == (
        window().start,
        window(0, 14400).end,
    )


def test_empty_provider_receipt_and_final_lease_check(gateway, db):
    i = instrument("NEWEMPTY")
    verifier = PostgresHistoryRangeVerifier(gateway)
    fence = verifier.acquire(instrument_id=i)
    calls = 0

    def late_loss():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("lost after mutations")

    with pytest.raises(RuntimeError, match="lost after"):
        verifier.confirm(
            instrument_id=i,
            time_range=window(),
            observed_minutes=set(),
            observed_at=BASE,
            accept_absence=True,
            check_lease=late_loss,
            fence=fence,
        )
    coverage = PostgresHistoryCoverageReader(gateway)
    assert (
        coverage.source_absent_counts(starts={(1, "NEWEMPTY"): BASE}, end_at=window().end.value)[
            (1, "NEWEMPTY")
        ]
        == 0
    )
    verifier.confirm(
        instrument_id=i,
        time_range=window(),
        observed_minutes=set(),
        observed_at=BASE,
        accept_absence=True,
        check_lease=lambda: None,
        fence=fence,
    )
    assert coverage.first_gap(instrument_id=i, time_range=window()) is None
    assert PostgresCanonicalCandleIndexReader(gateway).bounds(i) is None


def test_unprivileged_maintenance(gateway, db):
    from apps.migrations.market_data import provision_market_data_access

    app = "s04app" + uuid4().hex
    maintenance = "s04maint" + uuid4().hex
    i = instrument("ROLE")
    PostgresRawKlineWriter(gateway).write_1m([candle("ROLE")])
    obs = PostgresStoredArtifactCoordinates(gateway).observation(instrument=i, before=window().end)
    with psycopg.connect(db) as c:
        for role in (app, maintenance):
            c.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(role)))
        provision_market_data_access(c, application_role=app, maintenance_role=maintenance)
    g = IngestionGateway(make_conninfo(db, user=maintenance))
    try:
        m = PostgresCoverageMaintenance(g)
        gen = m.begin(instrument_id=i, expected_generation=obs.generation, reason="role-test")
        assert m.step(instrument_id=i, generation=gen)
    finally:
        g.close()
        with psycopg.connect(db) as c:
            for role in (app, maintenance):
                c.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
                c.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


def test_persisted_revision_and_empty_tombstone(gateway, db):
    from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.automatic_base_repository import (  # noqa: E501
        PostgresAutomaticBaseRepository,
    )
    from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.gateway import (
        PsycopgBacktestPostgresGateway,
    )
    from trading.contexts.backtest_artifacts.application.services.management import ManagementError

    # Capacity rejection is injected; all source reads/reconciliation and persisted SQL are real.
    class FullRepository(PostgresAutomaticBaseRepository):
        def enqueue_system(self, **kwargs):
            raise ManagementError("quota_exceeded")

    repo = FullRepository(
        gateway=PsycopgBacktestPostgresGateway(dsn=db),
        catalog=cast(Any, None),
        access=cast(Any, None),
    )
    i = instrument("TOMBSTONE")
    writer = PostgresRawKlineWriter(gateway)
    writer.write_1m([candle("TOMBSTONE")])
    stored = PostgresStoredArtifactCoordinates(gateway)
    reconciler = BaseSourceReconciler(
        uuid4(), PostgresCanonicalCandleIndexReader(gateway), stored, repo
    )
    now = BASE + timedelta(days=1)
    reconciler.observe(i, now=now)

    def state():
        with psycopg.connect(db) as c:
            return one(
                c.execute(
                    "SELECT "
                    "desired_target,source_generation,content_revision,"
                    "funding_revision,source_contract_version "
                    "FROM backtest_artifact_source_state WHERE symbol='TOMBSTONE'"
                ).fetchone()
            )

    before = state()
    early = stored.observation(instrument=i, before=UtcTimestamp(BASE))
    # An older cutoff sees no candles but shares generation/revisions with usable source.
    reconciler._observe(early, now=BASE)
    assert state() == before
    writer.write_1m([candle("TOMBSTONE", close=12, version=1)])
    reconciler.reconcile_page(after=(1, "TOMBSTON"), now=now, limit=1)
    after = state()
    assert before[0]["source"]["history"] == after[0]["source"]["history"]
    assert before[2] + 1 == after[2] and after[4] == 2
    maint = PostgresCoverageMaintenance(gateway)
    gen = maint.begin(
        instrument_id=i, expected_generation=after[1], reason="empty", delete_range=window(0, 1)
    )
    reconciler.observe(i, now=now)
    assert state()[0]["state"] == "unknown" and state()[0]["source"] is None
    stale = stored.observation(instrument=i, before=UtcTimestamp(now))
    assert maint.step(instrument_id=i, generation=gen)
    reconciler.observe(i, now=now)
    current = state()
    reconciler._observe(stale, now=now + timedelta(seconds=1))
    assert state() == current
    tombstone = state()
    assert tombstone[0]["source"] is None and tombstone[0]["state"] == "ready"
    assert tombstone[1] == gen


def test_generation_race_while_verifier_waits(gateway, db):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    i = instrument("FENCERACE")
    PostgresRawKlineWriter(gateway).write_1m([candle("FENCERACE")])
    verifier = PostgresHistoryRangeVerifier(gateway)
    fence = verifier.acquire(instrument_id=i)
    started = Event()

    def verify():
        started.set()
        verifier.confirm(
            instrument_id=i,
            time_range=window(),
            observed_minutes=set(),
            observed_at=BASE,
            accept_absence=True,
            check_lease=lambda: None,
            fence=fence,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        with psycopg.connect(db) as c:
            c.execute(
                "SELECT generation FROM market_data.instrument_generations "
                "WHERE symbol='FENCERACE' FOR UPDATE"
            )
            task = pool.submit(verify)
            assert started.wait(2)
            c.execute(
                "SELECT market_data.invalidate_instrument(1,'FENCERACE',%s,'race')",
                (fence.generation,),
            )
        with pytest.raises(CoverageUnavailable):
            task.result(timeout=5)
    with psycopg.connect(db) as c:
        assert (
            one(
                c.execute(
                    "SELECT count(*) FROM market_data.source_absent_candles_1m "
                    "WHERE symbol='FENCERACE'"
                ).fetchone()
            )[0]
            == 0
        )


def test_partial_day_missing_day_and_policy_refresh(gateway):
    i = instrument("POLICY")
    PostgresRawKlineWriter(gateway).write_1m([candle("POLICY", 1439), candle("POLICY", 2880)])
    v = PostgresHistoryRangeVerifier(gateway)
    fence = v.acquire(instrument_id=i)
    v.confirm(
        instrument_id=i,
        time_range=window(1438, 1440),
        observed_minutes=set(),
        observed_at=BASE,
        accept_absence=True,
        check_lease=lambda: None,
        fence=fence,
    )
    coverage = PostgresHistoryCoverageReader(gateway)
    assert coverage.first_gap(instrument_id=i, time_range=window(1438, 2881)) == window(1440, 2880)
    # A newer policy never trusts the old absence mask or certificate.
    changed = PostgresHistoryCoverageReader(gateway, policy_version=2)
    assert changed.first_gap(instrument_id=i, time_range=window(1438, 1440)) == window(1438, 1439)


def test_lease_renew_release_and_stale_release(gateway):
    i=instrument('LEASE')
    v=PostgresHistoryRangeVerifier(gateway)
    old=v.acquire(instrument_id=i)
    v.renew(instrument_id=i,fence=old)
    assert v.release(instrument_id=i,fence=old)
    current=v.acquire(instrument_id=i)
    assert not v.release(instrument_id=i,fence=old)
    assert v.release(instrument_id=i,fence=current)


def test_funding_universe_exceeds_single_transport_chunk(gateway,db):
    with psycopg.connect(db) as c:
        c.execute("INSERT INTO market_data.funding_instrument_universe "
            "(market_id,symbol,instrument_key,exchange,market_type,status,is_tradable) "
            "SELECT 4,'U'||lpad(n::text,5,'0'),'bybit:futures:U'||n," 
            "'bybit','futures','ENABLED',1 FROM generate_series(1,9000)n")
    result=PostgresFundingRateStore(gateway).list_tradable_funding_instruments(
        market_ids=[MarketId(4)])
    assert len(result)==9000
    assert [str(r.instrument_id.symbol) for r in result] == [
        'U' + str(n).zfill(5) for n in range(1, 9001)
    ]


def test_array_export_retries_real_disconnect_after_first_batch(gateway, db, monkeypatch):
    """A backend loss after4096 rows must not retain a partial output prefix."""
    count = 5000
    for offset in range(0, count, 1000):
        PostgresRawKlineWriter(gateway).write_1m(
            [candle("ARRAYRETRY", m, close=11.25 + (m % 64) / 8)
             for m in range(offset, offset + 1000)]
        )
    original = psycopg.ServerCursor.fetchmany
    fetches = 0
    disconnected = False

    def fetchmany(cursor, size=0):
        nonlocal fetches, disconnected
        if cursor.name == "market_arrays":
            fetches += 1
            if fetches == 2 and not disconnected:
                with psycopg.connect(db, autocommit=True) as admin:
                    terminated = admin.execute(
                        "SELECT pg_terminate_backend(%s)", (cursor.connection.info.backend_pid,)
                    ).fetchone()
                    assert terminated is not None and terminated[0]
                disconnected = True
        return original(cursor, size)

    monkeypatch.setattr(psycopg.ServerCursor, "fetchmany", fetchmany)
    result = PostgresCanonicalCandleReader(gateway).read_1m_arrays(
        instrument("ARRAYRETRY"), window(0, count)
    )
    expected = int(BASE.timestamp() * 1000) + np.arange(count, dtype=np.int64) * 60000
    assert disconnected and fetches >= 5
    np.testing.assert_array_equal(result.open_time_ms, expected)
    np.testing.assert_array_equal(result.close_time_ms, expected + 60000)
    np.testing.assert_array_equal(
        result.ohlcv_f32[:, 3], (11.25 + (np.arange(count) % 64) / 8).astype(np.float32)
    )
    assert result.row_count() == count
