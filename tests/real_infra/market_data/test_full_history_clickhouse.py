"""Explicit opt-in proof using a disposable ClickHouse database, never user candles."""

import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseConnectGateway,
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.history_coverage_reader import (  # noqa: E501
    ClickHouseHistoryCoverageReader,
)
from trading.contexts.market_data.application.ports.stores.enabled_instrument_reader import (
    EnabledInstrumentReader,
)
from trading.contexts.market_data.application.services.history_readiness import HistoryOrigins
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp

pytestmark = pytest.mark.skipif(
    os.environ.get("ROEHUB_HISTORY_CH_TEST") != "1",
    reason="explicit disposable ClickHouse target required",
)


@pytest.fixture
def history_db():
    import clickhouse_connect

    client = clickhouse_connect.get_client(
        host=os.environ["CH_HOST"],
        port=int(os.environ["CH_PORT"]),
        username=os.environ["CH_USER"],
        password=os.environ["CH_PASSWORD"],
    )
    database = "history_test_" + uuid4().hex
    client.command(f"CREATE DATABASE {database}")
    try:
        client.command(f"""CREATE TABLE {database}.canonical_candles_1m_stats (
            market_id UInt16, symbol String,
            rows_state AggregateFunction(sum, UInt64),
            first_ts_state AggregateFunction(min, DateTime64(3, 'UTC')),
            last_ts_state AggregateFunction(max, DateTime64(3, 'UTC')),
            minute_bitmap_state AggregateFunction(groupBitmap, UInt32))
            ENGINE=AggregatingMergeTree ORDER BY (market_id,symbol)""")
        client.command(f"""CREATE TABLE {database}.canonical_candles_1m (
            market_id UInt16, symbol String, ts_open DateTime64(3,'UTC'))
            ENGINE=MergeTree ORDER BY (market_id,symbol,ts_open)""")
        from pathlib import Path

        from apps.migrations.storage import split_clickhouse_sql
        ddl = Path("migrations/clickhouse/history_source_absences_ddl.sql").read_text()
        for statement in split_clickhouse_sql(ddl.replace("market_data.", f"{database}.")):
            client.command(statement)

        def insert(minutes, symbol="ACTUSDT"):
            client.command(
                f"""INSERT INTO {database}.canonical_candles_1m_stats
                SELECT toUInt16(1), %(symbol)s, sumState(toUInt64(1)),
                    minState(toDateTime64(n*60,3,'UTC')), maxState(toDateTime64(n*60,3,'UTC')),
                    groupBitmapState(toUInt32(n))
                FROM (SELECT arrayJoin(%(minutes)s) AS n)""",
                parameters={"symbol": symbol, "minutes": minutes},
            )

        yield SimpleNamespace(
            client=client,
            database=database,
            insert=insert,
            reader=ClickHouseHistoryCoverageReader(ClickHouseConnectGateway(client), database),
        )
    finally:
        client.command(f"DROP DATABASE {database}")
        client.close()


def test_empty_duplicates_internal_and_day_boundary_gaps(history_db):
    reader = history_db.reader
    first = datetime(2024, 1, 1, 10, tzinfo=UTC)
    base = int(first.timestamp() / 60)
    instrument = InstrumentId(MarketId(1), Symbol("ACTUSDT"))
    end = first + timedelta(minutes=3000)
    whole = TimeRange(UtcTimestamp(first), UtcTimestamp(end))
    assert reader.counts(starts={(1, "ACTUSDT"): first}, end_at=end).get((1, "ACTUSDT"), 0) == 0
    assert reader.first_gap(instrument_id=instrument, time_range=whole).start.value == first
    omitted = {5, 6, 1439, 1440, 2999}
    history_db.insert(
        [base + x for x in range(3000) if x not in omitted] + [base - 1, base + 3000, base + 100]
    )
    assert reader.counts(starts={(1, "ACTUSDT"): first}, end_at=end)[(1, "ACTUSDT")] == 2995
    expected = [(5, 7), (1439, 1440), (1440, 1441), (2999, 3000)]
    for a, b in expected:
        gap = reader.first_gap(instrument_id=instrument, time_range=whole)
        assert (gap.start.value, gap.end.value) == (
            first + timedelta(minutes=a),
            first + timedelta(minutes=b),
        )
        history_db.insert([base + x for x in range(a, b)])
    assert reader.first_gap(instrument_id=instrument, time_range=whole) is None
    assert reader.counts(starts={(1, "ACTUSDT"): first}, end_at=end)[(1, "ACTUSDT")] == 3000


def test_zero_history_preparation_admits_only_after_real_gap_is_filled(history_db):
    from trading.contexts.market_data.application.services.history_readiness import HistoryReadiness
    from trading.contexts.market_data.application.services.history_stream_gate import (
        HistoryReadyInstrumentReader,
    )

    end = datetime.now(UTC).replace(second=0, microsecond=0)
    first = end - timedelta(minutes=180)
    instrument = InstrumentId(MarketId(1), Symbol("ACTUSDT"))
    desired = [instrument]
    gate = HistoryReadyInstrumentReader(
        source=cast(
            EnabledInstrumentReader, SimpleNamespace(list_enabled_tradable=lambda: desired),
        ),
        readiness=HistoryReadiness(
            origins=cast(
                HistoryOrigins, SimpleNamespace(ready_starts=lambda **kw: {(1, "ACTUSDT"): first}),
            ),
            coverage=history_db.reader,
        ),
    )
    assert gate.list_enabled_tradable() == ()
    assert gate.repairs
    base = int(first.timestamp() / 60)
    history_db.insert(list(range(base, base + 179)))
    assert gate.list_enabled_tradable() == ()
    history_db.insert([base + 179])
    assert gate.list_enabled_tradable() == (instrument,)
    desired.clear()
    assert gate.list_enabled_tradable() == ()


def test_index_drift_source_halt_and_late_candle_are_distinct(history_db):
    from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.history_range_verifier import (  # noqa: E501
        ClickHouseHistoryRangeVerifier,
    )
    from trading.contexts.market_data.application.services.source_retry import (
        IncompleteHistoryError,
    )

    db = history_db
    first = datetime(2024, 1, 1, tzinfo=UTC)
    end = first + timedelta(minutes=3)
    base = int(first.timestamp() // 60)
    key = (1, "ACTUSDT")
    instrument = InstrumentId(MarketId(1), Symbol("ACTUSDT"))
    whole = TimeRange(UtcTimestamp(first), UtcTimestamp(end))
    db.client.insert(f"{db.database}.canonical_candles_1m", [(1, "ACTUSDT", first)],
                     column_names=["market_id", "symbol", "ts_open"])
    verifier = ClickHouseHistoryRangeVerifier(ClickHouseConnectGateway(db.client), db.database)
    def confirm(observed_minutes: set[int]):
        verifier.confirm(instrument_id=instrument, time_range=whole, observed_at=end,
                         accept_absence=True, check_lease=lambda: None,
                         observed_minutes=observed_minutes)
    # Provider-returned candle lost by storage must not become an accepted omission.
    with pytest.raises(IncompleteHistoryError):
        confirm(observed_minutes={base, base + 1})
    assert db.reader.source_absent_counts(starts={key: first}, end_at=end) == {}
    # A stale bitmap is repaired from canonical while the other two minutes are absent.
    confirm(observed_minutes={base})
    confirm(observed_minutes={base})  # replay is idempotent
    assert db.reader.counts(starts={key: first}, end_at=end)[key] == 1
    assert db.reader.source_absent_counts(starts={key: first}, end_at=end)[key] == 2
    assert db.reader.first_gap(instrument_id=instrument, time_range=whole) is None
    # A later real candle wins over an earlier absence, without double counting.
    db.insert([base + 1])
    assert db.reader.counts(starts={key: first}, end_at=end)[key] == 2
    assert db.reader.source_absent_counts(starts={key: first}, end_at=end)[key] == 1


def test_empty_source_range_and_lost_lease(history_db):
    from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.history_range_verifier import (  # noqa: E501
        ClickHouseHistoryRangeVerifier,
    )

    db = history_db
    first = datetime(2024, 1, 1, tzinfo=UTC)
    end = first + timedelta(hours=1)
    key = (1, "AAVEUSDT")
    instrument = InstrumentId(MarketId(1), Symbol("AAVEUSDT"))
    whole = TimeRange(UtcTimestamp(first), UtcTimestamp(end))
    verifier = ClickHouseHistoryRangeVerifier(ClickHouseConnectGateway(db.client), db.database)
    def confirm(check_lease):
        verifier.confirm(instrument_id=instrument, time_range=whole, observed_at=end,
                         observed_minutes=set(), accept_absence=True, check_lease=check_lease)

    def lost():
        raise RuntimeError("lease_lost")

    with pytest.raises(RuntimeError, match="lease_lost"):
        confirm(check_lease=lost)
    assert db.reader.source_absent_counts(starts={key: first}, end_at=end) == {}
    confirm(check_lease=lambda: None)
    assert db.reader.source_absent_counts(starts={key: first}, end_at=end)[key] == 60
    assert db.reader.first_gap(instrument_id=instrument, time_range=whole) is None


def test_latest_actual_candle_respects_exclusive_bound_duplicates_and_source_absences(history_db):
    from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.catalog_coverage_reader import (  # noqa: E501
        ClickHouseCatalogCoverageReader,
    )

    reader = ClickHouseCatalogCoverageReader(
        ClickHouseConnectGateway(history_db.client), history_db.database,
    )
    first = datetime(2024, 1, 1, tzinfo=UTC)
    base = int(first.timestamp() // 60)
    assert reader.latest(market_id=1, symbol="ACTUSDT", before=first) is None
    history_db.insert([base, base + 1, base + 1, base + 5])
    history_db.insert([base + 7], symbol="BTCUSDT")
    history_db.client.command(
        f"INSERT INTO {history_db.database}.source_absent_candles_1m VALUES "
        f"(1,'ACTUSDT',{base + 8},now64(3))"
    )
    assert reader.latest(market_id=1, symbol="ACTUSDT", before=first) is None
    assert reader.latest(
        market_id=1, symbol="ACTUSDT", before=first + timedelta(minutes=1),
    ) == first
    assert reader.latest(
        market_id=1, symbol="ACTUSDT", before=first + timedelta(minutes=1, seconds=1),
    ) == first + timedelta(minutes=1)
    assert reader.latest(
        market_id=1, symbol="ACTUSDT", before=first + timedelta(minutes=20),
    ) == first + timedelta(minutes=5)
    assert reader.latest(market_id=2, symbol="ACTUSDT", before=first) is None


def test_artifact_observation_exact_cutoff_empty_epoch_duplicates_and_repair(history_db):
    from trading.contexts.backtest_artifacts.adapters.outbound.persistence.clickhouse.stored_coordinates import (  # noqa: E501
        ClickHouseStoredArtifactCoordinates,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.canonical_candle_index_reader import (  # noqa: E501
        ClickHouseCanonicalCandleIndexReader,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.history_range_verifier import (  # noqa: E501
        ClickHouseHistoryRangeVerifier,
    )

    db = history_db
    gateway = ClickHouseConnectGateway(db.client)
    stored = ClickHouseStoredArtifactCoordinates(gateway, db.database)
    canonical = ClickHouseCanonicalCandleIndexReader(gateway, db.database)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    end = epoch + timedelta(minutes=10)
    instrument = InstrumentId(MarketId(1), Symbol('ACTUSDT'))
    assert canonical.bounds(instrument) is None
    assert canonical.max_ts_open_lt(instrument_id=instrument, before=UtcTimestamp(end)) is None
    assert canonical.bounds_1m(instrument_id=instrument, before=UtcTimestamp(end)) == (None, None)
    assert stored.observation_page(after=None, before=UtcTimestamp(end)) == ()
    db.client.insert(f'{db.database}.canonical_candles_1m',
                     [(1, 'ACTUSDT', epoch + timedelta(minutes=x)) for x in (0, 1, 1, 5)],
                     column_names=['market_id','symbol','ts_open'])
    # Deliberately missing index remains absent; it never triggers a hidden canonical scan.
    assert stored.observation_page(after=None, before=UtcTimestamp(end)) == ()
    verifier = ClickHouseHistoryRangeVerifier(gateway, db.database)
    verifier.confirm(instrument_id=instrument, time_range=TimeRange(UtcTimestamp(epoch),
                     UtcTimestamp(end)), observed_minutes={0, 1, 5}, observed_at=end,
                     accept_absence=True, check_lease=lambda: None)
    db.insert([0, 1, 5])  # duplicate states, exact bitmap union remains stable
    for seconds, expected in [(0, (None,None)), (1,(epoch,epoch)), (60,(epoch,epoch)),
                              (61,(epoch,epoch+timedelta(minutes=1))),
                              (600,(epoch,epoch+timedelta(minutes=5)))]:
        before = UtcTimestamp(epoch+timedelta(seconds=seconds))
        row = stored.observation(instrument=instrument, before=before)
        actual = tuple(t.value if t is not None else None for t in (row.first,row.last))
        assert actual == expected
        assert (row.first,row.last) == canonical.bounds_1m(instrument_id=instrument,before=before)
    # Absence-only coordinates and unrelated markets never become artifact source bounds.
    db.client.command(f"INSERT INTO {db.database}.source_absent_candles_1m VALUES "
                      "(1,'ABSUSDT',8,now64(3))")
    db.insert([100], symbol='FUTUSDT')
    page = stored.observation_page(after=None, before=UtcTimestamp(end), limit=1)
    assert str(page[0].instrument.symbol) == 'ACTUSDT'
    future = stored.observation_page(after=(1,'ACTUSDT'), before=UtcTimestamp(end), limit=1)
    assert str(future[0].instrument.symbol) == 'FUTUSDT'
    assert future[0].first is None and future[0].last is None
    assert stored.observation_page(after=(1,'FUTUSDT'), before=UtcTimestamp(end), limit=1) == ()

    from clickhouse_connect.driver.exceptions import DatabaseError

    # Missing schema must fail visibly, not silently return empty/fall back to canonical.
    db.client.command(f'DROP TABLE {db.database}.canonical_candles_1m_stats')
    with pytest.raises(DatabaseError):
        stored.observation_page(after=None, before=UtcTimestamp(end))
