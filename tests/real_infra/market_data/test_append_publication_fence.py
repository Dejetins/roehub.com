"""Concurrent tail appends preserve frozen artifacts; prefix mutations never do."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta

import psycopg
import pytest

from apps.cli.wiring.db.postgres import close_market_gateways, market_gateway
from tests.real_infra.market_data.test_postgres_consumers import db  # noqa: F401
from tests.real_infra.market_data.test_postgres_readers_frontier import (
    candle,
)
from tests.real_infra.market_data.test_postgres_readers_frontier import (
    db as database_fixture,  # noqa: F401
)
from tests.unit.contexts.backtest_artifacts.test_automatic_base import START, frozen, setup
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.backtest_job_repository import (  # noqa: E501
    PostgresBacktestJobRepository,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.gateway import (
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.publication_fence import (  # noqa: E501
    PostgresPublicationFence,
)
from trading.contexts.backtest_artifacts.application.services.management import ManagementError
from trading.contexts.backtest_artifacts.application.services.v2 import (
    artifact_slot_publisher as slot,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_reader import (  # noqa: E501
    PostgresCanonicalCandleReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.shared_kernel.primitives import UtcTimestamp


@pytest.mark.parametrize("change", ["append", "noop", "stale", "correction"])
def test_real_artifact_publication_with_concurrent_ingestion(db, tmp_path, monkeypatch, change):  # noqa: F811
    symbol = "APPEND" + change.upper()
    g = market_gateway({"STRATEGY_PG_DSN": db}, workload="bulk")
    writer = PostgresRawKlineWriter(g)
    rows = [candle(symbol, m, base=START) for m in range(180)]
    writer.write_1m(rows)
    control = PsycopgBacktestPostgresGateway(dsn=db)
    f, service = setup(
        tmp_path,
        repo=PostgresBacktestJobRepository(gateway=control),
        coordinates=ArtifactCoordinatesV2(exchange="binance", market_type="spot", symbol=symbol),
    )
    source = frozen(f, PostgresCanonicalCandleReader(g), 180)
    service = replace(
        service,
        publisher=replace(service.publisher, source_fence=PostgresPublicationFence(control)),
    )
    original = slot._fsync_candidate
    mutations = []

    def mutate(candidate):
        original(candidate)
        if candidate.name != "candidate" or mutations:
            return
        mutations.append(change)
        row = candle(symbol, 180, base=START) if change == "append" else rows[0]
        if change in {"stale", "correction"}:
            row = replace(
                row,
                candle=replace(row.candle, close=17),
                meta=replace(
                    row.meta,
                    ingested_at=UtcTimestamp(
                        row.meta.ingested_at.value
                        + timedelta(seconds=1 if change == "correction" else -1)
                    ),
                ),
            )
        # Actual independent PG transaction interleaved with the live build before final fence.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(writer.write_1m, [row]).result(timeout=30)

    monkeypatch.setattr(slot, "_fsync_candidate", mutate)
    try:
        if change == "correction":
            with pytest.raises(slot.ArtifactPublicationNotStarted, match="source_revision_changed"):
                service.publish(
                    coordinates=f.coordinates, source=source, generated_at=START + timedelta(days=1)
                )
        else:
            result = service.publish(
                coordinates=f.coordinates, source=source, generated_at=START + timedelta(days=1)
            )
            assert result.publication is not None
        assert mutations == [change]
    finally:
        close_market_gateways()


def test_gap_fill_and_unattested_old_writer_remain_fenced(db):  # noqa: F811
    g = market_gateway({"STRATEGY_PG_DSN": db}, workload="bulk")
    writer = PostgresRawKlineWriter(g)
    control = PsycopgBacktestPostgresGateway(dsn=db)
    fence = PostgresPublicationFence(control)
    coordinate = ArtifactCoordinatesV2(exchange="binance", market_type="spot", symbol="APPENDGAP")
    writer.write_1m([candle("APPENDGAP", m, base=START) for m in (0, 2)])
    before = fence.capture(coordinate, frozen_end=START + timedelta(minutes=3))
    writer.write_1m([candle("APPENDGAP", 1, base=START)])
    with (
        pytest.raises(ManagementError, match="source_revision_changed"),
        fence.guard(coordinate, before),
    ):
        pass
    before = fence.capture(coordinate, frozen_end=START + timedelta(minutes=3))
    with psycopg.connect(db) as conn:
        conn.execute(
            "UPDATE market_data.instrument_generations SET content_revision=content_revision+1 "
            "WHERE symbol='APPENDGAP'"
        )
    with (
        pytest.raises(ManagementError, match="source_revision_changed"),
        fence.guard(coordinate, before),
    ):
        pass
    writer.write_1m([candle("APPENDGAP", 3, base=START)])
    with (
        pytest.raises(ManagementError, match="source_revision_changed"),
        fence.guard(coordinate, before),
    ):
        pass
    close_market_gateways()


def test_future_export_range_and_rollback_cannot_claim_append_certificate(db):  # noqa: F811
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_common import (  # noqa: E501
        bounded,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (  # noqa: E501
        _canonical,
    )

    g = market_gateway({"STRATEGY_PG_DSN": db}, workload="bulk")
    writer = PostgresRawKlineWriter(g)
    coordinate = ArtifactCoordinatesV2(exchange="binance", market_type="spot", symbol="APPENDROLL")
    fence = PostgresPublicationFence(PsycopgBacktestPostgresGateway(dsn=db))
    writer.write_1m([candle("APPENDROLL", 0, base=START)])
    before = fence.capture(coordinate, frozen_end=START + timedelta(minutes=10))
    assert before["append_safe_prefix"] is False
    writer.write_1m([candle("APPENDROLL", 1, base=START)])
    with (
        pytest.raises(ManagementError, match="source_revision_changed"),
        fence.guard(coordinate, before),
    ):
        pass
    stable = fence._read(coordinate)
    batch = bounded([candle("APPENDROLL", 0, close=18, version=5, base=START)], _canonical)

    def rollback(conn):
        writer._write(conn, batch)
        raise ValueError("abort after canonical and counters")

    with pytest.raises(ValueError, match="abort after"):
        g.run(rollback)
    assert fence._read(coordinate) == stable
    with psycopg.connect(db) as conn:
        assert conn.execute(
            "SELECT close FROM market_data.canonical_candles_1m "
            "WHERE symbol='APPENDROLL' AND ts_open=%s",
            (START,),
        ).fetchone() == (11.0,)
    close_market_gateways()


def test_managed_frozen_target_allows_only_attested_tail(db):  # noqa: F811
    from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.management_repository import (  # noqa: E501
        PostgresArtifactManagementRepository,
    )

    g = market_gateway({"STRATEGY_PG_DSN": db}, workload="bulk")
    writer = PostgresRawKlineWriter(g)
    coordinate = ArtifactCoordinatesV2(
        exchange="binance", market_type="spot", symbol="APPENDMANAGED"
    )
    control = PsycopgBacktestPostgresGateway(dsn=db)
    fence = PostgresPublicationFence(control)
    repo = object.__new__(PostgresArtifactManagementRepository)
    repo.db = control
    writer.write_1m([candle("APPENDMANAGED", 0, base=START)])
    identity = fence.capture(coordinate)
    build = {
        "exchange": "binance",
        "market_type": "spot",
        "symbol": "APPENDMANAGED",
        "snapshot": {
            "source": {
                **identity,
                "history": {
                    "kind": "range",
                    "start_utc": START,
                    "end_utc": START + timedelta(minutes=1),
                },
                "watermark": START + timedelta(minutes=1),
                "revision": "test",
                "compute_identity": "native-base-v2",
                "funding_identity": "spot",
            }
        },
    }
    writer.write_1m([candle("APPENDMANAGED", 1, base=START)])
    with repo.source_guard(build):
        pass
    writer.write_1m([candle("APPENDMANAGED", 0, close=19, version=5, base=START)])
    with pytest.raises(ManagementError, match="source_revision_changed"), repo.source_guard(build):
        pass
    close_market_gateways()


def test_verified_prefix_survives_tail_but_not_late_actual_or_correction(db):  # noqa: F811
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_range_verifier import (  # noqa: E501
        PostgresHistoryRangeVerifier,
    )
    from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange

    g = market_gateway({"STRATEGY_PG_DSN": db}, workload="bulk")
    writer = PostgresRawKlineWriter(g)
    verifier = PostgresHistoryRangeVerifier(g)
    instrument = InstrumentId(MarketId(1), Symbol("APPENDPREFIX"))
    writer.write_1m([candle("APPENDPREFIX", m, base=START) for m in (0, 1)])

    def confirm(end, observed):
        lease = verifier.acquire(instrument_id=instrument)
        try:
            verifier.confirm(
                instrument_id=instrument,
                time_range=TimeRange(
                    UtcTimestamp(START), UtcTimestamp(START + timedelta(minutes=end))
                ),
                observed_minutes={int(START.timestamp()) // 60 + m for m in observed},
                observed_at=START + timedelta(days=1),
                accept_absence=True,
                check_lease=lambda: None,
                fence=lease,
                origin=START,
            )
        finally:
            verifier.release(instrument_id=instrument, fence=lease)

    def frontier():
        with psycopg.connect(db) as c:
            return c.execute(
                "SELECT verified_until FROM market_data.candle_history_verification "
                "WHERE symbol='APPENDPREFIX'"
            ).fetchone()

    confirm(2, (0, 1))
    before = frontier()
    assert before == (START + timedelta(minutes=2),)
    writer.write_1m([candle("APPENDPREFIX", 2, base=START)])
    assert frontier() == before
    confirm(5, (0, 1, 2))
    assert frontier() == (START + timedelta(minutes=5),)
    writer.write_1m([candle("APPENDPREFIX", 3, base=START)])
    assert frontier() is None  # strict actual tail, but inside certified absence
    confirm(5, (0, 1, 2, 3))
    writer.write_1m([candle("APPENDPREFIX", 0, close=20, version=8, base=START)])
    assert frontier() is None
    close_market_gateways()
