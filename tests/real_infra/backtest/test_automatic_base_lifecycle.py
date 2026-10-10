# ruff: noqa: F811
"""Disposable CH -> PG obligation -> 0028 filesystem publication; no live catalog access."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import clickhouse_connect
import pytest

from apps.worker.backtest_job_runner.wiring.automatic_base import AutomaticBaseWorker
from tests.real_infra.backtest.test_artifact_management import DSN, catalog, managed  # noqa: F401
from tests.unit.contexts.backtest_artifacts.test_automatic_base import START, Source, frozen, setup
from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
    PostgresBacktestJobRepository,  # noqa: E501
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.clickhouse.stored_coordinates import (  # noqa: E501
    ClickHouseStoredArtifactCoordinates,  # noqa: E501
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.automatic_base_repository import (  # noqa: E501
    PostgresAutomaticBaseRepository,  # noqa: E501
)
from trading.contexts.backtest_artifacts.application.services.base_source_reconciliation import (
    BaseSourceReconciler,  # noqa: E501
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.canonical_candle_index_reader import (  # noqa: E501
    ClickHouseCanonicalCandleIndexReader,  # noqa: E501
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.canonical_candle_reader import (  # noqa: E501
    ClickHouseCanonicalCandleReader,  # noqa: E501
)
from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseConnectGateway,  # noqa: E501
)
from trading.contexts.market_data.application.services.stored_candle_signal import (
    StoredCandleSignalWriter,  # noqa: E501
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol

PORT = os.environ.get("ROEHUB_BASE_TEST_CLICKHOUSE_PORT")
pytestmark = pytest.mark.skipif(
    not DSN or not PORT, reason="explicit disposable PostgreSQL/ClickHouse required"
)


@pytest.fixture
def source_db():
    assert PORT is not None
    client = clickhouse_connect.get_client(
        host=os.environ.get("CH_HOST", "127.0.0.1"), port=int(PORT),
        username=os.environ.get("CH_USER", "default"), password=os.environ.get("CH_PASSWORD", ""),
    )
    db = "s03_" + uuid4().hex
    client.command(f"CREATE DATABASE {db}")
    client.command(f"""CREATE TABLE {db}.canonical_candles_1m (
        market_id UInt16, symbol String, ts_open DateTime64(3,'UTC'), ts_close DateTime64(3,'UTC'),
        open Float32, high Float32, low Float32, close Float32, volume_base Float32,
        version UInt64) ENGINE=ReplacingMergeTree(version) ORDER BY (market_id,symbol,ts_open)""")
    # Use the production aggregate/MV definition; writes must make the exact index visible.
    from pathlib import Path

    from apps.migrations.storage import split_clickhouse_sql

    ddl = Path("migrations/clickhouse/market_data_ddl.sql").read_text()
    start = ddl.index("CREATE TABLE IF NOT EXISTS market_data.canonical_candles_1m_stats")
    end = ddl.index("/*", ddl.index("GROUP BY", start))
    for statement in split_clickhouse_sql(ddl[start:end].replace("market_data.", f"{db}.")):
        client.command(statement)
    try:
        yield client, db, ClickHouseConnectGateway(client)
    finally:
        client.command(f"DROP DATABASE {db}")
        client.close()


def insert(source_db, minutes=180, symbol="BTCUSDT"):
    client, db, _ = source_db
    client.insert(
        f"{db}.canonical_candles_1m",
        [
            [
                1,
                symbol,
                START + timedelta(minutes=i),
                START + timedelta(minutes=i + 1),
                100 + i,
                103 + i,
                99 + i,
                102 + i,
                2,
                1,
            ]
            for i in range(minutes)
        ],
        column_names=[
            "market_id",
            "symbol",
            "ts_open",
            "ts_close",
            "open",
            "high",
            "low",
            "close",
            "volume_base",
            "version",
        ],
    )


def repository(m):
    m["db"].execute(
        query="""ALTER TABLE backtest_jobs ADD COLUMN market_id INTEGER,
        ADD COLUMN symbol TEXT, ADD COLUMN artifact_slot TEXT,
        ADD COLUMN artifact_manifest_hash TEXT, ADD COLUMN state TEXT,
        ADD COLUMN execution_mode TEXT, ADD COLUMN request_json JSONB,
        ADD COLUMN spec_payload_json JSONB""",
        parameters={},
    )
    return PostgresAutomaticBaseRepository(
        gateway=m["db"], catalog=m["catalog"], access=m["repo"].access
    )


def test_lost_hint_reconciliation_frozen_request_and_real_publication(managed, source_db, tmp_path):
    m = managed
    repo = repository(m)
    _, db, gateway = source_db
    stored = ClickHouseStoredArtifactCoordinates(gateway, db)
    reconciler = BaseSourceReconciler(
        m["installation"], ClickHouseCanonicalCandleIndexReader(gateway, db), stored, repo
    )
    # No notification: discovery must find actual stored data, bounded to one coordinate/page.
    insert(source_db)
    insert(source_db, 15, "ETHUSDT")
    cursor = reconciler.reconcile_page(after=None, now=datetime.now(UTC), limit=1)
    assert cursor == (1, "BTCUSDT")
    reconciler.reconcile_page(after=cursor, now=datetime.now(UTC), limit=1)
    row = m["db"].fetch_one(
        query="SELECT * FROM backtest_artifact_requests WHERE symbol='BTCUSDT'", parameters={}
    )
    old_snapshot = row["snapshot"]
    insert(source_db, 181)
    for _ in range(3):
        reconciler.observe(InstrumentId(MarketId(1), Symbol("BTCUSDT")), now=datetime.now(UTC))
    assert (
        m["db"].fetch_one(
            query="SELECT count(*) AS n FROM backtest_artifact_requests WHERE symbol='BTCUSDT'",
            parameters={},
        )["n"]
        == 1
    )
    assert (
        repo.base_request(installation_id=m["installation"], request_id=row["request_id"])[
            "snapshot"
        ]
        == old_snapshot
    )
    f, pub = setup(tmp_path, PostgresBacktestJobRepository(gateway=m["db"]))
    checked = []
    worker = AutomaticBaseWorker(
        repo,
        ClickHouseCanonicalCandleReader(gateway, db),
        pub,
        lambda snapshot, size: checked.append(size),
    )
    result = worker.run(installation_id=m["installation"], request=row)
    assert result is not None
    assert result.state == "partial" and result.timeframes["1m"]["closed_bars"] == 180
    projection = m["db"].fetch_one(
        query="SELECT * FROM backtest_artifact_source_state WHERE symbol='BTCUSDT'", parameters={}
    )
    assert projection["published"]["state"] == "partial"
    assert projection["published"]["source_sha256"] == result.source_sha256
    assert (
        projection["desired_target"]["source"]["watermark"] != old_snapshot["source"]["watermark"]
    )
    assert len(checked) == 2
    assert not list(
        f.loader.resolve_current_pointer_path(f.coordinates).parent.glob(".publication-*")
    )


def test_commit_hint_failure_is_healed_and_quota_keeps_obligation(managed, source_db):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_artifact_precompute_runner_v2 as n,  # noqa: E501
    )

    m = managed
    repo = repository(m)
    _, db, gateway = source_db
    reconciler = BaseSourceReconciler(
        m["installation"],
        ClickHouseCanonicalCandleIndexReader(gateway, db),
        ClickHouseStoredArtifactCoordinates(gateway, db),
        repo,
    )

    class Writer:
        def write_1m(self, rows):
            insert(source_db, 15)

    def lost(_):
        raise OSError("lost post-commit hint")

    StoredCandleSignalWriter(Writer(), lost).write_1m(n._build_canonical_rows_v2(bar_indexes=(0,)))
    from tests.real_infra.backtest.test_artifact_management import item

    pending = item(m, "ETHUSDT")
    repo.enqueue_system(
        installation_id=m["installation"], coordinate=pending.coordinate, source=pending.source
    )
    m["db"].execute(
        query="UPDATE backtest_artifact_installation_settings SET global_queue_limit=1",
        parameters={},
    )
    reconciler.reconcile_page(after=None, now=datetime.now(UTC))
    assert (
        m["db"].fetch_one(
            query="SELECT count(*) AS n FROM backtest_artifact_source_state", parameters={}
        )["n"]
        == 2
    )
    m["db"].execute(
        query="UPDATE backtest_artifact_installation_settings SET global_queue_limit=2000",
        parameters={},
    )
    reconciler.reconcile_page(after=None, now=datetime.now(UTC))
    assert (
        m["db"].fetch_one(
            query="SELECT count(*) AS n FROM backtest_artifact_requests", parameters={}
        )["n"]
        == 2
    )


@pytest.mark.parametrize("mode", ["disk_full", "crash_before", "crash_after"])
def test_real_failure_windows_keep_valid_pointer_and_recover_ownership(managed, tmp_path, mode):
    import errno
    import multiprocessing

    from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
        ArtifactSlotValidationSpecV2,  # noqa: E501
    )

    m = managed
    repository(m)
    physical = PostgresBacktestJobRepository(gateway=m["db"])
    f, pub = setup(tmp_path, physical)
    first = pub.publish(
        coordinates=f.coordinates,
        source=frozen(f, Source(30), 30),
        generated_at=START + timedelta(days=1),
    )
    assert first.publication is not None
    old = first.publication.published_pointer
    source = frozen(f, Source(180), 180)
    original = pub.publisher.current_pointer_writer

    class FaultWriter:
        def write_current_pointer_atomically(self, *args, **kwargs):
            if mode == "disk_full":
                raise OSError(errno.ENOSPC, "disposable injected disk full")
            if mode == "crash_before":
                os._exit(17)
            result = original.write_current_pointer_atomically(*args, **kwargs)
            os._exit(18)
            return result

    faulty = replace(pub, publisher=replace(pub.publisher, current_pointer_writer=FaultWriter()))

    def child():
        try:
            faulty.publish(
                coordinates=f.coordinates, source=source, generated_at=START + timedelta(days=1)
            )
        except OSError:
            os._exit(19)

    process = multiprocessing.get_context("fork").Process(target=child)
    process.start()
    process.join(20)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("disposable fault child timed out")
    assert process.exitcode == {"disk_full": 19, "crash_before": 17, "crash_after": 18}[mode]
    recovery = replace(pub.publisher, recover_declared_inventory=True)
    recovery.recover_publications(
        coordinates=f.coordinates,
        validation_spec=ArtifactSlotValidationSpecV2(require_hit_times_manifest=False),
    )
    current = f.loader.load_current_pointer(f.coordinates)
    assert current.slot_generation == (
        old.slot_generation + 1 if mode == "crash_after" else old.slot_generation
    )
    manifest = f.loader.load_slot_manifest(f.coordinates, current.active_slot)
    assert manifest.slot_generation == current.slot_generation
    assert (
        m["db"].fetch_one(
            query=(
                "SELECT count(*) AS n FROM backtest_artifact_slot_ownership "
                "WHERE state!='available'"
            ),
            parameters={},
        )["n"]
        == 0
    )
    assert not list(current.path.parent.glob(".publication-*"))


def test_actual_pinned_reader_blocks_reuse_and_keeps_queue_available(managed, tmp_path):
    from tests.real_infra.backtest.test_artifact_management import item
    from trading.contexts.backtest.application.ports.backtest_job_repositories import (
        ArtifactReaderReservation,  # noqa: E501
    )
    from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (  # noqa: E501
        ArtifactPublicationNotStarted,  # noqa: E501
    )

    m = managed
    repo = repository(m)
    physical = PostgresBacktestJobRepository(gateway=m["db"])
    f, pub = setup(tmp_path, physical)
    result = pub.publish(
        coordinates=f.coordinates,
        source=frozen(f, Source(30), 30),
        generated_at=START + timedelta(days=1),
    )
    assert result.publication is not None
    old = result.publication.published_pointer
    token, parent = uuid4(), uuid4()
    reader = physical.reserve_artifact_reader(
        reader=ArtifactReaderReservation(
            "binance",
            "spot",
            "BTCUSDT",
            old.active_slot,
            old.slot_generation,
            old.manifest_sha256,
            "lazy",
            m["org"],
            uuid4(),
            token,
            1,
            parent,
        )
    )
    pub.publish(
        coordinates=f.coordinates,
        source=frozen(f, Source(60), 60),
        generated_at=START + timedelta(days=1),
    )
    assert f.loader.resolve_slot_manifest_path(f.coordinates, old.active_slot).exists()
    target = item(m)
    rid = repo.enqueue_system(
        installation_id=m["installation"], coordinate=target.coordinate, source=target.source
    )
    worker = AutomaticBaseWorker(repo, Source(180), pub, lambda snapshot, size: None)
    # item fixture's October source would be empty; use frozen January range in a separate
    # immutable request, never rewrite the stored snapshot.
    from trading.contexts.backtest_artifacts.application.services.management import (  # noqa: E501
        History,
        SourceTarget,
    )

    target_source = SourceTarget(
        history=History(kind="range", start_utc=START, end_utc=START + timedelta(minutes=180)),
        watermark=START + timedelta(minutes=180),
        revision="fixture",
        compute_identity="v2",
        funding_identity="spot",
    )
    rid = repo.enqueue_system(
        installation_id=m["installation"], coordinate=target.coordinate, source=target_source
    )
    row = repo.base_request(installation_id=m["installation"], request_id=rid)
    with pytest.raises(ArtifactPublicationNotStarted, match="pinned"):
        worker.run(installation_id=m["installation"], request=row)
    assert (
        m["db"].fetch_one(
            query="SELECT count(*) AS n FROM backtest_artifact_builds WHERE state='quarantined'",
            parameters={},
        )["n"]
        == 0
    )
    physical.release_artifact_reader(reader=reader)


def test_real_raw_commit_materialized_view_and_postcommit_first_request(managed):
    import re
    from pathlib import Path

    from tests.unit.contexts.backtest.application.services.v2 import (
        test_artifact_precompute_runner_v2 as n,  # noqa: E501
    )
    from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.raw_kline_writer import (  # noqa: E501
        ClickHouseRawKlineWriter,  # noqa: E501
    )

    m = managed
    repo = repository(m)
    assert PORT is not None
    client = clickhouse_connect.get_client(
        host=os.environ.get("CH_HOST", "127.0.0.1"), port=int(PORT),
        username=os.environ.get("CH_USER", "default"), password=os.environ.get("CH_PASSWORD", ""),
    )
    db = "s03_raw_" + uuid4().hex
    try:
        ddl = (
            Path("migrations/clickhouse/market_data_ddl.sql").read_text().replace("market_data", db)
        )
        ddl = re.sub(r"/\*.*?\*/", "", ddl, flags=re.S)
        ddl = re.sub(r"--[^\n]*", "", ddl)
        for statement in ddl.split(";"):
            if statement.strip() and not statement.strip().startswith("USE "):
                client.command(statement)
        gateway = ClickHouseConnectGateway(client)
        reconciler = BaseSourceReconciler(
            m["installation"],
            ClickHouseCanonicalCandleIndexReader(gateway, db),
            ClickHouseStoredArtifactCoordinates(gateway, db),
            repo,
        )
        # Empty canonical aggregate must never admit a fabricated epoch-zero source.
        reconciler.observe(InstrumentId(MarketId(1), Symbol("BTCUSDT")), now=datetime.now(UTC))
        assert (
            m["db"].fetch_one(
                query="SELECT count(*) AS n FROM backtest_artifact_requests", parameters={}
            )["n"]
            == 0
        )

        def committed(instrument):
            reconciler.observe(instrument, now=datetime.now(UTC))

        writer = StoredCandleSignalWriter(ClickHouseRawKlineWriter(gateway, db), committed)
        writer.write_1m(n._build_canonical_rows_v2(bar_indexes=tuple(range(15))))
        assert (
            client.query(f"SELECT count() FROM {db}.canonical_candles_1m FINAL").first_row[0] == 15
        )
        request = m["db"].fetch_one(query="SELECT * FROM backtest_artifact_requests", parameters={})
        assert request["state"] == "queued" and request["trigger"] == "initial"
        assert request["snapshot"]["source"]["revision"] == "content_unverified"
    finally:
        client.command(f"DROP DATABASE IF EXISTS {db}")
        client.close()
