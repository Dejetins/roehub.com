# ruff: noqa: F811
"""Disposable PostgreSQL obligation -> 0028 filesystem publication; no live catalog access."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from apps.worker.backtest_job_runner.wiring.automatic_base import AutomaticBaseWorker
from tests.real_infra.backtest.test_artifact_management import DSN, catalog, managed  # noqa: F401
from tests.unit.contexts.backtest_artifacts.test_automatic_base import START, Source, frozen, setup
from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
    PostgresBacktestJobRepository,  # noqa: E501
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.automatic_base_repository import (  # noqa: E501
    PostgresAutomaticBaseRepository,  # noqa: E501
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.stored_coordinates import (  # noqa: E501
    PostgresStoredArtifactCoordinates,  # noqa: E501
)
from trading.contexts.backtest_artifacts.application.services.base_source_reconciliation import (
    BaseSourceReconciler,  # noqa: E501
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_index_reader import (  # noqa: E501
    PostgresCanonicalCandleIndexReader,  # noqa: E501
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_reader import (  # noqa: E501
    PostgresCanonicalCandleReader,  # noqa: E501
)
from trading.contexts.market_data.application.services.stored_candle_signal import (
    StoredCandleSignalWriter,  # noqa: E501
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol

pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PostgreSQL required")


@pytest.fixture
def source_db(managed):
    return managed["source_gateway"]


def insert(source_db, minutes=180, symbol="BTCUSDT"):
    from tests.real_infra.market_data.test_postgres_readers_frontier import candle
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (  # noqa: E501
        PostgresRawKlineWriter,
    )

    PostgresRawKlineWriter(source_db).write_1m(
        [candle(symbol, i, base=START) for i in range(minutes)]
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
    m["db"].execute(
        query="""ALTER TABLE backtest_artifact_source_state
        ADD COLUMN IF NOT EXISTS source_generation uuid,
        ADD COLUMN IF NOT EXISTS content_revision bigint,
        ADD COLUMN IF NOT EXISTS funding_revision bigint,
        ADD COLUMN IF NOT EXISTS source_contract_version integer""",
        parameters={},
    )
    return PostgresAutomaticBaseRepository(
        gateway=m["db"], catalog=m["catalog"], access=m["repo"].access
    )


def test_lost_hint_reconciliation_frozen_request_and_real_publication(managed, source_db, tmp_path):
    m = managed
    repo = repository(m)
    gateway = source_db
    stored = PostgresStoredArtifactCoordinates(gateway)
    reconciler = BaseSourceReconciler(
        m["installation"], PostgresCanonicalCandleIndexReader(gateway), stored, repo
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
    f, pub = setup(tmp_path, PostgresBacktestJobRepository(gateway=m["db"]))
    checked = []
    worker = AutomaticBaseWorker(
        repo,
        PostgresCanonicalCandleReader(gateway),
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
    assert (
        m["db"].fetch_one(
            query=("SELECT desired_target FROM backtest_artifact_source_state "
                   "WHERE symbol='BTCUSDT'"),
            parameters={},
        )["desired_target"]["source"]["watermark"]
        != old_snapshot["source"]["watermark"]
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
    gateway = source_db
    reconciler = BaseSourceReconciler(
        m["installation"],
        PostgresCanonicalCandleIndexReader(gateway),
        PostgresStoredArtifactCoordinates(gateway),
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
    from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.publication_fence import (  # noqa: E501
        PostgresPublicationFence,
    )
    from trading.contexts.backtest_artifacts.application.services.management import (  # noqa: E501
        History,
        SourceTarget,
    )

    target_source = SourceTarget(
        **PostgresPublicationFence(m["db"]).capture(target.coordinate),
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


def test_real_raw_commit_and_postcommit_first_request(managed, source_db):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_artifact_precompute_runner_v2 as n,  # noqa: E501
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (  # noqa: E501
        PostgresRawKlineWriter,  # noqa: E501
    )

    m = managed
    repo = repository(m)
    gateway = source_db
    reconciler = BaseSourceReconciler(
        m["installation"],
        PostgresCanonicalCandleIndexReader(gateway),
        PostgresStoredArtifactCoordinates(gateway),
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

    writer = StoredCandleSignalWriter(PostgresRawKlineWriter(gateway), committed)
    writer.write_1m(n._build_canonical_rows_v2(bar_indexes=tuple(range(15))))
    assert (
        gateway.run(
            lambda c: c.execute("SELECT count(*) FROM market_data.canonical_candles_1m").fetchone()
        )[0]
        == 15
    )
    request = m["db"].fetch_one(query="SELECT * FROM backtest_artifact_requests", parameters={})
    assert request["state"] == "queued" and request["trigger"] == "initial"
    assert request["snapshot"]["source"]["revision"] != "content_unverified"


def test_actual_crash_after_old_candle_commit_recovers_without_hint(managed, source_db, tmp_path):
    """SIGKILL between durable writer return and hint; restart discovers same-bounds revision."""
    import json
    import multiprocessing
    import signal

    import numpy as np

    from tests.real_infra.market_data.test_postgres_readers_frontier import candle
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (  # noqa: E501
        IngestionGateway,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (  # noqa: E501
        PostgresRawKlineWriter,
    )
    from trading.shared_kernel.primitives import UtcTimestamp

    m = managed
    repository(m)
    from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.dispatch_repository import (  # noqa: E501
        PostgresArtifactDispatchRepository,
    )

    repo = PostgresArtifactDispatchRepository(
        gateway=m["db"], catalog=m["catalog"], access=m["repo"].access
    )
    insert(source_db)
    stored = PostgresStoredArtifactCoordinates(source_db)
    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    now = datetime.now(UTC)

    def reconcile_child():
        from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.gateway import (  # noqa: E501
            PsycopgBacktestPostgresGateway,  # noqa: E501
        )

        # A new process owns new connections; inherited parent sessions are never used.
        gateway = IngestionGateway(source_db._dsn, workload="reconcile")
        control = PsycopgBacktestPostgresGateway(dsn=m["dsn"])
        fresh_repo = PostgresArtifactDispatchRepository(
            gateway=control, catalog=m["catalog"], access=m["repo"].access)
        try:
            restarted = BaseSourceReconciler(
                m["installation"], PostgresCanonicalCandleIndexReader(gateway),
                PostgresStoredArtifactCoordinates(gateway), fresh_repo)
            assert restarted.reconcile_page(after=None, now=now, limit=1) == (1, "BTCUSDT")
            assert restarted.reconcile_page(after=(1, "BTCUSDT"), now=now, limit=1) is None
        finally:
            gateway.close()

    def reconcile():
        process = multiprocessing.get_context("fork").Process(target=reconcile_child)
        process.start()
        process.join(20)
        if process.is_alive():
            process.kill()
            process.join()
            pytest.fail("restarted reconciler exceeded bounded proof deadline")
        assert process.exitcode == 0

    def latest():
        return m["db"].fetch_one(
            query="SELECT * FROM backtest_artifact_requests WHERE symbol='BTCUSDT' "
            "ORDER BY created_at DESC,request_id DESC LIMIT 1",
            parameters={},
        )

    reconcile()
    f, pub = setup(tmp_path, PostgresBacktestJobRepository(gateway=m["db"]))
    worker = AutomaticBaseWorker(
        repo, PostgresCanonicalCandleReader(source_db), pub, lambda *a: None
    )
    first = worker.run(installation_id=m["installation"], request=latest())
    assert first is not None and first.publication is not None
    old_pointer = f.loader.load_current_pointer(f.coordinates)
    old_source = frozen(f, PostgresCanonicalCandleReader(source_db), 180)
    old_values = old_source.batch.ohlcv_f32.copy()
    before = stored.observation(instrument=instrument, before=UtcTimestamp(now))
    original = candle("BTCUSDT", 3, base=START)
    correction = replace(
        original,
        candle=replace(original.candle, close=19.0),
        meta=replace(original.meta, ingested_at=UtcTimestamp(now)),
    )
    marker = tmp_path / "hint-published"

    def crash_child():
        gateway = IngestionGateway(source_db._dsn)

        def before_hint(_):
            # Callback is reached only after the production writer's COMMIT returns.
            os.kill(os.getpid(), signal.SIGKILL)
            marker.write_text("unreachable")

        StoredCandleSignalWriter(PostgresRawKlineWriter(gateway), before_hint).write_1m(
            [correction]
        )

    child = multiprocessing.get_context("fork").Process(target=crash_child)
    child.start()
    child.join(20)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail("crash child exceeded deadline")
    assert child.exitcode == -signal.SIGKILL
    assert not marker.exists()
    after = stored.observation(instrument=instrument, before=UtcTimestamp(now))
    assert (before.first, before.last, before.actual_count, before.generation) == (
        after.first,
        after.last,
        after.actual_count,
        after.generation,
    )
    assert before.content_revision is not None
    assert after.content_revision == before.content_revision + 1
    assert f.loader.load_current_pointer(f.coordinates) == old_pointer
    # A fresh reconciler scans at most one coordinate/page and creates the new obligation.
    reconcile()
    assert repo.tick(installation=m["installation"], now=now, limit=1) == 1
    second_request = latest()
    second = worker.run(installation_id=m["installation"], request=second_request)
    assert second is not None and second.publication is not None
    new_pointer = f.loader.load_current_pointer(f.coordinates)
    new_source = frozen(f, PostgresCanonicalCandleReader(source_db), 180)
    assert new_pointer.slot_generation == old_pointer.slot_generation + 1
    assert new_pointer.manifest_sha256 != old_pointer.manifest_sha256
    assert second.source_sha256 != first.source_sha256
    assert new_source.sha256 != old_source.sha256
    expected = old_values.copy()
    expected[3, 3] = 19.0
    np.testing.assert_array_equal(new_source.batch.ohlcv_f32, expected)
    # Real exact-scoring engine: fixed signals/settings, PostgreSQL prices only.
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_no_risk_exact_scoring_service as engine,
    )
    from trading.contexts.backtest.application.dto import canonical_no_risk_top_results_hash
    from trading.contexts.backtest.application.services.v2 import BacktestNoRiskExactScoringService

    def score(values):
        prepared = replace(
            engine._prepared_result(indicator_ids=("alpha", "beta")),
            execution_open_1m=values[:5, 0],
            execution_close_1m=values[:5, 3],
        )
        result = BacktestNoRiskExactScoringService().execute(
            prepared_result=prepared,
            combo_planning_result=engine._combo_planning_result(prepared=prepared),
            normalized_request=engine._normalized_request(risk_mode="none"),
        )
        return canonical_no_risk_top_results_hash(result.top_results)

    backtest_before, backtest_after = score(old_values), score(new_source.batch.ohlcv_f32)
    assert backtest_before != backtest_after
    assert backtest_after == score(expected)
    # Observable fixed-input result changes exactly by the corrected close delta.
    old_result = float(np.sum(old_values[:, 3], dtype=np.float64))
    new_result = float(np.sum(new_source.batch.ohlcv_f32[:, 3], dtype=np.float64))
    assert new_result - old_result == 19.0 - float(old_values[3, 3])
    count_query = "SELECT count(*) AS n FROM backtest_artifact_requests WHERE symbol='BTCUSDT'"
    count = m["db"].fetch_one(query=count_query, parameters={})["n"]
    PostgresRawKlineWriter(source_db).write_1m([correction])
    for _ in range(3):
        reconcile()
        assert repo.tick(installation=m["installation"], now=now, limit=1) == 0
    replay = stored.observation(instrument=instrument, before=UtcTimestamp(now))
    assert replay == after
    assert m["db"].fetch_one(query=count_query, parameters={})["n"] == count
    assert f.loader.load_current_pointer(f.coordinates) == new_pointer
    print(
        json.dumps(
            {
                "backtest_before": backtest_before,
                "backtest_after": backtest_after,
                "crash_exitcode": child.exitcode,
                "hint_published": False,
                "actual_count": after.actual_count,
                "bounds_unchanged": True,
                "revision_before": before.content_revision,
                "revision_after": after.content_revision,
                "source_before": first.source_sha256,
                "source_after": second.source_sha256,
                "manifest_before": old_pointer.manifest_sha256,
                "manifest_after": new_pointer.manifest_sha256,
                "reconciler_restart": "new_process_new_connections",
                "bounded_page_size": 1,
                "replay_sweeps": 3,
                "requests_after_replay": count,
                "fixed_close_sum_before": old_result,
                "fixed_close_sum_after": new_result,
            },
            sort_keys=True,
        )
    )
