"""S05 activated PG composition, provider fences and native artifact parity on disposable PG."""

from dataclasses import replace
from datetime import timedelta
from typing import Any, LiteralString, cast

import numpy as np
import psycopg
import pytest

from apps.api.wiring.modules.market_data_reference import build_market_data_reference_use_cases
from apps.cli.wiring.db.postgres import close_market_gateways, market_gateway
from tests.real_infra.market_data.test_postgres_readers_frontier import (
    BASE,
    DSN,
    candle,
    instrument,
    window,
)
from tests.real_infra.market_data.test_postgres_readers_frontier import (
    db as database_fixture,  # noqa: F401
)
from tests.unit.contexts.backtest_artifacts.test_automatic_base import START, Source, frozen, setup
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.backtest_job_repository import (  # noqa: E501
    PostgresBacktestJobRepository,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.gateway import (
    PsycopgBacktestPostgresGateway,
)
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.publication_fence import (  # noqa: E501
    PostgresPublicationFence,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_index_reader import (  # noqa: E501
    PostgresCanonicalCandleIndexReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.canonical_candle_reader import (  # noqa: E501
    PostgresCanonicalCandleReader,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.history_range_verifier import (  # noqa: E501
    PostgresHistoryRangeVerifier,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.raw_kline_writer import (
    PostgresRawKlineWriter,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.reference_writers import (
    PostgresMarketRefWriter,
)
from trading.contexts.market_data.application.dto import RestFillTask
from trading.contexts.market_data.application.use_cases.rest_fill_range_1m import (
    RestFillRange1mUseCase,
)
from trading.contexts.market_data.application.use_cases.seed_ref_market import SeedRefMarketUseCase
from trading.shared_kernel.primitives import UtcTimestamp


class FixedClock:
    def __init__(self, value):
        self.value = value

    def now(self):
        return UtcTimestamp(self.value)


def one(cursor):
    row = cursor.fetchone()
    assert row is not None
    return row


pytestmark = pytest.mark.skipif(not DSN, reason="explicit disposable PG16 required")


@pytest.fixture(scope="module")
def db(database_fixture):  # noqa: F811
    with psycopg.connect(database_fixture) as c:
        c.execute("SELECT market_data.ensure_partitions('2026-01-01',1)")
    yield database_fixture


def test_hostname_composition_no_ch_and_seed_api(db):
    env = {
        "STRATEGY_PG_DSN": db.replace("127.0.0.1", "localhost"),
        "CH_HOST": "unreachable.invalid",
    }
    gateway = market_gateway(env, workload="reference")
    SeedRefMarketUseCase(
        writer=PostgresMarketRefWriter(gateway),
        clock=FixedClock(BASE),
    ).run()
    readers = build_market_data_reference_use_cases(environ=env)
    assert len(readers.list_enabled_markets.execute()) == 4
    # New physical connection repeats hostname resolution; no manual hostaddr required.
    close_market_gateways()
    assert market_gateway(env).run(lambda c: c.execute("SELECT 1").fetchone())[0] == 1
    close_market_gateways()


def test_real_rest_fence_origin_absence_and_rebuild(db):
    gateway = market_gateway({"STRATEGY_PG_DSN": db}, workload="ingestion")
    verifier = PostgresHistoryRangeVerifier(gateway)
    writer = PostgresRawKlineWriter(gateway)
    symbol = "S05REST"
    calls = []

    class Provider:
        def stream_1m(self, instrument_id, time_range):
            with psycopg.connect(db) as c:
                lease = c.execute(
                    "SELECT token FROM market_data.verification_leases WHERE symbol=%s", (symbol,)
                ).fetchone()
                assert lease is not None  # acquired before provider I/O
            calls.append(lease[0])
            yield candle(symbol, 0)

    fill = RestFillRange1mUseCase(
        Provider(),
        writer,
        FixedClock(BASE + timedelta(days=1)),
        1,
        500,
        PostgresCanonicalCandleIndexReader(gateway),
        verifier,
    )
    fill.run(RestFillTask(instrument(symbol), window(0, 3), "user_requested_fill"))
    with psycopg.connect(db) as c:
        assert (
            one(c.execute(
                "SELECT count(*) FROM market_data.source_absent_candles_1m WHERE symbol=%s",
                (symbol,),
            ))[0]
            == 2
        )
        assert (
            one(c.execute(
                "SELECT count(*) FROM market_data.verification_leases WHERE symbol=%s", (symbol,)
            ))[0]
            == 0
        )
        assert (
            one(c.execute(
                "SELECT count(*) FROM market_data.candle_history_verification WHERE symbol=%s",
                (symbol,),
            ))[0]
            == 0
        )
    assert len(calls) == 1
    close_market_gateways()


def test_real_pg_memmap_native_artifact_and_correction_fence(db, tmp_path):
    gateway = market_gateway({"STRATEGY_PG_DSN": db}, workload="bulk")
    writer = PostgresRawKlineWriter(gateway)
    rows = []
    for i in range(180):
        row = candle("BTCUSDT", i, base=START)
        rows.append(
            replace(
                row,
                candle=replace(
                    row.candle, open=100 + i, high=103 + i, low=99 + i, close=102 + i, volume_base=2
                ),
            )
        )
    writer.write_1m(rows)
    control = PsycopgBacktestPostgresGateway(dsn=db)
    f, service = setup(tmp_path, repo=PostgresBacktestJobRepository(gateway=control))
    reader = PostgresCanonicalCandleReader(gateway)
    source = frozen(f, reader, 180)
    expected = frozen(f, Source(180), 180)
    np.testing.assert_array_equal(source.batch.ohlcv_f32, expected.batch.ohlcv_f32)
    assert source.sha256 == expected.sha256
    batch = reader.read_1m_arrays(instrument("BTCUSDT"), source.time_range)
    assert isinstance(batch.ohlcv_f32, np.memmap) and not batch.ohlcv_f32.flags.writeable
    service = replace(
        service,
        publisher=replace(service.publisher, source_fence=PostgresPublicationFence(control)),
    )
    result = service.publish(
        coordinates=f.coordinates, source=source, generated_at=START + timedelta(days=1)
    )
    assert result.publication is not None
    # Validate every produced NPY field against fixed mathematical OHLCV inputs,
    # not just the source or manifest digest.
    from tests.unit.contexts.backtest_artifacts.test_automatic_base import DURATIONS

    pointer = result.publication.published_pointer
    manifest = f.loader.load_slot_manifest(f.coordinates, pointer.active_slot)
    artifact_root = f.loader.resolve_slot_manifest_path(f.coordinates, pointer.active_slot).parent
    for section in manifest.prices:
        width = DURATIONS[section.timeframe]
        starts = np.arange(0, 180 - width + 1, width)
        expected_values = np.array([
            [100 + i, 102 + i + width, 99 + i, 101 + i + width, 2 * width]
            for i in starts
        ], dtype=np.float32)
        opens = int(START.timestamp() * 1000) + starts * 60000
        np.testing.assert_array_equal(np.load(artifact_root / section.ohlcv.path), expected_values)
        np.testing.assert_array_equal(np.load(artifact_root / section.open_time.path), opens)
        np.testing.assert_array_equal(
            np.load(artifact_root / section.close_time.path), opens + width * 60000)
    for section in manifest.mappings:
        width = DURATIONS[section.timeframe]
        starts = np.arange(0, 180 - width + 1, width)
        np.testing.assert_array_equal(np.load(artifact_root / section.bar_open_1m_idx.path), starts)
        np.testing.assert_array_equal(
            np.load(artifact_root / section.bar_close_1m_idx.path), starts + width - 1)
    original = f.loader.load_current_pointer(f.coordinates)
    # A real value correction during compute must invalidate the captured source tuple.
    from trading.contexts.backtest_artifacts.application.services.management import ManagementError

    identity = service.publisher.source_fence.capture(f.coordinates)
    writer.write_1m(
        [
            replace(
                rows[0],
                candle=replace(rows[0].candle, close=101),
                meta=replace(
                    rows[0].meta, ingested_at=UtcTimestamp(BASE + timedelta(days=2, seconds=1))
                ),
            )
        ]
    )
    with pytest.raises(ManagementError, match="source_revision_changed"):
        with service.publisher.source_fence.guard(f.coordinates, identity):
            raise AssertionError("stale source admitted")
    assert f.loader.load_current_pointer(f.coordinates) == original
    close_market_gateways()


@pytest.mark.parametrize("change", ["candle", "funding", "generation"])
def test_change_after_compute_rejects_pointer_and_releases_candidate(
    db, tmp_path, monkeypatch, change
):
    from trading.contexts.backtest_artifacts.application.services.v2 import (
        artifact_slot_publisher as slot,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
        ArtifactCoordinatesV2,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.coverage_maintenance import (  # noqa: E501
        PostgresCoverageMaintenance,
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.funding_rate_store import (  # noqa: E501
        PostgresFundingRateStore,
    )
    from trading.contexts.market_data.application.dto import FundingRateRecord
    from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol

    symbol = "RACE" + change.upper()
    i = InstrumentId(MarketId(2), Symbol(symbol))
    g = market_gateway({"STRATEGY_PG_DSN": db}, workload="bulk")
    writer = PostgresRawKlineWriter(g)
    rows = [
        replace(
            candle(symbol, m, base=START),
            candle=replace(candle(symbol, m, base=START).candle, instrument_id=i),
        )
        for m in range(180)
    ]
    writer.write_1m(rows)
    control = PsycopgBacktestPostgresGateway(dsn=db)
    f, service = setup(
        tmp_path,
        repo=PostgresBacktestJobRepository(gateway=control),
        coordinates=ArtifactCoordinatesV2(exchange="binance", market_type="futures", symbol=symbol),
    )
    reader = PostgresCanonicalCandleReader(g)
    source = frozen(f, reader, 180)
    funding = PostgresFundingRateStore(g)
    funding.write_funding_rates(
        [
            FundingRateRecord(
                instrument_id=i,
                instrument_key="binance:futures:" + symbol,
                funding_time=UtcTimestamp(START),
                funding_rate=0.005,
                funding_interval_minutes=480,
                funding_interval_source="provider",
                source="rest",
                ingested_at=UtcTimestamp(BASE - timedelta(days=1)),
                ingest_id=None,
                mark_price=None,
                bybit_category=None,
            )
        ]
    )
    service = replace(
        service,
        runner=replace(service.runner, funding_rate_coverage_reader=funding),
        publisher=replace(service.publisher, source_fence=PostgresPublicationFence(control)),
    )
    service.publish(coordinates=f.coordinates, source=source, generated_at=BASE)
    before = f.loader.load_current_pointer(f.coordinates)
    original = slot._fsync_candidate
    observed = []

    def mutate_after_compute(candidate):
        original(candidate)
        if candidate.name != "candidate" or observed:
            return
        observed.append(change)
        if change == "candle":
            writer.write_1m(
                [
                    replace(
                        rows[0],
                        candle=replace(rows[0].candle, close=12),
                        meta=replace(
                            rows[0].meta, ingested_at=UtcTimestamp(BASE + timedelta(days=4))
                        ),
                    )
                ]
            )
        elif change == "funding":
            PostgresFundingRateStore(g).write_funding_rates(
                [
                    FundingRateRecord(
                        instrument_id=i,
                        instrument_key="binance:futures:" + symbol,
                        funding_time=UtcTimestamp(START),
                        funding_rate=0.01,
                        funding_interval_minutes=480,
                        funding_interval_source="provider",
                        source="rest",
                        ingested_at=UtcTimestamp(BASE),
                        ingest_id=None,
                        mark_price=None,
                        bybit_category=None,
                    )
                ]
            )
        else:
            identity = service.publisher.source_fence.capture(f.coordinates)
            maintenance = PostgresCoverageMaintenance(g)
            generation = maintenance.begin(
                instrument_id=i,
                expected_generation=identity["source_generation"],
                reason="S05 race",
            )
            assert maintenance.step(instrument_id=i, generation=generation)

    monkeypatch.setattr(slot, "_fsync_candidate", mutate_after_compute)
    with pytest.raises(slot.ArtifactPublicationNotStarted, match="source_revision_changed"):
        service.publish(coordinates=f.coordinates, source=source, generated_at=BASE)
    assert observed == [change]
    assert f.loader.load_current_pointer(f.coordinates) == before
    assert not list(before.path.parent.glob(".publication-*"))
    monkeypatch.setattr(slot, "_fsync_candidate", original)
    fresh = frozen(f, reader, 180)
    result = service.publish(coordinates=f.coordinates, source=fresh, generated_at=BASE)
    assert result.publication is not None
    assert result.publication.published_pointer.slot_generation > before.slot_generation
    close_market_gateways()


def test_ws_child_warmed_pg_receipt_and_restart(db):
    import asyncio

    from apps.worker.market_data_ws.wiring.persistent_writer import PersistentRawWriterProcess

    async def run():
        writer = PersistentRawWriterProcess(
            environ={"STRATEGY_PG_DSN": db, "CH_HOST": "unreachable.invalid"}
        )
        await writer.start()
        first = writer.active_pids
        try:
            await writer.write([candle("WSRECEIPT", 0)])
            await writer.write([candle("WSRECEIPT", 1)])
            assert writer.active_pids == first
        finally:
            await writer.close()
        assert not writer.active_pids
        restart = PersistentRawWriterProcess(environ={"STRATEGY_PG_DSN": db})
        try:
            await restart.write([candle("WSRECEIPT", 0)])
        finally:
            await restart.close()

    asyncio.run(run())
    with psycopg.connect(db) as c:
        assert (
            one(c.execute(
                "SELECT count(*) FROM market_data.canonical_candles_1m WHERE symbol='WSRECEIPT'"
            ))[0]
            == 2
        )


def test_pg_fixed_input_native_backtest_matches_independent_fixture(db):
    from datetime import UTC, datetime

    from tests.unit.contexts.backtest.application.services.v2 import (
        test_no_risk_exact_scoring_service as engine,  # noqa: E501
    )
    from trading.contexts.backtest.application.dto import canonical_no_risk_top_results_hash
    from trading.contexts.backtest.application.services.v2 import BacktestNoRiskExactScoringService

    reference = engine._prepared_result(indicator_ids=("alpha", "beta"))
    start = datetime(1970, 1, 1, tzinfo=UTC)
    rows = []
    assert reference.execution_open_1m is not None
    assert reference.execution_close_1m is not None
    for m, (op, cl) in enumerate(zip(reference.execution_open_1m, reference.execution_close_1m)):
        row = candle("FIXEDBACKTEST", m, base=start)
        rows.append(
            replace(
                row,
                candle=replace(
                    row.candle,
                    open=float(op),
                    close=float(cl),
                    high=float(max(op, cl) + 1),
                    low=float(min(op, cl) - 1),
                ),
            )
        )
    g = market_gateway({"STRATEGY_PG_DSN": db}, workload="bulk")
    PostgresRawKlineWriter(g).write_1m(rows)
    batch = PostgresCanonicalCandleReader(g).read_1m_arrays(
        instrument("FIXEDBACKTEST"), window(0, 5, base=start)
    )
    actual = replace(
        reference, execution_open_1m=batch.ohlcv_f32[:, 0], execution_close_1m=batch.ohlcv_f32[:, 3]
    )
    request = engine._normalized_request(risk_mode="none")
    scorer = BacktestNoRiskExactScoringService()
    expected = scorer.execute(
        prepared_result=reference,
        combo_planning_result=engine._combo_planning_result(prepared=reference),
        normalized_request=request,
    )
    result = scorer.execute(
        prepared_result=actual,
        combo_planning_result=engine._combo_planning_result(prepared=actual),
        normalized_request=request,
    )
    assert canonical_no_risk_top_results_hash(
        result.top_results
    ) == canonical_no_risk_top_results_hash(expected.top_results)
    assert result.top_results
    from trading.contexts.backtest.application.dto import canonical_no_risk_top_results_payload

    expected_payload = canonical_no_risk_top_results_payload(expected.top_results)
    actual_payload = canonical_no_risk_top_results_payload(result.top_results)
    assert actual_payload == expected_payload
    assert result.execution_context == expected.execution_context
    import json
    print(json.dumps({"fixed_input_full_parity": True, "expected": expected_payload,
                      "actual": actual_payload,
                      "context": result.execution_context.as_mapping()}, sort_keys=True))
    close_market_gateways()


def test_real_redis_pg_strategy_tail_and_audit(db):
    import os

    from redis import Redis

    from trading.contexts.market_data.adapters.outbound.config import (  # noqa: E501
        RedisHotCacheConfig,
        RedisStreamsConfig,
    )
    from trading.contexts.market_data.adapters.outbound.messaging.redis.redis_candle_hot_cache import (  # noqa: E501
        RedisCandleHotCache,  # noqa: E501
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.candle_repair_audit_repository import (  # noqa: E501
        PostgresCandleRepairAuditRepository,  # noqa: E501
    )
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.gateway import (
        PsycopgMarketDataPostgresGateway,  # noqa: E501
    )
    from trading.contexts.market_data.application.services.closed_candle_tail_provider import (
        MarketDataClosedCandleTailProvider,  # noqa: E501
    )

    if not os.environ.get("ROEHUB_S05_REDIS_PORT"):
        pytest.skip("explicit disposable Redis required")
    client = Redis(host="127.0.0.1", port=int(os.environ["ROEHUB_S05_REDIS_PORT"]))
    hot = RedisCandleHotCache(
        connection_config=RedisStreamsConfig(
            True, "127.0.0.1", 57918, 0, None, 1, 1, "per_instrument", "s05", 1, 100
        ),
        config=RedisHotCacheConfig(True, "s05tail", 24),
        environ={},
        redis_client=client,
    )
    g = market_gateway({"STRATEGY_PG_DSN": db}, workload="interactive")
    rows = [candle("TAILREPAIR", m) for m in range(3)]
    PostgresRawKlineWriter(g).write_1m(rows)
    assert hot.write_closed_1m(rows[0])

    class NoRest:
        def stream_1m(self, instrument_id, time_range):
            raise AssertionError("PG tail is complete; REST must not run")

    provider = MarketDataClosedCandleTailProvider(
        hot_cache=hot,
        canonical_reader=PostgresCanonicalCandleReader(g),
        rest_source=NoRest(),
        audit_repository=PostgresCandleRepairAuditRepository(
            gateway=PsycopgMarketDataPostgresGateway(dsn=db)
        ),
        clock=FixedClock(BASE + timedelta(minutes=4)),
    )
    result = provider.get_closed_1m_tail(
        instrument_id=instrument("TAILREPAIR"),
        instrument_key="binance:spot:TAILREPAIR",
        start_ts_open=UtcTimestamp(BASE),
        end_ts_open=UtcTimestamp(BASE + timedelta(minutes=3)),
        correlation_id="s05-real-tail",
    )
    assert len(result.candles) == 3
    assert [a.source for a in result.sources_attempted] == ["redis_hot_cache", "canonical_store"]
    with psycopg.connect(db) as c:
        audit = c.execute(
            "SELECT sources_attempted_json FROM market_data_candle_repair_events "
            "WHERE correlation_id='s05-real-tail'"
        ).fetchone()
        assert audit and audit[0][-1]["source"] == "canonical_store"
    client.close()
    close_market_gateways()


def test_hostname_tls_verify_full_and_wrong_identity(db):
    import os

    from psycopg.conninfo import make_conninfo

    from trading.contexts.market_data.adapters.outbound.persistence.postgres.ingestion_gateway import (  # noqa: E501
        IngestionGateway,  # noqa: E501
    )
    from trading.contexts.market_data.application.services.ingestion_retry import (
        TemporaryIngestionError,  # noqa: E501
    )

    root = os.environ.get("ROEHUB_S05_TLS_ROOT")
    if not root:
        pytest.skip("explicit disposable TLS certificate required")
    good = make_conninfo(db, host="localhost", sslmode="verify-full", sslrootcert=root)
    for _ in range(2):
        g = IngestionGateway(good, deadline_s=2)
        try:
            assert g.run(
                lambda c: c.execute(
                    "SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid()"
                ).fetchone()
            )[0]
        finally:
            g.close()
    bad = IngestionGateway(make_conninfo(good, host="127.0.0.1"), deadline_s=1)
    try:
        with pytest.raises(TemporaryIngestionError):
            bad.run(lambda c: c.execute("SELECT 1"))
    finally:
        bad.close()


def test_cli_parquet_and_rl_exports_use_pg(db, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from apps.cli.commands.backfill_1m import Backfill1mCli
    from scripts.rl_trading import stage05_roehub_dataset_builder as dataset
    from scripts.rl_trading import stage06_dataset_qa_session_extractor as sessions
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.research_export import (  # noqa: E501
        coverage_row,
        feature_rows,
    )

    parquet = tmp_path / "source.parquet"
    records = [
        dict(
            market_id=1,
            symbol="PARQUETPG",
            ts_open=BASE + timedelta(minutes=m),
            ts_close=BASE + timedelta(minutes=m + 1),
            open=10.0,
            high=20.0,
            low=1.0,
            close=11.0,
            volume_base=2.0,
            volume_quote=22.0,
            trades_count=5,
            taker_buy_volume_base=None,
            taker_buy_volume_quote=None,
        )
        for m in range(3)
    ]
    pq.write_table(pa.Table.from_pylist(records), parquet)
    assert (
        Backfill1mCli(environ={"STRATEGY_PG_DSN": db, "CH_HOST": "unreachable.invalid"}).run(
            [
                "--market-id",
                "1",
                "--symbol",
                "PARQUETPG",
                "--start",
                BASE.isoformat(),
                "--end",
                (BASE + timedelta(minutes=3)).isoformat(),
                "--parquet",
                str(parquet),
                "--batch-size",
                "None",
            ]
        )
        == 0
    )
    g = market_gateway({"STRATEGY_PG_DSN": db}, workload="bulk")
    kwargs: dict[str, Any] = dict(
        gateway=g, market_id=1, symbol="PARQUETPG", start=BASE, end=BASE + timedelta(minutes=3)
    )
    rows = feature_rows(**kwargs)
    assert isinstance(rows.data, np.memmap) and not rows.data.flags.writeable
    assert len(rows) == 3 and rows[0]["trades_count_i64"] == 5
    batch = dataset._read_candle_batch(**kwargs, database="market_data", expected_rows=3)
    assert list(batch.close_f32) == [11.0, 11.0, 11.0]
    slab = sessions._read_slab_from_postgresql(**kwargs, database="market_data")
    assert slab is not None
    assert coverage_row(**kwargs)["distinct_minutes"] == 3
    close_market_gateways()


def test_sync_cli_large_whitelist_and_disabled_semantics(db, tmp_path, monkeypatch):
    from apps.cli.commands.sync_instruments import SyncInstrumentsCli

    whitelist = tmp_path / "whitelist.csv"
    whitelist.write_text(
        "market_id,symbol,is_enabled\n"
        + "".join(f"1,S05META{k}," + ("0" if k == 1000 else "1") + "\n" for k in range(1001))
    )
    monkeypatch.setenv("STRATEGY_PG_DSN", db)
    monkeypatch.setenv("CH_HOST", "unreachable.invalid")
    assert SyncInstrumentsCli().run(["--whitelist", str(whitelist), "--report-format", "json"]) == 0
    assert SyncInstrumentsCli().run(["--whitelist", str(whitelist), "--report-format", "json"]) == 0
    with psycopg.connect(db) as c:
        assert (
            one(c.execute(
                "SELECT count(*) FROM market_data.ref_instruments WHERE symbol LIKE 'S05META%%'"
            ))[0]
            == 1001
        )
        assert c.execute(
            "SELECT status,is_tradable FROM market_data.ref_instruments WHERE symbol='S05META1000'"
        ).fetchone() == ("DISABLED", 0)
    close_market_gateways()


def test_additive_repair_label_migration_preserves_audit_and_legacy_target_rejected(db):
    from pathlib import Path

    from psycopg import sql

    from trading.contexts.backtest_artifacts.application.services.management import SourceTarget

    with psycopg.connect(db) as c:
        before = c.execute(
            """INSERT INTO market_data_candle_repair_events
            (event_id,correlation_id,market_id,symbol,instrument_key,range_start_ts_open,
             range_end_ts_open,status,sources_attempted_json,restored_ts_opens_json,
             missing_ts_opens_json,error_code,created_at)
            VALUES(gen_random_uuid(),'s05-migrate-label',1,'MIGRATE','binance:spot:MIGRATE',
             %s,%s,'failed','[{"source":"clickhouse","status":"failed","error_code":"clickhouse_exception"}]',
             '[]','[]','clickhouse_exception',%s) RETURNING event_id,created_at""",
            (BASE, BASE + timedelta(minutes=1), BASE),
        ).fetchone()
        assert before
        c.execute(
            sql.SQL(
                cast(
                    LiteralString,
                    Path("migrations/postgres/0039_market_data_consumer_contract_v1.sql").read_text(),
                )
            )
        )
        after = c.execute(
            "SELECT event_id,created_at,error_code,sources_attempted_json "
            "FROM market_data_candle_repair_events WHERE event_id=%s",
            (before[0],),
        ).fetchone()
        assert after is not None
        assert after[:2] == before and after[2] == "canonical_store_exception"
        assert after[3][-1]["source"] == "canonical_store"
    with pytest.raises(ValueError):
        SourceTarget.model_validate(
            dict(
                history=dict(kind="range", start_utc=BASE, end_utc=BASE + timedelta(minutes=1)),
                watermark=BASE + timedelta(minutes=1),
                revision="content_unverified",
                compute_identity="v2",
                funding_identity="spot",
            )
        )


def test_s05_owned_server_restart_preserves_committed_source(db):
    import os
    import subprocess
    import time

    name = os.environ.get("ROEHUB_S05_DISPOSABLE_CONTAINER")
    if not name:
        pytest.skip("explicit S05 disposable container required")
    assert name == "roehub-s05-consumers"
    assert (
        subprocess.check_output(
            ["docker", "inspect", "--format", '{{index .Config.Labels "roehub.disposable"}}', name],
            text=True,
        ).strip()
        == "s05"
    )
    g = market_gateway(
        {"STRATEGY_PG_DSN": db.replace("127.0.0.1", "localhost")}, workload="ingestion"
    )
    writer = PostgresRawKlineWriter(g)
    writer.write_1m([candle("S05RESTART", 0)])
    with psycopg.connect(db) as c:
        before = c.execute(
            "SELECT generation,content_revision "
            "FROM market_data.instrument_generations "
            "WHERE symbol='S05RESTART'"
        ).fetchone()
    uncommitted = psycopg.connect(db)
    uncommitted.execute("UPDATE market_data.ref_market SET count_symbols=123456 WHERE market_id=4")
    try:
        subprocess.run(
            ["docker", "restart", "-t", "1", name], check=True, capture_output=True, timeout=20
        )
        with pytest.raises(psycopg.OperationalError):
            uncommitted.commit()
    finally:
        uncommitted.close()
    for _ in range(50):
        try:
            with psycopg.connect(db) as c:
                assert (
                    c.execute(
                        "SELECT generation,content_revision "
                        "FROM market_data.instrument_generations "
                        "WHERE symbol='S05RESTART'"
                    ).fetchone()
                    == before
                )
                assert (
                    one(c.execute(
                        "SELECT count(*) FROM market_data.ref_market "
                        "WHERE market_id=4 AND count_symbols=123456"
                    ))[0]
                    == 0
                )
            break
        except psycopg.OperationalError:
            time.sleep(0.1)
    else:
        pytest.fail("owned PG did not restart")
    writer.write_1m([candle("S05RESTART", 1)])
    close_market_gateways()


def test_managed_ready_projection_and_legacy_dispatch_are_revision_fenced(db, monkeypatch):
    from uuid import uuid4

    from tests.real_infra.backtest import test_artifact_management as fixture
    from tests.unit.contexts.backtest_artifacts.test_management_profiles import catalog
    from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.dispatch_repository import (  # noqa: E501
        PostgresArtifactDispatchRepository,
    )
    from trading.contexts.backtest_artifacts.application.services.management import (
        Coordinate,
        History,
        ManagementError,
        PolicyItem,
        SourceTarget,
    )

    monkeypatch.setattr(fixture, "DSN", db)
    setup_fixture = cast(Any, fixture.managed).__wrapped__(cast(Any, catalog).__wrapped__())
    m = next(setup_fixture)
    try:
        coordinate = Coordinate(exchange="binance", market_type="spot", symbol="BTCUSDT")
        fixture.item(m)
        identity = PostgresPublicationFence(m["db"]).capture(coordinate)
        source = SourceTarget(
            **identity,
            history=History(kind="range", start_utc=START, end_utc=START + timedelta(minutes=180)),
            watermark=START + timedelta(minutes=180),
            revision="v2:s05",
            compute_identity="native-base-v2",
            funding_identity="spot",
        )
        item = PolicyItem(
            coordinate=coordinate,
            expected_version=0,
            profile=m["catalog"].default_profile(),
            schedule=m["schedule"],
            source=source,
        )
        request = fixture.submit(m, (item,))["items"][0]["request_id"]
        build = fixture.claim(m, request)
        assert build
        g = m["source_gateway"]
        PostgresRawKlineWriter(g).write_1m([candle("BTCUSDT", 0, close=17, version=5, base=START)])
        with pytest.raises(ManagementError, match="source_revision_changed"):
            m["repo"].finish(
                build=build,
                outcome="ready",
                publication=dict(slot="slot_a", generation=1, manifest_sha256="a" * 64),
            )
        assert (
            m["db"].fetch_one(
                query="SELECT state FROM backtest_artifact_builds WHERE build_id=%(id)s",
                parameters={"id": build["build_id"]},
            )["state"]
            == "running"
        )
        m["repo"].finish(build=build, outcome="error", reason="source_revision_changed")
        request2 = fixture.submit(m, (item,), organization_id=m["other"])["items"][0]["request_id"]
        m["db"].execute(
            query=(
                "UPDATE backtest_artifact_requests SET snapshot=jsonb_set(snapshot,'{source}',"
                "(snapshot->'source')-'source_generation'-'content_revision'-'funding_revision') "
                "WHERE request_id=%(r)s"
            ),
            parameters={"r": request2},
        )
        row = m["db"].fetch_one(
            query="SELECT * FROM backtest_artifact_requests WHERE request_id=%(r)s",
            parameters={"r": request2},
        )
        dispatch = PostgresArtifactDispatchRepository(
            gateway=m["db"], catalog=m["catalog"], access=fixture.Access()
        )
        assert (
            dispatch.claim_group(
                installation=m["installation"],
                request=row,
                owner=uuid4(),
                incarnation=uuid4(),
                reservation=1024,
            )
            is None
        )
        assert m["db"].fetch_one(
            query="SELECT state,reason FROM backtest_artifact_requests WHERE request_id=%(r)s",
            parameters={"r": request2},
        ) == {"state": "error", "reason": "legacy_source_target_invalid"}
    finally:
        close_market_gateways()
        try:
            next(setup_fixture)
        except StopIteration:
            pass


def test_catalog_last_candle_is_nullable_before_first_ingestion(db):
    from trading.contexts.market_data.adapters.outbound.persistence.postgres.catalog_coverage_reader import (  # noqa: E501
        PostgresCatalogCoverageReader,
    )

    gateway = market_gateway({"STRATEGY_PG_DSN": db}, workload="reference")
    reader = PostgresCatalogCoverageReader(gateway)
    assert reader.latest(market_id=1, symbol="S08FRESH", before=BASE + timedelta(days=1)) is None
    PostgresRawKlineWriter(gateway).write_1m([candle("S08FRESH", 0)])
    assert reader.latest(market_id=1, symbol="S08FRESH", before=BASE + timedelta(days=1)) == BASE
