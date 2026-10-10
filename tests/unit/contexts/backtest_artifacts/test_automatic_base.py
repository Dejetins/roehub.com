"""Independent OHLCV expectations and actual schema-aware production reader proof."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2 import (
    test_artifact_precompute_runner_v2 as n,
)
from tests.unit.contexts.backtest.application.services.v2.test_artifact_slot_publisher_v2 import (
    AtomicArtifactCurrentPointerWriterV2,
    _FakeJobRepository,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs.artifact_context_resolver import (
    FilesystemBacktestArtifactContextResolver,
)
from trading.contexts.backtest.application.dto import BacktestCoordinates
from trading.contexts.backtest_artifacts.application.services.automatic_base import (
    AutomaticBasePublisher,
    freeze_source,
)
from trading.contexts.backtest_artifacts.application.services.management import ManagementError
from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (
    BacktestArtifactSlotPublisherV2,
)
from trading.contexts.market_data.application.dto import CanonicalCandleBatch1m
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp

START = datetime(2026, 1, 1, tzinfo=UTC)
DURATIONS = {
    "1m": 1,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "1d": 1440,
    "2d": 2880,
    "3d": 4320,
}


class Source:
    def __init__(self, minutes, missing=(), offset=0):
        ix = np.array([i for i in range(minutes) if i not in missing], dtype=np.int64)
        self.opens = int(START.timestamp() * 1000) + ix * 60000
        self.values = np.array(
            [
                [i + 100 + offset, i + 103 + offset, i + 99 + offset, i + 102 + offset, 2]
                for i in ix
            ],
            dtype=np.float32,
        ).reshape(-1, 5)
        self.calls = []

    def read_1m_arrays(self, instrument_id, time_range):
        self.calls.append(time_range)
        mask = (self.opens >= int(time_range.start.value.timestamp() * 1000)) & (
            self.opens < int(time_range.end.value.timestamp() * 1000)
        )
        return CanonicalCandleBatch1m(self.opens[mask], self.opens[mask] + 60000, self.values[mask])

    def read_1m(self, instrument_id, time_range):
        raise AssertionError("must use columnar source")


def setup(tmp_path, repo=None, coordinates=None):
    f = n.build_artifact_precompute_fixture_v2(tmp_path=tmp_path, coordinates=coordinates)
    f.loader.resolve_current_pointer_path(f.coordinates).unlink()
    runner = n.BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=f.runtime_settings,
        artifact_loader=f.loader,
        canonical_candle_reader=Source(1),
    )
    pub = BacktestArtifactSlotPublisherV2(
        artifact_loader=f.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=f.builder),
        job_repository=cast(
            Any, repo or _FakeJobRepository()
        ),  # Deliberately partial publisher fake.
    )
    return f, AutomaticBasePublisher(runner, pub)


def frozen(f, source, minutes, start=START, chunk=20000):
    return freeze_source(
        reader=source,
        coordinates=f.coordinates,
        time_range=TimeRange(UtcTimestamp(start), UtcTimestamp(START + timedelta(minutes=minutes))),
        chunk_minutes=chunk,
    )


@pytest.mark.parametrize(
    "minutes,missing", [(1, ()), (14, ()), (15, ()), (180, (31,)), (10080, (1007,))]
)
def test_native_all_timeframes_and_subset_readers(tmp_path, minutes, missing):
    f, service = setup(tmp_path)
    src = Source(minutes, missing)
    snapshot = frozen(f, src, minutes, chunk=97)
    result = service.publish(
        coordinates=f.coordinates, source=snapshot, generated_at=START + timedelta(days=20)
    )
    assert result.publication is not None
    pointer = result.publication.published_pointer
    root = f.loader.resolve_slot_manifest_path(f.coordinates, pointer.active_slot).parent
    manifest = f.loader.load_slot_manifest(f.coordinates, pointer.active_slot)
    assert manifest.schema_version == 2 and manifest.hit_times is None
    assert not manifest.signals.manifests
    assert len(result.timeframes) == 11
    seen = []
    for tf, size in DURATIONS.items():
        duration = size * 60000
        expected = []
        for bucket in sorted(set(src.opens // duration)):
            idx = np.flatnonzero(src.opens // duration == bucket)
            if len(idx) == size and src.opens[idx[0]] == bucket * duration:
                data = src.values[idx]
                expected.append(
                    (
                        bucket * duration,
                        [
                            data[0, 0],
                            max(data[:, 1]),
                            min(data[:, 2]),
                            data[-1, 3],
                            sum(data[:, 4]),
                        ],
                    )
                )
        if not expected:
            assert result.timeframes[tf]["state"] == "insufficient_history"
            continue
        seen.append(tf)
        section = next(p for p in manifest.prices if p.timeframe == tf)
        opens = np.load(root / section.open_time.path)
        values = np.load(root / section.ohlcv.path)
        np.testing.assert_array_equal(opens, [row[0] for row in expected])
        np.testing.assert_allclose(values, [row[1] for row in expected])
        if tf != "1m":
            mapping = next(p for p in manifest.mappings if p.timeframe == tf)
            oi = np.load(root / mapping.bar_open_1m_idx.path)
            ci = np.load(root / mapping.bar_close_1m_idx.path)
            np.testing.assert_array_equal(src.opens[oi], opens)
            np.testing.assert_array_equal(src.opens[ci] + 60000, opens + duration)
    assert tuple(p.timeframe for p in manifest.prices) == tuple(seen)
    context = FilesystemBacktestArtifactContextResolver(f.loader).resolve_context(
        coordinates=BacktestCoordinates(exchange="binance", market_type="spot", symbol="BTCUSDT")
    )
    assert context.artifact_manifest_hash == pointer.manifest_sha256
    assert result.state == ("ready" if len(seen) == 11 else "partial")
    assert not list(root.rglob("signals.npy")) and not (root / "hit_times").exists()


def test_empty_no_publication_and_frozen_payload(tmp_path):
    f, service = setup(tmp_path)
    src = Source(0)
    result = service.publish(
        coordinates=f.coordinates, source=frozen(f, src, 30), generated_at=START + timedelta(days=1)
    )
    assert result.state == "waiting_for_source" and result.publication is None
    assert not f.loader.resolve_current_pointer_path(f.coordinates).exists()
    src = Source(30)
    snapshot = frozen(f, src, 30)
    src.values[:] = 99
    assert snapshot.batch.ohlcv_f32[0, 0] == 100
    with pytest.raises(ValueError):
        snapshot.batch.ohlcv_f32[0, 0] = 22


def test_changes_during_snapshot_and_budget_do_not_publish(tmp_path):
    f, service = setup(tmp_path)
    src = Source(30)
    original = src.read_1m_arrays

    def changed(instrument_id, time_range):
        if src.calls:
            src.values[0, 0] += 1
        return original(instrument_id, time_range)

    src.read_1m_arrays = changed
    with pytest.raises(ManagementError, match="source_changed"):
        frozen(f, src, 30)
    with pytest.raises(ManagementError, match="resource_conflict"):
        freeze_source(
            reader=src,
            coordinates=f.coordinates,
            time_range=TimeRange(UtcTimestamp(START), UtcTimestamp(START + timedelta(days=2))),
            max_minutes=30,
        )


def test_historical_extension_and_correction_replace_all_source(tmp_path):
    f, service = setup(tmp_path)
    first = service.publish(
        coordinates=f.coordinates,
        source=frozen(f, Source(180), 180, start=START + timedelta(minutes=60)),
        generated_at=START + timedelta(days=1),
    )
    second = service.publish(
        coordinates=f.coordinates,
        source=frozen(f, Source(180, offset=5), 180),
        generated_at=START + timedelta(days=1),
    )
    assert first.source_sha256 != second.source_sha256
    assert second.timeframes["1m"]["start_ms"] == int(START.timestamp() * 1000)
    assert second.publication is not None
    assert second.publication.published_pointer.slot_generation == 2


def test_real_preflight_and_array_reader_gate_for_missing_timeframe(tmp_path):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_backtest_preflight_service as pre,  # noqa: E501
    )
    from trading.contexts.backtest.adapters.outbound.artifacts_fs.artifact_array_loader import (
        FilesystemBacktestArtifactArrayLoader,  # noqa: E501
    )
    from trading.contexts.backtest.application.services.v2.preflight import (
        BacktestPreflightRejected,  # noqa: E501
    )

    f, service = setup(tmp_path)
    service.publish(
        coordinates=f.coordinates,
        source=frozen(f, Source(180), 180),
        generated_at=START + timedelta(days=1),
    )
    resolver = FilesystemBacktestArtifactContextResolver(f.loader)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=f.loader)
    preflight = replace(
        pre._service(),
        artifact_context_resolver=resolver,
        artifact_array_loader=loader,
        indicator_grid_builder=n._signal_grid_builder_v2(),
    )
    request = pre._valid_request()
    request["time_range"] = {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T03:00:00Z"}
    result = preflight.execute(request)
    metadata = resolver.resolve_context(
        coordinates=BacktestCoordinates(exchange="binance", market_type="spot", symbol="BTCUSDT")
    )
    context = loader.resolve_context(
        coordinates=BacktestCoordinates(exchange="binance", market_type="spot", symbol="BTCUSDT"),
        artifact_metadata=metadata,
    )
    assert loader.load_price_arrays(context=context, timeframe="15m").ohlcv.shape == (12, 5)
    assert len(loader.load_mapping_arrays(context=context, timeframe="15m").bar_open_1m_idx) == 12
    assert result.input_recipe is not None
    request["timeframe"] = "1d"
    with pytest.raises(BacktestPreflightRejected) as error:
        preflight.execute(request)
    assert error.value.error_code == "backtest.artifacts_unavailable"


def test_futures_funding_unavailable_preserves_pointer_and_snapshot_is_read_once(tmp_path):
    from trading.contexts.backtest_artifacts.application.services.automatic_base import (
        BasePrepublicationError,  # noqa: E501
    )
    from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
        ArtifactCoordinatesV2,  # noqa: E501
    )
    from trading.contexts.market_data.application.ports.stores.funding_rate_coverage_reader import (
        FundingRateArtifactRecord,  # noqa: E501
        FundingRateCoverageSnapshot,
    )

    f, service = setup(
        tmp_path,
        coordinates=ArtifactCoordinatesV2(
            exchange="binance", market_type="futures", symbol="BTCUSDT"
        ),
    )
    source = frozen(f, Source(180), 180)

    class Funding:
        calls = 0
        status: Literal["ready", "unavailable"] = "ready"

        def read_funding_coverage(self, *, instrument_id, time_range):
            self.calls += 1
            return FundingRateCoverageSnapshot(
                self.status,
                "degraded_with_warning",
                time_range,
                UtcTimestamp(START),
                UtcTimestamp(START),
                1,
                1,
                0,
                () if self.status == "ready" else ("provider_missing",),
                (
                    FundingRateArtifactRecord(
                        instrument_id,
                        "binance:futures:BTCUSDT",
                        UtcTimestamp(START),
                        0.0001,
                        100.0,
                        480,
                        0,
                    ),
                ),
            )

    funding = Funding()
    service = replace(service, runner=replace(service.runner, funding_rate_coverage_reader=funding))
    result = service.publish(
        coordinates=f.coordinates, source=source, generated_at=START + timedelta(days=1)
    )
    assert funding.calls == 1
    path = f.loader.resolve_current_pointer_path(f.coordinates)
    old = path.read_bytes()
    funding.status = "unavailable"
    with pytest.raises(BasePrepublicationError, match="funding_unavailable"):
        service.publish(
            coordinates=f.coordinates, source=source, generated_at=START + timedelta(days=1)
        )
    assert old == path.read_bytes()
    assert result.source_sha256 != source.sha256

    class EmptyFunding:
        def read_funding_coverage(self, *, instrument_id, time_range):
            return FundingRateCoverageSnapshot(
                "ready", "degraded_with_warning", time_range, None, None, 0, 0, 0, (), ()
            )

    service = replace(
        service, runner=replace(service.runner, funding_rate_coverage_reader=EmptyFunding())
    )
    with pytest.raises(BasePrepublicationError, match="funding_insufficient_history"):
        service.publish(
            coordinates=f.coordinates, source=source, generated_at=START + timedelta(days=1)
        )
    assert old == path.read_bytes()


def test_storage_admission_counts_retained_files_and_rejects_symlinks(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace

    from trading.contexts.backtest_artifacts.adapters.outbound.artifacts_fs.base_storage_admission import (  # noqa: E501
        BaseStorageAdmission,  # noqa: E501
    )

    root = tmp_path / "root"
    root.mkdir()
    (root / "retained").write_bytes(b"x" * 1024)
    check = BaseStorageAdmission(root, "primary", 1, 1_073_741_824 + 2048)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))
    check({"root_id": "primary", "root_revision": 1}, 1024)
    with pytest.raises(ManagementError, match="resource_conflict"):
        check({"root_id": "primary", "root_revision": 1}, 1025)
    (root / "link").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ManagementError, match="untrusted_root"):
        check({"root_id": "primary", "root_revision": 1}, 1)


def test_reconciliation_consumes_batched_bounds_without_fanout_and_keeps_empty_cursor():
    from types import SimpleNamespace
    from uuid import uuid4

    from trading.contexts.backtest_artifacts.application.services.base_source_reconciliation import (  # noqa: E501
        BaseSourceReconciler,
        StoredSourceObservation,
    )
    from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol

    observed, cutoffs = [], []
    instrument = InstrumentId(MarketId(1), Symbol('BTCUSDT'))
    row = StoredSourceObservation(instrument, UtcTimestamp(START), UtcTimestamp(START))

    def page(**kw):
        cutoffs.append(kw['before'])
        return (row,)

    def forbidden(*args, **kwargs):
        raise AssertionError('canonical fanout forbidden')

    reconciler = BaseSourceReconciler(
        uuid4(), cast(Any, SimpleNamespace(bounds_1m=forbidden,distinct_ts_opens=forbidden)),
        cast(Any, SimpleNamespace(observation_page=page,observation=lambda **kw: row)),
        cast(Any, SimpleNamespace(observe_source=lambda **kw: observed.append(kw))),
    )
    now = START + timedelta(minutes=2,seconds=17)
    assert reconciler.reconcile_page(after=None, now=now,limit=1) == (1,'BTCUSDT')
    assert cutoffs == [UtcTimestamp(START+timedelta(minutes=2))]
    assert observed[0]['source'].history.end_utc == START+timedelta(minutes=1)
    row = StoredSourceObservation(instrument,None,None)
    assert reconciler.reconcile_page(after=None, now=now,limit=1) == (1,'BTCUSDT')
    assert len(observed)==1
    with pytest.raises(ValueError, match='utc_required'):
        reconciler.reconcile_page(after=None,now=now.replace(tzinfo=None))
