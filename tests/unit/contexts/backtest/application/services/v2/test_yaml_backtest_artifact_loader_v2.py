from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, cast

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    ArtifactPrecomputeFixtureV2,
    SyntheticArtifactStoreV2,
    build_artifact_precompute_fixture_v2,
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.adapters.outbound import (
    AtomicArtifactCurrentPointerWriterV2,
    BacktestArtifactPathBuilderV2,
)
from trading.contexts.backtest.application.ports import BacktestJobRepository
from trading.contexts.backtest.application.services import (
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
    ARTIFACT_PRICE_TIMEFRAMES_V2,
    SIGNAL_FEATURE_NAMES_V2,
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCoordinatesV2,
    BacktestArtifactPrecomputeRunnerV2,
    BacktestArtifactSlotPublisherV2,
    YamlBacktestArtifactLoaderV2,
)
from trading.contexts.market_data.application.dto import (
    CandleWithMeta,
    CanonicalCandleBatch1m,
)
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)

_PRECOMPUTE_BASE_TIME_UTC = datetime(2026, 3, 26, 0, 0, tzinfo=timezone.utc)
_PRECOMPUTE_INTEGRATION_MINUTES_V2 = 3 * 24 * 60


class _PrecomputeCanonicalReaderForLoaderTest:
    """
    Deterministic in-memory canonical reader used by loader integration coverage.

    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/
        test_yaml_backtest_artifact_loader_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    def __init__(self, *, rows: tuple[CandleWithMeta, ...]) -> None:
        """
        Store deterministic canonical rows for later `read_1m(...)` filtering.

        Args:
            rows: Full in-memory canonical candle sequence available to the fake reader.
        Returns:
            None.
        Assumptions:
            Loader integration only needs deterministic source rows, not instrument branching.
        Raises:
            None.
        Side Effects:
            Stores the rows in memory for later range filtering.
        """
        self._rows = rows

    def read_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        """
        Return rows whose `ts_open` belongs to the requested half-open range.

        Args:
            instrument_id: Ignored shared-kernel identity passed by the production runner.
            time_range: Source reread window requested by the runner.
        Returns:
            Iterator[CandleWithMeta]: Filtered canonical candle iterator.
        Assumptions:
            Integration test validates loader compatibility, not instrument dispatch.
        Raises:
            None.
        Side Effects:
            None.
        """
        del instrument_id
        return iter(
            tuple(
                row
                for row in self._rows
                if time_range.start.value <= row.candle.ts_open.value < time_range.end.value
            )
        )

    def read_1m_arrays(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> CanonicalCandleBatch1m:
        """
        Return one strict columnar canonical batch for loader integration coverage.

        Args:
            instrument_id: Ignored shared-kernel identity passed by the production runner.
            time_range: Source reread window requested by the runner.
        Returns:
            CanonicalCandleBatch1m: Filtered canonical candle batch.
        Assumptions:
            Integration coverage needs deterministic arrays, not transport-specific row DTOs.
        Raises:
            None.
        Side Effects:
            None.
        """
        del instrument_id
        rows = tuple(
            row
            for row in self._rows
            if time_range.start.value <= row.candle.ts_open.value < time_range.end.value
        )
        if len(rows) == 0:
            return CanonicalCandleBatch1m(
                open_time_ms=np.empty(0, dtype=np.int64),
                close_time_ms=np.empty(0, dtype=np.int64),
                ohlcv_f32=np.empty((0, 5), dtype=np.float32),
            )
        return CanonicalCandleBatch1m(
            open_time_ms=np.ascontiguousarray(
                [int(row.candle.ts_open.value.timestamp() * 1000) for row in rows],
                dtype=np.int64,
            ),
            close_time_ms=np.ascontiguousarray(
                [int(row.candle.ts_close.value.timestamp() * 1000) for row in rows],
                dtype=np.int64,
            ),
            ohlcv_f32=np.ascontiguousarray(
                [
                    [
                        float(row.candle.open),
                        float(row.candle.high),
                        float(row.candle.low),
                        float(row.candle.close),
                        float(row.candle.volume_base),
                    ]
                    for row in rows
                ],
                dtype=np.float32,
            ),
        )


class _ZeroBlockingRepositoryForLoaderTest:
    """
    Fake job repository returning zero inactive-slot pins for loader+publisher smoke coverage.
    """

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        """
        Return zero blocking jobs for the explicit publish-guard query.

        Args:
            market_id: Canonical market id for the symbol under publish.
            symbol: Instrument symbol under publish.
            artifact_slot: Candidate inactive slot literal.
            artifact_manifest_hash: SHA-256 hash of the inactive slot root manifest.
        Returns:
            int: Always `0`.
        Assumptions:
            Loader smoke coverage exercises successful publish flow rather than pin-guard failure.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        del market_id, symbol, artifact_slot, artifact_manifest_hash
        return 0


@pytest.fixture()
def synthetic_artifact_store_v2(tmp_path: Path) -> SyntheticArtifactStoreV2:
    """
    Build a strict synthetic artifact store used by loader tests.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        SyntheticArtifactStoreV2: Deterministic strict artifact store fixture.
    Assumptions:
        Loader tests require valid root/signal/hit-times manifests and real `.npy` payloads.
    Raises:
        OSError: If the synthetic artifact tree cannot be created.
    Side Effects:
        Creates a temporary artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    return build_synthetic_artifact_store_v2(tmp_path=tmp_path)


def test_yaml_backtest_artifact_loader_v2_reads_current_and_strict_manifests(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify the loader reads strict root/signal/hit-times manifests via explicit paths.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Loader must expose typed manifest metadata without runtime schema inference.
    Raises:
        AssertionError: If typed loader outputs or resolved paths are incorrect.
    Side Effects:
        Reads deterministic YAML and `.npy` metadata from the synthetic artifact tree.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    store = synthetic_artifact_store_v2
    loader = store.loader
    current = loader.load_current_pointer(store.coordinates)
    root_manifest = loader.load_slot_manifest(store.coordinates, store.active_slot)
    active_manifest = loader.load_active_slot_manifest(store.coordinates)
    signal_manifest = loader.load_signal_manifest(
        store.coordinates,
        store.inactive_slot,
        "15m",
        "ma.ema",
    )
    signal_features_manifest = loader.load_signal_features_manifest(
        store.coordinates,
        store.inactive_slot,
        "15m",
        "ma.ema",
    )
    hit_times_manifest = loader.load_hit_times_manifest(
        store.coordinates,
        store.inactive_slot,
    )

    assert current.active_slot == store.active_slot
    assert current.slot_generation == 4
    assert root_manifest.slot == store.active_slot
    assert active_manifest.path == store.builder.slot_manifest_path(
        store.coordinates,
        store.active_slot,
    )
    assert tuple(item.timeframe for item in root_manifest.prices) == ("1m", "15m")
    assert root_manifest.signal_encoding.value_set == (-1, 0, 1)
    assert root_manifest.signals.manifests[0].indicator_id == "ma.ema"
    assert signal_manifest.rows_count == 2
    assert signal_manifest.signals.axis_order == ("variant", "time")
    assert signal_manifest.grid.variant_key_version == 1
    assert signal_manifest.signal_features is not None
    assert signal_features_manifest.rows_count == 2
    assert signal_features_manifest.features.axis_order == ("variant", "feature")
    assert signal_features_manifest.feature_names == SIGNAL_FEATURE_NAMES_V2
    assert hit_times_manifest.timeline_bar_count == 4
    assert hit_times_manifest.long_tp.array.axis_order == ("level", "time")

    explicit_root_manifest = loader.load_manifest_from_path(
        loader.resolve_slot_manifest_path(store.coordinates, store.inactive_slot),
        slot=store.inactive_slot,
    )
    explicit_signal_manifest = loader.load_signal_manifest_from_path(
        loader.resolve_signal_paths(
            store.coordinates,
            store.inactive_slot,
            "15m",
            "ma.ema",
        ).manifest,
        slot=store.inactive_slot,
    )
    explicit_signal_features_manifest = loader.load_signal_features_manifest_from_path(
        loader.resolve_signal_features_paths(
            store.coordinates,
            store.inactive_slot,
            "15m",
            "ma.ema",
        ).manifest,
        slot=store.inactive_slot,
    )
    explicit_hit_times_manifest = loader.load_hit_times_manifest_from_path(
        loader.resolve_hit_times_manifest_path(store.coordinates, store.inactive_slot),
        slot=store.inactive_slot,
    )
    hit_times_paths = loader.resolve_hit_times_paths(store.coordinates, store.inactive_slot)
    signal_features_paths = loader.resolve_signal_features_paths(
        store.coordinates,
        store.inactive_slot,
        "15m",
        "ma.ema",
    )

    assert explicit_root_manifest.slot_generation == 5
    assert explicit_signal_manifest.indicator_id == "ma.ema"
    assert explicit_signal_features_manifest.indicator_id == "ma.ema"
    assert explicit_hit_times_manifest.sentinel_index == 4
    assert (
        hit_times_paths.long_tp
        == store.builder.hit_times_paths(
            store.coordinates,
            store.inactive_slot,
        ).long_tp
    )
    assert (
        signal_features_paths.features
        == store.builder.signal_features_paths(
            store.coordinates,
            store.inactive_slot,
            "15m",
            "ma.ema",
        ).features
    )


def test_yaml_backtest_artifact_loader_v2_accepts_legacy_slot_without_signal_features(
    tmp_path: Path,
) -> None:
    """
    Verify the loader keeps reading legacy signal manifests that omit `signal_features`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Old published slots remain readable while the additive feature family is still optional.
    Raises:
        AssertionError: If the optional field becomes mandatory.
    Side Effects:
        Builds one synthetic legacy-style inactive slot under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        inactive_include_signal_features=False,
    )
    signal_manifest = store.loader.load_signal_manifest(
        store.coordinates,
        store.inactive_slot,
        "15m",
        "ma.ema",
    )

    assert signal_manifest.signal_features is None


def test_yaml_backtest_artifact_loader_v2_avoids_directory_scanning(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify loader reads and resolves manifests without directory scanning helpers.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Loader must stay on explicit deterministic paths in runtime-facing code paths.
    Raises:
        AssertionError: If one loader method relies on directory scanning.
    Side Effects:
        Temporarily replaces scanning helpers on `Path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    store = synthetic_artifact_store_v2
    monkeypatch.setattr(Path, "iterdir", _forbid_directory_scan)
    monkeypatch.setattr(Path, "glob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "rglob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "walk", _forbid_directory_scan)

    current = store.loader.load_current_pointer(store.coordinates)
    manifest = store.loader.load_active_slot_manifest(store.coordinates)
    price_paths = store.loader.resolve_price_paths(store.coordinates, current.active_slot, "1m")
    signal_paths = store.loader.resolve_signal_paths(
        store.coordinates,
        store.inactive_slot,
        "15m",
        "ma.ema",
    )
    mapping_paths = store.loader.resolve_mapping_paths(
        store.coordinates,
        store.inactive_slot,
        "15m",
    )
    hit_times_manifest_path = store.loader.resolve_hit_times_manifest_path(
        store.coordinates,
        store.inactive_slot,
    )

    assert manifest.slot == store.active_slot
    assert (
        price_paths.ohlcv
        == store.builder.price_paths(
            store.coordinates,
            store.active_slot,
            "1m",
        ).ohlcv
    )
    assert (
        signal_paths.signals
        == store.builder.signal_paths(
            store.coordinates,
            store.inactive_slot,
            "15m",
            "ma.ema",
        ).signals
    )
    assert (
        mapping_paths.bar_close_1m_idx
        == store.builder.mapping_paths(
            store.coordinates,
            store.inactive_slot,
            "15m",
        ).bar_close_1m_idx
    )
    assert hit_times_manifest_path == store.builder.hit_times_manifest_path(
        store.coordinates,
        store.inactive_slot,
    )


def test_yaml_backtest_artifact_loader_v2_loads_all_root_listed_signal_manifests_without_scanning(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify every root-manifest-listed signal manifest loads from explicit paths without scanning.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Root manifest already provides the canonical signal manifest catalog for runtime loading.
    Raises:
        AssertionError: If one listed signal manifest cannot be loaded explicitly.
    Side Effects:
        Temporarily replaces scanning helpers on `Path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    store = synthetic_artifact_store_v2
    loader = store.loader
    root_manifest = loader.load_slot_manifest(store.coordinates, store.active_slot)
    slot_root = store.builder.slot_root(store.coordinates, store.active_slot)
    monkeypatch.setattr(Path, "iterdir", _forbid_directory_scan)
    monkeypatch.setattr(Path, "glob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "rglob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "walk", _forbid_directory_scan)

    loaded = tuple(
        loader.load_signal_manifest_from_path(
            slot_root / entry.manifest_path,
            slot=store.active_slot,
        )
        for entry in root_manifest.signals.manifests
    )

    assert tuple((manifest.timeframe, manifest.indicator_id) for manifest in loaded) == tuple(
        (entry.timeframe, entry.indicator_id) for entry in root_manifest.signals.manifests
    )


def test_yaml_backtest_artifact_loader_v2_loads_runner_generated_rollup_manifest(
    tmp_path: Path,
) -> None:
    """
    Verify the loader parses a real R3-03 runner-generated root manifest with all price and
    mapping TFs.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Loader must remain schema-compatible with the root manifest written by the runner.
    Raises:
        AssertionError: If the loader cannot parse or order runner-generated rollup metadata.
    Side Effects:
        Builds one inactive-slot artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    fixture = build_artifact_precompute_fixture_v2(tmp_path=tmp_path, price_tail_bars_1m=2)
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_PrecomputeCanonicalReaderForLoaderTest(
            rows=_build_loader_canonical_rows_v2(bar_count=_PRECOMPUTE_INTEGRATION_MINUTES_V2)
        ),
    )

    runner.export_canonical_price_1m(
        _loader_request_v2(
            fixture=fixture,
            end_minute=_PRECOMPUTE_INTEGRATION_MINUTES_V2,
        )
    )
    manifest = fixture.loader.load_slot_manifest(fixture.coordinates, fixture.inactive_slot)
    three_day_paths = fixture.loader.resolve_price_paths(
        fixture.coordinates,
        fixture.inactive_slot,
        "3d",
    )
    fifteen_minute_mapping_paths = fixture.loader.resolve_mapping_paths(
        fixture.coordinates,
        fixture.inactive_slot,
        "15m",
    )

    assert tuple(item.timeframe for item in manifest.prices) == ARTIFACT_PRICE_TIMEFRAMES_V2
    assert tuple(item.timeframe for item in manifest.mappings) == ARTIFACT_MAPPING_TIMEFRAMES_V2
    assert manifest.prices[-1].timeframe == "3d"
    assert manifest.prices[-1].coverage.bar_count == 1
    assert (
        manifest.prices[-1].open_time.path
        == three_day_paths.open_time.relative_to(three_day_paths.open_time.parents[2]).as_posix()
    )
    assert manifest.mappings[0].bar_open_1m_idx.path == (
        fifteen_minute_mapping_paths.bar_open_1m_idx.relative_to(
            fifteen_minute_mapping_paths.bar_open_1m_idx.parents[2]
        ).as_posix()
    )


def test_yaml_backtest_artifact_loader_v2_rejects_invalid_pointer_shape(tmp_path: Path) -> None:
    """
    Verify loader fails fast when strict `current.yaml` misses required fields.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Strict pointer parsing rejects missing keys before runtime can pin the slot.
    Raises:
        AssertionError: If an invalid pointer document is accepted.
    Side Effects:
        Creates and reads one temporary invalid YAML document.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    builder = BacktestArtifactPathBuilderV2(root=tmp_path / "artifacts" / "backtest" / "v2")
    coordinates = ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )
    pointer_path = builder.current_pointer_path(coordinates)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text("schema_version: 1\n", encoding="utf-8")

    loader = YamlBacktestArtifactLoaderV2(path_resolver=builder)

    with pytest.raises(ValueError, match="active_slot"):
        loader.load_current_pointer(coordinates)


def test_yaml_backtest_artifact_loader_v2_reads_runner_built_published_prices_mappings_slot(
    tmp_path: Path,
) -> None:
    """
    Verify loader reads the active slot after a successful R3-04 prices+mappings publish.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Publish flow derives an explicit prices+mappings validation spec from a runtime config
        that still contains later-stage `signals/hit_times` targets.
    Raises:
        AssertionError: If loader cannot follow the switched `current.yaml` pointer.
    Side Effects:
        Builds and publishes one inactive slot under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        price_tail_bars_1m=2,
        validation_signal_artifacts=(("15m", "ma.ema"),),
        require_hit_times_manifest=True,
    )
    runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=_PrecomputeCanonicalReaderForLoaderTest(
            rows=_build_loader_canonical_rows_v2(bar_count=_PRECOMPUTE_INTEGRATION_MINUTES_V2)
        ),
    )
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=fixture.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=fixture.builder),
        job_repository=cast(BacktestJobRepository, _ZeroBlockingRepositoryForLoaderTest()),
        now_provider=lambda: datetime(2026, 3, 26, 3, 4, 5, tzinfo=timezone.utc),
    )

    publish_result = publisher.build_publish_prices_mappings_slot(
        request=_loader_request_v2(
            fixture=fixture,
            end_minute=_PRECOMPUTE_INTEGRATION_MINUTES_V2,
        ),
        precompute_runner=runner,
        validation_spec=fixture.runtime_config.to_prices_mappings_publish_validation_spec(),
    )
    current = fixture.loader.load_current_pointer(fixture.coordinates)
    active_manifest = fixture.loader.load_active_slot_manifest(fixture.coordinates)

    assert current.active_slot == fixture.inactive_slot
    assert current.manifest_sha256 == publish_result.build_result.manifest_sha256
    assert active_manifest.path == publish_result.build_result.manifest_path
    assert tuple(item.timeframe for item in active_manifest.prices) == ARTIFACT_PRICE_TIMEFRAMES_V2
    assert (
        tuple(item.timeframe for item in active_manifest.mappings) == ARTIFACT_MAPPING_TIMEFRAMES_V2
    )


@pytest.mark.parametrize(
    ("payload_text", "error_pattern"),
    (
        (
            "\n".join(
                (
                    "schema_version: 2",
                    "active_slot: slot_a",
                    "slot_generation: 1",
                    'asof_date: "2026-03-24"',
                    'manifest_sha256: "' + ("a" * 64) + '"',
                    'published_at_utc: "2026-03-24T02:00:00Z"',
                )
            ),
            "schema_version",
        ),
        (
            "\n".join(
                (
                    "schema_version: 1",
                    "active_slot: slot_a",
                    "slot_generation: 1",
                    'asof_date: "2026-03-24"',
                    'manifest_sha256: "not-a-sha"',
                    'published_at_utc: "2026-03-24T02:00:00Z"',
                )
            ),
            "manifest_sha256",
        ),
        (
            "\n".join(
                (
                    "schema_version: 1",
                    "active_slot: slot_a",
                    "slot_generation: 1",
                    'asof_date: "2026-03-24"',
                    'manifest_sha256: "' + ("a" * 64) + '"',
                    'published_at_utc: "2026-03-24T02:00:00+00:00"',
                )
            ),
            "published_at_utc",
        ),
    ),
)
def test_yaml_backtest_artifact_loader_v2_rejects_invalid_strict_pointer_fields(
    tmp_path: Path,
    payload_text: str,
    error_pattern: str,
) -> None:
    """
    Verify strict pointer parsing rejects malformed schema/hash/timestamp literals.

    Args:
        tmp_path: pytest temporary path fixture.
        payload_text: Invalid strict pointer YAML payload.
        error_pattern: Stable error substring expected from strict validation.
    Returns:
        None.
    Assumptions:
        Pointer identity must remain fully validated before any publish/runtime usage.
    Raises:
        AssertionError: If invalid strict pointer fields are accepted.
    Side Effects:
        Creates and reads one temporary invalid YAML document.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    builder = BacktestArtifactPathBuilderV2(root=tmp_path / "artifacts" / "backtest" / "v2")
    coordinates = ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )
    pointer_path = builder.current_pointer_path(coordinates)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(payload_text + "\n", encoding="utf-8")

    loader = YamlBacktestArtifactLoaderV2(path_resolver=builder)

    with pytest.raises(ValueError, match=error_pattern):
        loader.load_current_pointer(coordinates)


def _loader_request_v2(
    *,
    fixture: ArtifactPrecomputeFixtureV2,
    end_minute: int,
) -> ArtifactCanonicalPriceExportRequestV2:
    """
    Build one deterministic R3-02 precompute request for loader integration coverage.

    Args:
        fixture: Strict precompute fixture providing artifact coordinates.
        end_minute: Exclusive end minute offset relative to `_PRECOMPUTE_BASE_TIME_UTC`.
    Returns:
        ArtifactCanonicalPriceExportRequestV2: Explicit runner request DTO.
    Assumptions:
        Loader integration uses the same aligned UTC base as the runner unit tests.
    Raises:
        ValueError: If request identity violates strict export contracts.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return ArtifactCanonicalPriceExportRequestV2(
        coordinates=fixture.coordinates,
        time_range=TimeRange(
            start=UtcTimestamp(_PRECOMPUTE_BASE_TIME_UTC),
            end=UtcTimestamp(_PRECOMPUTE_BASE_TIME_UTC + timedelta(minutes=end_minute)),
        ),
        asof_date="2026-03-26",
        generated_at_utc="2026-03-26T03:00:00Z",
    )


def _build_loader_canonical_rows_v2(*, bar_count: int) -> tuple[CandleWithMeta, ...]:
    """
    Build deterministic aligned canonical rows for loader integration coverage.

    Args:
        bar_count: Number of contiguous `1m` bars to build from `_PRECOMPUTE_BASE_TIME_UTC`.
    Returns:
        tuple[CandleWithMeta, ...]: Deterministic canonical candle rows.
    Assumptions:
        Loader integration needs only stable aligned rows, not complex update scenarios.
    Raises:
        ValueError: If one constructed candle violates shared-kernel invariants.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/market_data/application/dto/candle_with_meta.py
    """
    return tuple(
        _build_loader_canonical_row_v2(bar_index=bar_index) for bar_index in range(bar_count)
    )


def _build_loader_canonical_row_v2(*, bar_index: int) -> CandleWithMeta:
    """
    Build one deterministic aligned canonical `1m` row for loader integration tests.

    Args:
        bar_index: Minute offset relative to `_PRECOMPUTE_BASE_TIME_UTC`.
    Returns:
        CandleWithMeta: Deterministic canonical candle row.
    Assumptions:
        The integration test only needs monotonically increasing aligned minute candles.
    Raises:
        ValueError: If one constructed candle violates shared-kernel invariants.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/shared_kernel/primitives/candle.py
    """
    ts_open = _PRECOMPUTE_BASE_TIME_UTC + timedelta(minutes=bar_index)
    ts_close = ts_open + timedelta(minutes=1)
    base_price = float(bar_index + 1)
    return CandleWithMeta(
        candle=Candle(
            instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
            ts_open=UtcTimestamp(ts_open),
            ts_close=UtcTimestamp(ts_close),
            open=base_price,
            high=base_price + 0.5,
            low=base_price - 0.25,
            close=base_price + 0.25,
            volume_base=10.0 + float(bar_index),
            volume_quote=None,
        ),
        meta=CandleMeta(
            source="rest",
            ingested_at=UtcTimestamp(ts_close),
            ingest_id=None,
            instrument_key="binance:spot:BTCUSDT",
            trades_count=1,
            taker_buy_volume_base=None,
            taker_buy_volume_quote=None,
        ),
    )


def _forbid_directory_scan(*_args: object, **_kwargs: object) -> None:
    """
    Fail the test immediately if a loader code path attempts directory scanning.

    Args:
        *_args: Positional arguments ignored by the failure stub.
        **_kwargs: Keyword arguments ignored by the failure stub.
    Returns:
        None.
    Assumptions:
        Explicit-path loader methods must never need scanning helpers.
    Raises:
        AssertionError: Always, to signal forbidden scanning usage.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    raise AssertionError("directory scanning is forbidden in artifact loader v2")
