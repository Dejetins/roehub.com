from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    SyntheticArtifactStoreV2,
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.adapters.outbound import BacktestArtifactPathBuilderV2
from trading.contexts.backtest.application.services import (
    ArtifactCoordinatesV2,
    YamlBacktestArtifactLoaderV2,
)


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
    explicit_hit_times_manifest = loader.load_hit_times_manifest_from_path(
        loader.resolve_hit_times_manifest_path(store.coordinates, store.inactive_slot),
        slot=store.inactive_slot,
    )
    hit_times_paths = loader.resolve_hit_times_paths(store.coordinates, store.inactive_slot)

    assert explicit_root_manifest.slot_generation == 5
    assert explicit_signal_manifest.indicator_id == "ma.ema"
    assert explicit_hit_times_manifest.sentinel_index == 4
    assert hit_times_paths.long_tp == store.builder.hit_times_paths(
        store.coordinates,
        store.inactive_slot,
    ).long_tp


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
    assert price_paths.ohlcv == store.builder.price_paths(
        store.coordinates,
        store.active_slot,
        "1m",
    ).ohlcv
    assert signal_paths.signals == store.builder.signal_paths(
        store.coordinates,
        store.inactive_slot,
        "15m",
        "ma.ema",
    ).signals
    assert mapping_paths.bar_close_1m_idx == store.builder.mapping_paths(
        store.coordinates,
        store.inactive_slot,
        "15m",
    ).bar_close_1m_idx
    assert hit_times_manifest_path == store.builder.hit_times_manifest_path(
        store.coordinates,
        store.inactive_slot,
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
