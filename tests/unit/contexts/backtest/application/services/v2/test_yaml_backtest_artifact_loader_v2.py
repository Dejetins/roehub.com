from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.backtest.adapters.outbound import BacktestArtifactPathBuilderV2
from trading.contexts.backtest.application.services import (
    ArtifactCoordinatesV2,
    YamlBacktestArtifactLoaderV2,
)


@pytest.fixture()
def synthetic_artifact_store_v2(
    tmp_path: Path,
) -> tuple[BacktestArtifactPathBuilderV2, ArtifactCoordinatesV2]:
    """
    Build a minimal synthetic artifact tree for loader tests under `tmp_path`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        tuple[BacktestArtifactPathBuilderV2, ArtifactCoordinatesV2]: Builder and coordinates
            pointing at the created test tree.
    Assumptions:
        Loader tests need only minimal placeholder files because path resolution does not parse
        binary payloads in R2-01.
    Raises:
        OSError: If one temporary file cannot be written.
    Side Effects:
        Creates a minimal deterministic artifact directory tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    builder = BacktestArtifactPathBuilderV2(root=tmp_path / "artifacts" / "backtest" / "v2")
    coordinates = ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )

    current_path = builder.current_pointer_path(coordinates)
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "active_slot: slot_b",
                "slot_generation: 42",
                'asof_date: "2026-03-24"',
                'manifest_sha256: "' + ("b" * 64) + '"',
                'published_at_utc: "2026-03-24T02:00:00Z"',
            )
        ),
        encoding="utf-8",
    )

    slot_a_manifest = builder.slot_manifest_path(coordinates, "slot_a")
    slot_a_manifest.parent.mkdir(parents=True, exist_ok=True)
    slot_a_manifest.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "slot: slot_a",
                "timeframes:",
                "  - 1m",
                "  - 15m",
            )
        ),
        encoding="utf-8",
    )

    slot_b_manifest = builder.slot_manifest_path(coordinates, "slot_b")
    slot_b_manifest.parent.mkdir(parents=True, exist_ok=True)
    slot_b_manifest.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "slot: slot_b",
                "timeframes:",
                "  - 1m",
                "  - 15m",
            )
        ),
        encoding="utf-8",
    )

    _write_placeholder_file(builder.price_paths(coordinates, "slot_b", "1m").open_time)
    _write_placeholder_file(builder.price_paths(coordinates, "slot_b", "1m").close_time)
    _write_placeholder_file(builder.price_paths(coordinates, "slot_b", "1m").ohlcv)
    _write_placeholder_file(builder.signal_paths(coordinates, "slot_b", "15m", "ma.ema").signals)
    _write_placeholder_file(builder.signal_paths(coordinates, "slot_b", "15m", "ma.ema").manifest)
    _write_placeholder_file(builder.mapping_paths(coordinates, "slot_b", "15m").bar_open_1m_idx)
    _write_placeholder_file(builder.mapping_paths(coordinates, "slot_b", "15m").bar_close_1m_idx)
    _write_placeholder_file(builder.hit_times_manifest_path(coordinates, "slot_b"))

    return builder, coordinates


def test_yaml_backtest_artifact_loader_v2_reads_current_and_manifests(
    synthetic_artifact_store_v2: tuple[BacktestArtifactPathBuilderV2, ArtifactCoordinatesV2],
) -> None:
    """
    Verify the loader reads `current.yaml` and slot manifests through known deterministic paths.

    Args:
        synthetic_artifact_store_v2: Fixture with a minimal synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        `current.yaml` remains the only active-slot source for `load_active_slot_manifest`.
    Raises:
        AssertionError: If the loader returns incorrect typed metadata or paths.
    Side Effects:
        Reads deterministic YAML files from the synthetic artifact tree.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    builder, coordinates = synthetic_artifact_store_v2
    loader = YamlBacktestArtifactLoaderV2(path_resolver=builder)

    current = loader.load_current_pointer(coordinates)
    assert current.path == builder.current_pointer_path(coordinates)
    assert current.active_slot == "slot_b"
    assert current.slot_generation == 42
    assert current.asof_date == "2026-03-24"
    assert current.manifest_sha256 == "b" * 64
    assert current.published_at_utc == "2026-03-24T02:00:00Z"

    slot_manifest = loader.load_slot_manifest(coordinates, "slot_a")
    assert slot_manifest.path == builder.slot_manifest_path(coordinates, "slot_a")
    assert slot_manifest.slot == "slot_a"
    assert slot_manifest.raw_payload["schema_version"] == 1

    active_manifest = loader.load_active_slot_manifest(coordinates)
    assert active_manifest.path == builder.slot_manifest_path(coordinates, "slot_b")
    assert active_manifest.slot == "slot_b"
    assert active_manifest.raw_payload["slot"] == "slot_b"

    explicit_pointer = loader.load_current_pointer_from_path(
        loader.resolve_current_pointer_path(coordinates)
    )
    explicit_manifest = loader.load_manifest_from_path(
        loader.resolve_slot_manifest_path(coordinates, "slot_b"),
        slot="slot_b",
    )
    assert explicit_pointer.active_slot == "slot_b"
    assert explicit_manifest.path == builder.slot_manifest_path(coordinates, "slot_b")


def test_yaml_backtest_artifact_loader_v2_avoids_directory_scanning(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifact_store_v2: tuple[BacktestArtifactPathBuilderV2, ArtifactCoordinatesV2],
) -> None:
    """
    Verify loader hot-path methods do not rely on directory scanning helpers.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        synthetic_artifact_store_v2: Fixture with a minimal synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Explicit path computation and direct file reads are allowed; scanning is not.
    Raises:
        AssertionError: If one loader method uses directory scanning.
    Side Effects:
        Temporarily replaces `Path` scanning helpers with failure stubs.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    builder, coordinates = synthetic_artifact_store_v2
    loader = YamlBacktestArtifactLoaderV2(path_resolver=builder)

    monkeypatch.setattr(Path, "iterdir", _forbid_directory_scan)
    monkeypatch.setattr(Path, "glob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "rglob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "walk", _forbid_directory_scan)

    current = loader.load_current_pointer(coordinates)
    manifest = loader.load_active_slot_manifest(coordinates)
    price_paths = loader.resolve_price_paths(coordinates, current.active_slot, "1m")
    signal_paths = loader.resolve_signal_paths(coordinates, current.active_slot, "15m", "ma.ema")
    mapping_paths = loader.resolve_mapping_paths(coordinates, current.active_slot, "15m")
    hit_times_manifest_path = loader.resolve_hit_times_manifest_path(
        coordinates,
        current.active_slot,
    )

    assert manifest.path == builder.slot_manifest_path(coordinates, "slot_b")
    assert price_paths.ohlcv == builder.price_paths(coordinates, "slot_b", "1m").ohlcv
    assert (
        signal_paths.signals == builder.signal_paths(coordinates, "slot_b", "15m", "ma.ema").signals
    )
    assert mapping_paths.bar_close_1m_idx == (
        builder.mapping_paths(coordinates, "slot_b", "15m").bar_close_1m_idx
    )
    assert hit_times_manifest_path == builder.hit_times_manifest_path(coordinates, "slot_b")


def test_yaml_backtest_artifact_loader_v2_rejects_invalid_pointer_shape(tmp_path: Path) -> None:
    """
    Verify the loader fails fast when `current.yaml` misses the required `active_slot`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        R2-01 validates only the minimal pointer fields needed for deterministic reads.
    Raises:
        AssertionError: If an invalid pointer document is accepted.
    Side Effects:
        Creates and reads one temporary invalid YAML document.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
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
    Verify strict `current.yaml` parsing rejects unsupported schema/hash/timestamp literals.

    Args:
        tmp_path: pytest temporary path fixture.
        payload_text: Invalid strict pointer YAML payload.
        error_pattern: Stable error substring expected from strict validation.
    Returns:
        None.
    Assumptions:
        R2-02 rejects malformed strict pointer literals fail-fast before runtime can pin them.
    Raises:
        AssertionError: If invalid strict pointer fields are accepted.
    Side Effects:
        Creates and reads one temporary invalid YAML document.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
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
    pointer_path.write_text(payload_text, encoding="utf-8")

    loader = YamlBacktestArtifactLoaderV2(path_resolver=builder)

    with pytest.raises(ValueError, match=error_pattern):
        loader.load_current_pointer(coordinates)


def _write_placeholder_file(path: Path) -> None:
    """
    Create one placeholder file and its parent directories for synthetic artifact tests.

    Args:
        path: File path to create.
    Returns:
        None.
    Assumptions:
        Placeholder bytes are sufficient because loader tests never parse artifact binaries.
    Raises:
        OSError: If the file cannot be written.
    Side Effects:
        Creates parent directories and writes one placeholder file.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".yaml"):
        path.write_text("schema_version: 1\n", encoding="utf-8")
        return
    path.write_bytes(b"placeholder")


def _forbid_directory_scan(*args: object, **kwargs: object) -> None:
    """
    Fail a test immediately when one directory-scanning helper is invoked.

    Args:
        *args: Positional arguments passed to the patched `Path` helper.
        **kwargs: Keyword arguments passed to the patched `Path` helper.
    Returns:
        None.
    Assumptions:
        Any call into scanning helpers inside R2-01 loader hot paths is a contract violation.
    Raises:
        AssertionError: Always, to mark forbidden directory scanning.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    del args, kwargs
    raise AssertionError("directory scanning is forbidden in R2-01 loader hot paths")
