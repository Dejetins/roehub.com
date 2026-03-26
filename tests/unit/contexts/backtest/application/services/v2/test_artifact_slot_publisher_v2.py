from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    AtomicArtifactCurrentPointerWriterV2,
    BacktestArtifactPathBuilderV2,
)
from trading.contexts.backtest.application.ports import BacktestJobRepository
from trading.contexts.backtest.application.services import (
    ArtifactCoordinatesV2,
    ArtifactSignalValidationSpecV2,
    ArtifactSlotPublishErrorV2,
    ArtifactSlotValidationSpecV2,
    BacktestArtifactSlotPublisherV2,
    YamlBacktestArtifactLoaderV2,
)


class _FakeJobRepository:
    """
    Deterministic fake repository exposing only the publish-guard pin-count query.
    """

    def __init__(self, *, blocked_total: int = 0) -> None:
        """
        Initialize fake repository with fixed inactive-slot blocking count.

        Args:
            blocked_total: Count returned for `count_active_for_artifact_manifest(...)`.
        Returns:
            None.
        Assumptions:
            Publisher unit tests exercise only the pin-count query.
        Raises:
            None.
        Side Effects:
            Stores last query payload for assertions.
        """
        self.blocked_total = blocked_total
        self.last_call: dict[str, object] | None = None

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        """
        Return configured blocking count and record the publish-guard query payload.

        Args:
            market_id: Canonical market id for the published symbol root.
            symbol: Instrument symbol.
            artifact_slot: Candidate inactive slot literal.
            artifact_manifest_hash: SHA-256 of inactive slot `manifest.yaml`.
        Returns:
            int: Configured blocking active jobs count.
        Assumptions:
            Other repository methods are not needed in these publisher unit tests.
        Raises:
            None.
        Side Effects:
            Records last query payload in memory.
        """
        self.last_call = {
            "market_id": market_id,
            "symbol": symbol,
            "artifact_slot": artifact_slot,
            "artifact_manifest_hash": artifact_manifest_hash,
        }
        return self.blocked_total


def test_backtest_artifact_slot_publisher_v2_switches_current_yaml_after_explicit_validation(
    tmp_path: Path,
) -> None:
    """
    Verify publisher validates explicit paths and atomically switches strict `current.yaml`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Inactive slot was already rebuilt before publish and no active jobs pin it.
    Raises:
        AssertionError: If publish result or written pointer identity is incorrect.
    Side Effects:
        Creates and replaces artifact files under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    builder, loader, coordinates = _artifact_store(tmp_path=tmp_path)
    repository = _FakeJobRepository(blocked_total=0)
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=builder),
        job_repository=cast(BacktestJobRepository, repository),
        now_provider=lambda: datetime(2026, 3, 26, 3, 4, 5, tzinfo=timezone.utc),
    )

    precheck = publisher.precheck_publish(coordinates)
    result = publisher.publish(
        precheck=precheck,
        validation_spec=_validation_spec(),
        asof_date="2026-03-26",
    )

    reloaded_pointer = loader.load_current_pointer(coordinates)
    assert precheck.ready is True
    assert precheck.inactive_slot == "slot_b"
    assert repository.last_call is not None
    assert repository.last_call["market_id"] == 1
    assert repository.last_call["symbol"] == "BTCUSDT"
    assert result.previous_pointer.active_slot == "slot_a"
    assert result.published_pointer.active_slot == "slot_b"
    assert result.published_pointer.slot_generation == 5
    assert result.published_pointer.asof_date == "2026-03-26"
    assert result.published_pointer.published_at_utc == "2026-03-26T03:04:05Z"
    assert reloaded_pointer == result.published_pointer
    assert result.validation.slot_manifest.path == builder.slot_manifest_path(
        coordinates,
        "slot_b",
    )


def test_backtest_artifact_slot_publisher_v2_blocks_publish_when_inactive_slot_is_pinned(
    tmp_path: Path,
) -> None:
    """
    Verify precheck/publish fail fast when the inactive slot is pinned by active jobs.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Blocking count comes from persisted queued/running jobs with pinned slot metadata.
    Raises:
        AssertionError: If blocked precheck does not yield stable diagnostics.
    Side Effects:
        Creates temporary artifact files under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    builder, loader, coordinates = _artifact_store(tmp_path=tmp_path)
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=builder),
        job_repository=cast(BacktestJobRepository, _FakeJobRepository(blocked_total=2)),
    )

    precheck = publisher.precheck_publish(coordinates)

    assert precheck.ready is False
    assert precheck.failure_code == "inactive_slot_pinned"
    with pytest.raises(ArtifactSlotPublishErrorV2, match="slot_b"):
        publisher.publish(
            precheck=precheck,
            validation_spec=_validation_spec(),
            asof_date="2026-03-26",
        )


def test_backtest_artifact_slot_publisher_v2_rejects_missing_explicit_validation_path(
    tmp_path: Path,
) -> None:
    """
    Verify publish validation fails when one required explicit inactive-slot path is missing.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Validation step must check all caller-provided explicit paths before pointer switch.
    Raises:
        AssertionError: If missing path does not raise stable validation error.
    Side Effects:
        Creates temporary artifact files under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    builder, loader, coordinates = _artifact_store(tmp_path=tmp_path, omit_slot_b_ohlcv=True)
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=builder),
        job_repository=cast(BacktestJobRepository, _FakeJobRepository(blocked_total=0)),
    )

    precheck = publisher.precheck_publish(coordinates)

    with pytest.raises(ArtifactSlotPublishErrorV2, match="price ohlcv"):
        publisher.publish(
            precheck=precheck,
            validation_spec=_validation_spec(),
            asof_date="2026-03-26",
        )


def _artifact_store(
    *,
    tmp_path: Path,
    omit_slot_b_ohlcv: bool = False,
) -> tuple[BacktestArtifactPathBuilderV2, YamlBacktestArtifactLoaderV2, ArtifactCoordinatesV2]:
    """
    Build a minimal two-slot artifact tree used by publisher unit tests.

    Args:
        tmp_path: pytest temporary path fixture.
        omit_slot_b_ohlcv: Whether to omit one validated explicit file in `slot_b`.
    Returns:
        tuple[BacktestArtifactPathBuilderV2, YamlBacktestArtifactLoaderV2,
            ArtifactCoordinatesV2]: Builder, loader, and coordinates for the synthetic store.
    Assumptions:
        `slot_a` is the active pointer target and `slot_b` is the inactive publish target.
    Raises:
        OSError: If one synthetic artifact file cannot be written.
    Side Effects:
        Creates a deterministic artifact directory tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    builder = BacktestArtifactPathBuilderV2(root=tmp_path / "artifacts" / "backtest" / "v2")
    loader = YamlBacktestArtifactLoaderV2(path_resolver=builder)
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
                "active_slot: slot_a",
                "slot_generation: 4",
                'asof_date: "2026-03-25"',
                'manifest_sha256: "' + ("c" * 64) + '"',
                'published_at_utc: "2026-03-25T02:00:00Z"',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    _write_text(
        builder.slot_manifest_path(coordinates, "slot_a"),
        "\n".join(("schema_version: 1", "slot: slot_a")) + "\n",
    )
    _write_text(
        builder.slot_manifest_path(coordinates, "slot_b"),
        "\n".join(("schema_version: 1", "slot: slot_b")) + "\n",
    )
    _write_bytes(builder.price_paths(coordinates, "slot_b", "1m").open_time, b"open")
    _write_bytes(builder.price_paths(coordinates, "slot_b", "1m").close_time, b"close")
    if not omit_slot_b_ohlcv:
        _write_bytes(builder.price_paths(coordinates, "slot_b", "1m").ohlcv, b"ohlcv")
    _write_bytes(builder.mapping_paths(coordinates, "slot_b", "15m").bar_open_1m_idx, b"open")
    _write_bytes(builder.mapping_paths(coordinates, "slot_b", "15m").bar_close_1m_idx, b"close")
    _write_bytes(builder.signal_paths(coordinates, "slot_b", "15m", "ma.ema").manifest, b"{}")
    _write_bytes(builder.signal_paths(coordinates, "slot_b", "15m", "ma.ema").signals, b"sig")
    _write_text(builder.hit_times_manifest_path(coordinates, "slot_b"), "schema_version: 1\n")

    return builder, loader, coordinates


def _validation_spec() -> ArtifactSlotValidationSpecV2:
    """
    Build explicit deterministic validation plan used by publisher tests.

    Args:
        None.
    Returns:
        ArtifactSlotValidationSpecV2: Explicit validation plan fixture.
    Assumptions:
        Validation plan remains fully explicit because R2-04 config loading is out of scope.
    Raises:
        ValueError: If one validation literal is invalid.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    return ArtifactSlotValidationSpecV2(
        price_timeframes=("1m",),
        mapping_timeframes=("15m",),
        signal_artifacts=(
            ArtifactSignalValidationSpecV2(timeframe="15m", indicator_id="ma.ema"),
        ),
        require_hit_times_manifest=True,
    )


def _write_text(path: Path, content: str) -> None:
    """
    Create one UTF-8 text file and its parent directories.

    Args:
        path: Target text file path.
        content: UTF-8 text content to write.
    Returns:
        None.
    Assumptions:
        Publisher unit tests use only deterministic local fixture files.
    Raises:
        OSError: If file cannot be written.
    Side Effects:
        Creates parent directories and writes file content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    """
    Create one binary fixture file and its parent directories.

    Args:
        path: Target binary file path.
        payload: Binary payload bytes.
    Returns:
        None.
    Assumptions:
        Placeholder bytes are sufficient because publisher validates path existence only.
    Raises:
        OSError: If file cannot be written.
    Side Effects:
        Creates parent directories and writes binary content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
