from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    AtomicArtifactCurrentPointerWriterV2,
    BacktestArtifactPathBuilderV2,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    current_pointer_writer as pointer_writer_module,
)
from trading.contexts.backtest.application.services import (
    ArtifactCoordinatesV2,
    ArtifactCurrentPointerV2,
    ArtifactSlotLiteralV2,
)


def test_atomic_artifact_current_pointer_writer_v2_replaces_pointer_file_atomically(
    tmp_path: Path,
) -> None:
    """
    Verify writer serializes strict pointer payload and replaces `current.yaml` in one step.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Writer uses temp-file write plus atomic rename within the target directory.
    Raises:
        AssertionError: If serialized pointer content is not deterministic.
    Side Effects:
        Creates and replaces temporary `current.yaml` files under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py
    """
    builder = BacktestArtifactPathBuilderV2(root=tmp_path / "artifacts" / "backtest" / "v2")
    coordinates = ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )
    current_path = builder.current_pointer_path(coordinates)
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text("schema_version: 1\n", encoding="utf-8")

    writer = AtomicArtifactCurrentPointerWriterV2(path_resolver=builder)
    pointer = _pointer(path=current_path, slot="slot_b", generation=12)

    written_path = writer.write_current_pointer_atomically(coordinates, pointer)

    assert written_path == current_path
    assert current_path.read_text(encoding="utf-8") == "\n".join(
        (
            "schema_version: 1",
            "active_slot: slot_b",
            "slot_generation: 12",
            'asof_date: "2026-03-24"',
            'manifest_sha256: "' + ("a" * 64) + '"',
            'published_at_utc: "2026-03-24T02:00:00Z"',
            "",
        )
    )


def test_atomic_artifact_current_pointer_writer_v2_keeps_original_pointer_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify original `current.yaml` remains unchanged if atomic replace step fails.

    Args:
        tmp_path: pytest temporary path fixture.
        monkeypatch: pytest monkeypatch fixture.
    Returns:
        None.
    Assumptions:
        Temp-file write must never partially overwrite the active pointer file.
    Raises:
        AssertionError: If original pointer content changes after replace failure.
    Side Effects:
        Creates temporary files and injects one failing `os.replace` stub.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py
    """
    builder = BacktestArtifactPathBuilderV2(root=tmp_path / "artifacts" / "backtest" / "v2")
    coordinates = ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )
    current_path = builder.current_pointer_path(coordinates)
    current_path.parent.mkdir(parents=True, exist_ok=True)
    original_content = "\n".join(
        (
            "schema_version: 1",
            "active_slot: slot_a",
            "slot_generation: 11",
            'asof_date: "2026-03-23"',
            'manifest_sha256: "' + ("b" * 64) + '"',
            'published_at_utc: "2026-03-23T02:00:00Z"',
            "",
        )
    )
    current_path.write_text(original_content, encoding="utf-8")

    writer = AtomicArtifactCurrentPointerWriterV2(path_resolver=builder)
    pointer = _pointer(path=current_path, slot="slot_b", generation=12)

    def _raise_replace_failure(source: str | bytes | Path, target: str | bytes | Path) -> None:
        """
        Raise one deterministic `OSError` from the atomic replace stub.

        Args:
            source: Temp-file source path.
            target: Target `current.yaml` path.
        Returns:
            None.
        Assumptions:
            Test needs only to simulate replace failure after temp file was written.
        Raises:
            OSError: Always raised.
        Side Effects:
            None.
        """
        _ = source, target
        raise OSError("replace failed")

    monkeypatch.setattr(pointer_writer_module.os, "replace", _raise_replace_failure)

    with pytest.raises(OSError, match="replace failed"):
        writer.write_current_pointer_atomically(coordinates, pointer)

    assert current_path.read_text(encoding="utf-8") == original_content


def _pointer(
    *,
    path: Path,
    slot: ArtifactSlotLiteralV2,
    generation: int,
) -> ArtifactCurrentPointerV2:
    """
    Build deterministic strict pointer payload fixture for writer tests.

    Args:
        path: Canonical target `current.yaml` path.
        slot: Active slot literal.
        generation: Positive slot generation.
    Returns:
        ArtifactCurrentPointerV2: Strict pointer payload fixture.
    Assumptions:
        Fixture literals satisfy the R2-02 strict pointer contract.
    Raises:
        ValueError: If one fixture literal violates pointer validation.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py
    """
    published_at_utc = datetime(2026, 3, 24, 2, 0, tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    payload = {
        "schema_version": 1,
        "active_slot": slot,
        "slot_generation": generation,
        "asof_date": "2026-03-24",
        "manifest_sha256": "a" * 64,
        "published_at_utc": published_at_utc,
    }
    return ArtifactCurrentPointerV2(
        path=path,
        active_slot=slot,
        raw_payload=payload,
        schema_version=1,
        slot_generation=generation,
        asof_date="2026-03-24",
        manifest_sha256="a" * 64,
        published_at_utc=published_at_utc,
    )
