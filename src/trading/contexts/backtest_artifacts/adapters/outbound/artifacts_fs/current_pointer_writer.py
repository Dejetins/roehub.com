"""Atomic writer for strict `current.yaml` replacements in backtest artifact store v2."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
    ArtifactCurrentPointerV2,
    BacktestArtifactCurrentPointerWriterV2,
    BacktestArtifactPathResolverV2,
)


@dataclass(frozen=True, slots=True)
class AtomicArtifactCurrentPointerWriterV2(BacktestArtifactCurrentPointerWriterV2):
    """
    Replace `current.yaml` via temp-file write and atomic rename within one symbol root.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    path_resolver: BacktestArtifactPathResolverV2

    def write_current_pointer_atomically(
        self,
        coordinates: ArtifactCoordinatesV2,
        pointer: ArtifactCurrentPointerV2,
    ) -> Path:
        """
        Atomically replace one symbol-root `current.yaml` with deterministic payload bytes.

        Args:
            coordinates: Symbol-root coordinates used to resolve canonical target path.
            pointer: Strict pointer payload to serialize and publish.
        Returns:
            Path: Canonical `current.yaml` path that was replaced.
        Assumptions:
            Temp file is written in the same directory so `os.replace` remains atomic.
        Raises:
            ValueError: If pointer path does not match the canonical target path.
            OSError: If temp-file write or atomic replace fails.
        Side Effects:
            Creates or replaces `current.yaml` under the resolved artifact symbol root.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        target_path = self.path_resolver.current_pointer_path(coordinates)
        if pointer.path != target_path:
            raise ValueError(
                "AtomicArtifactCurrentPointerWriterV2 pointer.path must match "
                f"{target_path}; got {pointer.path}"
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_pointer = _serialize_current_pointer_v2(pointer)
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{CURRENT_POINTER_PREFIX_V2}{target_path.name}.",
            suffix=".tmp",
            dir=target_path.parent,
            text=True,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized_pointer)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise
        return target_path


CURRENT_POINTER_PREFIX_V2 = "current-pointer-"


def _serialize_current_pointer_v2(pointer: ArtifactCurrentPointerV2) -> str:
    """
    Serialize one strict pointer payload into deterministic YAML bytes.

    Args:
        pointer: Strict pointer identity payload.
    Returns:
        str: Deterministic UTF-8 YAML text terminated by a trailing newline.
    Assumptions:
        All pointer fields were already validated by `ArtifactCurrentPointerV2`.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    return "\n".join(
        (
            f"schema_version: {pointer.schema_version}",
            f"active_slot: {pointer.active_slot}",
            f"slot_generation: {pointer.slot_generation}",
            f'asof_date: "{pointer.asof_date}"',
            f'manifest_sha256: "{pointer.manifest_sha256}"',
            f'published_at_utc: "{pointer.published_at_utc}"',
        )
    ) + "\n"
