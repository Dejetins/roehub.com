"""Atomic writer for root-level backtest artifact availability summaries."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    AVAILABILITY_SUMMARY_FILENAME_V2,
)


@dataclass(frozen=True, slots=True)
class AtomicArtifactAvailabilitySummaryWriterV2:
    """
    Replace `<artifact_root>/availability_summary.yaml` via temp-file write and rename.

    Docs:
      - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
    Related:
      - src/trading/contexts/backtest_artifacts/application/services/v2/
        artifact_availability_summary.py
    """

    def write_availability_summary_atomically(
        self,
        *,
        artifact_root: Path,
        payload: Mapping[str, Any],
    ) -> Path:
        """
        Atomically replace root-level `availability_summary.yaml` with deterministic YAML.

        Args:
            artifact_root: Artifact v2 root directory.
            payload: Complete summary payload, including `summary_hash`.
        Returns:
            Path: Final summary path.
        Assumptions:
            Caller owns serialization ordering and host-level concurrency control.
        Raises:
            OSError: If the temp write, fsync, or rename fails.
        Side Effects:
            Creates/replaces `<artifact_root>/availability_summary.yaml`.
        """
        root = Path(artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        target_path = root / AVAILABILITY_SUMMARY_FILENAME_V2
        temp_path = root / f"{AVAILABILITY_SUMMARY_FILENAME_V2}.tmp"
        serialized = _serialize_availability_summary_v2(payload)
        try:
            with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target_path)
            _fsync_directory_v2(root)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return target_path


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: object) -> bool:
        return True


def _serialize_availability_summary_v2(payload: Mapping[str, Any]) -> str:
    text = yaml.dump(
        dict(payload),
        Dumper=_NoAliasSafeDumper,
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )
    return text if text.endswith("\n") else f"{text}\n"


def _fsync_directory_v2(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


__all__ = ["AtomicArtifactAvailabilitySummaryWriterV2"]
