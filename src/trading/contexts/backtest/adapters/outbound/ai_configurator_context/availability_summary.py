from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True, slots=True)
class FilesystemBacktestAiAvailabilitySummaryRepository:
    summary_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary_path", Path(self.summary_path))

    def load_availability_summary(self) -> Mapping[str, Any]:
        """
        Load `availability_summary.yaml` from the trusted runtime artifact root.

        The returned payload is raw trusted input for the application snapshot builder.
        The adapter never adds the local path to the payload, so model-facing context can
        keep provenance sanitized.
        """
        if not self.summary_path.is_file():
            raise FileNotFoundError(f"availability_summary.yaml not found: {self.summary_path}")
        payload = yaml.safe_load(self.summary_path.read_text(encoding="utf-8"))
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError("availability_summary.yaml must be a mapping")
        return payload


__all__ = ["FilesystemBacktestAiAvailabilitySummaryRepository"]
