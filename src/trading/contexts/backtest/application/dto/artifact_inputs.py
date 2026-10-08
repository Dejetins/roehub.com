from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast

import numpy as np

from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCandleSnapshot,
    ArtifactHitTimesManifestDocumentV2,
    ArtifactInputFileReference,
    ArtifactSignalManifestDocumentV2,
    ArtifactSlotPinnedRuntimeContextV2,
    BacktestPreparedArtifactSet,
)


@dataclass(frozen=True, slots=True)
class BacktestSignalMetadata:
    """Canonical grid cardinality is independent of the stored physical rows."""

    canonical_rows_count: int | None
    manifest: ArtifactSignalManifestDocumentV2 | None = None

    def __post_init__(self) -> None:
        if self.canonical_rows_count is not None and self.canonical_rows_count < 1:
            raise ValueError("canonical_rows_count must be positive")


@dataclass(frozen=True, slots=True)
class BacktestArtifactRuntimeContext:
    """Explicit inventory tied to the real published source and internally trusted roots."""

    source: ArtifactSlotPinnedRuntimeContextV2
    references: tuple[ArtifactInputFileReference, ...]
    trusted_roots: Mapping[str, Path]
    signal_metadata: Mapping[tuple[str, str], BacktestSignalMetadata]
    signal_manifests: Mapping[tuple[str, str], ArtifactSignalManifestDocumentV2]
    prepared: BacktestPreparedArtifactSet | None = None
    source_snapshot: ArtifactCandleSnapshot | None = None
    hit_times_manifest: ArtifactHitTimesManifestDocumentV2 | None = None
    hit_times_manifest_hash: str | None = None

    mmap_owners: list[np.memmap] = field(default_factory=list, compare=False, repr=False)

    def close_mmaps(self) -> None:
        for owner in self.mmap_owners:
            if cast(Any, owner)._mmap is not None:
                cast(Any, owner)._mmap.close()
        self.mmap_owners.clear()

    def __post_init__(self) -> None:
        roots = {key: Path(path).resolve(strict=True) for key, path in self.trusted_roots.items()}
        if any(not root.is_dir() for root in roots.values()):
            raise ValueError("trusted roots must be directories")
        if any(ref.root_id not in roots for ref in self.references):
            raise ValueError("input reference has an untrusted root")
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "trusted_roots", MappingProxyType(roots))
        object.__setattr__(self, "signal_metadata", MappingProxyType(dict(self.signal_metadata)))
        object.__setattr__(self, "signal_manifests", MappingProxyType(dict(self.signal_manifests)))


@dataclass(frozen=True, slots=True)
class BacktestSignalMatrix:
    timeframe: str
    indicator_id: str
    row_ids: tuple[int, ...]
    canonical_rows_count: int | None
    matrix: np.ndarray
    manifest: ArtifactSignalManifestDocumentV2 | None = None

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.dtype != np.dtype(np.int8):
            raise ValueError("signal matrix must be two-dimensional int8")
        if len(self.row_ids) != self.matrix.shape[0] or len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError("signal row IDs must uniquely identify physical rows")
        if any(row < 0 for row in self.row_ids):
            raise ValueError("canonical row IDs must be nonnegative")
        if self.canonical_rows_count is not None and any(
            row >= self.canonical_rows_count for row in self.row_ids
        ):
            raise ValueError("canonical row ID exceeds full grid")


@dataclass(frozen=True, slots=True)
class BacktestAttemptInputs:
    """Trusted child composition, never deserialized from a public request."""

    directory: Path
    organization_id: str
    job_id: str
    owner_token: str
    attempt: int
    max_generated_bytes: int
    max_compute_bytes: int
    acknowledge: Callable[[BacktestPreparedArtifactSet], str]
