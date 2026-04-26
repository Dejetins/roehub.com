from __future__ import annotations

from typing import Protocol

import numpy as np

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactMappingArraysV2,
    ArtifactPriceArraysV2,
    ArtifactSignalMatrixV2,
    ArtifactSlotPinnedRuntimeContextV2,
)


class BacktestArtifactArrayLoader(Protocol):
    """
    Application port for trusted mmap artifact array loading.
    """

    def resolve_context(
        self,
        *,
        coordinates: BacktestCoordinates,
        artifact_metadata: BacktestArtifactMetadata,
    ) -> ArtifactSlotPinnedRuntimeContextV2:
        """
        Resolve one slot-pinned runtime context from normalized coordinates and preflight metadata.
        """
        ...

    def load_price_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
    ) -> ArtifactPriceArraysV2:
        """
        Load one `prices/<tf>` family through `np.load(..., mmap_mode="r")`.
        """
        ...

    def load_mapping_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
    ) -> ArtifactMappingArraysV2:
        """
        Load one `mappings/<tf>` family through `np.load(..., mmap_mode="r")`.
        """
        ...

    def load_signal_matrix(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalMatrixV2:
        """
        Load one `signals/<tf>/<indicator_id>` matrix through `np.load(..., mmap_mode="r")`.
        """
        ...

    def load_signal_rows(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
        row_ids: np.ndarray,
        time_slice: slice,
    ) -> np.ndarray:
        """
        Copy requested signal rows and `[start, end)` bars into one contiguous int8 matrix.
        """
        ...


__all__ = ["BacktestArtifactArrayLoader"]
