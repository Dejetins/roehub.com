from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestTpSlHitTimesGridArrays,
    BacktestTpSlHitTimesTableArrays,
)
from trading.contexts.backtest.application.dto.artifact_inputs import (
    BacktestArtifactRuntimeContext,
    BacktestSignalMatrix,
    BacktestSignalMetadata,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactFundingArraysV2,
    ArtifactMappingArraysV2,
    ArtifactPriceArraysV2,
    BacktestPreparedArtifactSet,
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
        metadata_only: bool = False,
    ) -> BacktestArtifactRuntimeContext:
        """
        Resolve one slot-pinned runtime context from normalized coordinates and preflight metadata.
        """
        ...

    def with_prepared_inputs(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
        prepared: BacktestPreparedArtifactSet,
        output_directory: Path,
        acknowledged_prepared_sha256: str | None = None,
        signal_metadata: Mapping[tuple[str, str], BacktestSignalMetadata] | None = None,
    ) -> BacktestArtifactRuntimeContext:
        """Bind validated selected references to the actual pinned source."""
        ...

    def write_prepared_manifest(
        self,
        *,
        prepared: BacktestPreparedArtifactSet,
        output_directory: Path,
    ) -> Path:
        """Atomically write the validated job manifest after preparation succeeds."""
        ...

    def load_price_arrays(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
        timeframe: str,
    ) -> ArtifactPriceArraysV2:
        """
        Load one `prices/<tf>` family through `np.load(..., mmap_mode="r")`.
        """
        ...

    def load_mapping_arrays(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
        timeframe: str,
    ) -> ArtifactMappingArraysV2:
        """
        Load one `mappings/<tf>` family through `np.load(..., mmap_mode="r")`.
        """
        ...

    def load_funding_arrays(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
    ) -> ArtifactFundingArraysV2:
        """
        Load the `funding/` family only when the selected futures job needs it.
        """
        ...

    def load_signal_matrix(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
        timeframe: str,
        indicator_id: str,
    ) -> BacktestSignalMatrix:
        """
        Load one `signals/<tf>/<indicator_id>` matrix through `np.load(..., mmap_mode="r")`.
        """
        ...

    def load_signal_rows(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
        timeframe: str,
        indicator_id: str,
        row_ids: np.ndarray,
        time_slice: slice,
    ) -> np.ndarray:
        """
        Copy requested signal rows and `[start, end)` bars into one contiguous int8 matrix.
        """
        ...

    def load_hit_times_grid_arrays(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
    ) -> BacktestTpSlHitTimesGridArrays:
        """
        Load the small `hit_times/15m` manifest and TP/SL level arrays.
        """
        ...

    def load_hit_times_table_arrays(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
    ) -> BacktestTpSlHitTimesTableArrays:
        """
        Load heavy `hit_times/15m` table arrays after request grid validation.
        """
        ...


__all__ = ["BacktestArtifactArrayLoader"]
