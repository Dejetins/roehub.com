from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CanonicalCandleBatch1m:
    """
    Columnar canonical `1m` candle batch for offline/precompute workloads.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/backtest-service-artifact-runtime-v1.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/canonical_candle_reader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
    """

    open_time_ms: np.ndarray
    close_time_ms: np.ndarray
    ohlcv_f32: np.ndarray

    def __post_init__(self) -> None:
        """
        Validate the strict columnar shape/dtype contract for precompute readers.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Arrays are already aligned to the same ordered timeline and may be empty.
        Raises:
            ValueError: If array dtypes or shapes violate the strict batch contract.
        Side Effects:
            None.
        """
        if self.open_time_ms.dtype != np.int64:
            raise ValueError("CanonicalCandleBatch1m.open_time_ms must use int64 dtype")
        if self.close_time_ms.dtype != np.int64:
            raise ValueError("CanonicalCandleBatch1m.close_time_ms must use int64 dtype")
        if self.ohlcv_f32.dtype != np.float32:
            raise ValueError("CanonicalCandleBatch1m.ohlcv_f32 must use float32 dtype")
        if len(self.open_time_ms.shape) != 1:
            raise ValueError("CanonicalCandleBatch1m.open_time_ms must be 1-dimensional")
        if len(self.close_time_ms.shape) != 1:
            raise ValueError("CanonicalCandleBatch1m.close_time_ms must be 1-dimensional")
        if self.ohlcv_f32.ndim != 2 or self.ohlcv_f32.shape[1] != 5:
            raise ValueError("CanonicalCandleBatch1m.ohlcv_f32 must have shape [T, 5]")
        timeline_bar_count = int(self.open_time_ms.shape[0])
        if int(self.close_time_ms.shape[0]) != timeline_bar_count:
            raise ValueError(
                "CanonicalCandleBatch1m.close_time_ms length must match open_time_ms length"
            )
        if int(self.ohlcv_f32.shape[0]) != timeline_bar_count:
            raise ValueError(
                "CanonicalCandleBatch1m.ohlcv_f32 row count must match open_time_ms length"
            )

    def row_count(self) -> int:
        """
        Return the number of canonical `1m` bars stored in the batch.

        Args:
            None.
        Returns:
            int: Timeline row count.
        Assumptions:
            Batch validation already guaranteed aligned lengths.
        Raises:
            None.
        Side Effects:
            None.
        """
        return int(self.open_time_ms.shape[0])
