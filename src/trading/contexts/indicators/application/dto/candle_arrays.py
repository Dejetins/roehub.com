from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from trading.shared_kernel.primitives import MarketId, Symbol, Timeframe, TimeRange


@dataclass(frozen=True, slots=True)
class CandleArrays:
    """
    Dense candle arrays consumed by indicator compute.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: ...domain.errors.missing_input_series_error, ..ports.feeds.candle_feed
    """

    market_id: MarketId
    symbol: Symbol
    time_range: TimeRange
    timeframe: Timeframe
    ts_open: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __post_init__(self) -> None:
        """
        Validate array contracts for dense OHLCV transport.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            All arrays represent the same dense timeline and are already aligned by index.
        Raises:
            ValueError: If metadata is missing or array shape, dtype,
                length, or ordering invariants are violated.
        Side Effects:
            None.
        """
        if self.market_id is None:  # type: ignore[truthy-bool]
            raise ValueError("CandleArrays requires market_id")
        if self.symbol is None:  # type: ignore[truthy-bool]
            raise ValueError("CandleArrays requires symbol")
        if self.time_range is None:  # type: ignore[truthy-bool]
            raise ValueError("CandleArrays requires time_range")
        if self.timeframe is None:  # type: ignore[truthy-bool]
            raise ValueError("CandleArrays requires timeframe")

        length = self._validate_array("ts_open", self.ts_open, np.int64, None)
        self._validate_array("open", self.open, np.float32, length)
        self._validate_array("high", self.high, np.float32, length)
        self._validate_array("low", self.low, np.float32, length)
        self._validate_array("close", self.close, np.float32, length)
        self._validate_array("volume", self.volume, np.float32, length)
        self._validate_timestamp_order()

    def _validate_array(
        self,
        name: str,
        values: np.ndarray,
        expected_dtype: npt.DTypeLike,
        expected_length: int | None,
    ) -> int:
        """
        Validate one ndarray shape/dtype contract and optional expected length.

        Args:
            name: Human-readable field name for diagnostics.
            values: Candidate numpy array.
            expected_dtype: Required dtype for the array.
            expected_length: Expected array length, or None when this array
                defines the baseline length.
        Returns:
            int: Validated array length.
        Assumptions:
            Arrays are one-dimensional vectors.
        Raises:
            ValueError: If array is not ndarray-like, not 1D, has unexpected
                dtype, or has a length mismatch.
        Side Effects:
            None.
        """
        normalized_expected_dtype = np.dtype(expected_dtype)
        try:
            if values.ndim != 1:
                raise ValueError(f"{name} must be a 1D array")
            if values.dtype != normalized_expected_dtype:
                raise ValueError(
                    f"{name} must have dtype {normalized_expected_dtype}, got {values.dtype}"
                )
        except AttributeError as error:
            raise ValueError(f"{name} must be a numpy ndarray") from error

        length = values.shape[0]
        if expected_length is not None and length != expected_length:
            raise ValueError(
                f"{name} length must match baseline length {expected_length}, got {length}"
            )
        return length

    def _validate_timestamp_order(self) -> None:
        """
        Validate stable non-decreasing timestamp ordering.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `ts_open` uses integer timestamp units and strict unit semantics are outside this epic.
        Raises:
            ValueError: If timestamps are not monotonically non-decreasing.
        Side Effects:
            None.
        """
        if self.ts_open.shape[0] <= 1:
            return
        if not np.all(self.ts_open[1:] >= self.ts_open[:-1]):
            raise ValueError("ts_open must be stably ordered in non-decreasing order")
