from __future__ import annotations

from typing import Protocol

from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import MarketId, Symbol, TimeRange


class CandleFeed(Protocol):
    """
    Port for loading dense 1m candle arrays for indicator compute.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: ....application.dto.candle_arrays, ....domain.errors.missing_input_series_error
    """

    def load_1m_dense(
        self,
        market_id: MarketId,
        symbol: Symbol,
        time_range: TimeRange,
    ) -> CandleArrays:
        """
        Load dense 1m candles for one instrument and time range.

        Args:
            market_id: Stable market identifier.
            symbol: Market-local instrument symbol.
            time_range: Requested half-open interval `[start, end)`.
        Returns:
            CandleArrays: Dense 1m arrays ready for compute.
        Assumptions:
            Missing physical candles are represented by NaN in OHLCV arrays.
        Raises:
            MissingInputSeriesError: If required input series cannot be produced.
            GridValidationError: If requested range cannot satisfy feed invariants.
        Side Effects:
            None.
        """
        ...
