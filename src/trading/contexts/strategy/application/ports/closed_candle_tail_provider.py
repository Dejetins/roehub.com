from __future__ import annotations

from typing import Protocol

from trading.contexts.market_data.application.dto import ClosedCandleTailResult
from trading.shared_kernel.primitives import InstrumentId, UtcTimestamp


class ClosedCandleTailProvider(Protocol):
    """
    Strategy-side port for requesting a continuous closed 1m candle tail.

    The implementation belongs to Market Data adapters/use cases. Strategy must not call
    exchange REST providers or ClickHouse directly through this contract.
    """

    def get_closed_1m_tail(
        self,
        *,
        instrument_id: InstrumentId,
        instrument_key: str,
        start_ts_open: UtcTimestamp,
        end_ts_open: UtcTimestamp,
        correlation_id: str,
    ) -> ClosedCandleTailResult:
        """
        Return a half-open closed-candle range `[start_ts_open, end_ts_open)`.

        Args:
            instrument_id: Domain market/symbol identity.
            instrument_key: Canonical operational instrument key.
            start_ts_open: Inclusive 1m candle-open boundary.
            end_ts_open: Exclusive 1m candle-open boundary.
            correlation_id: Redacted caller correlation id for audit linkage.
        Returns:
            ClosedCandleTailResult: Continuous or missing result with source/audit metadata.
        Assumptions:
            The provider returns only already closed 1m candles and redacted errors.
        Raises:
            Exception: Implementations may propagate bounded storage/provider failures.
        Side Effects:
            Implementation may read Market Data stores and append repair audit events.
        """
        ...
