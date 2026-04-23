from __future__ import annotations

from typing import Protocol

from trading.shared_kernel.primitives import InstrumentId, UtcTimestamp


class InstrumentHistoryStartSource(Protocol):
    """
    Source port resolving earliest confirmed exchange history start for one instrument.

    Contract:
    - Returns minute-level UTC timestamp when exchange can confirm historical candles.
    - Returns `None` when the source cannot determine a symbol-specific lower bound.
    - Callers must fall back to market-wide `earliest_available_ts_utc` on `None`.
    """

    def get_history_start(self, instrument_id: InstrumentId) -> UtcTimestamp | None:
        """
        Resolve symbol-specific earliest available historical candle minute.

        Parameters:
        - instrument_id: target instrument identity `(market_id, symbol)`.

        Returns:
        - UTC timestamp for first known exchange history minute, or `None` when unknown.
        """
        ...
