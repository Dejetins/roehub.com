from __future__ import annotations

from typing import Protocol

from trading.contexts.market_data.application.dto import CandleWithMeta


class LiveCandlePublisher(Protocol):
    """
    Publish closed 1m websocket candles to live strategy feed.

    Contract:
    - publish_1m_closed(candle: CandleWithMeta) -> None

    Semantics:
    - Implementations are expected to be best-effort and must not break WS ingestion.
    """

    def publish_1m_closed(self, candle: CandleWithMeta) -> None:
        """
        Publish one closed 1m candle event.

        Parameters:
        - candle: normalized closed candle with ingestion metadata.

        Returns:
        - None.

        Assumptions/Invariants:
        - Candle originates from WS closed 1m stream.

        Errors/Exceptions:
        - Implementations should avoid propagating runtime failures.

        Side effects:
        - Emits live event to outbound feed transport.
        """
        ...
