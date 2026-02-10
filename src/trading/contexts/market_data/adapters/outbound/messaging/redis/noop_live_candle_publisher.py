from __future__ import annotations

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.feeds import LiveCandlePublisher


class NoopLiveCandlePublisher(LiveCandlePublisher):
    """
    No-op live candle publisher used when Redis streams feed is disabled.
    """

    def publish_1m_closed(self, candle: CandleWithMeta) -> None:
        """
        Ignore WS candle publication request.

        Parameters:
        - candle: closed candle event passed by WS worker.

        Returns:
        - None.

        Assumptions/Invariants:
        - Method is intentionally side-effect free.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        _ = candle
