from __future__ import annotations

import logging
from collections.abc import Sequence

from trading.contexts.market_data.application.dto import CandleWithMeta
from trading.contexts.market_data.application.ports.feeds import LiveCandlePublisher

log = logging.getLogger(__name__)


class FanoutLiveCandlePublisher(LiveCandlePublisher):
    """
    Best-effort fan-out for multiple live closed-candle publishers.
    """

    def __init__(self, publishers: Sequence[LiveCandlePublisher]) -> None:
        """
        Initialize fan-out publisher.

        Parameters:
        - publishers: non-empty sequence of live feed publishers.

        Returns:
        - None.

        Assumptions/Invariants:
        - Publishers are called in the supplied order.

        Errors/Exceptions:
        - Raises `ValueError` when no publishers are supplied.

        Side effects:
        - None.
        """
        if not publishers:
            raise ValueError("FanoutLiveCandlePublisher requires at least one publisher")
        self._publishers = tuple(publishers)

    def publish_1m_closed(self, candle: CandleWithMeta) -> None:
        """
        Publish one closed 1m candle to all configured live feed publishers.

        Parameters:
        - candle: normalized closed candle with ingestion metadata.

        Returns:
        - None.

        Assumptions/Invariants:
        - One publisher failure must not prevent the remaining publishers from running.

        Errors/Exceptions:
        - Unexpected publisher exceptions are logged and suppressed.

        Side effects:
        - Calls each configured publisher in order.
        """
        for publisher in self._publishers:
            try:
                publisher.publish_1m_closed(candle)
            except Exception:  # noqa: BLE001
                log.exception(
                    "live candle fanout publisher failed for instrument_key=%s",
                    candle.meta.instrument_key,
                )
