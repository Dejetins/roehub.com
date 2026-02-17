from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from trading.contexts.market_data.application.dto import CandleWithMeta


@dataclass(frozen=True, slots=True)
class StrategyLiveCandleMessage:
    """
    StrategyLiveCandleMessage — one parsed Redis stream message with closed 1m candle payload.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md
      - src/trading/contexts/strategy/adapters/outbound/messaging/redis/
        redis_streams_live_candle_stream.py
    """

    message_id: str
    candle: CandleWithMeta


class StrategyLiveCandleStream(Protocol):
    """
    StrategyLiveCandleStream — port for consuming closed 1m candles for Strategy live-runner.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - docs/architecture/market_data/market-data-live-feed-redis-streams-v1.md
      - src/trading/contexts/strategy/application/services/live_runner.py
      - src/trading/contexts/strategy/adapters/outbound/messaging/redis/
        redis_streams_live_candle_stream.py
    """

    def read_closed_1m(self, *, instrument_key: str) -> tuple[StrategyLiveCandleMessage, ...]:
        """
        Read pending live candle messages for one instrument stream.

        Args:
            instrument_key: Canonical key used in Redis stream suffix.
        Returns:
            tuple[StrategyLiveCandleMessage, ...]: Parsed messages to process.
        Assumptions:
            Stream name format is `md.candles.1m.<instrument_key>`.
        Raises:
            Exception: Adapter-specific Redis/network/serialization errors.
        Side Effects:
            May create consumer group for stream on first access.
        """
        ...

    def ack(self, *, instrument_key: str, message_id: str) -> None:
        """
        Acknowledge one processed stream message in consumer group.

        Args:
            instrument_key: Canonical key used in Redis stream suffix.
            message_id: Redis stream message id.
        Returns:
            None.
        Assumptions:
            Ack is called only after successful processing of message side effects.
        Raises:
            Exception: Adapter-specific Redis/network errors.
        Side Effects:
            Mutates Redis consumer-group pending entries list.
        """
        ...
