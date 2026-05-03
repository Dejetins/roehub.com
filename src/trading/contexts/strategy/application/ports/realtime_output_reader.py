from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import UserId


class StrategyRealtimeStreamUnavailableError(Exception):
    """
    Raised when the realtime output stream substrate is not available for UI reads.
    """


@dataclass(frozen=True, slots=True)
class StrategyRealtimeStreamMessageV1:
    """
    One browser-facing realtime output message read from Strategy Redis Streams v1.
    """

    stream: str
    stream_kind: str
    message_id: str
    payload: Mapping[str, str]


class StrategyRealtimeOutputReader(Protocol):
    """
    Read-only port for browser-facing Strategy realtime output.
    """

    def read_for_user(
        self,
        *,
        user_id: UserId,
        strategy_id: UUID | None,
        last_event_id: str,
        count: int,
        block_ms: int,
    ) -> tuple[StrategyRealtimeStreamMessageV1, ...]:
        """
        Read bounded realtime messages from current user's metric/event streams.
        """
        ...


class UnavailableStrategyRealtimeOutputReader:
    """
    Fallback reader used when Strategy realtime Redis output is disabled or unwired.
    """

    def __init__(self, *, reason: str) -> None:
        if not reason.strip():
            raise ValueError("UnavailableStrategyRealtimeOutputReader requires reason")
        self._reason = reason.strip()

    def read_for_user(
        self,
        *,
        user_id: UserId,
        strategy_id: UUID | None,
        last_event_id: str,
        count: int,
        block_ms: int,
    ) -> tuple[StrategyRealtimeStreamMessageV1, ...]:
        _ = (user_id, strategy_id, last_event_id, count, block_ms)
        raise StrategyRealtimeStreamUnavailableError(self._reason)
