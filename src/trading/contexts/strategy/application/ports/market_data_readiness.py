from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

MarketDataReadinessState = Literal["ready", "missing", "stale", "pending"]


@dataclass(frozen=True, slots=True)
class MarketDataReadinessSnapshot:
    state: MarketDataReadinessState
    reason_code: str
    stream_name: str
    stream_length: int | None
    last_message_id: str | None
    last_observed_at: datetime | None
    age_seconds: int | None


class MarketDataReadinessReader(Protocol):
    def check(
        self,
        *,
        instrument_key: str,
        timeframe: str,
        observed_at: datetime,
    ) -> MarketDataReadinessSnapshot: ...
