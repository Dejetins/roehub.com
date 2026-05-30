from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class ExchangeConnectionReadiness:
    eligible: bool
    reason: str
    exchange_name: str | None = None
    market_type: str | None = None


class ExchangeConnectionReadinessChecker(Protocol):
    def check_trading_ready(
        self, *, owner_user_id: UserId, exchange_connection_id: UUID
    ) -> ExchangeConnectionReadiness: ...
