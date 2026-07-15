from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol
from uuid import UUID

from trading.shared_kernel.primitives import OrganizationId, UserId


@dataclass(frozen=True, slots=True)
class ExchangeConnectionReadinessContext:
    mode: str
    market_type: str
    symbol: str
    direction: str
    notional: Decimal


@dataclass(frozen=True, slots=True)
class ExchangeConnectionReadiness:
    eligible: bool
    reason: str
    exchange_name: str | None = None
    market_type: str | None = None


class ExchangeConnectionReadinessChecker(Protocol):
    def check_trading_ready(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        context: ExchangeConnectionReadinessContext | None = None,
    ) -> ExchangeConnectionReadiness: ...
