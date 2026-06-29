from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.contexts.market_data.application.dto import MarketDataCandleRepairAuditEvent


class CandleRepairAuditRepository(Protocol):
    """
    Persistence port for durable Market Data live-tail repair audit events.
    """

    def record(
        self,
        *,
        event: MarketDataCandleRepairAuditEvent,
    ) -> MarketDataCandleRepairAuditEvent:
        """
        Append one repair audit event and return the persisted row.
        """
        ...

    def get_by_id(self, *, event_id: UUID) -> MarketDataCandleRepairAuditEvent | None:
        """
        Return one repair audit event by id.
        """
        ...

    def list_for_correlation(
        self,
        *,
        correlation_id: str,
    ) -> tuple[MarketDataCandleRepairAuditEvent, ...]:
        """
        Return repair audit events for one redacted correlation id in deterministic order.
        """
        ...
