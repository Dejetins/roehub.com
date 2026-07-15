from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Protocol
from uuid import UUID

from trading.contexts.live_execution.domain import (
    CapitalReservation,
    PaperFill,
    PaperOrder,
    StrategyPaperAccountingSnapshot,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class PaperAccountingRepository(Protocol):
    def record_reservation(self, *, reservation: CapitalReservation) -> CapitalReservation: ...

    def release_reservation_for_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        changed_at: datetime,
        reason: str,
    ) -> CapitalReservation | None: ...

    def get_active_reservation_for_run(
        self, *, organization_id: OrganizationId, owner_user_id: UserId, strategy_run_id: UUID
    ) -> CapitalReservation | None: ...

    def sum_active_reserved(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        asset: str,
    ) -> Decimal: ...

    def record_paper_execution(
        self,
        *,
        order: PaperOrder,
        fill: PaperFill,
        accounting: StrategyPaperAccountingSnapshot,
    ) -> StrategyPaperAccountingSnapshot: ...

    def get_latest_accounting_for_strategy(
        self, *, organization_id: OrganizationId, owner_user_id: UserId, strategy_id: UUID
    ) -> StrategyPaperAccountingSnapshot | None: ...
