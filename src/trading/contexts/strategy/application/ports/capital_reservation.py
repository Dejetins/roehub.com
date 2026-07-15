from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Protocol
from uuid import UUID

from trading.contexts.strategy.domain.entities import StrategySignal
from trading.shared_kernel.primitives import OrganizationId, UserId


class StrategyCapitalReservationCoordinator(Protocol):
    def reserve_virtual_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_id: UUID,
        live_profile_id: UUID | None,
        strategy_run_id: UUID,
        requested_amount: Decimal,
        now: datetime,
    ) -> object: ...

    def reserve_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        strategy_id: UUID,
        live_profile_id: UUID | None,
        strategy_run_id: UUID,
        requested_amount: Decimal,
        now: datetime,
    ) -> object: ...

    def release_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        now: datetime,
        reason: str,
    ) -> object | None: ...


class StrategyPaperAccountingRecorder(Protocol):
    def record_paper_signal(self, *, signal: StrategySignal) -> object | None: ...
