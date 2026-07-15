from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import OrganizationId, UserId


class StrategyPositionOwnershipCoordinator(Protocol):
    def reserve_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        strategy_id: UUID,
        live_profile_id: UUID | None,
        strategy_run_id: UUID,
        market_type: str,
        instrument_key: str,
        position_mode: str,
        now: datetime,
        reason: str = "run_started",
    ) -> Any: ...

    def activate_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        now: datetime,
    ) -> Any: ...

    def mark_releasing_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        now: datetime,
        reason: str = "run_stopping",
    ) -> Any: ...

    def release_for_strategy_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        now: datetime,
        reason: str = "run_stopped",
    ) -> Any: ...
