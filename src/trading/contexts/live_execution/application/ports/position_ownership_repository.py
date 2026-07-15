from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from trading.contexts.live_execution.domain import StrategyPositionOwnership
from trading.shared_kernel.primitives import OrganizationId, UserId


class StrategyPositionOwnershipRepository(Protocol):
    def reserve(self, *, ownership: StrategyPositionOwnership) -> StrategyPositionOwnership: ...

    def update_state(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        state: str,
        reason: str,
        changed_at: datetime,
    ) -> StrategyPositionOwnership | None: ...

    def get_for_run(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_run_id: UUID,
    ) -> StrategyPositionOwnership | None: ...

    def get_blocking_for_scope(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        market_type: str,
        instrument_key: str,
    ) -> StrategyPositionOwnership | None: ...
