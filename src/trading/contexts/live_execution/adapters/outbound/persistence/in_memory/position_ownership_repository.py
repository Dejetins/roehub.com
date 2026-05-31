from __future__ import annotations

from datetime import datetime
from typing import cast
from uuid import UUID

from trading.contexts.live_execution.application.ports import (
    StrategyPositionOwnershipRepository,
)
from trading.contexts.live_execution.domain import (
    BLOCKING_POSITION_OWNERSHIP_STATES,
    StrategyPositionOwnership,
    StrategyPositionOwnershipConflictError,
    StrategyPositionOwnershipState,
)
from trading.shared_kernel.primitives import UserId


class InMemoryStrategyPositionOwnershipRepository(StrategyPositionOwnershipRepository):
    def __init__(self) -> None:
        self.ownerships: list[StrategyPositionOwnership] = []

    def reserve(self, *, ownership: StrategyPositionOwnership) -> StrategyPositionOwnership:
        existing = self.get_blocking_for_scope(
            owner_user_id=ownership.owner_user_id,
            exchange_connection_id=ownership.exchange_connection_id,
            market_type=ownership.market_type,
            instrument_key=ownership.instrument_key,
        )
        if existing is not None:
            raise StrategyPositionOwnershipConflictError(existing=existing)
        self.ownerships.append(ownership)
        return ownership

    def update_state(
        self,
        *,
        owner_user_id: UserId,
        strategy_run_id: UUID,
        state: str,
        reason: str,
        changed_at: datetime,
    ) -> StrategyPositionOwnership | None:
        for index, item in enumerate(self.ownerships):
            if item.owner_user_id == owner_user_id and item.strategy_run_id == strategy_run_id:
                updated = item.with_state(
                    state=cast(StrategyPositionOwnershipState, state),
                    reason=reason,
                    changed_at=changed_at,
                )
                self.ownerships[index] = updated
                return updated
        return None

    def get_for_run(
        self, *, owner_user_id: UserId, strategy_run_id: UUID
    ) -> StrategyPositionOwnership | None:
        matches = [
            item
            for item in self.ownerships
            if item.owner_user_id == owner_user_id and item.strategy_run_id == strategy_run_id
        ]
        if not matches:
            return None
        return matches[-1]

    def get_blocking_for_scope(
        self,
        *,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        market_type: str,
        instrument_key: str,
    ) -> StrategyPositionOwnership | None:
        matches = [
            item
            for item in self.ownerships
            if item.owner_user_id == owner_user_id
            and item.exchange_connection_id == exchange_connection_id
            and item.market_type == market_type
            and item.instrument_key == instrument_key
            and item.state in BLOCKING_POSITION_OWNERSHIP_STATES
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: (item.acquired_at, str(item.ownership_id)))
