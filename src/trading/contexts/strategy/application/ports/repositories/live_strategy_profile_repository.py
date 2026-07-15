from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.contexts.strategy.domain.entities.live_strategy_profile import (
    LiveStrategyProfile,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class LiveStrategyProfileRepository(Protocol):
    def create(self, *, profile: LiveStrategyProfile) -> LiveStrategyProfile | None: ...

    def get_for_strategy(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_id: UUID,
    ) -> LiveStrategyProfile | None: ...

    def update(self, *, profile: LiveStrategyProfile) -> LiveStrategyProfile: ...
