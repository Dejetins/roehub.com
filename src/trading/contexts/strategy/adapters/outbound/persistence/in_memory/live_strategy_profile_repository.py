from __future__ import annotations

from uuid import UUID

from trading.contexts.strategy.application.ports.repositories import (
    LiveStrategyProfileRepository,
)
from trading.contexts.strategy.domain.entities.live_strategy_profile import (
    LiveStrategyProfile,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class InMemoryLiveStrategyProfileRepository(LiveStrategyProfileRepository):
    def __init__(self) -> None:
        self._profiles: dict[UUID, LiveStrategyProfile] = {}

    def create(self, *, profile: LiveStrategyProfile) -> LiveStrategyProfile | None:
        if (
            self.get_for_strategy(
                organization_id=profile.organization_id,
                owner_user_id=profile.owner_user_id,
                strategy_id=profile.strategy_id,
            )
            is not None
        ):
            return None
        self._profiles[profile.profile_id] = profile
        return profile

    def get_for_strategy(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_id: UUID,
    ) -> LiveStrategyProfile | None:
        for profile in self._profiles.values():
            if (
                profile.organization_id == organization_id
                and profile.owner_user_id == owner_user_id
                and profile.strategy_id == strategy_id
            ):
                return profile
        return None

    def update(self, *, profile: LiveStrategyProfile) -> LiveStrategyProfile:
        self._profiles[profile.profile_id] = profile
        return profile
