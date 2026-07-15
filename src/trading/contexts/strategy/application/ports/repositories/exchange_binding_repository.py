from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from trading.contexts.strategy.domain.entities import StrategyExchangeBinding
from trading.shared_kernel.primitives import OrganizationId, UserId


class StrategyExchangeBindingRepository(Protocol):
    def create(
        self, *, binding: StrategyExchangeBinding
    ) -> StrategyExchangeBinding | None: ...

    def get(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_id: UUID,
        binding_id: UUID,
    ) -> StrategyExchangeBinding | None: ...

    def list_for_strategy(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_id: UUID,
    ) -> tuple[StrategyExchangeBinding, ...]: ...

    def disable(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        strategy_id: UUID,
        binding_id: UUID,
        disabled_at: datetime,
    ) -> StrategyExchangeBinding | None: ...
