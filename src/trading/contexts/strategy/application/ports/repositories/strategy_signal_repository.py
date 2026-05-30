from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.contexts.strategy.domain.entities import StrategySignal
from trading.shared_kernel.primitives import UserId


class StrategySignalRepository(Protocol):
    def record(self, *, signal: StrategySignal) -> StrategySignal: ...

    def list_latest_for_strategy(
        self,
        *,
        owner_user_id: UserId,
        strategy_id: UUID,
        limit: int,
    ) -> tuple[StrategySignal, ...]: ...
