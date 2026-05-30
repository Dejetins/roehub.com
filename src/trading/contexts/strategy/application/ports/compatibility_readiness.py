from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from trading.contexts.strategy.application.ports.current_user import CurrentUser

if TYPE_CHECKING:
    from trading.contexts.strategy.application.use_cases.compatibility_readiness import (
        StrategyCompatibilityReadinessReport,
    )


class StrategyCompatibilityReadinessChecker(Protocol):
    def check_strategy(
        self, *, strategy_id: UUID, current_user: CurrentUser
    ) -> StrategyCompatibilityReadinessReport: ...
