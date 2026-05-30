from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from trading.contexts.strategy.application.use_cases.compatibility_readiness import (
        StrategyCompatibilityReadinessReport,
    )


class StrategyCompatibilityReadinessRepository(Protocol):
    def record(
        self, *, report: StrategyCompatibilityReadinessReport
    ) -> StrategyCompatibilityReadinessReport: ...
