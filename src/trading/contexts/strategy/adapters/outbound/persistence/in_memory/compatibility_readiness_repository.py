from __future__ import annotations

from trading.contexts.strategy.application.ports.repositories import (
    StrategyCompatibilityReadinessRepository,
)
from trading.contexts.strategy.application.use_cases.compatibility_readiness import (
    StrategyCompatibilityReadinessReport,
)


class InMemoryStrategyCompatibilityReadinessRepository(
    StrategyCompatibilityReadinessRepository
):
    def __init__(self) -> None:
        self.compatibility_reports: list[StrategyCompatibilityReadinessReport] = []

    def record(
        self, *, report: StrategyCompatibilityReadinessReport
    ) -> StrategyCompatibilityReadinessReport:
        self.compatibility_reports.append(report)
        return report
