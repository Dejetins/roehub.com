from __future__ import annotations

from trading.contexts.strategy.application.ports.repositories import (
    StrategyVariantScenarioMatrixRepository,
)
from trading.contexts.strategy.application.use_cases.scenario_matrix import (
    StrategyVariantScenarioMatrixReport,
)


class InMemoryStrategyVariantScenarioMatrixRepository(
    StrategyVariantScenarioMatrixRepository
):
    def __init__(self) -> None:
        self.reports: list[StrategyVariantScenarioMatrixReport] = []

    def record(
        self, *, report: StrategyVariantScenarioMatrixReport
    ) -> StrategyVariantScenarioMatrixReport:
        self.reports.append(report)
        return report
