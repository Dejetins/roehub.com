from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from trading.contexts.strategy.application.use_cases.scenario_matrix import (
        StrategyVariantScenarioMatrixReport,
    )


class StrategyVariantScenarioMatrixRepository(Protocol):
    def record(
        self, *, report: StrategyVariantScenarioMatrixReport
    ) -> StrategyVariantScenarioMatrixReport: ...
