from __future__ import annotations

from trading.contexts.live_execution.application.ports import (
    PaperScenarioCoverageRepository,
)
from trading.contexts.live_execution.domain import PaperScenarioCoverageResult
from trading.shared_kernel.primitives import UserId


class InMemoryPaperScenarioCoverageRepository(PaperScenarioCoverageRepository):
    def __init__(self) -> None:
        self.results: list[PaperScenarioCoverageResult] = []

    def record(
        self, *, result: PaperScenarioCoverageResult
    ) -> PaperScenarioCoverageResult:
        for index, item in enumerate(self.results):
            if (
                item.owner_user_id == result.owner_user_id
                and item.scenario_key == result.scenario_key
            ):
                self.results[index] = result
                return result
        self.results.append(result)
        return result

    def get_latest_by_scenario_key(
        self, *, owner_user_id: UserId, scenario_key: str
    ) -> PaperScenarioCoverageResult | None:
        matches = [
            item
            for item in self.results
            if item.owner_user_id == owner_user_id and item.scenario_key == scenario_key
        ]
        if not matches:
            return None
        return max(matches, key=lambda item: (item.checked_at, str(item.coverage_result_id)))
