from __future__ import annotations

from typing import Protocol

from trading.contexts.live_execution.domain import PaperScenarioCoverageResult
from trading.shared_kernel.primitives import UserId


class PaperScenarioCoverageRepository(Protocol):
    def record(
        self, *, result: PaperScenarioCoverageResult
    ) -> PaperScenarioCoverageResult: ...

    def get_latest_by_scenario_key(
        self, *, owner_user_id: UserId, scenario_key: str
    ) -> PaperScenarioCoverageResult | None: ...
