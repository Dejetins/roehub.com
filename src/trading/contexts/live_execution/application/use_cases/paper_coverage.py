from __future__ import annotations

from trading.contexts.live_execution.application.ports import (
    PaperScenarioCoverageRepository,
)
from trading.contexts.live_execution.domain import PaperScenarioCoverageResult
from trading.shared_kernel.primitives import OrganizationId, UserId


class PaperScenarioCoverageService:
    def __init__(self, *, repository: PaperScenarioCoverageRepository) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("PaperScenarioCoverageService requires repository")
        self._repository = repository

    def record(
        self, *, result: PaperScenarioCoverageResult
    ) -> PaperScenarioCoverageResult:
        return self._repository.record(result=result)

    def get_latest_by_scenario_key(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        scenario_key: str,
    ) -> PaperScenarioCoverageResult | None:
        return self._repository.get_latest_by_scenario_key(
            organization_id=organization_id,
            owner_user_id=owner_user_id,
            scenario_key=scenario_key,
        )
