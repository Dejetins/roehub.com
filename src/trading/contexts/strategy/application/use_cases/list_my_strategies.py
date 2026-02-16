from __future__ import annotations

from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.repositories import StrategyRepository
from trading.contexts.strategy.application.use_cases.errors import map_strategy_exception
from trading.contexts.strategy.domain.entities import Strategy


class ListMyStrategiesUseCase:
    """
    ListMyStrategiesUseCase — list non-deleted strategies owned by current user.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
      - apps/api/routes/strategies.py
      - apps/api/wiring/modules/strategy.py
    """

    def __init__(self, *, repository: StrategyRepository) -> None:
        """
        Initialize use-case with strategy repository dependency.

        Args:
            repository: Strategy repository port.
        Returns:
            None.
        Assumptions:
            Repository provides deterministic list ordering when requested.
        Raises:
            ValueError: If repository dependency is missing.
        Side Effects:
            None.
        """
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("ListMyStrategiesUseCase requires repository")
        self._repository = repository

    def execute(self, *, current_user: CurrentUser) -> tuple[Strategy, ...]:
        """
        Return deterministic owner-local list of non-deleted strategies.

        Args:
            current_user: Authenticated current user context.
        Returns:
            tuple[Strategy, ...]: Deterministically sorted strategy snapshots.
        Assumptions:
            Only owner strategies are visible and soft-deleted rows are excluded.
        Raises:
            RoehubError: If repository read fails.
        Side Effects:
            Reads strategy snapshots from storage.
        """
        try:
            strategies = self._repository.list_for_user(
                user_id=current_user.user_id,
                include_deleted=False,
            )
        except Exception as error:  # noqa: BLE001
            raise map_strategy_exception(error=error) from error

        return tuple(
            sorted(
                strategies,
                key=lambda strategy: (strategy.created_at, str(strategy.strategy_id)),
            )
        )
