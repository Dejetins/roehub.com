from __future__ import annotations

from uuid import UUID

from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.repositories import StrategyRepository
from trading.contexts.strategy.application.use_cases._shared import require_owned_strategy
from trading.contexts.strategy.domain.entities import Strategy


class GetMyStrategyUseCase:
    """
    GetMyStrategyUseCase — load one strategy with explicit owner-only visibility checks.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/use_cases/_shared.py
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
      - apps/api/routes/strategies.py
    """

    def __init__(self, *, repository: StrategyRepository) -> None:
        """
        Initialize use-case with strategy repository dependency.

        Args:
            repository: Strategy repository port.
        Returns:
            None.
        Assumptions:
            Repository can load strategy by id without owner filter for explicit checks.
        Raises:
            ValueError: If repository dependency is missing.
        Side Effects:
            None.
        """
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("GetMyStrategyUseCase requires repository")
        self._repository = repository

    def execute(self, *, strategy_id: UUID, current_user: CurrentUser) -> Strategy:
        """
        Load one strategy and enforce owner-only visibility in use-case layer.

        Args:
            strategy_id: Target strategy identifier.
            current_user: Authenticated current user context.
        Returns:
            Strategy: Owned non-deleted strategy snapshot.
        Assumptions:
            Forbidden access should be deterministic and independent from SQL query tricks.
        Raises:
            RoehubError: If strategy is missing, forbidden, deleted, or storage fails.
        Side Effects:
            Reads one strategy from storage.
        """
        return require_owned_strategy(
            repository=self._repository,
            strategy_id=strategy_id,
            current_user=current_user,
        )
