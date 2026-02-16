from __future__ import annotations

from uuid import UUID

from trading.contexts.strategy.application.ports.clock import StrategyClock
from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.repositories import (
    StrategyEventRepository,
    StrategyRepository,
)
from trading.contexts.strategy.application.use_cases._shared import (
    append_strategy_event,
    ensure_utc_datetime,
    require_owned_strategy,
)
from trading.contexts.strategy.application.use_cases.errors import (
    map_strategy_exception,
    strategy_conflict,
)
from trading.platform.errors import RoehubError


class DeleteStrategyUseCase:
    """
    DeleteStrategyUseCase — soft-delete owned strategy snapshot.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
      - src/trading/contexts/strategy/domain/entities/strategy.py
      - apps/api/routes/strategies.py
    """

    def __init__(
        self,
        *,
        repository: StrategyRepository,
        clock: StrategyClock,
        event_repository: StrategyEventRepository | None = None,
    ) -> None:
        """
        Initialize strategy soft-delete use-case dependencies.

        Args:
            repository: Strategy repository port.
            clock: Clock port for deterministic UTC timestamps.
            event_repository: Optional append-only event repository port.
        Returns:
            None.
        Assumptions:
            Soft-delete toggles `is_deleted` only and preserves immutable strategy spec.
        Raises:
            ValueError: If required dependencies are missing.
        Side Effects:
            None.
        """
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("DeleteStrategyUseCase requires repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("DeleteStrategyUseCase requires clock")
        self._repository = repository
        self._clock = clock
        self._event_repository = event_repository

    def execute(self, *, strategy_id: UUID, current_user: CurrentUser) -> None:
        """
        Soft-delete one owned strategy and append deterministic strategy-deleted event.

        Args:
            strategy_id: Target strategy identifier.
            current_user: Authenticated current user context.
        Returns:
            None.
        Assumptions:
            Deleted strategies are excluded from list/get queries by default.
        Raises:
            RoehubError: If strategy is missing/forbidden or soft-delete write fails.
        Side Effects:
            Updates strategy row and appends strategy-deleted event when repository is configured.
        """
        strategy = require_owned_strategy(
            repository=self._repository,
            strategy_id=strategy_id,
            current_user=current_user,
        )
        deleted_at = ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")

        try:
            changed = self._repository.soft_delete(
                user_id=current_user.user_id,
                strategy_id=strategy.strategy_id,
            )
            if not changed:
                raise strategy_conflict(
                    message="Strategy is already deleted",
                    details={"strategy_id": str(strategy.strategy_id)},
                )
            append_strategy_event(
                repository=self._event_repository,
                strategy_id=strategy.strategy_id,
                current_user=current_user,
                event_type="strategy_deleted",
                ts=deleted_at,
                payload_json={
                    "strategy_id": str(strategy.strategy_id),
                },
            )
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_strategy_exception(error=error) from error
