from __future__ import annotations

from uuid import UUID, uuid4

from trading.contexts.strategy.application.ports.clock import StrategyClock
from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.repositories import (
    StrategyEventRepository,
    StrategyRepository,
    StrategyRunRepository,
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
from trading.contexts.strategy.domain.entities import StrategyRun
from trading.platform.errors import RoehubError


class RunStrategyUseCase:
    """
    RunStrategyUseCase — create new Strategy run in `starting` state.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy_run.py
      - src/trading/contexts/strategy/application/services/live_runner.py
      - apps/api/routes/strategies.py
    """

    def __init__(
        self,
        *,
        strategy_repository: StrategyRepository,
        run_repository: StrategyRunRepository,
        clock: StrategyClock,
        event_repository: StrategyEventRepository | None = None,
    ) -> None:
        """
        Initialize strategy run-control use-case dependencies.

        Args:
            strategy_repository: Strategy repository port.
            run_repository: Strategy run repository port.
            clock: Clock port for deterministic UTC timestamps.
            event_repository: Optional append-only event repository port.
        Returns:
            None.
        Assumptions:
            Runner is responsible for warmup computation and `starting -> running` progression.
        Raises:
            ValueError: If required dependencies are missing.
        Side Effects:
            None.
        """
        if strategy_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RunStrategyUseCase requires strategy_repository")
        if run_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RunStrategyUseCase requires run_repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("RunStrategyUseCase requires clock")

        self._strategy_repository = strategy_repository
        self._run_repository = run_repository
        self._clock = clock
        self._event_repository = event_repository

    def execute(self, *, strategy_id: UUID, current_user: CurrentUser) -> StrategyRun:
        """
        Start strategy run by creating one immutable snapshot in `starting` state.

        Args:
            strategy_id: Target strategy identifier.
            current_user: Authenticated current user context.
        Returns:
            StrategyRun: Persisted run snapshot in `starting` state.
        Assumptions:
            Live runner worker performs warmup and runtime transitions after API creates run.
        Raises:
            RoehubError: If ownership, state, or storage invariants are violated.
        Side Effects:
            Writes run rows and appends run lifecycle events.
        """
        strategy = require_owned_strategy(
            repository=self._strategy_repository,
            strategy_id=strategy_id,
            current_user=current_user,
        )
        run_started_at = ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")

        try:
            active_run = self._run_repository.find_active_for_strategy(
                user_id=current_user.user_id,
                strategy_id=strategy.strategy_id,
            )
            if active_run is not None:
                raise strategy_conflict(
                    message="Strategy already has active run",
                    details={
                        "strategy_id": str(strategy.strategy_id),
                        "run_id": str(active_run.run_id),
                        "current_state": active_run.state,
                    },
                )

            started = StrategyRun.start(
                run_id=uuid4(),
                user_id=current_user.user_id,
                strategy_id=strategy.strategy_id,
                started_at=run_started_at,
                metadata_json={},
            )
            persisted_started = self._run_repository.create(run=started)

            append_strategy_event(
                repository=self._event_repository,
                strategy_id=strategy.strategy_id,
                current_user=current_user,
                event_type="run_started",
                ts=run_started_at,
                payload_json={
                    "strategy_id": str(strategy.strategy_id),
                    "run_id": str(persisted_started.run_id),
                    "state": persisted_started.state,
                },
                run_id=persisted_started.run_id,
            )
            return persisted_started
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_strategy_exception(error=error) from error
