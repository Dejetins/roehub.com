from __future__ import annotations

from typing import Any, Callable
from uuid import UUID, uuid4

from trading.contexts.strategy.application.ports.clock import StrategyClock
from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.repositories import (
    StrategyEventRepository,
    StrategyRepository,
    StrategyRunRepository,
)
from trading.contexts.strategy.application.services import estimate_strategy_warmup_bars
from trading.contexts.strategy.application.use_cases._shared import (
    append_strategy_event,
    ensure_utc_datetime,
    require_owned_strategy,
)
from trading.contexts.strategy.application.use_cases.errors import (
    map_strategy_exception,
    strategy_conflict,
)
from trading.contexts.strategy.domain.entities import StrategyRun, StrategySpecV1
from trading.platform.errors import RoehubError


class RunStrategyUseCase:
    """
    RunStrategyUseCase — start strategy run lifecycle and transition to running state.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy_run.py
      - src/trading/contexts/strategy/application/services/warmup_estimator.py
      - apps/api/routes/strategies.py
    """

    def __init__(
        self,
        *,
        strategy_repository: StrategyRepository,
        run_repository: StrategyRunRepository,
        clock: StrategyClock,
        event_repository: StrategyEventRepository | None = None,
        warmup_estimator: Callable[[StrategySpecV1], int] | None = None,
    ) -> None:
        """
        Initialize strategy run-control use-case dependencies.

        Args:
            strategy_repository: Strategy repository port.
            run_repository: Strategy run repository port.
            clock: Clock port for deterministic UTC timestamps.
            event_repository: Optional append-only event repository port.
            warmup_estimator: Optional deterministic warmup estimator callable.
        Returns:
            None.
        Assumptions:
            Run repository enforces one-active-run invariant with transaction-safe storage guard.
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
        self._warmup_estimator = warmup_estimator or _default_warmup_estimator

    def execute(self, *, strategy_id: UUID, current_user: CurrentUser) -> StrategyRun:
        """
        Start strategy run with deterministic warmup metadata and transition to running state.

        Args:
            strategy_id: Target strategy identifier.
            current_user: Authenticated current user context.
        Returns:
            StrategyRun: Persisted run snapshot in `running` state.
        Assumptions:
            Run sequence follows deterministic transitions: starting -> warming_up -> running.
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

            warmup_bars = self._warmup_estimator(strategy.spec)
            metadata_json = _build_run_metadata(warmup_bars=warmup_bars)

            started = StrategyRun.start(
                run_id=uuid4(),
                user_id=current_user.user_id,
                strategy_id=strategy.strategy_id,
                started_at=run_started_at,
                metadata_json=metadata_json,
            )
            persisted_started = self._run_repository.create(run=started)

            warmed_up = persisted_started.transition_to(
                next_state="warming_up",
                changed_at=run_started_at,
                checkpoint_ts_open=None,
                last_error=None,
            )
            persisted_warming = self._run_repository.update(run=warmed_up)

            running = persisted_warming.transition_to(
                next_state="running",
                changed_at=run_started_at,
                checkpoint_ts_open=None,
                last_error=None,
            )
            persisted_running = self._run_repository.update(run=running)

            append_strategy_event(
                repository=self._event_repository,
                strategy_id=strategy.strategy_id,
                current_user=current_user,
                event_type="run_started",
                ts=run_started_at,
                payload_json={
                    "strategy_id": str(strategy.strategy_id),
                    "run_id": str(persisted_running.run_id),
                    "state": persisted_running.state,
                    "warmup_bars": warmup_bars,
                },
                run_id=persisted_running.run_id,
            )
            return persisted_running
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_strategy_exception(error=error) from error



def _default_warmup_estimator(spec: StrategySpecV1) -> int:
    """
    Delegate deterministic warmup estimation to default strategy warmup estimator service.

    Args:
        spec: Strategy spec used to derive warmup bars.
    Returns:
        int: Deterministic warmup bars count.
    Assumptions:
        Service estimator is pure and deterministic for identical strategy specs.
    Raises:
        ValueError: If estimator cannot parse indicator params payload.
    Side Effects:
        None.
    """
    return estimate_strategy_warmup_bars(spec=spec)



def _build_run_metadata(*, warmup_bars: int) -> dict[str, Any]:
    """
    Build deterministic run metadata payload including warmup traceability attributes.

    Args:
        warmup_bars: Deterministic warmup bars count.
    Returns:
        dict[str, Any]: Metadata mapping persisted with run snapshot.
    Assumptions:
        Warmup bars must be positive integer (`>= 1`).
    Raises:
        RoehubError: If warmup bars value is invalid.
    Side Effects:
        None.
    """
    if warmup_bars <= 0:
        raise RoehubError(
            code="validation_error",
            message="Warmup bars must be positive",
            details={
                "errors": [
                    {
                        "path": "warmup_bars",
                        "code": "invalid_value",
                        "message": "warmup_bars must be > 0",
                    }
                ]
            },
        )

    return {
        "warmup": {
            "algorithm": "numeric_max_param_v1",
            "bars": warmup_bars,
        }
    }
