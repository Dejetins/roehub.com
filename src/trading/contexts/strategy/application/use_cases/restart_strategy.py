from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
from uuid import UUID, uuid4

from trading.contexts.strategy.application.ports.clock import StrategyClock
from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.position_ownership import (
    StrategyPositionOwnershipCoordinator,
)
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

RESTART_METADATA_KEY = "restart"
RESTART_PENDING_STATE = "pending_start"
RESTART_SUCCESSOR_STATE = "successor_started"


class RestartStrategyUseCase:
    """
    RestartStrategyUseCase — persist explicit restart operation on an active Strategy run.

    Docs:
      - docs/architecture/live_execution/live-execution-universal-order-gateway-v1.md
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - src/trading/contexts/strategy/application/use_cases/run_strategy.py
      - src/trading/contexts/strategy/application/use_cases/stop_strategy.py
    """

    def __init__(
        self,
        *,
        strategy_repository: StrategyRepository,
        run_repository: StrategyRunRepository,
        clock: StrategyClock,
        event_repository: StrategyEventRepository | None = None,
        position_ownership_coordinator: StrategyPositionOwnershipCoordinator | None = None,
    ) -> None:
        if strategy_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RestartStrategyUseCase requires strategy_repository")
        if run_repository is None:  # type: ignore[truthy-bool]
            raise ValueError("RestartStrategyUseCase requires run_repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("RestartStrategyUseCase requires clock")

        self._strategy_repository = strategy_repository
        self._run_repository = run_repository
        self._clock = clock
        self._event_repository = event_repository
        self._position_ownership_coordinator = position_ownership_coordinator

    def execute(self, *, strategy_id: UUID, current_user: CurrentUser) -> StrategyRun:
        """
        Queue restart by transitioning current active run to `stopping`.

        The live-runner worker owns drain and successor creation. This keeps the
        one-active-run invariant intact because the successor is created only
        after the previous run reaches terminal `stopped`.
        """
        strategy = require_owned_strategy(
            repository=self._strategy_repository,
            strategy_id=strategy_id,
            current_user=current_user,
        )

        try:
            active_run = self._run_repository.find_active_for_strategy(
                user_id=current_user.user_id,
                strategy_id=strategy.strategy_id,
            )
            if active_run is None:
                raise strategy_conflict(
                    message="Strategy has no active run to restart",
                    details={"strategy_id": str(strategy.strategy_id)},
                )
            if _restart_is_pending(metadata_json=active_run.metadata_json):
                restart = _restart_metadata(metadata_json=active_run.metadata_json)
                raise strategy_conflict(
                    message="Strategy restart is already pending",
                    details={
                        "strategy_id": str(strategy.strategy_id),
                        "run_id": str(active_run.run_id),
                        "current_state": active_run.state,
                        "restart_operation_id": str(restart.get("operation_id", "")),
                    },
                )
            if active_run.state == "stopping":
                raise strategy_conflict(
                    message="Strategy run is already stopping",
                    details={
                        "strategy_id": str(strategy.strategy_id),
                        "run_id": str(active_run.run_id),
                        "current_state": active_run.state,
                    },
                )

            requested_at = ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
            operation_id = uuid4()
            metadata_json = _with_restart_metadata(
                metadata_json=active_run.metadata_json,
                operation_id=operation_id,
                requested_at=requested_at,
            )
            stopping = active_run.transition_to(
                next_state="stopping",
                changed_at=requested_at,
                checkpoint_ts_open=active_run.checkpoint_ts_open,
                last_error=active_run.last_error,
            )
            stopping = StrategyRun(
                run_id=stopping.run_id,
                user_id=stopping.user_id,
                strategy_id=stopping.strategy_id,
                state=stopping.state,
                started_at=stopping.started_at,
                stopped_at=stopping.stopped_at,
                checkpoint_ts_open=stopping.checkpoint_ts_open,
                last_error=stopping.last_error,
                updated_at=stopping.updated_at,
                metadata_json=metadata_json,
            )
            persisted_stopping = self._run_repository.update(run=stopping)
            if self._position_ownership_coordinator is not None:
                self._position_ownership_coordinator.mark_releasing_for_strategy_run(
                    owner_user_id=current_user.user_id,
                    strategy_run_id=persisted_stopping.run_id,
                    now=requested_at,
                    reason="run_restart_requested",
                )

            append_strategy_event(
                repository=self._event_repository,
                strategy_id=strategy.strategy_id,
                current_user=current_user,
                event_type="run_restart_requested",
                ts=requested_at,
                payload_json={
                    "strategy_id": str(strategy.strategy_id),
                    "run_id": str(persisted_stopping.run_id),
                    "state": persisted_stopping.state,
                    "restart_operation_id": str(operation_id),
                },
                run_id=persisted_stopping.run_id,
            )
            return persisted_stopping
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_strategy_exception(error=error) from error


def _with_restart_metadata(
    *,
    metadata_json: Mapping[str, Any],
    operation_id: UUID,
    requested_at: datetime,
) -> dict[str, Any]:
    metadata = dict(metadata_json)
    metadata[RESTART_METADATA_KEY] = {
        "operation_id": str(operation_id),
        "state": RESTART_PENDING_STATE,
        "requested_at": requested_at.isoformat().replace("+00:00", "Z"),
    }
    return metadata


def _restart_metadata(*, metadata_json: Mapping[str, Any]) -> Mapping[str, Any]:
    value = metadata_json.get(RESTART_METADATA_KEY)
    if isinstance(value, Mapping):
        return value
    return {}


def _restart_is_pending(*, metadata_json: Mapping[str, Any]) -> bool:
    return _restart_metadata(metadata_json=metadata_json).get("state") == RESTART_PENDING_STATE
