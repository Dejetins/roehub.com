from __future__ import annotations

from typing import Any, Mapping

from trading.contexts.strategy.application.ports.clock import StrategyClock
from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.repositories import (
    StrategyEventRepository,
    StrategyRepository,
)
from trading.contexts.strategy.application.use_cases._shared import (
    append_strategy_event,
    ensure_utc_datetime,
)
from trading.contexts.strategy.application.use_cases.errors import (
    map_strategy_exception,
    validation_error,
)
from trading.contexts.strategy.domain.entities import Strategy, StrategySpecV1
from trading.platform.errors import RoehubError


class CreateStrategyUseCase:
    """
    CreateStrategyUseCase — create immutable Strategy v1 aggregate for authenticated owner.

    Docs:
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy.py
      - src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py
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
        Initialize immutable strategy creation use-case dependencies.

        Args:
            repository: Strategy repository port.
            clock: Clock port for deterministic UTC timestamps.
            event_repository: Optional append-only event repository port.
        Returns:
            None.
        Assumptions:
            Creation writes one strategy snapshot and optional strategy-created event.
        Raises:
            ValueError: If required dependencies are missing.
        Side Effects:
            None.
        """
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateStrategyUseCase requires repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("CreateStrategyUseCase requires clock")
        self._repository = repository
        self._clock = clock
        self._event_repository = event_repository

    def execute(self, *, spec_payload: Mapping[str, Any], current_user: CurrentUser) -> Strategy:
        """
        Create immutable strategy snapshot from API/domain payload for current owner.

        Args:
            spec_payload: Strategy spec JSON mapping.
            current_user: Authenticated current user context.
        Returns:
            Strategy: Persisted immutable strategy snapshot.
        Assumptions:
            Strategy updates are forbidden; modifications happen through clone use-case only.
        Raises:
            RoehubError: If payload is invalid or repository/event operations fail.
        Side Effects:
            Persists strategy row and appends strategy-created event when repository is configured.
        """
        if not isinstance(spec_payload, Mapping):
            raise validation_error(
                message="Strategy spec payload must be object",
                errors=(
                    {
                        "path": "body.spec",
                        "code": "type_error",
                        "message": "Strategy spec payload must be object",
                    },
                ),
            )

        try:
            spec = StrategySpecV1.from_json(payload=dict(spec_payload))
            created_at = ensure_utc_datetime(value=self._clock.now(), field_name="clock.now")
            strategy = Strategy.create(
                user_id=current_user.user_id,
                spec=spec,
                created_at=created_at,
            )
            persisted = self._repository.create(strategy=strategy)
            append_strategy_event(
                repository=self._event_repository,
                strategy_id=persisted.strategy_id,
                current_user=current_user,
                event_type="strategy_created",
                ts=created_at,
                payload_json={
                    "strategy_id": str(persisted.strategy_id),
                    "schema_version": persisted.spec.schema_version,
                    "spec_kind": persisted.spec.spec_kind,
                },
            )
            return persisted
        except RoehubError:
            raise
        except Exception as error:  # noqa: BLE001
            raise map_strategy_exception(error=error) from error
