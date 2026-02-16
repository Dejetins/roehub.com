from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.strategy.application.ports.current_user import CurrentUser
from trading.contexts.strategy.application.ports.repositories import (
    StrategyEventRepository,
    StrategyRepository,
)
from trading.contexts.strategy.application.use_cases.errors import (
    map_strategy_exception,
    strategy_forbidden,
    strategy_not_found,
)
from trading.contexts.strategy.domain.entities import Strategy, StrategyEvent
from trading.platform.errors import RoehubError


def ensure_utc_datetime(*, value: datetime, field_name: str) -> datetime:
    """
    Validate that datetime value is timezone-aware UTC and return it unchanged.

    Args:
        value: Datetime value to validate.
        field_name: Field label used in deterministic validation message.
    Returns:
        datetime: Same validated datetime object.
    Assumptions:
        Strategy timestamps are stored in UTC only.
    Raises:
        RoehubError: If datetime is naive or non-UTC.
    Side Effects:
        None.
    """
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise RoehubError(
            code="validation_error",
            message=f"{field_name} must be timezone-aware UTC datetime",
            details={},
        )
    if offset.total_seconds() != 0:
        raise RoehubError(
            code="validation_error",
            message=f"{field_name} must be UTC datetime",
            details={},
        )
    return value



def require_owned_strategy(
    *,
    repository: StrategyRepository,
    strategy_id: UUID,
    current_user: CurrentUser,
) -> Strategy:
    """
    Load strategy by id and enforce explicit owner-only visibility rule in use-case layer.

    Args:
        repository: Strategy repository port.
        strategy_id: Requested strategy identifier.
        current_user: Authenticated current user context.
    Returns:
        Strategy: Owned and non-deleted strategy aggregate.
    Assumptions:
        Ownership/visibility checks are business rules and must not rely solely on SQL filters.
    Raises:
        RoehubError: If strategy is missing, deleted, forbidden, or storage mapping fails.
    Side Effects:
        Reads one strategy snapshot from storage.
    """
    try:
        strategy = repository.find_any_by_strategy_id(strategy_id=strategy_id)
    except Exception as error:  # noqa: BLE001
        raise map_strategy_exception(error=error) from error

    if strategy is None or strategy.is_deleted:
        raise strategy_not_found(strategy_id=strategy_id)
    if strategy.user_id != current_user.user_id:
        raise strategy_forbidden(strategy_id=strategy_id)
    return strategy



def append_strategy_event(
    *,
    repository: StrategyEventRepository | None,
    strategy_id: UUID,
    current_user: CurrentUser,
    event_type: str,
    ts: datetime,
    payload_json: Mapping[str, Any],
    run_id: UUID | None = None,
) -> None:
    """
    Append strategy/run event when event repository is configured.

    Args:
        repository: Optional strategy event repository port.
        strategy_id: Strategy identifier for event stream.
        current_user: Authenticated current user context.
        event_type: Deterministic event type literal.
        ts: Event timestamp in UTC.
        payload_json: Event payload mapping.
        run_id: Optional run identifier for run-scoped events.
    Returns:
        None.
    Assumptions:
        Event append failures should fail-fast and surface as RoehubError.
    Raises:
        RoehubError: If event append operation fails.
    Side Effects:
        Writes one event row when repository is configured.
    """
    if repository is None:
        return

    try:
        event = StrategyEvent.create(
            user_id=current_user.user_id,
            strategy_id=strategy_id,
            run_id=run_id,
            ts=ts,
            event_type=event_type,
            payload_json=payload_json,
        )
        repository.append(event=event)
    except Exception as error:  # noqa: BLE001
        raise map_strategy_exception(error=error) from error
