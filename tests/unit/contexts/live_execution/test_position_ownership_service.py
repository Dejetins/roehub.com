from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryStrategyPositionOwnershipRepository,
)
from trading.contexts.live_execution.application import StrategyPositionOwnershipService
from trading.contexts.live_execution.domain import StrategyPositionOwnershipConflictError
from trading.shared_kernel.primitives import OrganizationId, UserId

_FIRST_ORGANIZATION_ID = OrganizationId(UUID("00000000-0000-4000-8000-000000014000"))
_SECOND_ORGANIZATION_ID = OrganizationId(UUID("00000000-0000-4000-8000-000000014999"))
_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000014001")
_CONNECTION_ID = UUID("00000000-0000-0000-0000-000000014101")
_STRATEGY_ID = UUID("00000000-0000-0000-0000-000000014201")
_RUN_ID = UUID("00000000-0000-0000-0000-000000014301")
_NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


def test_position_ownership_scope_isolated_by_organization() -> None:
    repository = InMemoryStrategyPositionOwnershipRepository()
    service = StrategyPositionOwnershipService(repository=repository)

    first = _reserve(service=service, organization_id=_FIRST_ORGANIZATION_ID)
    second = _reserve(service=service, organization_id=_SECOND_ORGANIZATION_ID)

    assert first.ownership_id != second.ownership_id
    assert (
        repository.get_for_run(
            organization_id=_FIRST_ORGANIZATION_ID,
            owner_user_id=_USER_ID,
            strategy_run_id=_RUN_ID,
        )
        == first
    )
    assert (
        repository.get_for_run(
            organization_id=_SECOND_ORGANIZATION_ID,
            owner_user_id=_USER_ID,
            strategy_run_id=_RUN_ID,
        )
        == second
    )

    with pytest.raises(StrategyPositionOwnershipConflictError):
        _reserve(service=service, organization_id=_FIRST_ORGANIZATION_ID)


def _reserve(
    *,
    service: StrategyPositionOwnershipService,
    organization_id: OrganizationId,
):
    return service.reserve_for_strategy_run(
        organization_id=organization_id,
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        strategy_id=_STRATEGY_ID,
        live_profile_id=None,
        strategy_run_id=_RUN_ID,
        market_type="spot",
        instrument_key="binance:spot:BTCUSDT",
        position_mode="net",
        now=_NOW,
    )
