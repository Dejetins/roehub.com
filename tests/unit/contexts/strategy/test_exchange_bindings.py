from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from trading.contexts.strategy.adapters.outbound.persistence.in_memory import (
    InMemoryStrategyEventRepository,
    InMemoryStrategyExchangeBindingRepository,
    InMemoryStrategyRepository,
)
from trading.contexts.strategy.application import (
    CreateStrategyUseCase,
    CurrentUser,
    StrategyExchangeBindingService,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("00000000-0000-4000-8000-000000000122"))


class _Clock:
    def __init__(self, *, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now


def test_strategy_exchange_binding_lifecycle_is_owner_scoped() -> None:
    strategy_repository = InMemoryStrategyRepository()
    service = StrategyExchangeBindingService(
        strategy_repository=strategy_repository,
        binding_repository=InMemoryStrategyExchangeBindingRepository(),
    )
    owner = CurrentUser(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
        organization_id=_ORGANIZATION_ID,
    )
    other_user_id = UserId.from_string("00000000-0000-0000-0000-000000000456")
    strategy = CreateStrategyUseCase(
        repository=strategy_repository,
        event_repository=InMemoryStrategyEventRepository(),
        clock=_Clock(now=datetime(2026, 5, 27, 9, 0, tzinfo=timezone.utc)),
    ).execute(spec_payload=_strategy_spec_payload(), current_user=owner)
    connection_id = UUID("00000000-0000-0000-0000-000000000999")

    binding = service.create_binding(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner.user_id,
        strategy_id=strategy.strategy_id,
        exchange_connection_id=connection_id,
        usage_mode="trading",
        now=datetime(2026, 5, 27, 9, 1, tzinfo=timezone.utc),
    )

    assert binding.binding_status == "active"
    assert service.list_bindings(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner.user_id,
        strategy_id=strategy.strategy_id,
    ) == (binding,)
    with pytest.raises(RoehubError) as duplicate_error:
        service.create_binding(
            organization_id=_ORGANIZATION_ID,
            owner_user_id=owner.user_id,
            strategy_id=strategy.strategy_id,
            exchange_connection_id=connection_id,
            usage_mode="trading",
            now=datetime(2026, 5, 27, 9, 2, tzinfo=timezone.utc),
        )
    assert duplicate_error.value.code == "strategy_exchange_binding_already_active"

    with pytest.raises(RoehubError) as owner_error:
        service.list_bindings(
            organization_id=_ORGANIZATION_ID,
            owner_user_id=other_user_id,
            strategy_id=strategy.strategy_id,
        )
    assert owner_error.value.code == "not_found"

    disabled = service.disable_binding(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner.user_id,
        strategy_id=strategy.strategy_id,
        binding_id=binding.binding_id,
        now=datetime(2026, 5, 27, 9, 3, tzinfo=timezone.utc),
    )
    assert disabled.binding_status == "disabled"
    assert disabled.disabled_at is not None


def _strategy_spec_payload() -> dict[str, object]:
    return {
        "instrument_id": {
            "market_id": 1,
            "symbol": "BTCUSDT",
        },
        "instrument_key": "binance:spot:BTCUSDT",
        "market_type": "spot",
        "timeframe": "1m",
        "indicators": [
            {
                "name": "MA",
                "params": {
                    "fast": 20,
                    "slow": 50,
                },
            }
        ],
        "signal_template": "MA(20,50)",
    }
