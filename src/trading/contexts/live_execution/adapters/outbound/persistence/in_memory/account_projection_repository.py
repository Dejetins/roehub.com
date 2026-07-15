from __future__ import annotations

from uuid import UUID

from trading.contexts.live_execution.application.ports import (
    ExchangeAccountProjectionRepository,
)
from trading.contexts.live_execution.domain import (
    AccountConfigGuardResult,
    ExchangeAccountProjection,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class InMemoryExchangeAccountProjectionRepository(ExchangeAccountProjectionRepository):
    def __init__(self) -> None:
        self.projections: list[ExchangeAccountProjection] = []
        self.config_results: list[AccountConfigGuardResult] = []

    def record_projection(
        self, *, projection: ExchangeAccountProjection
    ) -> ExchangeAccountProjection:
        self.projections.append(projection)
        return projection

    def record_config_guard_result(
        self, *, result: AccountConfigGuardResult
    ) -> AccountConfigGuardResult:
        self.config_results.append(result)
        return result

    def get_latest_projection(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
    ) -> ExchangeAccountProjection | None:
        matches = [
            item
            for item in self.projections
            if item.organization_id == organization_id
            and item.owner_user_id == owner_user_id
            and item.exchange_connection_id == exchange_connection_id
        ]
        if not matches:
            return None
        return max(enumerate(matches), key=lambda item: (item[1].observed_at, item[0]))[1]

    def get_latest_config_guard_result(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        instrument_key: str,
        market_type: str,
    ) -> AccountConfigGuardResult | None:
        matches = [
            item
            for item in self.config_results
            if item.organization_id == organization_id
            and item.owner_user_id == owner_user_id
            and item.exchange_connection_id == exchange_connection_id
            and item.instrument_key == instrument_key
            and item.market_type == market_type
        ]
        if not matches:
            return None
        return max(enumerate(matches), key=lambda item: (item[1].checked_at, item[0]))[1]
