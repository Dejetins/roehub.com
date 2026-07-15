from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.contexts.live_execution.domain import (
    AccountConfigGuardResult,
    ExchangeAccountProjection,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


class ExchangeAccountProjectionRepository(Protocol):
    def record_projection(
        self, *, projection: ExchangeAccountProjection
    ) -> ExchangeAccountProjection: ...

    def record_config_guard_result(
        self, *, result: AccountConfigGuardResult
    ) -> AccountConfigGuardResult: ...

    def get_latest_projection(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
    ) -> ExchangeAccountProjection | None: ...

    def get_latest_config_guard_result(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        instrument_key: str,
        market_type: str,
    ) -> AccountConfigGuardResult | None: ...
