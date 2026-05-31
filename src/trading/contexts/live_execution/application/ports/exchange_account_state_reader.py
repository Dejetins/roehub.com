from __future__ import annotations

from typing import Protocol
from uuid import UUID

from trading.contexts.live_execution.domain import ExchangeAccountProjection
from trading.shared_kernel.primitives import UserId


class ExchangeAccountStateReader(Protocol):
    def read_account_projection(
        self, *, owner_user_id: UserId, exchange_connection_id: UUID
    ) -> ExchangeAccountProjection: ...
