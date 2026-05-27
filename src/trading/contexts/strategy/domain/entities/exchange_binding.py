from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from trading.shared_kernel.primitives import UserId

StrategyExchangeBindingStatus = Literal["active", "paused", "disabled", "archived"]
StrategyExchangeBindingUsageMode = Literal["trading"]


@dataclass(frozen=True, slots=True)
class StrategyExchangeBinding:
    binding_id: UUID
    owner_user_id: UserId
    strategy_id: UUID
    exchange_connection_id: UUID
    usage_mode: StrategyExchangeBindingUsageMode
    binding_status: StrategyExchangeBindingStatus
    created_at: datetime
    updated_at: datetime
    disabled_at: datetime | None = None
    archived_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.usage_mode != "trading":
            raise ValueError("StrategyExchangeBinding usage_mode must be trading")
        if self.binding_status not in {"active", "paused", "disabled", "archived"}:
            raise ValueError("StrategyExchangeBinding status is unsupported")
        if self.binding_status == "active" and (
            self.disabled_at is not None or self.archived_at is not None
        ):
            raise ValueError("Active strategy exchange binding cannot have lifecycle timestamps")

    def disabled(self, *, disabled_at: datetime) -> "StrategyExchangeBinding":
        return StrategyExchangeBinding(
            binding_id=self.binding_id,
            owner_user_id=self.owner_user_id,
            strategy_id=self.strategy_id,
            exchange_connection_id=self.exchange_connection_id,
            usage_mode=self.usage_mode,
            binding_status="disabled",
            created_at=self.created_at,
            updated_at=disabled_at,
            disabled_at=disabled_at,
            archived_at=self.archived_at,
        )
