from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from trading.shared_kernel.primitives import UserId

LiveStrategyProfileMode = Literal["monitor_only", "paper", "live"]
LiveStrategyProfileSizingMethod = Literal["fixed_quote", "fixed_equity_pct"]
LiveStrategyProfileReadinessStatus = Literal["ready", "blocked"]

_ALLOWED_MODES = {"monitor_only", "paper", "live"}
_ALLOWED_SIZING_METHODS = {"fixed_quote", "fixed_equity_pct"}
_ALLOWED_READINESS_STATUSES = {"ready", "blocked"}


@dataclass(frozen=True, slots=True)
class LiveStrategyProfile:
    profile_id: UUID
    owner_user_id: UserId
    strategy_id: UUID
    mode: LiveStrategyProfileMode
    exchange_connection_id: UUID | None
    sizing_method: LiveStrategyProfileSizingMethod
    sizing_value: Decimal
    max_position_notional: Decimal | None
    max_orders_per_run: int
    max_notional_per_run: Decimal
    readiness_status: LiveStrategyProfileReadinessStatus
    readiness_reason: str
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.mode not in _ALLOWED_MODES:
            raise ValueError("LiveStrategyProfile mode is unsupported")
        if self.sizing_method not in _ALLOWED_SIZING_METHODS:
            raise ValueError("LiveStrategyProfile sizing_method is unsupported")
        if self.readiness_status not in _ALLOWED_READINESS_STATUSES:
            raise ValueError("LiveStrategyProfile readiness_status is unsupported")
        if not self.readiness_reason.strip():
            raise ValueError("LiveStrategyProfile readiness_reason must be non-empty")
        _ensure_non_negative_decimal(name="sizing_value", value=self.sizing_value)
        if self.max_position_notional is not None:
            _ensure_non_negative_decimal(
                name="max_position_notional", value=self.max_position_notional
            )
        if self.max_orders_per_run < 0:
            raise ValueError("LiveStrategyProfile max_orders_per_run must be >= 0")
        _ensure_non_negative_decimal(
            name="max_notional_per_run", value=self.max_notional_per_run
        )
        _ensure_utc_datetime(name="created_at", value=self.created_at)
        _ensure_utc_datetime(name="updated_at", value=self.updated_at)
        if self.updated_at < self.created_at:
            raise ValueError("LiveStrategyProfile updated_at cannot be before created_at")

    def with_readiness(
        self,
        *,
        readiness_status: LiveStrategyProfileReadinessStatus,
        readiness_reason: str,
        updated_at: datetime,
    ) -> "LiveStrategyProfile":
        return replace(
            self,
            readiness_status=readiness_status,
            readiness_reason=readiness_reason,
            updated_at=updated_at,
        )


def _ensure_non_negative_decimal(*, name: str, value: Decimal) -> None:
    if value < Decimal("0"):
        raise ValueError(f"LiveStrategyProfile {name} must be >= 0")


def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{name} must be UTC datetime")
