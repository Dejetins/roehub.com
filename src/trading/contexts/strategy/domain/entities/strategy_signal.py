from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Mapping
from uuid import UUID

from trading.contexts.strategy.domain.entities.live_strategy_profile import (
    LiveStrategyProfileMode,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

StrategySignalAction = Literal["none", "open", "close", "reduce", "reverse"]
StrategySignalSide = Literal["buy", "sell"]
StrategySignalOutcome = Literal["warmup", "no_signal", "signal", "blocked"]

_ALLOWED_ACTIONS = {"none", "open", "close", "reduce", "reverse"}
_ALLOWED_SIDES = {"buy", "sell"}
_ALLOWED_OUTCOMES = {"warmup", "no_signal", "signal", "blocked"}
_EXPECTED_ORDER_SCHEMA_V1 = "strategy_signal_expected_order_v1"
_ALLOWED_EXPECTED_ORDER_KEYS = frozenset(
    {
        "schema",
        "mode",
        "quote_notional",
        "exchange_connection_id",
        "paper_no_exchange_submit",
    }
)


@dataclass(frozen=True, slots=True)
class StrategySignal:
    """
    StrategySignal — durable Stage 05 signal/no-signal journal entry.

    The entity is an explanatory Strategy-context decision record only. It never
    represents an order intent or exchange-side effect.
    """

    signal_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    strategy_id: UUID
    strategy_run_id: UUID
    live_profile_id: UUID | None
    mode: LiveStrategyProfileMode
    instrument_key: str
    market_type: str
    timeframe: str
    bar_ts_open: datetime
    bar_ts_close: datetime
    signal_action: StrategySignalAction
    side: StrategySignalSide | None
    outcome: StrategySignalOutcome
    reason_code: str
    reference_price: Decimal
    confidence: Decimal | None
    source_message_id: str
    evaluator_version: str
    expected_order_json: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.signal_action not in _ALLOWED_ACTIONS:
            raise ValueError("StrategySignal signal_action is unsupported")
        if self.side is not None and self.side not in _ALLOWED_SIDES:
            raise ValueError("StrategySignal side is unsupported")
        if self.outcome not in _ALLOWED_OUTCOMES:
            raise ValueError("StrategySignal outcome is unsupported")
        if self.signal_action == "none" and self.side is not None:
            raise ValueError("StrategySignal side must be None for action=none")
        if self.signal_action != "none" and self.side is None:
            raise ValueError("StrategySignal side is required for actionable signals")
        if not self.instrument_key.strip():
            raise ValueError("StrategySignal instrument_key must be non-empty")
        if not self.market_type.strip():
            raise ValueError("StrategySignal market_type must be non-empty")
        if not self.timeframe.strip():
            raise ValueError("StrategySignal timeframe must be non-empty")
        if not self.reason_code.strip():
            raise ValueError("StrategySignal reason_code must be non-empty")
        if not self.source_message_id.strip():
            raise ValueError("StrategySignal source_message_id must be non-empty")
        if not self.evaluator_version.strip():
            raise ValueError("StrategySignal evaluator_version must be non-empty")
        if self.reference_price < Decimal("0"):
            raise ValueError("StrategySignal reference_price must be >= 0")
        if self.confidence is not None and (
            self.confidence < Decimal("0") or self.confidence > Decimal("1")
        ):
            raise ValueError("StrategySignal confidence must be between 0 and 1")
        _ensure_utc_datetime(name="bar_ts_open", value=self.bar_ts_open)
        _ensure_utc_datetime(name="bar_ts_close", value=self.bar_ts_close)
        if self.bar_ts_open >= self.bar_ts_close:
            raise ValueError("StrategySignal requires bar_ts_open < bar_ts_close")
        if self.created_at is not None:
            _ensure_utc_datetime(name="created_at", value=self.created_at)
        if self.expected_order_json:
            _validate_expected_order_json(value=self.expected_order_json)


def _ensure_utc_datetime(*, name: str, value: datetime) -> None:
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{name} must be timezone-aware UTC datetime")
    if offset.total_seconds() != 0:
        raise ValueError(f"{name} must be UTC datetime")


def _validate_expected_order_json(*, value: Mapping[str, Any]) -> None:
    keys = frozenset(str(key) for key in value)
    unknown = keys - _ALLOWED_EXPECTED_ORDER_KEYS
    if unknown:
        raise ValueError("StrategySignal expected_order_json has unsupported keys")
    if value.get("schema") != _EXPECTED_ORDER_SCHEMA_V1:
        raise ValueError("StrategySignal expected_order_json schema is unsupported")
    if value.get("mode") != "paper":
        raise ValueError("StrategySignal expected_order_json mode is unsupported")
    if value.get("paper_no_exchange_submit") is not True:
        raise ValueError("StrategySignal expected_order_json must be paper no-dispatch")
    try:
        quote_notional = Decimal(str(value.get("quote_notional", "")))
    except Exception as error:  # noqa: BLE001
        raise ValueError("StrategySignal expected_order_json quote_notional is invalid") from error
    if quote_notional <= 0:
        raise ValueError("StrategySignal expected_order_json quote_notional must be > 0")
    exchange_connection_id = value.get("exchange_connection_id")
    if exchange_connection_id is not None:
        try:
            UUID(str(exchange_connection_id))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "StrategySignal expected_order_json exchange_connection_id is invalid"
            ) from error
