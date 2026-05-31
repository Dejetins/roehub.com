from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, Mapping
from uuid import UUID

from trading.shared_kernel.primitives import UserId

ExecutionSourceType = Literal[
    "strategy_signal",
    "manual_request",
    "ml_agent_decision",
    "ops_test",
]
ExecutionSourceOutcome = Literal[
    "recorded",
    "intent_created",
    "order_model_rejected",
    "no_intent",
]
ExecutionIntentStatus = Literal[
    "recorded",
    "accepted",
    "rejected",
    "dispatching",
    "dispatched",
    "retry",
    "quarantined",
]
ExecutionSide = Literal["buy", "sell"]
ExecutionOrderType = Literal["market", "limit"]

SUPPORTED_SOURCE_TYPES: frozenset[str] = frozenset(
    {"strategy_signal", "manual_request", "ml_agent_decision", "ops_test"}
)
SUPPORTED_ORDER_TYPES: frozenset[str] = frozenset({"market", "limit"})
_SENSITIVE_REF_PARTS = frozenset(
    {"secret", "token", "authorization", "cookie", "api_key", "apikey", "signature", "passphrase"}
)


class ExecutionSourceValidationError(ValueError):
    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ExecutionOrderModelRejectedError(ValueError):
    def __init__(self, *, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ExecutionSourceEvent:
    source_event_id: UUID
    owner_user_id: UserId
    source_type: ExecutionSourceType
    source_event_ref: str
    source_ref_json: Mapping[str, str]
    strategy_signal_id: UUID | None
    idempotency_key_hash: str
    outcome: ExecutionSourceOutcome
    outcome_reason: str
    intent_id: UUID | None
    received_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionOrderModelV1:
    order_type: ExecutionOrderType
    side: ExecutionSide
    quantity: Decimal | None
    quote_notional: Decimal | None
    limit_price: Decimal | None


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    source_event_id: UUID
    source_type: ExecutionSourceType
    idempotency_key_hash: str
    exchange_connection_id: UUID
    market_type: str
    instrument_key: str
    order: ExecutionOrderModelV1


@dataclass(frozen=True, slots=True)
class ExecutionIntent:
    intent_id: UUID
    source_event_id: UUID
    owner_user_id: UserId
    source_type: ExecutionSourceType
    strategy_signal_id: UUID | None
    exchange_connection_id: UUID
    market_type: str
    instrument_key: str
    side: ExecutionSide
    order_type: ExecutionOrderType
    quantity: Decimal | None
    quote_notional: Decimal | None
    limit_price: Decimal | None
    status: ExecutionIntentStatus
    status_reason: str
    risk_status: str
    risk_reason: str
    idempotency_key_hash: str
    created_at: datetime
    dispatch_attempt_count: int = 0
    dispatch_stream_name: str | None = None
    dispatch_redis_message_id: str | None = None
    dispatch_last_error: str | None = None
    dispatch_updated_at: datetime | None = None


def hash_idempotency_key(raw_value: str) -> str:
    normalized = raw_value.strip()
    if not normalized:
        raise ExecutionSourceValidationError(reason="idempotency_key_required")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_source_event_fields(
    *,
    source_type: str,
    source_event_ref: str,
    source_ref_json: Mapping[str, str],
    strategy_signal_id: UUID | None,
) -> ExecutionSourceType:
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ExecutionSourceValidationError(reason="unsupported_source_type")
    normalized_ref = source_event_ref.strip()
    if not normalized_ref:
        raise ExecutionSourceValidationError(reason="source_event_ref_required")
    _validate_source_refs(source_ref_json=source_ref_json)
    if source_type == "strategy_signal" and strategy_signal_id is None:
        raise ExecutionSourceValidationError(reason="strategy_signal_id_required")
    if source_type != "strategy_signal" and strategy_signal_id is not None:
        raise ExecutionSourceValidationError(reason="strategy_signal_id_only_for_strategy_signal")
    return source_type  # type: ignore[return-value]


def validate_order_model(
    *,
    order_type: str,
    side: str,
    quantity: Decimal | None,
    quote_notional: Decimal | None,
    limit_price: Decimal | None,
    advanced_order_flags: Mapping[str, object],
) -> ExecutionOrderModelV1:
    rejected_reason = _advanced_order_rejection_reason(flags=advanced_order_flags)
    if rejected_reason is not None:
        raise ExecutionOrderModelRejectedError(reason=rejected_reason)
    if order_type not in SUPPORTED_ORDER_TYPES:
        raise ExecutionOrderModelRejectedError(reason="unsupported_order_type")
    if side not in {"buy", "sell"}:
        raise ExecutionSourceValidationError(reason="invalid_order_side")
    if quantity is None and quote_notional is None:
        raise ExecutionSourceValidationError(reason="order_size_required")
    if quantity is not None and quantity <= 0:
        raise ExecutionSourceValidationError(reason="invalid_order_quantity")
    if quote_notional is not None and quote_notional <= 0:
        raise ExecutionSourceValidationError(reason="invalid_order_quote_notional")
    if order_type == "limit" and (limit_price is None or limit_price <= 0):
        raise ExecutionSourceValidationError(reason="limit_price_required")
    if order_type == "market" and limit_price is not None:
        raise ExecutionOrderModelRejectedError(reason="market_order_limit_price_not_supported")
    return ExecutionOrderModelV1(
        order_type=order_type,  # type: ignore[arg-type]
        side=side,  # type: ignore[arg-type]
        quantity=quantity,
        quote_notional=quote_notional,
        limit_price=limit_price,
    )


def _validate_source_refs(*, source_ref_json: Mapping[str, str]) -> None:
    if len(source_ref_json) > 12:
        raise ExecutionSourceValidationError(reason="source_ref_too_large")
    for key, value in source_ref_json.items():
        normalized_key = key.strip().lower()
        if not normalized_key:
            raise ExecutionSourceValidationError(reason="source_ref_key_required")
        if any(part in normalized_key for part in _SENSITIVE_REF_PARTS):
            raise ExecutionSourceValidationError(reason="sensitive_source_ref_key_rejected")
        if not isinstance(value, str) or not value.strip():
            raise ExecutionSourceValidationError(reason="source_ref_value_required")
        if len(value.strip()) > 256:
            raise ExecutionSourceValidationError(reason="source_ref_value_too_long")


def _advanced_order_rejection_reason(*, flags: Mapping[str, object]) -> str | None:
    if flags.get("oco") is not None:
        return "oco_not_supported"
    if flags.get("trailing") is not None:
        return "trailing_not_supported"
    if flags.get("take_profit") is not None or flags.get("stop_loss") is not None:
        return "tp_sl_not_supported"
    if flags.get("amend_replace") is not None:
        return "amend_replace_not_supported"
    legs = flags.get("legs")
    if isinstance(legs, list) and len(legs) > 1:
        return "multi_leg_not_supported"
    if legs is not None:
        return "multi_leg_not_supported"
    return None
