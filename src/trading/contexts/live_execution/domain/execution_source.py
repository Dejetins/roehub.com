from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal, Mapping
from uuid import UUID

from trading.shared_kernel.primitives import OrganizationId, UserId

EXECUTION_IDEMPOTENCY_NAMESPACE = "io.roehub.execution-idempotency/v1"
EXECUTION_INTENT_HASH_NAMESPACE = "io.roehub.execution-intent/v1"
SOURCE_EVENT_ACCOUNT_NAMESPACE = "source-event"

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
    "risk_rejected",
    "submitted",
    "filled",
    "cancelled",
    "failed",
    "reconciliation_required",
    "handoff_failed",
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
    organization_id: OrganizationId
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
    organization_id: OrganizationId
    source_type: ExecutionSourceType
    idempotency_key_hash: str
    exchange_connection_id: UUID
    market_type: str
    instrument_key: str
    order: ExecutionOrderModelV1
    constraints: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ExecutionIntent:
    intent_id: UUID
    source_event_id: UUID
    organization_id: OrganizationId
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
    constraints: Mapping[str, str] = field(default_factory=dict)
    canonical_intent_hash: str = ""
    dispatch_attempt_count: int = 0
    dispatch_stream_name: str | None = None
    dispatch_redis_message_id: str | None = None
    dispatch_last_error: str | None = None
    dispatch_updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.canonical_intent_hash:
            object.__setattr__(
                self,
                "canonical_intent_hash",
                hash_canonical_execution_intent(
                    organization_id=self.organization_id,
                    exchange_connection_id=self.exchange_connection_id,
                    market_type=self.market_type,
                    instrument_key=self.instrument_key,
                    side=self.side,
                    order_type=self.order_type,
                    quantity=self.quantity,
                    quote_notional=self.quote_notional,
                    limit_price=self.limit_price,
                    constraints=self.constraints,
                    idempotency_key_hash=self.idempotency_key_hash,
                ),
            )

@dataclass(frozen=True, slots=True)
class ExecutionProducerOutcomeLink:
    source_event_id: UUID
    organization_id: OrganizationId
    owner_user_id: UserId
    source_type: ExecutionSourceType
    source_event_ref: str
    source_event_received_at: datetime | None
    strategy_signal_id: UUID | None
    outcome: str
    outcome_reason: str
    intent_id: UUID | None
    intent_status: str | None
    intent_status_reason: str | None
    risk_status: str | None
    risk_reason: str | None
    order_status: str | None
    order_status_reason: str | None
    fill_count: int | None
    latest_fill_at: datetime | None
    reconciliation_status: str | None
    reconciliation_reason: str | None
    notification_event_type: str | None
    notification_reason: str | None
    updated_at: datetime


def hash_idempotency_key(
    raw_value: str,
    *,
    organization_id: OrganizationId,
    account_namespace: UUID | str,
) -> str:
    normalized = raw_value.strip()
    if not normalized:
        raise ExecutionSourceValidationError(reason="idempotency_key_required")
    normalized_account = str(account_namespace).strip()
    if not normalized_account:
        raise ExecutionSourceValidationError(reason="account_namespace_required")
    namespaced = "\x1f".join(
        (
            EXECUTION_IDEMPOTENCY_NAMESPACE,
            str(organization_id),
            normalized_account,
            normalized,
        )
    )
    return hashlib.sha256(namespaced.encode("utf-8")).hexdigest()


def canonicalize_execution_constraints(
    *,
    constraints: Mapping[str, object],
    market_type: str,
    order_type: str,
) -> dict[str, str]:
    supported = {"time_in_force", "reduce_only", "expires_at"}
    unknown = sorted(set(constraints) - supported)
    if unknown:
        raise ExecutionSourceValidationError(reason="unsupported_execution_constraint")

    time_in_force = str(constraints.get("time_in_force", "GTC")).strip().upper()
    if time_in_force != "GTC":
        raise ExecutionSourceValidationError(reason="unsupported_time_in_force")
    if order_type != "limit" and "time_in_force" in constraints:
        raise ExecutionSourceValidationError(reason="time_in_force_only_for_limit")

    raw_reduce_only = constraints.get("reduce_only", False)
    if not isinstance(raw_reduce_only, bool):
        raise ExecutionSourceValidationError(reason="reduce_only_must_be_boolean")
    if raw_reduce_only and market_type != "futures":
        raise ExecutionSourceValidationError(reason="reduce_only_requires_futures")

    normalized: dict[str, str] = {
        "reduce_only": "true" if raw_reduce_only else "false",
        "time_in_force": time_in_force,
    }
    if "expires_at" in constraints:
        expires_at = str(constraints["expires_at"]).strip()
        try:
            parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ExecutionSourceValidationError(reason="invalid_execution_expiry") from error
        if parsed.tzinfo is None:
            raise ExecutionSourceValidationError(reason="execution_expiry_timezone_required")
        normalized["expires_at"] = parsed.isoformat()
    return normalized


def hash_canonical_execution_intent(
    *,
    organization_id: OrganizationId,
    exchange_connection_id: UUID,
    market_type: str,
    instrument_key: str,
    side: str,
    order_type: str,
    quantity: Decimal | None,
    quote_notional: Decimal | None,
    limit_price: Decimal | None,
    constraints: Mapping[str, str],
    idempotency_key_hash: str,
) -> str:
    payload = {
        "schema": EXECUTION_INTENT_HASH_NAMESPACE,
        "organization_id": str(organization_id),
        "account_id": str(exchange_connection_id),
        "market_type": market_type,
        "instrument_key": instrument_key,
        "side": side,
        "order_type": order_type,
        "quantity": str(quantity) if quantity is not None else None,
        "quote_notional": str(quote_notional) if quote_notional is not None else None,
        "limit_price": str(limit_price) if limit_price is not None else None,
        "constraints": dict(sorted(constraints.items())),
        "idempotency_key_hash": idempotency_key_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
