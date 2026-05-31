from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ExecutionSourceEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["strategy_signal", "manual_request", "ml_agent_decision", "ops_test"]
    source_event_ref: str = Field(min_length=1, max_length=256)
    source_ref: dict[str, str] = Field(default_factory=dict)
    strategy_signal_id: UUID | None = None
    idempotency_key: str = Field(min_length=1, max_length=256)


class ExecutionSourceEventResponse(BaseModel):
    source_event_id: UUID
    source_type: str
    source_event_ref: str
    source_ref: dict[str, str]
    strategy_signal_id: UUID | None
    outcome: str
    outcome_reason: str
    intent_id: UUID | None
    received_at: datetime
    duplicate: bool = False


class ExecutionOrderModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_type: str
    side: str
    quantity: Decimal | None = Field(default=None, gt=0)
    quote_notional: Decimal | None = Field(default=None, gt=0)
    limit_price: Decimal | None = Field(default=None, gt=0)
    oco: dict[str, object] | None = None
    trailing: dict[str, object] | None = None
    take_profit: dict[str, object] | None = None
    stop_loss: dict[str, object] | None = None
    amend_replace: dict[str, object] | None = None
    legs: list[dict[str, object]] | None = None


class ExecutionRiskContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exchange_connection_active: bool = False
    secret_custody_ready: bool = False
    source_authorized: bool = False
    strategy_variant_compatible: bool = False
    market_data_state: Literal["ready", "missing", "stale", "pending"] = "missing"
    strategy_binding_active: bool = False
    strategy_live_profile_ready: bool = False
    strategy_run_active: bool = False
    exchange_config_verified: bool = False
    account_state_fresh: bool = False
    position_ownership_active: bool = False
    capital_reservation_active: bool = False
    capital_reservation_sufficient: bool = False
    paper_accounting_ready: bool = False
    manual_recent_auth: bool = False
    ml_agent_policy_active: bool = False
    kill_switch_open: bool = False
    environment_policy_allows: bool = False
    max_order_size_ok: bool = False
    daily_limit_ok: bool = False


class ExecutionIntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_event_id: UUID
    idempotency_key: str = Field(min_length=1, max_length=256)
    exchange_connection_id: UUID
    market_type: Literal["spot", "futures"]
    instrument_key: str = Field(min_length=1, max_length=128)
    order: ExecutionOrderModelRequest
    risk_context: ExecutionRiskContextRequest | None = None


class ExecutionIntentResponse(BaseModel):
    intent_id: UUID
    source_event_id: UUID
    source_type: str
    strategy_signal_id: UUID | None
    exchange_connection_id: UUID
    market_type: str
    instrument_key: str
    side: str
    order_type: str
    quantity: Decimal | None
    quote_notional: Decimal | None
    limit_price: Decimal | None
    status: str
    status_reason: str
    risk_status: str
    risk_reason: str
    dispatch_attempt_count: int = 0
    dispatch_stream_name: str | None = None
    dispatch_redis_message_id: str | None = None
    dispatch_last_error: str | None = None
    dispatch_updated_at: datetime | None = None
    created_at: datetime
    duplicate: bool = False
    source_event: ExecutionSourceEventResponse
