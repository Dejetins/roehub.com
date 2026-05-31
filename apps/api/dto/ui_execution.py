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


class ExecutionIntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_event_id: UUID
    idempotency_key: str = Field(min_length=1, max_length=256)
    exchange_connection_id: UUID
    market_type: Literal["spot", "futures"]
    instrument_key: str = Field(min_length=1, max_length=128)
    order: ExecutionOrderModelRequest


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
    created_at: datetime
    duplicate: bool = False
    source_event: ExecutionSourceEventResponse
