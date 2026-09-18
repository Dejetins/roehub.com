"""Bounded strategy execution projection, separate from research/backtest data."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OperationFill(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fill_id: str
    time: datetime
    price: float
    reference_price: float | None = None
    quantity: float
    fee: float | None
    action: Literal["entry", "exit"]
    reason: str
    side: Literal["buy", "sell"]


class OperationTrade(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trade_id: str
    side: Literal["long", "short"]
    entry_time: datetime
    exit_time: datetime | None = None
    entry: float
    exit: float | None = None
    quantity: float
    remaining_quantity: float
    entry_reason: str
    exit_reason: str | None = None
    gross_pnl: float | None = None
    fees: float | None = None
    net_pnl: float | None = None
    fills: list[OperationFill]


class OperationPoint(BaseModel):
    timestamp: datetime
    value: float


class StrategyOperationsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = "execution_fills"
    state: Literal["ready", "empty", "degraded", "unavailable"] = "unavailable"
    scope: Literal["current_run", "history"] = "current_run"
    partial: bool = False
    reason: str | None = None
    initial_cash: float | None = None
    trades: list[OperationTrade] = Field(default_factory=list)
    equity: list[OperationPoint] = Field(default_factory=list)
    drawdown: list[OperationPoint] = Field(default_factory=list)
    current_price: float | None = None
    price_at: datetime | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
