from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from .raw_feature_dataset import hash_json_payload_v1
from .stage08k_monitor_policy import Stage08kMonitorPolicyConfig


class Stage08kMonitorRuntimeError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        super().__init__(reason if field is None else f"{reason}: {field}")


@dataclass(frozen=True, slots=True)
class Stage08kPendingVirtualTrade:
    instrument_key: str
    symbol: str
    entry_decision_id: str
    entry_time_utc: datetime
    expected_exit_time_utc: datetime
    entry_price: float
    notional_quote: float
    policy_hash: str

    def __post_init__(self) -> None:
        for name in ("instrument_key", "symbol", "entry_decision_id", "policy_hash"):
            if not str(getattr(self, name)).strip():
                raise Stage08kMonitorRuntimeError(reason="virtual_trade_field_required", field=name)
        for name in ("entry_time_utc", "expected_exit_time_utc"):
            if getattr(self, name).tzinfo is None:
                raise Stage08kMonitorRuntimeError(
                    reason="virtual_trade_timestamp_must_be_aware",
                    field=name,
                )
        if self.expected_exit_time_utc <= self.entry_time_utc:
            raise Stage08kMonitorRuntimeError(reason="invalid_virtual_exit_time")
        for name in ("entry_price", "notional_quote"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise Stage08kMonitorRuntimeError(reason="invalid_virtual_trade_value", field=name)

    def as_payload(self) -> dict[str, object]:
        return {
            "entry_decision_id": self.entry_decision_id,
            "entry_price": self.entry_price,
            "entry_time_utc": _format_utc(self.entry_time_utc),
            "expected_exit_time_utc": _format_utc(self.expected_exit_time_utc),
            "instrument_key": self.instrument_key,
            "notional_quote": self.notional_quote,
            "policy_hash": self.policy_hash,
            "symbol": self.symbol,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Stage08kPendingVirtualTrade:
        return cls(
            instrument_key=str(payload["instrument_key"]),
            symbol=str(payload["symbol"]),
            entry_decision_id=str(payload["entry_decision_id"]),
            entry_time_utc=_parse_utc(str(payload["entry_time_utc"])),
            expected_exit_time_utc=_parse_utc(str(payload["expected_exit_time_utc"])),
            entry_price=float(payload["entry_price"]),
            notional_quote=float(payload["notional_quote"]),
            policy_hash=str(payload["policy_hash"]),
        )


@dataclass(frozen=True, slots=True)
class Stage08kVirtualExit:
    exit_decision_id: str
    entry_decision_id: str
    instrument_key: str
    symbol: str
    exit_time_utc: datetime
    exit_price: float
    gross_return: float
    net_return: float
    pnl_quote: float
    hold_seconds: float
    valid_for_policy_evaluation: bool
    reason: str

    def metadata(self) -> dict[str, str]:
        return {
            "entry_decision_id": self.entry_decision_id,
            "exit_price": _decimal_text(self.exit_price),
            "funding_model": "not_modeled_for_1m_monitor",
            "gross_return": _decimal_text(self.gross_return),
            "hold_seconds": _decimal_text(self.hold_seconds),
            "net_return": _decimal_text(self.net_return),
            "pnl_quote": _decimal_text(self.pnl_quote),
            "policy_reason": self.reason,
            "valid_for_policy_evaluation": str(self.valid_for_policy_evaluation).lower(),
            "virtual_trade_phase": "close",
        }


def open_stage08k_virtual_trade_v1(
    *,
    instrument_key: str,
    symbol: str,
    entry_decision_id: str,
    entry_time_utc: datetime,
    entry_price: float,
    policy: Stage08kMonitorPolicyConfig,
) -> Stage08kPendingVirtualTrade:
    if entry_time_utc.tzinfo is None:
        raise Stage08kMonitorRuntimeError(reason="entry_time_must_be_aware")
    return Stage08kPendingVirtualTrade(
        instrument_key=instrument_key,
        symbol=symbol,
        entry_decision_id=entry_decision_id,
        entry_time_utc=entry_time_utc.astimezone(UTC),
        expected_exit_time_utc=entry_time_utc.astimezone(UTC)
        + timedelta(minutes=policy.virtual_hold_minutes),
        entry_price=float(entry_price),
        notional_quote=policy.virtual_notional_quote,
        policy_hash=policy.policy_hash(),
    )


def close_stage08k_virtual_trade_v1(
    *,
    trade: Stage08kPendingVirtualTrade,
    exit_time_utc: datetime,
    exit_price: float,
    policy: Stage08kMonitorPolicyConfig,
) -> Stage08kVirtualExit:
    if exit_time_utc.tzinfo is None:
        raise Stage08kMonitorRuntimeError(reason="exit_time_must_be_aware")
    if trade.policy_hash != policy.policy_hash():
        raise Stage08kMonitorRuntimeError(reason="virtual_trade_policy_hash_mismatch")
    normalized_exit = exit_time_utc.astimezone(UTC)
    if normalized_exit < trade.expected_exit_time_utc:
        raise Stage08kMonitorRuntimeError(reason="virtual_exit_before_expected_time")
    price = float(exit_price)
    if not math.isfinite(price) or price <= 0.0:
        raise Stage08kMonitorRuntimeError(reason="invalid_virtual_exit_price")
    gross_return = (price / trade.entry_price) - 1.0
    round_trip_cost = 2.0 * (policy.taker_fee_rate + policy.slippage_rate)
    net_return = gross_return - round_trip_cost
    pnl_quote = trade.notional_quote * net_return
    hold_seconds = (normalized_exit - trade.entry_time_utc).total_seconds()
    valid = normalized_exit == trade.expected_exit_time_utc
    reason = "virtual_close_after_1m" if valid else "late_virtual_close_excluded"
    exit_id = hash_json_payload_v1(
        {
            "entry_decision_id": trade.entry_decision_id,
            "exit_price": price,
            "exit_time_utc": _format_utc(normalized_exit),
            "policy_hash": policy.policy_hash(),
        }
    )
    return Stage08kVirtualExit(
        exit_decision_id=exit_id,
        entry_decision_id=trade.entry_decision_id,
        instrument_key=trade.instrument_key,
        symbol=trade.symbol,
        exit_time_utc=normalized_exit,
        exit_price=price,
        gross_return=gross_return,
        net_return=net_return,
        pnl_quote=pnl_quote,
        hold_seconds=hold_seconds,
        valid_for_policy_evaluation=valid,
        reason=reason,
    )


def stage08k_entry_decision_id_v1(
    *, instrument_key: str, candle_close_utc: datetime, feature_hash: str, policy_hash: str
) -> str:
    if candle_close_utc.tzinfo is None:
        raise Stage08kMonitorRuntimeError(reason="candle_close_must_be_aware")
    return hash_json_payload_v1(
        {
            "candle_close_utc": _format_utc(candle_close_utc.astimezone(UTC)),
            "feature_hash": feature_hash,
            "instrument_key": instrument_key,
            "policy_hash": policy_hash,
        }
    )


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise Stage08kMonitorRuntimeError(reason="timestamp_must_be_aware")
    return parsed.astimezone(UTC)


def _decimal_text(value: float) -> str:
    return format(float(value), ".12g")
