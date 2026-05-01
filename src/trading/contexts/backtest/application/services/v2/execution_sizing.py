from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numba as nb
import numpy as np

DIRECTION_MODE_LONG_ONLY = "long_only"
DIRECTION_MODE_LONG_SHORT_REVERSAL = "long_short_reversal"
DIRECTION_MODE_LONG_ONLY_CODE = np.int8(1)
DIRECTION_MODE_LONG_SHORT_REVERSAL_CODE = np.int8(2)

SIZING_MODE_ALL_IN = "all_in"
SIZING_MODE_FIXED_QUOTE = "fixed_quote"
SIZING_MODE_FIXED_EQUITY_PCT = "fixed_equity_pct"
SIZING_MODE_FIXED_EQUITY_PCT_MIN_QUOTE = "fixed_equity_pct_min_quote"
SIZING_MODE_FIXED_EQUITY_PCT_MAX_QUOTE = "fixed_equity_pct_max_quote"

SIZING_MODE_ALL_IN_CODE = np.int8(1)
SIZING_MODE_FIXED_QUOTE_CODE = np.int8(2)
SIZING_MODE_FIXED_EQUITY_PCT_CODE = np.int8(3)
SIZING_MODE_FIXED_EQUITY_PCT_MIN_QUOTE_CODE = np.int8(4)
SIZING_MODE_FIXED_EQUITY_PCT_MAX_QUOTE_CODE = np.int8(5)

SUPPORTED_SIZING_MODES = (
    SIZING_MODE_ALL_IN,
    SIZING_MODE_FIXED_QUOTE,
    SIZING_MODE_FIXED_EQUITY_PCT,
    SIZING_MODE_FIXED_EQUITY_PCT_MIN_QUOTE,
    SIZING_MODE_FIXED_EQUITY_PCT_MAX_QUOTE,
)


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    direction_mode: str
    direction_mode_code: np.int8
    fee_rate: float
    slippage_rate: float
    initial_cash_quote: float
    sizing_mode: str
    sizing_mode_code: np.int8
    quote_amount: float
    equity_pct: float
    min_quote: float
    max_quote: float
    safe_profit_percent: float
    use_profit_lock: np.int8
    close_on_end: np.int8

    @property
    def uses_optimized_all_in_path(self) -> bool:
        return (
            self.sizing_mode_code == SIZING_MODE_ALL_IN_CODE
            and self.use_profit_lock == 0
            and self.close_on_end == 1
        )


class BacktestExecutionSizingRejected(ValueError):
    """
    Internal rejection for malformed normalized execution settings.
    """


def execution_settings_from_normalized(
    normalized_request: Mapping[str, Any],
    *,
    expected_direction_mode: str,
    config: Any,
    rejection_cls: type[ValueError] = BacktestExecutionSizingRejected,
) -> ExecutionSettings:
    execution = normalized_request.get("execution")
    if not isinstance(execution, Mapping):
        raise rejection_cls("normalized_request.execution must be a mapping")
    direction_mode = str(execution.get("direction_mode", expected_direction_mode))
    if direction_mode != expected_direction_mode:
        raise rejection_cls(
            f"execution direction_mode {direction_mode!r} does not match "
            f"backend direction_mode {expected_direction_mode!r}"
        )
    sizing = execution.get("sizing", {"mode": SIZING_MODE_ALL_IN})
    if not isinstance(sizing, Mapping):
        raise rejection_cls("normalized_request.execution.sizing must be a mapping")
    sizing_mode = str(sizing.get("mode", SIZING_MODE_ALL_IN))
    sizing_mode_code = sizing_mode_code_from_literal(
        sizing_mode=sizing_mode,
        rejection_cls=rejection_cls,
    )
    profit_lock = execution.get("profit_lock", {"enabled": False})
    if not isinstance(profit_lock, Mapping):
        raise rejection_cls("normalized_request.execution.profit_lock must be a mapping")
    return ExecutionSettings(
        direction_mode=direction_mode,
        direction_mode_code=direction_mode_code(
            direction_mode=direction_mode,
            rejection_cls=rejection_cls,
        ),
        fee_rate=_float_from_mapping(
            execution,
            "fee_rate",
            default=config.default_fee_rate,
            minimum=0.0,
            rejection_cls=rejection_cls,
        ),
        slippage_rate=_float_from_mapping(
            execution,
            "slippage_rate",
            default=config.default_slippage_rate,
            minimum=0.0,
            rejection_cls=rejection_cls,
        ),
        initial_cash_quote=_float_from_mapping(
            execution,
            "initial_cash_quote",
            default=config.default_initial_cash_quote,
            minimum=0.0,
            minimum_inclusive=False,
            rejection_cls=rejection_cls,
        ),
        sizing_mode=sizing_mode,
        sizing_mode_code=sizing_mode_code,
        quote_amount=_float_from_mapping(
            sizing,
            "quote_amount",
            legacy_key="fixed_quote",
            default=config.default_fixed_quote,
            minimum=0.0,
            minimum_inclusive=False,
            rejection_cls=rejection_cls,
        ),
        equity_pct=_float_from_mapping(
            sizing,
            "equity_pct",
            legacy_key="pct",
            default=100.0,
            minimum=0.0,
            minimum_inclusive=False,
            rejection_cls=rejection_cls,
        ),
        min_quote=_float_from_mapping(
            sizing,
            "min_quote",
            default=config.default_fixed_quote,
            minimum=0.0,
            minimum_inclusive=False,
            rejection_cls=rejection_cls,
        ),
        max_quote=_float_from_mapping(
            sizing,
            "max_quote",
            default=config.default_fixed_quote,
            minimum=0.0,
            minimum_inclusive=False,
            rejection_cls=rejection_cls,
        ),
        safe_profit_percent=_float_from_mapping(
            profit_lock,
            "safe_profit_percent",
            default=config.default_safe_profit_percent,
            minimum=0.0,
            rejection_cls=rejection_cls,
        ),
        use_profit_lock=np.int8(1 if bool(profit_lock.get("enabled", False)) else 0),
        close_on_end=np.int8(1 if bool(execution.get("close_on_end", True)) else 0),
    )


def direction_mode_code(
    *,
    direction_mode: str,
    rejection_cls: type[ValueError] = BacktestExecutionSizingRejected,
) -> np.int8:
    if direction_mode == DIRECTION_MODE_LONG_ONLY:
        return DIRECTION_MODE_LONG_ONLY_CODE
    if direction_mode == DIRECTION_MODE_LONG_SHORT_REVERSAL:
        return DIRECTION_MODE_LONG_SHORT_REVERSAL_CODE
    raise rejection_cls(
        f"Unsupported direction_mode={direction_mode!r}; expected "
        f"{(DIRECTION_MODE_LONG_ONLY, DIRECTION_MODE_LONG_SHORT_REVERSAL)!r}"
    )


def sizing_mode_code_from_literal(
    *,
    sizing_mode: str,
    rejection_cls: type[ValueError] = BacktestExecutionSizingRejected,
) -> np.int8:
    if sizing_mode == SIZING_MODE_ALL_IN:
        return SIZING_MODE_ALL_IN_CODE
    if sizing_mode == SIZING_MODE_FIXED_QUOTE:
        return SIZING_MODE_FIXED_QUOTE_CODE
    if sizing_mode == SIZING_MODE_FIXED_EQUITY_PCT:
        return SIZING_MODE_FIXED_EQUITY_PCT_CODE
    if sizing_mode == SIZING_MODE_FIXED_EQUITY_PCT_MIN_QUOTE:
        return SIZING_MODE_FIXED_EQUITY_PCT_MIN_QUOTE_CODE
    if sizing_mode == SIZING_MODE_FIXED_EQUITY_PCT_MAX_QUOTE:
        return SIZING_MODE_FIXED_EQUITY_PCT_MAX_QUOTE_CODE
    raise rejection_cls(
        f"unsupported execution sizing mode {sizing_mode!r}; "
        f"expected {SUPPORTED_SIZING_MODES!r}"
    )


def _float_from_mapping(
    mapping: Mapping[str, Any],
    key: str,
    *,
    default: float,
    minimum: float,
    rejection_cls: type[ValueError],
    legacy_key: str | None = None,
    minimum_inclusive: bool = True,
) -> float:
    raw_value = mapping.get(key)
    if raw_value is None and legacy_key is not None:
        raw_value = mapping.get(legacy_key)
    if raw_value is None:
        raw_value = default
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise rejection_cls(f"{key} must be numeric")
    value = float(raw_value)
    if not np.isfinite(value):
        raise rejection_cls(f"{key} must be finite")
    if minimum_inclusive:
        invalid = value < minimum
    else:
        invalid = value <= minimum
    if invalid:
        op = ">=" if minimum_inclusive else ">"
        raise rejection_cls(f"{key} must be {op} {minimum}")
    return value


@nb.njit(cache=True, inline="always")
def execution_quote_amount(
    available_quote: float,
    equity: float,
    sizing_mode_code: np.int8,
    quote_amount: float,
    equity_pct: float,
    min_quote: float,
    max_quote: float,
) -> float:
    if available_quote <= 0.0:
        return 0.0
    requested_quote = available_quote
    if sizing_mode_code == SIZING_MODE_FIXED_QUOTE_CODE:
        requested_quote = quote_amount
    elif sizing_mode_code == SIZING_MODE_FIXED_EQUITY_PCT_CODE:
        requested_quote = equity * (equity_pct / 100.0)
    elif sizing_mode_code == SIZING_MODE_FIXED_EQUITY_PCT_MIN_QUOTE_CODE:
        requested_quote = equity * (equity_pct / 100.0)
        if requested_quote < min_quote:
            requested_quote = min_quote
    elif sizing_mode_code == SIZING_MODE_FIXED_EQUITY_PCT_MAX_QUOTE_CODE:
        requested_quote = equity * (equity_pct / 100.0)
        if requested_quote > max_quote:
            requested_quote = max_quote
    if requested_quote > available_quote:
        requested_quote = available_quote
    if requested_quote <= 0.0:
        return 0.0
    return requested_quote


def execution_quote_amount_py(
    *,
    available_quote: float,
    equity: float,
    sizing_mode_code: np.int8,
    quote_amount: float,
    equity_pct: float,
    min_quote: float,
    max_quote: float,
) -> float:
    return float(
        execution_quote_amount(
            available_quote,
            equity,
            sizing_mode_code,
            quote_amount,
            equity_pct,
            min_quote,
            max_quote,
        )
    )


__all__ = [
    "BacktestExecutionSizingRejected",
    "DIRECTION_MODE_LONG_ONLY",
    "DIRECTION_MODE_LONG_ONLY_CODE",
    "DIRECTION_MODE_LONG_SHORT_REVERSAL",
    "DIRECTION_MODE_LONG_SHORT_REVERSAL_CODE",
    "ExecutionSettings",
    "SIZING_MODE_ALL_IN",
    "SIZING_MODE_ALL_IN_CODE",
    "SIZING_MODE_FIXED_EQUITY_PCT",
    "SIZING_MODE_FIXED_EQUITY_PCT_CODE",
    "SIZING_MODE_FIXED_EQUITY_PCT_MAX_QUOTE",
    "SIZING_MODE_FIXED_EQUITY_PCT_MAX_QUOTE_CODE",
    "SIZING_MODE_FIXED_EQUITY_PCT_MIN_QUOTE",
    "SIZING_MODE_FIXED_EQUITY_PCT_MIN_QUOTE_CODE",
    "SIZING_MODE_FIXED_QUOTE",
    "SIZING_MODE_FIXED_QUOTE_CODE",
    "SUPPORTED_SIZING_MODES",
    "direction_mode_code",
    "execution_quote_amount",
    "execution_quote_amount_py",
    "execution_settings_from_normalized",
    "sizing_mode_code_from_literal",
]
