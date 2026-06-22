from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from trading.contexts.backtest.application.services.v2.execution_sizing import (
    execution_quote_amount_py,
)

from .no_risk_funding import (
    FUNDING_EVENTS_COUNT,
    FUNDING_PNL_QUOTE,
    FUNDING_RETURN_PCT,
    NO_RISK_FUNDING_METRIC_NAMES,
    TOTAL_RETURN_PCT_NET_OF_FUNDING,
)

TP_SL_FUNDING_ADJUSTMENT_STAGE_NAME = "tp_sl_funding_adjustment"
TP_SL_FUNDING_METRIC_NAMES = NO_RISK_FUNDING_METRIC_NAMES


@dataclass(frozen=True, slots=True)
class TpSlSelectedExit:
    exit_abs: int
    reason: str
    closed: bool


@dataclass(frozen=True, slots=True)
class TpSlFundingAdjustmentSummary:
    funding_pnl_quote: float
    funding_events_count: int
    funding_data_quality: str
    funding_warning_codes: tuple[str, ...]

    def metric_payload(
        self,
        *,
        gross_total_return_pct: float,
        initial_cash_quote: float,
    ) -> dict[str, float]:
        funding_return_pct = (
            (self.funding_pnl_quote / initial_cash_quote) * 100.0
            if initial_cash_quote > 0.0
            else 0.0
        )
        return {
            TOTAL_RETURN_PCT_NET_OF_FUNDING: gross_total_return_pct + funding_return_pct,
            FUNDING_RETURN_PCT: funding_return_pct,
            FUNDING_PNL_QUOTE: self.funding_pnl_quote,
            FUNDING_EVENTS_COUNT: float(self.funding_events_count),
        }


def resolve_tp_sl_selected_exit(
    *,
    direction: int,
    entry_abs: int,
    signal_exit_abs: int,
    best_tp_idx: int,
    best_sl_idx: int,
    hit_times: Any,
    runtime: Any,
) -> TpSlSelectedExit:
    start = entry_abs + 1
    t_exec_abs = int(runtime.t_exec_abs_15m)
    stop_abs = signal_exit_abs if signal_exit_abs < t_exec_abs else t_exec_abs
    if start < t_exec_abs:
        if direction == 1:
            t_tp = int(hit_times.long_tp[best_tp_idx, start])
            t_sl = int(hit_times.long_sl[best_sl_idx, start])
        else:
            t_tp = int(hit_times.short_tp[best_tp_idx, start])
            t_sl = int(hit_times.short_sl[best_sl_idx, start])
        if t_tp < stop_abs and t_tp < t_sl:
            return TpSlSelectedExit(exit_abs=t_tp, reason="take_profit", closed=True)
        if t_sl < stop_abs and t_sl <= t_tp:
            return TpSlSelectedExit(exit_abs=t_sl, reason="stop_loss", closed=True)
    if signal_exit_abs < t_exec_abs:
        return TpSlSelectedExit(exit_abs=signal_exit_abs, reason="signal", closed=True)
    if runtime.close_on_end == 1 and t_exec_abs > 0:
        return TpSlSelectedExit(
            exit_abs=t_exec_abs - 1,
            reason="close_on_end",
            closed=True,
        )
    return TpSlSelectedExit(exit_abs=entry_abs, reason="open", closed=False)


def calculate_tp_sl_funding_adjustment(
    *,
    entry_abs: np.ndarray,
    dir_arr: np.ndarray,
    sig_exit_abs: np.ndarray,
    trade_returns: Sequence[float],
    best_tp_idx: int,
    best_sl_idx: int,
    hit_times: Any,
    runtime: Any,
    open_time_15m: np.ndarray,
    close_time_15m: np.ndarray,
    execution_close_1m: np.ndarray,
    execution_close_time_1m: np.ndarray,
    funding_time: np.ndarray,
    funding_rate: np.ndarray,
    mark_price: np.ndarray,
    data_quality: np.ndarray,
    funding_data_quality: str,
    warning_codes: Sequence[str] = (),
) -> TpSlFundingAdjustmentSummary:
    """
    Calculate post-hoc funding PnL for TP/SL selected-cell open windows.

    Funding events are included exactly when `entry_time < funding_time <= exit_time`.
    The TP/SL exit resolver is shared with lazy detail and selected-cell metrics, including
    the same-bar stop-loss precedence rule.
    """

    funding_pnl_quote = 0.0
    funding_events_count = 0
    warnings = set(str(code) for code in warning_codes if str(code))
    available_quote = float(runtime.initial_cash_quote)
    safe_quote = 0.0
    equity = float(runtime.initial_cash_quote)
    funding_time_i64 = np.asarray(funding_time, dtype=np.int64)
    funding_rate_f64 = np.asarray(funding_rate, dtype=np.float64)
    mark_price_f64 = np.asarray(mark_price, dtype=np.float64)
    data_quality_u8 = np.asarray(data_quality, dtype=np.uint8)
    trade_return_index = 0

    for trade_index in range(int(entry_abs.size)):
        direction = int(dir_arr[trade_index])
        entry_idx = int(entry_abs[trade_index])
        entry_price_raw = float(runtime.price_open_15m[entry_idx])
        if not math.isfinite(entry_price_raw) or entry_price_raw <= 0.0:
            continue
        selected_exit = resolve_tp_sl_selected_exit(
            direction=direction,
            entry_abs=entry_idx,
            signal_exit_abs=int(sig_exit_abs[trade_index]),
            best_tp_idx=best_tp_idx,
            best_sl_idx=best_sl_idx,
            hit_times=hit_times,
            runtime=runtime,
        )
        if not selected_exit.closed:
            continue
        if trade_return_index >= len(trade_returns):
            raise ValueError("trade_returns shorter than closed TP/SL trade list")
        trade_return = float(trade_returns[trade_return_index])
        trade_return_index += 1
        quote_amount = execution_quote_amount_py(
            available_quote=available_quote,
            equity=equity,
            sizing_mode_code=runtime.sizing_mode_code,
            quote_amount=runtime.quote_amount,
            equity_pct=runtime.equity_pct,
            min_quote=runtime.min_quote,
            max_quote=runtime.max_quote,
        )
        if quote_amount <= 0.0:
            continue
        qty_base = quote_amount / entry_price_raw
        entry_time_ms = int(open_time_15m[entry_idx])
        exit_time_ms = _tp_sl_exit_time_ms(
            selected_exit=selected_exit,
            open_time_15m=open_time_15m,
            close_time_15m=close_time_15m,
        )
        event_start = int(np.searchsorted(funding_time_i64, entry_time_ms, side="right"))
        event_stop = int(np.searchsorted(funding_time_i64, exit_time_ms, side="right"))
        for event_idx in range(event_start, event_stop):
            event_time_ms = int(funding_time_i64[event_idx])
            rate = float(funding_rate_f64[event_idx])
            price, mark_price_fallback_used = _funding_event_mark_price(
                event_time_ms=event_time_ms,
                mark_price=float(mark_price_f64[event_idx]),
                execution_close_1m=execution_close_1m,
                execution_close_time_1m=execution_close_time_1m,
            )
            if mark_price_fallback_used:
                warnings.add("funding_mark_price_fallback_used")
            if not (math.isfinite(rate) and math.isfinite(price)) or price <= 0.0:
                warnings.add("invalid_funding_event_values")
                continue
            if int(data_quality_u8[event_idx]) == 0:
                warnings.add("funding_event_data_quality_degraded")
            funding_pnl_quote += -float(direction) * qty_base * price * rate
            funding_events_count += 1

        pnl = quote_amount * trade_return
        available_quote += pnl
        if runtime.use_profit_lock == 1 and pnl > 0.0:
            locked_profit_quote = pnl * (float(runtime.safe_profit_percent) / 100.0)
            available_quote -= locked_profit_quote
            safe_quote += locked_profit_quote
        equity = available_quote + safe_quote

    if trade_return_index != len(trade_returns):
        raise ValueError("trade_returns longer than closed TP/SL trade list")

    quality = funding_data_quality
    if (
        "funding_event_data_quality_degraded" in warnings
        or "funding_mark_price_fallback_used" in warnings
    ):
        quality = "degraded"
    return TpSlFundingAdjustmentSummary(
        funding_pnl_quote=funding_pnl_quote,
        funding_events_count=funding_events_count,
        funding_data_quality=quality,
        funding_warning_codes=tuple(sorted(warnings)),
    )


def _tp_sl_exit_time_ms(
    *,
    selected_exit: TpSlSelectedExit,
    open_time_15m: np.ndarray,
    close_time_15m: np.ndarray,
) -> int:
    time_array = close_time_15m if selected_exit.reason == "close_on_end" else open_time_15m
    return int(time_array[selected_exit.exit_abs])


def _funding_event_mark_price(
    *,
    event_time_ms: int,
    mark_price: float,
    execution_close_1m: np.ndarray,
    execution_close_time_1m: np.ndarray,
) -> tuple[float, bool]:
    if math.isfinite(mark_price) and mark_price > 0.0:
        return mark_price, False

    close_time = np.asarray(execution_close_time_1m, dtype=np.int64)
    if int(close_time.size) == 0:
        return mark_price, False
    fallback_idx = int(np.searchsorted(close_time, event_time_ms, side="right")) - 1
    if fallback_idx < 0:
        return mark_price, False
    fallback_price = float(execution_close_1m[fallback_idx])
    if not math.isfinite(fallback_price) or fallback_price <= 0.0:
        return mark_price, False
    return fallback_price, True


__all__ = [
    "TP_SL_FUNDING_ADJUSTMENT_STAGE_NAME",
    "TP_SL_FUNDING_METRIC_NAMES",
    "TpSlFundingAdjustmentSummary",
    "TpSlSelectedExit",
    "calculate_tp_sl_funding_adjustment",
    "resolve_tp_sl_selected_exit",
]
