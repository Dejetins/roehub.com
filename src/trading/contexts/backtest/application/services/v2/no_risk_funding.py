from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from trading.contexts.backtest.application.services.v2.execution_sizing import (
    ExecutionSettings,
    execution_quote_amount,
)

TOTAL_RETURN_PCT_NET_OF_FUNDING = "total_return_pct_net_of_funding"
FUNDING_RETURN_PCT = "funding_return_pct"
FUNDING_PNL_QUOTE = "funding_pnl_quote"
FUNDING_EVENTS_COUNT = "funding_events_count"
FUNDING_DATA_QUALITY = "funding_data_quality"
FUNDING_WARNING_CODES = "funding_warning_codes"
FUNDING_INCLUDED = "funding_included"
FUNDING_ADJUSTMENT_SCOPE = "funding_adjustment_scope"
FUNDING_ADJUSTMENT_EXACT_GLOBAL_RANKING = "funding_adjustment_exact_global_ranking"
FUNDING_ADJUSTMENT_SCOPE_CANDIDATE_POOL = "bounded_candidate_pool"
FUNDING_ADJUSTMENT_SCOPE_UNAVAILABLE = "unavailable"
NO_RISK_FUNDING_METRIC_NAMES = (
    TOTAL_RETURN_PCT_NET_OF_FUNDING,
    FUNDING_RETURN_PCT,
    FUNDING_PNL_QUOTE,
    FUNDING_EVENTS_COUNT,
)


@dataclass(frozen=True, slots=True)
class NoRiskFundingAdjustmentSummary:
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


def calculate_no_risk_funding_adjustment(
    *,
    entry_exec_idx: np.ndarray,
    dir_arr: np.ndarray,
    sig_exit_exec_idx: np.ndarray,
    execution_open_1m: np.ndarray,
    execution_close_1m: np.ndarray,
    execution_open_time_1m: np.ndarray,
    execution_close_time_1m: np.ndarray,
    funding_time: np.ndarray,
    funding_rate: np.ndarray,
    mark_price: np.ndarray,
    data_quality: np.ndarray,
    execution_settings: ExecutionSettings,
    t_exec: int,
    funding_data_quality: str,
    warning_codes: Sequence[str] = (),
) -> NoRiskFundingAdjustmentSummary:
    """
    Calculate post-hoc funding PnL for no-risk open-position windows.

    Positive funding rates make long positions pay and short positions receive.
    Funding is applied after base scoring, so it does not resize later trades.
    """

    funding_pnl_quote = 0.0
    funding_events_count = 0
    warnings = set(str(code) for code in warning_codes if str(code))
    available_quote = execution_settings.initial_cash_quote
    safe_quote = 0.0
    equity = execution_settings.initial_cash_quote
    funding_time_i64 = np.asarray(funding_time, dtype=np.int64)
    funding_rate_f64 = np.asarray(funding_rate, dtype=np.float64)
    mark_price_f64 = np.asarray(mark_price, dtype=np.float64)
    data_quality_u8 = np.asarray(data_quality, dtype=np.uint8)

    for trade_index in range(int(entry_exec_idx.size)):
        entry_idx = int(entry_exec_idx[trade_index])
        if entry_idx >= t_exec:
            continue
        exit_idx = int(sig_exit_exec_idx[trade_index])
        if exit_idx < t_exec:
            exit_exec_idx = exit_idx
            exit_price_raw = float(execution_open_1m[exit_exec_idx])
            exit_time_ms = int(execution_open_time_1m[exit_exec_idx])
        elif execution_settings.close_on_end == 1 and t_exec > 0:
            exit_exec_idx = t_exec - 1
            exit_price_raw = float(execution_close_1m[exit_exec_idx])
            exit_time_ms = int(execution_close_time_1m[exit_exec_idx])
        else:
            continue
        if available_quote <= 0.0:
            continue
        quote_amount = execution_quote_amount(
            available_quote,
            equity,
            execution_settings.sizing_mode_code,
            execution_settings.quote_amount,
            execution_settings.equity_pct,
            execution_settings.min_quote,
            execution_settings.max_quote,
        )
        if quote_amount <= 0.0:
            continue
        trade_direction = int(dir_arr[trade_index])
        entry_price_raw = float(execution_open_1m[entry_idx])
        if trade_direction == 1:
            entry_fill_price = entry_price_raw * (1.0 + execution_settings.slippage_rate)
            exit_fill_price = exit_price_raw * (1.0 - execution_settings.slippage_rate)
        else:
            entry_fill_price = entry_price_raw * (1.0 - execution_settings.slippage_rate)
            exit_fill_price = exit_price_raw * (1.0 + execution_settings.slippage_rate)
        qty_base = quote_amount / entry_fill_price
        entry_time_ms = int(execution_open_time_1m[entry_idx])
        event_start = int(np.searchsorted(funding_time_i64, entry_time_ms, side="right"))
        event_stop = int(np.searchsorted(funding_time_i64, exit_time_ms, side="right"))
        for event_idx in range(event_start, event_stop):
            rate = float(funding_rate_f64[event_idx])
            price = float(mark_price_f64[event_idx])
            if not (math.isfinite(rate) and math.isfinite(price)) or price <= 0.0:
                warnings.add("invalid_funding_event_values")
                continue
            if int(data_quality_u8[event_idx]) == 0:
                warnings.add("funding_event_data_quality_degraded")
            funding_pnl_quote += -float(trade_direction) * qty_base * price * rate
            funding_events_count += 1

        entry_fee_quote = quote_amount * execution_settings.fee_rate
        available_quote -= quote_amount + entry_fee_quote
        exit_quote_amount = qty_base * exit_fill_price
        exit_fee_quote = exit_quote_amount * execution_settings.fee_rate
        if trade_direction == 1:
            gross_pnl_quote = exit_quote_amount - quote_amount
        else:
            gross_pnl_quote = quote_amount - exit_quote_amount
        available_quote += quote_amount + gross_pnl_quote - exit_fee_quote
        net_pnl_quote = gross_pnl_quote - entry_fee_quote - exit_fee_quote
        if execution_settings.use_profit_lock == 1 and net_pnl_quote > 0.0:
            locked_profit_quote = net_pnl_quote * (
                execution_settings.safe_profit_percent / 100.0
            )
            available_quote -= locked_profit_quote
            safe_quote += locked_profit_quote
        equity = available_quote + safe_quote

    quality = funding_data_quality
    if "funding_event_data_quality_degraded" in warnings:
        quality = "degraded"
    return NoRiskFundingAdjustmentSummary(
        funding_pnl_quote=funding_pnl_quote,
        funding_events_count=funding_events_count,
        funding_data_quality=quality,
        funding_warning_codes=tuple(sorted(warnings)),
    )


__all__ = [
    "FUNDING_ADJUSTMENT_EXACT_GLOBAL_RANKING",
    "FUNDING_ADJUSTMENT_SCOPE",
    "FUNDING_ADJUSTMENT_SCOPE_CANDIDATE_POOL",
    "FUNDING_ADJUSTMENT_SCOPE_UNAVAILABLE",
    "FUNDING_DATA_QUALITY",
    "FUNDING_EVENTS_COUNT",
    "FUNDING_INCLUDED",
    "FUNDING_PNL_QUOTE",
    "FUNDING_RETURN_PCT",
    "FUNDING_WARNING_CODES",
    "NO_RISK_FUNDING_METRIC_NAMES",
    "NoRiskFundingAdjustmentSummary",
    "TOTAL_RETURN_PCT_NET_OF_FUNDING",
    "calculate_no_risk_funding_adjustment",
]
