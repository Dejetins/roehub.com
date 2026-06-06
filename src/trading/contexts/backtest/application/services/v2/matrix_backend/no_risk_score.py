from __future__ import annotations

import numba as nb
import numpy as np

from trading.contexts.backtest.application.services.v2.execution_sizing import (
    execution_quote_amount,
)

BITS_PER_WORD = 64
ALL_BITS = np.uint64(18446744073709551615)


@nb.njit(cache=True, parallel=True, fastmath=True)
def matrix_bitset_no_risk_long_only(
    combo_idx_by_indicator: np.ndarray,
    pos_bits_0: np.ndarray,
    pos_bits_1: np.ndarray,
    pos_bits_2: np.ndarray,
    arity: np.int32,
    signal_length: np.int32,
    word_count: np.int32,
    last_word_mask: np.uint64,
    sig_entry_exec_idx: np.ndarray,
    exec_open_1m: np.ndarray,
    exec_close_1m: np.ndarray,
    t_exec: np.int32,
    init_cash_quote: float,
    sizing_mode_code: np.int8,
    configured_quote_amount: float,
    equity_pct: float,
    min_quote: float,
    max_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    use_profit_lock: np.int8,
    bars_per_year_exec: float,
    close_on_end: np.int8,
    out_total_return_pct: np.ndarray,
    out_max_drawdown_pct: np.ndarray,
    out_return_over_max_drawdown: np.ndarray,
    out_profit_factor: np.ndarray,
    out_trade_count: np.ndarray,
    out_sharpe_trades: np.ndarray,
    out_win_rate_pct: np.ndarray,
    out_avg_trade_ret_pct: np.ndarray,
    out_avg_trade_exec_bars: np.ndarray,
    out_exposure_pct: np.ndarray,
) -> None:
    combo_count = combo_idx_by_indicator.shape[1]
    for k in nb.prange(combo_count):
        row_0 = combo_idx_by_indicator[0, k]
        row_1 = combo_idx_by_indicator[1, k]
        row_2 = np.int32(0)
        if arity == 3:
            row_2 = combo_idx_by_indicator[2, k]

        available_quote = init_cash_quote
        safe_quote = 0.0
        equity = init_cash_quote
        peak_equity = equity
        max_drawdown_pct = 0.0
        gross_profit_quote = 0.0
        gross_loss_quote = 0.0
        closed_trade_count = np.int32(0)
        win_count = np.int32(0)
        sum_trade_return = 0.0
        sum_trade_return_squared = 0.0
        total_trade_return_pct = 0.0
        total_trade_exec_bars = 0.0
        exposure_bars = 0.0
        current_dir = np.int8(0)
        current_entry = np.int32(0)
        stopped = False

        for word_idx in range(word_count):
            word_mask = ALL_BITS
            if word_idx == word_count - 1:
                word_mask = last_word_mask
            bits = pos_bits_0[row_0, word_idx] & pos_bits_1[row_1, word_idx]
            if arity == 3:
                bits &= pos_bits_2[row_2, word_idx]
            bits &= word_mask

            if current_dir == 1:
                zero_bits = word_mask ^ bits
                if zero_bits != 0:
                    exit_bit = _ctz_u64(zero_bits)
                    exit_signal_idx = np.int32(word_idx * BITS_PER_WORD + exit_bit)
                    if exit_signal_idx >= signal_length:
                        stopped = True
                        break
                    exit_exec = sig_entry_exec_idx[exit_signal_idx]
                    if exit_exec >= t_exec:
                        stopped = True
                        break
                    (
                        available_quote,
                        safe_quote,
                        equity,
                        peak_equity,
                        max_drawdown_pct,
                        gross_profit_quote,
                        gross_loss_quote,
                        closed_trade_count,
                        win_count,
                        sum_trade_return,
                        sum_trade_return_squared,
                        total_trade_return_pct,
                        total_trade_exec_bars,
                        exposure_bars,
                    ) = _apply_long_trade_to_state(
                        current_entry,
                        np.int32(exit_exec),
                        float(exec_open_1m[exit_exec]),
                        exec_open_1m,
                        available_quote,
                        safe_quote,
                        equity,
                        peak_equity,
                        max_drawdown_pct,
                        gross_profit_quote,
                        gross_loss_quote,
                        closed_trade_count,
                        win_count,
                        sum_trade_return,
                        sum_trade_return_squared,
                        total_trade_return_pct,
                        total_trade_exec_bars,
                        exposure_bars,
                        init_cash_quote,
                        sizing_mode_code,
                        configured_quote_amount,
                        equity_pct,
                        min_quote,
                        max_quote,
                        fee_rate,
                        slippage_rate,
                        safe_profit_percent,
                        use_profit_lock,
                    )
                    current_dir = np.int8(0)
                    current_entry = np.int32(0)
                    bits &= _mask_from_bit(exit_bit)

            while bits != 0 and not stopped:
                entry_bit = _ctz_u64(bits)
                entry_signal_idx = np.int32(word_idx * BITS_PER_WORD + entry_bit)
                if entry_signal_idx >= signal_length:
                    stopped = True
                    break
                entry_exec = sig_entry_exec_idx[entry_signal_idx]
                if entry_exec >= t_exec:
                    stopped = True
                    break
                current_dir = np.int8(1)
                current_entry = np.int32(entry_exec)

                remaining_mask = _mask_from_bit(entry_bit)
                zero_bits = (word_mask ^ bits) & remaining_mask
                if zero_bits == 0:
                    break

                exit_bit = _ctz_u64(zero_bits)
                exit_signal_idx = np.int32(word_idx * BITS_PER_WORD + exit_bit)
                if exit_signal_idx >= signal_length:
                    stopped = True
                    break
                exit_exec = sig_entry_exec_idx[exit_signal_idx]
                if exit_exec >= t_exec:
                    stopped = True
                    break
                (
                    available_quote,
                    safe_quote,
                    equity,
                    peak_equity,
                    max_drawdown_pct,
                    gross_profit_quote,
                    gross_loss_quote,
                    closed_trade_count,
                    win_count,
                    sum_trade_return,
                    sum_trade_return_squared,
                    total_trade_return_pct,
                    total_trade_exec_bars,
                    exposure_bars,
                ) = _apply_long_trade_to_state(
                    current_entry,
                    np.int32(exit_exec),
                    float(exec_open_1m[exit_exec]),
                    exec_open_1m,
                    available_quote,
                    safe_quote,
                    equity,
                    peak_equity,
                    max_drawdown_pct,
                    gross_profit_quote,
                    gross_loss_quote,
                    closed_trade_count,
                    win_count,
                    sum_trade_return,
                    sum_trade_return_squared,
                    total_trade_return_pct,
                    total_trade_exec_bars,
                    exposure_bars,
                    init_cash_quote,
                    sizing_mode_code,
                    configured_quote_amount,
                    equity_pct,
                    min_quote,
                    max_quote,
                    fee_rate,
                    slippage_rate,
                    safe_profit_percent,
                    use_profit_lock,
                )
                current_dir = np.int8(0)
                current_entry = np.int32(0)
                bits &= _mask_from_bit(exit_bit)

            if stopped:
                break

        if current_dir != 0 and close_on_end == 1 and t_exec > 0:
            exit_exec_idx = np.int32(t_exec - 1)
            (
                available_quote,
                safe_quote,
                equity,
                peak_equity,
                max_drawdown_pct,
                gross_profit_quote,
                gross_loss_quote,
                closed_trade_count,
                win_count,
                sum_trade_return,
                sum_trade_return_squared,
                total_trade_return_pct,
                total_trade_exec_bars,
                exposure_bars,
            ) = _apply_long_trade_to_state(
                current_entry,
                exit_exec_idx,
                float(exec_close_1m[exit_exec_idx]),
                exec_open_1m,
                available_quote,
                safe_quote,
                equity,
                peak_equity,
                max_drawdown_pct,
                gross_profit_quote,
                gross_loss_quote,
                closed_trade_count,
                win_count,
                sum_trade_return,
                sum_trade_return_squared,
                total_trade_return_pct,
                total_trade_exec_bars,
                exposure_bars,
                init_cash_quote,
                sizing_mode_code,
                configured_quote_amount,
                equity_pct,
                min_quote,
                max_quote,
                fee_rate,
                slippage_rate,
                safe_profit_percent,
                use_profit_lock,
            )

        _write_final_metrics(
            k,
            equity,
            init_cash_quote,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
            t_exec,
            bars_per_year_exec,
            out_total_return_pct,
            out_max_drawdown_pct,
            out_return_over_max_drawdown,
            out_profit_factor,
            out_trade_count,
            out_sharpe_trades,
            out_win_rate_pct,
            out_avg_trade_ret_pct,
            out_avg_trade_exec_bars,
            out_exposure_pct,
        )


@nb.njit(cache=True, inline="always")
def _mask_from_bit(bit_idx: int) -> np.uint64:
    return ALL_BITS << np.uint64(bit_idx)


@nb.njit(cache=True, inline="always")
def _ctz_u64(value: np.uint64) -> int:
    idx = 0
    while (value & np.uint64(1)) == np.uint64(0):
        value >>= np.uint64(1)
        idx += 1
    return idx


@nb.njit(cache=True, inline="always")
def _apply_long_trade_to_state(
    entry_idx: np.int32,
    exit_exec_idx: np.int32,
    exit_price_raw: float,
    exec_open_1m: np.ndarray,
    available_quote: float,
    safe_quote: float,
    equity: float,
    peak_equity: float,
    max_drawdown_pct: float,
    gross_profit_quote: float,
    gross_loss_quote: float,
    closed_trade_count: np.int32,
    win_count: np.int32,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    total_trade_return_pct: float,
    total_trade_exec_bars: float,
    exposure_bars: float,
    init_cash_quote: float,
    sizing_mode_code: np.int8,
    configured_quote_amount: float,
    equity_pct: float,
    min_quote: float,
    max_quote: float,
    fee_rate: float,
    slippage_rate: float,
    safe_profit_percent: float,
    use_profit_lock: np.int8,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    np.int32,
    np.int32,
    float,
    float,
    float,
    float,
    float,
]:
    if available_quote <= 0.0:
        return (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
        )

    quote_amount = execution_quote_amount(
        available_quote,
        equity,
        sizing_mode_code,
        configured_quote_amount,
        equity_pct,
        min_quote,
        max_quote,
    )
    if quote_amount <= 0.0:
        return (
            available_quote,
            safe_quote,
            equity,
            peak_equity,
            max_drawdown_pct,
            gross_profit_quote,
            gross_loss_quote,
            closed_trade_count,
            win_count,
            sum_trade_return,
            sum_trade_return_squared,
            total_trade_return_pct,
            total_trade_exec_bars,
            exposure_bars,
        )

    entry_price_raw = float(exec_open_1m[entry_idx])
    entry_fill_price = entry_price_raw * (1.0 + slippage_rate)
    exit_fill_price = exit_price_raw * (1.0 - slippage_rate)
    qty_base = quote_amount / entry_fill_price
    entry_fee_quote = quote_amount * fee_rate
    available_quote -= quote_amount + entry_fee_quote

    exit_quote_amount = qty_base * exit_fill_price
    exit_fee_quote = exit_quote_amount * fee_rate
    gross_pnl_quote = exit_quote_amount - quote_amount
    available_quote += quote_amount + gross_pnl_quote - exit_fee_quote
    net_pnl_quote = gross_pnl_quote - entry_fee_quote - exit_fee_quote

    if use_profit_lock == 1 and net_pnl_quote > 0.0:
        locked_profit_quote = net_pnl_quote * (safe_profit_percent / 100.0)
        available_quote -= locked_profit_quote
        safe_quote += locked_profit_quote

    equity = available_quote + safe_quote
    if equity > peak_equity:
        peak_equity = equity
    elif peak_equity > 0.0:
        drawdown_pct = ((peak_equity - equity) / peak_equity) * 100.0
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct

    trade_return_pct = (net_pnl_quote / quote_amount) * 100.0
    trade_return = net_pnl_quote / quote_amount
    bars_held = float(exit_exec_idx - entry_idx)
    if bars_held < 0.0:
        bars_held = 0.0

    closed_trade_count += 1
    if net_pnl_quote > 0.0:
        win_count += 1
        gross_profit_quote += net_pnl_quote
    elif net_pnl_quote < 0.0:
        gross_loss_quote += abs(net_pnl_quote)
    sum_trade_return += trade_return
    sum_trade_return_squared += trade_return * trade_return
    total_trade_return_pct += trade_return_pct
    total_trade_exec_bars += bars_held
    exposure_bars += bars_held

    return (
        available_quote,
        safe_quote,
        equity,
        peak_equity,
        max_drawdown_pct,
        gross_profit_quote,
        gross_loss_quote,
        closed_trade_count,
        win_count,
        sum_trade_return,
        sum_trade_return_squared,
        total_trade_return_pct,
        total_trade_exec_bars,
        exposure_bars,
    )


@nb.njit(cache=True, inline="always")
def _trade_sharpe(
    trade_count: np.int32,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    bars_per_year_exec: float,
    sentinel_index: np.int32,
) -> float:
    if trade_count <= 1:
        return 0.0
    mean_trade_return = sum_trade_return / float(trade_count)
    variance = (sum_trade_return_squared / float(trade_count)) - (
        mean_trade_return * mean_trade_return
    )
    if variance <= 0.0:
        return 0.0
    years = float(sentinel_index) / float(bars_per_year_exec)
    if years <= 0.0:
        years = 1.0
    trades_per_year = float(trade_count) / years
    return (mean_trade_return / np.sqrt(variance)) * np.sqrt(trades_per_year)


@nb.njit(cache=True, inline="always")
def _write_final_metrics(
    k: int,
    equity: float,
    init_cash_quote: float,
    max_drawdown_pct: float,
    gross_profit_quote: float,
    gross_loss_quote: float,
    closed_trade_count: np.int32,
    win_count: np.int32,
    sum_trade_return: float,
    sum_trade_return_squared: float,
    total_trade_return_pct: float,
    total_trade_exec_bars: float,
    exposure_bars: float,
    t_exec: np.int32,
    bars_per_year_exec: float,
    out_total_return_pct: np.ndarray,
    out_max_drawdown_pct: np.ndarray,
    out_return_over_max_drawdown: np.ndarray,
    out_profit_factor: np.ndarray,
    out_trade_count: np.ndarray,
    out_sharpe_trades: np.ndarray,
    out_win_rate_pct: np.ndarray,
    out_avg_trade_ret_pct: np.ndarray,
    out_avg_trade_exec_bars: np.ndarray,
    out_exposure_pct: np.ndarray,
) -> None:
    total_return_pct = ((equity / init_cash_quote) - 1.0) * 100.0
    out_total_return_pct[k] = total_return_pct
    out_trade_count[k] = closed_trade_count
    if out_max_drawdown_pct.shape[0] == 0:
        return

    if gross_loss_quote > 0.0:
        profit_factor = gross_profit_quote / gross_loss_quote
    elif gross_profit_quote > 0.0:
        profit_factor = np.inf
    else:
        profit_factor = 0.0

    if max_drawdown_pct > 0.0:
        return_over_max_drawdown = total_return_pct / max_drawdown_pct
    elif total_return_pct > 0.0:
        return_over_max_drawdown = np.inf
    else:
        return_over_max_drawdown = 0.0

    if closed_trade_count > 0:
        win_rate_pct = (float(win_count) / float(closed_trade_count)) * 100.0
        avg_trade_ret_pct = total_trade_return_pct / float(closed_trade_count)
        avg_trade_exec_bars = total_trade_exec_bars / float(closed_trade_count)
    else:
        win_rate_pct = 0.0
        avg_trade_ret_pct = 0.0
        avg_trade_exec_bars = 0.0
    exposure_pct = (exposure_bars / float(t_exec)) * 100.0 if t_exec > 0 else 0.0
    sharpe_trades = _trade_sharpe(
        closed_trade_count,
        sum_trade_return,
        sum_trade_return_squared,
        bars_per_year_exec,
        t_exec,
    )
    out_max_drawdown_pct[k] = max_drawdown_pct
    out_return_over_max_drawdown[k] = return_over_max_drawdown
    out_profit_factor[k] = profit_factor
    out_sharpe_trades[k] = sharpe_trades
    out_win_rate_pct[k] = win_rate_pct
    out_avg_trade_ret_pct[k] = avg_trade_ret_pct
    out_avg_trade_exec_bars[k] = avg_trade_exec_bars
    out_exposure_pct[k] = exposure_pct


__all__ = ["matrix_bitset_no_risk_long_only"]
