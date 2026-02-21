from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading.contexts.backtest.domain.entities import (
    AccountStateV1,
    ExecutionOutcomeV1,
    PositionV1,
    TradeV1,
)
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1, RiskParamsV1, SignalV1
from trading.contexts.indicators.application.dto import CandleArrays

_SIGNAL_EXIT_REASON = "signal_exit"
_FORCED_CLOSE_REASON = "forced_close"
_SL_EXIT_REASON = "sl"
_TP_EXIT_REASON = "tp"


@dataclass(frozen=True, slots=True)
class _EntryDecision:
    """
    Internal deterministic entry decision for one bar evaluation step.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/execution_engine_v1.py
      - src/trading/contexts/backtest/domain/value_objects/signal_v1.py
      - tests/unit/contexts/backtest/application/services/test_execution_engine_v1.py
    """

    direction: str
    quote_amount: float


class BacktestExecutionEngineV1:
    """
    Deterministic close-fill execution engine v1 for one-variant backtest simulation.

    Docs:
      - docs/architecture/backtest/backtest-execution-engine-close-fill-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/close_fill_scorer_v1.py
      - src/trading/contexts/backtest/domain/entities/execution_v1.py
      - tests/unit/contexts/backtest/application/services/test_execution_engine_v1.py
    """

    def run(
        self,
        *,
        candles: CandleArrays,
        target_slice: slice,
        final_signal: np.ndarray,
        execution_params: ExecutionParamsV1,
        risk_params: RiskParamsV1,
    ) -> ExecutionOutcomeV1:
        """
        Execute deterministic close-fill simulation on target slice and return outcome.

        Args:
            candles: Warmup-inclusive candles used for close-price execution.
            target_slice: Half-open slice of bars included in trading/reporting window.
            final_signal: Aggregated signal vector (`LONG|SHORT|NEUTRAL`) for all bars.
            execution_params: Runtime execution and sizing settings.
            risk_params: Close-based SL/TP settings for current stage.
        Returns:
            ExecutionOutcomeV1: Deterministic trade log and equity summary.
        Assumptions:
            `final_signal` length equals candles length and target slice boundaries are valid.
        Raises:
            ValueError: If signal shape or target slice boundaries are invalid.
        Side Effects:
            None.
        """
        if final_signal.ndim != 1:
            raise ValueError("final_signal must be a 1D array")
        if final_signal.shape[0] != candles.close.shape[0]:
            raise ValueError("final_signal length must match candles.close length")

        start_index = 0 if target_slice.start is None else target_slice.start
        stop_index = candles.close.shape[0] if target_slice.stop is None else target_slice.stop
        if start_index < 0:
            raise ValueError("target_slice.start must be >= 0")
        if stop_index < start_index:
            raise ValueError("target_slice.stop must be >= target_slice.start")
        if stop_index > candles.close.shape[0]:
            raise ValueError("target_slice.stop must be <= candles bars count")

        account = AccountStateV1(available_quote=execution_params.init_cash_quote, safe_quote=0.0)
        position: PositionV1 | None = None
        trades: list[TradeV1] = []
        next_trade_id = 1
        prev_signal = SignalV1.NEUTRAL.value

        for bar_index in range(start_index, stop_index):
            close_price = float(candles.close[bar_index])
            signal_value = _normalize_signal_value(raw_value=final_signal[bar_index])
            is_last_target_bar = bar_index == (stop_index - 1)

            if position is not None:
                risk_exit_reason = _risk_exit_reason(
                    position=position,
                    close_price=close_price,
                    risk_params=risk_params,
                )
                if risk_exit_reason is not None:
                    account, closed_trade = self._close_position(
                        account=account,
                        position=position,
                        close_price=close_price,
                        bar_index=bar_index,
                        reason=risk_exit_reason,
                        execution_params=execution_params,
                    )
                    trades.append(closed_trade)
                    position = None

            if position is not None:
                should_exit_by_signal = _should_exit_by_signal(
                    position=position,
                    signal_value=signal_value,
                )
                if should_exit_by_signal:
                    account, closed_trade = self._close_position(
                        account=account,
                        position=position,
                        close_price=close_price,
                        bar_index=bar_index,
                        reason=_SIGNAL_EXIT_REASON,
                        execution_params=execution_params,
                    )
                    trades.append(closed_trade)
                    position = None

            signal_changed = signal_value != prev_signal
            if position is None and signal_changed:
                entry_decision = _entry_decision_from_signal(
                    signal_value=signal_value,
                    execution_params=execution_params,
                    account=account,
                )
                if entry_decision is not None:
                    account, position = self._open_position(
                        account=account,
                        trade_id=next_trade_id,
                        direction=entry_decision.direction,
                        quote_amount=entry_decision.quote_amount,
                        close_price=close_price,
                        bar_index=bar_index,
                        execution_params=execution_params,
                    )
                    next_trade_id += 1

            if is_last_target_bar and position is not None:
                account, closed_trade = self._close_position(
                    account=account,
                    position=position,
                    close_price=close_price,
                    bar_index=bar_index,
                    reason=_FORCED_CLOSE_REASON,
                    execution_params=execution_params,
                )
                trades.append(closed_trade)
                position = None

            prev_signal = signal_value

        equity_end_quote = account.available_quote + account.safe_quote
        total_return_pct = (
            ((equity_end_quote / execution_params.init_cash_quote) - 1.0) * 100.0
        )
        return ExecutionOutcomeV1(
            trades=tuple(trades),
            equity_end_quote=equity_end_quote,
            available_quote=account.available_quote,
            safe_quote=account.safe_quote,
            total_return_pct=total_return_pct,
        )

    def _open_position(
        self,
        *,
        account: AccountStateV1,
        trade_id: int,
        direction: str,
        quote_amount: float,
        close_price: float,
        bar_index: int,
        execution_params: ExecutionParamsV1,
    ) -> tuple[AccountStateV1, PositionV1]:
        """
        Open one position on current bar close with deterministic slippage/fee semantics.

        Args:
            account: Current strategy-account state.
            trade_id: Deterministic trade identifier.
            direction: Entry direction (`long` or `short`).
            quote_amount: Position budget in quote currency.
            close_price: Current bar raw close price.
            bar_index: Current bar index.
            execution_params: Execution and fee/slippage settings.
        Returns:
            tuple[AccountStateV1, PositionV1]: Updated account and newly opened position.
        Assumptions:
            Quote amount is positive and direction is already validated by caller.
        Raises:
            ValueError: If one numeric input is non-positive.
        Side Effects:
            None.
        """
        if quote_amount <= 0.0:
            raise ValueError("quote_amount must be > 0")
        if close_price <= 0.0:
            raise ValueError("close_price must be > 0")

        is_buy = direction == "long"
        fill_price = _fill_price_from_close(
            close_price=close_price,
            slippage_rate=execution_params.slippage_rate,
            is_buy=is_buy,
        )
        qty_base = quote_amount / fill_price
        entry_fee_quote = quote_amount * execution_params.fee_rate
        updated_account = account.reserve_for_entry(
            quote_amount=quote_amount,
            entry_fee_quote=entry_fee_quote,
        )
        position = PositionV1(
            trade_id=trade_id,
            direction=direction,
            qty_base=qty_base,
            entry_fill_price=fill_price,
            entry_bar_index=bar_index,
            entry_quote_amount=quote_amount,
            entry_fee_quote=entry_fee_quote,
        )
        return updated_account, position

    def _close_position(
        self,
        *,
        account: AccountStateV1,
        position: PositionV1,
        close_price: float,
        bar_index: int,
        reason: str,
        execution_params: ExecutionParamsV1,
    ) -> tuple[AccountStateV1, TradeV1]:
        """
        Close open position on current bar close with deterministic fee/slippage accounting.

        Args:
            account: Current strategy-account state.
            position: Currently open position.
            close_price: Current bar raw close price.
            bar_index: Current bar index.
            reason: Exit reason literal.
            execution_params: Execution and fee/slippage settings.
        Returns:
            tuple[AccountStateV1, TradeV1]: Updated account and closed trade snapshot.
        Assumptions:
            Close operation is called only when a position exists.
        Raises:
            ValueError: If close price is non-positive.
        Side Effects:
            None.
        """
        if close_price <= 0.0:
            raise ValueError("close_price must be > 0")

        is_buy = position.direction == "short"
        exit_fill_price = _fill_price_from_close(
            close_price=close_price,
            slippage_rate=execution_params.slippage_rate,
            is_buy=is_buy,
        )
        exit_quote_amount = position.qty_base * exit_fill_price
        exit_fee_quote = exit_quote_amount * execution_params.fee_rate

        if position.direction == "long":
            gross_pnl_quote = exit_quote_amount - position.entry_quote_amount
        else:
            gross_pnl_quote = position.entry_quote_amount - exit_quote_amount

        updated_account = account.release_after_close(
            quote_amount=position.entry_quote_amount,
            gross_pnl_quote=gross_pnl_quote,
            exit_fee_quote=exit_fee_quote,
        )
        net_pnl_quote = gross_pnl_quote - position.entry_fee_quote - exit_fee_quote

        locked_profit_quote = 0.0
        if (
            execution_params.sizing_mode == "strategy_compound_profit_lock"
            and net_pnl_quote > 0.0
        ):
            locked_profit_quote = net_pnl_quote * (execution_params.safe_profit_percent / 100.0)
            updated_account = updated_account.lock_profit(locked_quote=locked_profit_quote)

        trade = TradeV1(
            trade_id=position.trade_id,
            direction=position.direction,
            entry_bar_index=position.entry_bar_index,
            exit_bar_index=bar_index,
            entry_fill_price=position.entry_fill_price,
            exit_fill_price=exit_fill_price,
            qty_base=position.qty_base,
            entry_quote_amount=position.entry_quote_amount,
            exit_quote_amount=exit_quote_amount,
            entry_fee_quote=position.entry_fee_quote,
            exit_fee_quote=exit_fee_quote,
            gross_pnl_quote=gross_pnl_quote,
            net_pnl_quote=net_pnl_quote,
            locked_profit_quote=locked_profit_quote,
            exit_reason=reason,
        )
        return updated_account, trade


def _normalize_signal_value(*, raw_value: object) -> str:
    """
    Normalize one signal value to canonical `LONG|SHORT|NEUTRAL` literal.

    Args:
        raw_value: Raw signal value from aggregated signal array.
    Returns:
        str: Canonical signal literal.
    Assumptions:
        Signal array may contain `SignalV1` enum or uppercase string literal.
    Raises:
        ValueError: If signal literal is unsupported.
    Side Effects:
        None.
    """
    normalized = str(raw_value).strip().upper()
    if normalized not in {
        SignalV1.LONG.value,
        SignalV1.SHORT.value,
        SignalV1.NEUTRAL.value,
    }:
        raise ValueError("final_signal values must be LONG, SHORT, or NEUTRAL")
    return normalized


def _fill_price_from_close(*, close_price: float, slippage_rate: float, is_buy: bool) -> float:
    """
    Build fill price from raw close using deterministic buy/sell slippage semantics.

    Args:
        close_price: Raw close price.
        slippage_rate: Decimal slippage rate (`0.0001 == 0.01%`).
        is_buy: Whether current fill side is buy.
    Returns:
        float: Slippage-adjusted fill price.
    Assumptions:
        Buy fill uses `+slippage`, sell fill uses `-slippage`.
    Raises:
        ValueError: If close price is non-positive.
    Side Effects:
        None.
    """
    if close_price <= 0.0:
        raise ValueError("close_price must be > 0")
    if is_buy:
        return close_price * (1.0 + slippage_rate)
    return close_price * (1.0 - slippage_rate)


def _entry_decision_from_signal(
    *,
    signal_value: str,
    execution_params: ExecutionParamsV1,
    account: AccountStateV1,
) -> _EntryDecision | None:
    """
    Resolve deterministic entry decision from current signal and direction/sizing settings.

    Args:
        signal_value: Canonical current signal literal.
        execution_params: Execution settings with mode literals and sizing params.
        account: Current strategy-account state.
    Returns:
        _EntryDecision | None: Entry decision or `None` when entry is forbidden/empty.
    Assumptions:
        Entry gating (edge semantics) is handled by engine loop before this call.
    Raises:
        None.
    Side Effects:
        None.
    """
    direction: str | None = None
    if signal_value == SignalV1.LONG.value:
        if execution_params.direction_mode in {"long-only", "long-short"}:
            direction = "long"
    elif signal_value == SignalV1.SHORT.value:
        if execution_params.direction_mode in {"short-only", "long-short"}:
            direction = "short"

    if direction is None:
        return None

    quote_amount = _entry_quote_amount(account=account, execution_params=execution_params)
    if quote_amount <= 0.0:
        return None
    return _EntryDecision(direction=direction, quote_amount=quote_amount)


def _entry_quote_amount(*, account: AccountStateV1, execution_params: ExecutionParamsV1) -> float:
    """
    Compute deterministic entry notional for selected sizing mode.

    Args:
        account: Current strategy-account state.
        execution_params: Execution settings containing sizing mode and defaults.
    Returns:
        float: Entry quote budget (non-negative scalar).
    Assumptions:
        `strategy_compound` and `all_in` both consume full available quote in v1.
    Raises:
        None.
    Side Effects:
        None.
    """
    available_quote = account.available_quote
    if available_quote <= 0.0:
        return 0.0

    if execution_params.sizing_mode == "fixed_quote":
        return min(available_quote, execution_params.fixed_quote)

    if execution_params.sizing_mode in {
        "all_in",
        "strategy_compound",
        "strategy_compound_profit_lock",
    }:
        return available_quote

    return 0.0


def _risk_exit_reason(
    *,
    position: PositionV1,
    close_price: float,
    risk_params: RiskParamsV1,
) -> str | None:
    """
    Resolve deterministic risk exit reason (`sl`/`tp`) for current close price.

    Args:
        position: Open position snapshot.
        close_price: Raw close price for current bar.
        risk_params: Close-based risk settings for current stage.
    Returns:
        str | None: Exit reason literal or `None` when no risk trigger is hit.
    Assumptions:
        SL has priority when both SL and TP conditions are true on the same bar.
    Raises:
        ValueError: If close price is non-positive.
    Side Effects:
        None.
    """
    if close_price <= 0.0:
        raise ValueError("close_price must be > 0")

    sl_hit = False
    tp_hit = False

    if risk_params.sl_enabled and risk_params.sl_rate is not None:
        if position.direction == "long":
            sl_hit = close_price <= position.entry_fill_price * (1.0 - risk_params.sl_rate)
        else:
            sl_hit = close_price >= position.entry_fill_price * (1.0 + risk_params.sl_rate)

    if risk_params.tp_enabled and risk_params.tp_rate is not None:
        if position.direction == "long":
            tp_hit = close_price >= position.entry_fill_price * (1.0 + risk_params.tp_rate)
        else:
            tp_hit = close_price <= position.entry_fill_price * (1.0 - risk_params.tp_rate)

    if sl_hit:
        return _SL_EXIT_REASON
    if tp_hit:
        return _TP_EXIT_REASON
    return None


def _should_exit_by_signal(*, position: PositionV1, signal_value: str) -> bool:
    """
    Decide whether current open position must be closed by signal semantics.

    Args:
        position: Current open position.
        signal_value: Canonical current signal literal.
    Returns:
        bool: True when signal requires immediate close of current position.
    Assumptions:
        Forbidden opposite signals in one-side modes are handled as exit-only by caller.
    Raises:
        None.
    Side Effects:
        None.
    """
    if position.direction == "long":
        return signal_value in {SignalV1.NEUTRAL.value, SignalV1.SHORT.value}
    return signal_value in {SignalV1.NEUTRAL.value, SignalV1.LONG.value}


__all__ = [
    "BacktestExecutionEngineV1",
]
