from __future__ import annotations

import pytest

from trading.contexts.backtest.domain.entities import AccountStateV1, PositionV1, TradeV1


def test_position_v1_rejects_invalid_direction() -> None:
    """
    Verify position entity rejects unsupported direction literals.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        v1 supports only `long` and `short` open-position directions.
    Raises:
        AssertionError: If invalid direction does not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="direction"):
        PositionV1(
            trade_id=1,
            direction="neutral",
            qty_base=1.0,
            entry_fill_price=100.0,
            entry_bar_index=0,
            entry_quote_amount=100.0,
            entry_fee_quote=0.1,
        )


def test_account_state_v1_reserve_release_and_lock_are_deterministic() -> None:
    """
    Verify account-state transitions follow deterministic reserve/release/lock arithmetic.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Entry fee is charged on reserve and exit fee is charged on release.
    Raises:
        AssertionError: If resulting balances mismatch expected arithmetic.
    Side Effects:
        None.
    """
    initial = AccountStateV1(available_quote=1000.0, safe_quote=0.0)

    reserved = initial.reserve_for_entry(quote_amount=500.0, entry_fee_quote=1.0)
    assert reserved.available_quote == 499.0
    assert reserved.safe_quote == 0.0

    released = reserved.release_after_close(
        quote_amount=500.0,
        gross_pnl_quote=100.0,
        exit_fee_quote=2.0,
    )
    assert released.available_quote == 1097.0
    assert released.safe_quote == 0.0

    locked = released.lock_profit(locked_quote=25.0)
    assert locked.available_quote == 1072.0
    assert locked.safe_quote == 25.0


def test_trade_v1_rejects_invalid_exit_order() -> None:
    """
    Verify trade entity rejects exit bar index lower than entry bar index.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Trade timeline indexes must be monotonic for deterministic replay.
    Raises:
        AssertionError: If invalid trade payload does not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="exit_bar_index"):
        TradeV1(
            trade_id=1,
            direction="long",
            entry_bar_index=5,
            exit_bar_index=4,
            entry_fill_price=100.0,
            exit_fill_price=101.0,
            qty_base=1.0,
            entry_quote_amount=100.0,
            exit_quote_amount=101.0,
            entry_fee_quote=0.1,
            exit_fee_quote=0.1,
            gross_pnl_quote=1.0,
            net_pnl_quote=0.8,
            locked_profit_quote=0.0,
            exit_reason="signal_exit",
        )
