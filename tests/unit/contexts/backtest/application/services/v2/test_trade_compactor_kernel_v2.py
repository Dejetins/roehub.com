from __future__ import annotations

import math

import numpy as np
import pytest

from trading.contexts.backtest.application.services import (
    build_compact_trade_list_v2,
    compute_no_risk_metrics_v2,
    no_risk_metrics_to_ranking_payload_v2,
)
from trading.contexts.backtest.application.services.v2.contracts import StageACompactTradeV2
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1


def test_build_compact_trade_list_v2_builds_signal_exit_and_sentinel_trades() -> None:
    """
    Verify Stage A compact trades keep opposite-signal exits and sentinel carry semantics.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Repeated same-direction confirmations are ignored and neutral bars do not close trades.
    Raises:
        AssertionError: If compact trade fields drift from the deterministic contract.
    Side Effects:
        None.
    """
    compact = build_compact_trade_list_v2(
        final_signal=np.array([[1, 1, -1, 0]], dtype=np.int8),
        bar_close_1m_idx=np.array([0, 1, 2, 3], dtype=np.int64),
        sentinel_index=5,
    )

    assert compact == (
        (
            StageACompactTradeV2(
                entry_signal_idx=0,
                entry_exec_idx=1,
                direction=1,
                sig_exit_signal_idx=2,
                sig_exit_exec_idx=3,
            ),
            StageACompactTradeV2(
                entry_signal_idx=2,
                entry_exec_idx=3,
                direction=-1,
                sig_exit_signal_idx=None,
                sig_exit_exec_idx=5,
            ),
        ),
    )


def test_build_compact_trade_list_v2_respects_exit_only_direction_modes() -> None:
    """
    Verify forbidden opposite signals close open trades but do not open new ones in one-side mode.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `long-only` must treat short confirmations as exit-only events.
    Raises:
        AssertionError: If one-side mode opens forbidden trades.
    Side Effects:
        None.
    """
    compact = build_compact_trade_list_v2(
        final_signal=np.array([[1, -1, -1, 1]], dtype=np.int8),
        bar_close_1m_idx=np.array([0, 1, 2, 3], dtype=np.int64),
        sentinel_index=5,
        direction_mode="long-only",
    )

    assert compact == (
        (
            StageACompactTradeV2(
                entry_signal_idx=0,
                entry_exec_idx=1,
                direction=1,
                sig_exit_signal_idx=1,
                sig_exit_exec_idx=2,
            ),
            StageACompactTradeV2(
                entry_signal_idx=3,
                entry_exec_idx=4,
                direction=1,
                sig_exit_signal_idx=None,
                sig_exit_exec_idx=5,
            ),
        ),
    )


def test_compute_no_risk_metrics_v2_is_deterministic_and_shortlist_ready() -> None:
    """
    Verify no-risk Stage A metrics are deterministic and expose stable shortlist ranking fields.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        All-in sizing with zero fees/slippage compounds returns across closed compact trades.
    Raises:
        AssertionError: If metric values or ranking payload drift.
    Side Effects:
        None.
    """
    metrics = compute_no_risk_metrics_v2(
        compact_trades=(
            StageACompactTradeV2(
                entry_signal_idx=0,
                entry_exec_idx=0,
                direction=1,
                sig_exit_signal_idx=1,
                sig_exit_exec_idx=1,
            ),
            StageACompactTradeV2(
                entry_signal_idx=2,
                entry_exec_idx=2,
                direction=1,
                sig_exit_signal_idx=None,
                sig_exit_exec_idx=4,
            ),
        ),
        exec_open=np.array([100.0, 110.0, 100.0, 120.0], dtype=np.float64),
        exec_close=np.array([105.0, 111.0, 110.0, 120.0], dtype=np.float64),
        sentinel_index=4,
        execution_params=ExecutionParamsV1(
            direction_mode="long-short",
            sizing_mode="all_in",
            init_cash_quote=1000.0,
            fixed_quote=100.0,
            safe_profit_percent=30.0,
            fee_pct=0.0,
            slippage_pct=0.0,
        ),
    )
    ranking_payload = no_risk_metrics_to_ranking_payload_v2(metrics=metrics)

    assert round(metrics.total_return_pct, 6) == 32.0
    assert metrics.max_drawdown_pct == 0.0
    assert math.isinf(metrics.return_over_max_drawdown)
    assert math.isinf(metrics.profit_factor)
    assert metrics.trade_count == 2
    assert metrics.win_rate_pct == 100.0
    assert round(metrics.avg_trade_ret_pct, 6) == 15.0
    assert metrics.avg_trade_exec_bars == 1.0
    assert metrics.exposure_pct == 50.0
    assert tuple(ranking_payload.keys()) == (
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "trade_count",
        "win_rate_pct",
        "avg_trade_ret_pct",
        "avg_trade_exec_bars",
        "exposure_pct",
    )


def test_build_compact_trade_list_v2_rejects_out_of_range_mapping_indexes() -> None:
    """
    Verify compact trade construction fails fast when local `bar_close_1m_idx` leaves bounds.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Rebased `bar_close_1m_idx` must stay inside the local execution window.
    Raises:
        AssertionError: If invalid local mapping indexes are accepted.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="bar_close_1m_idx values must stay within"):
        build_compact_trade_list_v2(
            final_signal=np.array([[1, 0]], dtype=np.int8),
            bar_close_1m_idx=np.array([0, 2], dtype=np.int64),
            sentinel_index=2,
        )
