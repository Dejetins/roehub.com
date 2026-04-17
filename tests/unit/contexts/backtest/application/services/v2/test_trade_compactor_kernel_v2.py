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
from trading.contexts.backtest.application.services.v2.signal_aggregator_kernel import (
    aggregate_signal_pairs_v2,
)
from trading.contexts.backtest.application.services.v2.trade_compactor_kernel import (
    build_compact_exact_payloads_v2,
    build_compact_trade_batch_for_signal_pairs_v2,
    build_compact_trade_batch_v2,
    compute_no_risk_metrics_for_trade_batch_v2,
)
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


def test_build_compact_exact_payloads_v2_wraps_internal_compact_trade_representation() -> None:
    """
    Verify retained-candidate exact payloads keep compact-trade arrays without signal-row baggage.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The compactor kernel should expose an internal payload container while preserving the
        existing compact trade representation as its only payload body.
    Raises:
        AssertionError: If the payload wrapper drifts from the underlying compact trade rows.
    Side Effects:
        None.
    """
    compact_trades = build_compact_trade_list_v2(
        final_signal=np.array([[1, 1, -1, 0]], dtype=np.int8),
        bar_close_1m_idx=np.array([0, 1, 2, 3], dtype=np.int64),
        sentinel_index=5,
    )
    payloads = build_compact_exact_payloads_v2(
        final_signal=np.array([[1, 1, -1, 0]], dtype=np.int8),
        bar_close_1m_idx=np.array([0, 1, 2, 3], dtype=np.int64),
        sentinel_index=5,
    )

    assert len(payloads) == 1
    payload = payloads[0]

    np.testing.assert_array_equal(payload.entry_signal_idx, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(payload.entry_exec_idx, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(payload.direction, np.array([1, -1], dtype=np.int8))
    np.testing.assert_array_equal(payload.sig_exit_signal_idx, np.array([2, -1], dtype=np.int64))
    np.testing.assert_array_equal(payload.sig_exit_exec_idx, np.array([3, 5], dtype=np.int64))
    assert payload.trade_count == 2
    assert payload.memory_shape_bucket == "compact_trade_arrays"
    assert payload.entry_signal_idx.flags.writeable is False
    assert payload.entry_exec_idx.flags.writeable is False
    assert payload.direction.flags.writeable is False
    assert payload.sig_exit_signal_idx.flags.writeable is False
    assert payload.sig_exit_exec_idx.flags.writeable is False
    assert hasattr(payload, "final_signal_row") is False
    assert payload.compact_trades == compact_trades[0]


def test_build_compact_trade_batch_v2_bounds_internal_width_by_actual_trade_count() -> None:
    """
    Verify retained compact-trade batch storage width follows `max_trade_count`, not `signal_count`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Repeated same-direction confirmations should collapse into one open trade, so retained
        batch storage can stay narrower than the full signal timeline.
    Raises:
        AssertionError: If bounded allocation regresses back to dense `[V, T_signal]` storage.
    Side Effects:
        None.
    """
    final_signal = np.array(
        [[1, 1, 1, -1, -1, 0, 0], [0, 1, 1, 1, 1, 1, 0]],
        dtype=np.int8,
    )
    batch = build_compact_trade_batch_v2(
        final_signal=final_signal,
        bar_close_1m_idx=np.arange(final_signal.shape[1], dtype=np.int64),
        sentinel_index=8,
    )

    assert tuple(int(value) for value in batch.trade_count) == (2, 1)
    assert batch.max_trade_count == 2
    assert batch.max_trade_count < int(final_signal.shape[1])
    assert batch.entry_signal_idx.shape == (2, 2)
    assert batch.entry_exec_idx.shape == (2, 2)
    assert batch.direction.shape == (2, 2)
    assert batch.sig_exit_signal_idx.shape == (2, 2)
    assert batch.sig_exit_exec_idx.shape == (2, 2)
    assert batch.exact_payload_at(row_index=1).trade_count == 1


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
    assert metrics.sharpe_trades > 0.0
    assert metrics.win_rate_pct == 100.0
    assert round(metrics.avg_trade_ret_pct, 6) == 15.0
    assert metrics.avg_trade_exec_bars == 1.0
    assert metrics.exposure_pct == 50.0
    assert tuple(ranking_payload.keys()) == (
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "profit_factor",
        "sharpe_trades",
        "win_rate_pct",
        "trade_count",
        "avg_trade_ret_pct",
        "avg_trade_exec_bars",
        "exposure_pct",
    )


def test_compute_no_risk_metrics_for_trade_batch_v2_matches_scalar_rows() -> None:
    """
    Verify batched retained exact scoring matches the scalar no-risk metric contract row-by-row.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Dense trade-list-first batch scoring must preserve the scalar metric semantics used by the
        deterministic shortlist.
    Raises:
        AssertionError: If one batched metric drifts from the scalar row result.
    Side Effects:
        None.
    """
    execution_params = ExecutionParamsV1(
        direction_mode="long-short",
        sizing_mode="all_in",
        init_cash_quote=1000.0,
        fixed_quote=100.0,
        safe_profit_percent=30.0,
        fee_pct=0.0,
        slippage_pct=0.0,
    )
    batch = build_compact_trade_batch_v2(
        final_signal=np.array([[1, 1, -1, 0], [0, -1, -1, 1]], dtype=np.int8),
        bar_close_1m_idx=np.array([0, 1, 2, 3], dtype=np.int64),
        sentinel_index=5,
    )

    batch_metrics = compute_no_risk_metrics_for_trade_batch_v2(
        compact_trade_batch=batch,
        exec_open=np.array([100.0, 110.0, 120.0, 105.0, 115.0], dtype=np.float64),
        exec_close=np.array([102.0, 112.0, 118.0, 108.0, 117.0], dtype=np.float64),
        sentinel_index=5,
        execution_params=execution_params,
    )
    scalar_metrics = tuple(
        compute_no_risk_metrics_v2(
            compact_trades=batch.exact_payload_at(row_index=row_index).compact_trades,
            exec_open=np.array([100.0, 110.0, 120.0, 105.0, 115.0], dtype=np.float64),
            exec_close=np.array([102.0, 112.0, 118.0, 108.0, 117.0], dtype=np.float64),
            sentinel_index=5,
            execution_params=execution_params,
        )
        for row_index in range(2)
    )

    payload = batch.exact_payload_at(row_index=0)
    assert payload.trade_count == 2
    assert payload.memory_shape_bucket == "compact_trade_arrays"
    np.testing.assert_array_equal(payload.entry_signal_idx, np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(payload.entry_exec_idx, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(payload.direction, np.array([1, -1], dtype=np.int8))
    np.testing.assert_array_equal(payload.sig_exit_signal_idx, np.array([2, -1], dtype=np.int64))
    np.testing.assert_array_equal(payload.sig_exit_exec_idx, np.array([3, 5], dtype=np.int64))
    assert payload.compact_trades == build_compact_exact_payloads_v2(
        final_signal=np.array([[1, 1, -1, 0]], dtype=np.int8),
        bar_close_1m_idx=np.array([0, 1, 2, 3], dtype=np.int64),
        sentinel_index=5,
    )[0].compact_trades
    for batched, scalar in zip(batch_metrics, scalar_metrics, strict=True):
        assert batched.total_return_pct == pytest.approx(scalar.total_return_pct)
        assert batched.max_drawdown_pct == pytest.approx(scalar.max_drawdown_pct)
        assert batched.return_over_max_drawdown == pytest.approx(
            scalar.return_over_max_drawdown
        )
        assert batched.profit_factor == pytest.approx(scalar.profit_factor)
        assert batched.trade_count == scalar.trade_count
        assert batched.sharpe_trades == pytest.approx(scalar.sharpe_trades)
        assert batched.win_rate_pct == pytest.approx(scalar.win_rate_pct)
        assert batched.avg_trade_ret_pct == pytest.approx(scalar.avg_trade_ret_pct)
        assert batched.avg_trade_exec_bars == pytest.approx(scalar.avg_trade_exec_bars)
        assert batched.exposure_pct == pytest.approx(scalar.exposure_pct)


def test_build_compact_trade_batch_for_signal_pairs_v2_matches_generic_batch_shape() -> None:
    """
    Verify pair-first compact batching matches generic batching for two-indicator consensus rows.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        D4 pair-first compaction must preserve exact trade ordering and payload semantics while
        skipping the broad retained `final_signal[V, T_signal]` batch on the parity path.
    Raises:
        AssertionError: If pair-first batching drifts from the generic compact-trade contract.
    Side Effects:
        None.
    """
    left_signal_rows = np.array([[1, 1, -1, 0], [1, 0, -1, 1]], dtype=np.int8)
    right_signal_rows = np.array([[1, 1, -1, 0], [1, -1, -1, 1]], dtype=np.int8)
    final_signal = aggregate_signal_pairs_v2(
        left_signal_rows=left_signal_rows,
        right_signal_rows=right_signal_rows,
        indicator_ids=("ma.dema", "ma.hma"),
    )

    generic_batch = build_compact_trade_batch_v2(
        final_signal=final_signal,
        bar_close_1m_idx=np.array([0, 1, 2, 3], dtype=np.int64),
        sentinel_index=5,
    )
    pair_batch = build_compact_trade_batch_for_signal_pairs_v2(
        left_signal_rows=left_signal_rows,
        right_signal_rows=right_signal_rows,
        bar_close_1m_idx=np.array([0, 1, 2, 3], dtype=np.int64),
        sentinel_index=5,
    )

    np.testing.assert_array_equal(pair_batch.trade_count, generic_batch.trade_count)
    np.testing.assert_array_equal(pair_batch.entry_signal_idx, generic_batch.entry_signal_idx)
    np.testing.assert_array_equal(pair_batch.entry_exec_idx, generic_batch.entry_exec_idx)
    np.testing.assert_array_equal(pair_batch.direction, generic_batch.direction)
    np.testing.assert_array_equal(
        pair_batch.sig_exit_signal_idx,
        generic_batch.sig_exit_signal_idx,
    )
    np.testing.assert_array_equal(
        pair_batch.sig_exit_exec_idx,
        generic_batch.sig_exit_exec_idx,
    )
    assert pair_batch.max_trade_count == generic_batch.max_trade_count
    assert tuple(
        pair_batch.exact_payload_at(row_index=row_index).compact_trades
        for row_index in range(int(pair_batch.trade_count.shape[0]))
    ) == tuple(
        generic_batch.exact_payload_at(row_index=row_index).compact_trades
        for row_index in range(int(generic_batch.trade_count.shape[0]))
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
