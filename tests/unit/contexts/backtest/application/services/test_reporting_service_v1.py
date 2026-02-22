from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from trading.contexts.backtest.application.services import (
    BACKTEST_METRIC_ORDER_V1,
    BacktestReportingServiceV1,
)
from trading.contexts.backtest.domain.entities import ExecutionOutcomeV1
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1
from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import (
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)


def test_backtest_reporting_service_v1_builds_ordered_rows_and_markdown_table() -> None:
    """
    Verify reporting service assembles deterministic rows and canonical markdown table header.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        No-trades outcome is valid and still produces full metric table payload.
    Raises:
        AssertionError: If row ordering or markdown header contract is violated.
    Side Effects:
        None.
    """
    candles = _candles_from_closes(closes=(100.0, 102.0, 101.0))
    service = BacktestReportingServiceV1()
    report = service.build_report(
        requested_time_range=candles.time_range,
        candles=candles,
        target_slice=slice(0, 3),
        execution_params=_execution_params(),
        execution_outcome=ExecutionOutcomeV1(
            trades=(),
            equity_end_quote=1000.0,
            available_quote=1000.0,
            safe_quote=0.0,
            total_return_pct=0.0,
        ),
        include_table_md=True,
        include_trades=True,
    )

    assert tuple(row.metric for row in report.rows) == BACKTEST_METRIC_ORDER_V1
    assert report.table_md is not None
    assert report.table_md.startswith("|Metric|Value|")
    assert report.trades == ()
    num_trades_row = next(row for row in report.rows if row.metric == "Num. Trades")
    assert num_trades_row.value == "0"


def _execution_params() -> ExecutionParamsV1:
    """
    Build deterministic execution parameters for reporting-service integration test.

    Args:
        None.
    Returns:
        ExecutionParamsV1: Immutable execution settings fixture.
    Assumptions:
        Zero fee/slippage keeps integration fixture simple and deterministic.
    Raises:
        ValueError: If execution settings violate domain invariants.
    Side Effects:
        None.
    """
    return ExecutionParamsV1(
        direction_mode="long-short",
        sizing_mode="all_in",
        init_cash_quote=1000.0,
        fixed_quote=100.0,
        safe_profit_percent=30.0,
        fee_pct=0.0,
        slippage_pct=0.0,
    )


def _candles_from_closes(*, closes: tuple[float, ...]) -> CandleArrays:
    """
    Build deterministic `1m` candle arrays fixture from close-price tuple.

    Args:
        closes: Ordered close-price values.
    Returns:
        CandleArrays: Dense finite candle arrays fixture.
    Assumptions:
        Open/high/low equal close values in this integration test fixture.
    Raises:
        ValueError: If generated primitive values violate invariants.
    Side Effects:
        Allocates numpy arrays for fixture construction.
    """
    bars = len(closes)
    ts_open = np.arange(bars, dtype=np.int64) * np.int64(60_000)
    close_array = np.asarray(closes, dtype=np.float32)
    start = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)
    end = start + (timedelta(minutes=bars))
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(start),
            end=UtcTimestamp(end),
        ),
        timeframe=Timeframe("1m"),
        ts_open=ts_open,
        open=close_array.copy(),
        high=close_array.copy(),
        low=close_array.copy(),
        close=close_array,
        volume=np.ones(bars, dtype=np.float32),
    )
