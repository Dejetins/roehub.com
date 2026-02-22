from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from trading.contexts.backtest.application.dto import BacktestReportV1
from trading.contexts.backtest.application.services import (
    BacktestExecutionEngineV1,
    BacktestReportingServiceV1,
)
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1, RiskParamsV1
from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import (
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)

_GOLDEN_DIR = Path(__file__).resolve().parents[2] / "golden"
_EPOCH_UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_backtest_reporting_table_md_matches_golden_no_trades() -> None:
    """
    Verify no-trades scenario markdown table is byte-stable against golden fixture.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Golden fixture was generated from deterministic engine/reporting contracts only.
    Raises:
        AssertionError: If formatter output diverges from approved golden fixture.
    Side Effects:
        None.
    """
    report = _build_report_for_scenario(
        closes=(100.0, 102.0, 101.0, 104.0, 103.0),
        signal_values=("NEUTRAL", "NEUTRAL", "NEUTRAL", "NEUTRAL", "NEUTRAL"),
    )
    assert report.table_md is not None
    assert _canonical_table(report.table_md) == _read_golden("no-trades.md")


def test_backtest_reporting_table_md_matches_golden_multi_trade() -> None:
    """
    Verify deterministic multi-trade scenario markdown table matches golden fixture 1:1.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Scenario is constructed to produce multiple closed trades without indicator compute.
    Raises:
        AssertionError: If report table differs from golden fixture.
    Side Effects:
        None.
    """
    report = _build_report_for_scenario(
        closes=(100.0, 110.0, 120.0, 130.0, 90.0, 80.0),
        signal_values=("LONG", "NEUTRAL", "SHORT", "NEUTRAL", "LONG", "NEUTRAL"),
    )
    assert report.table_md is not None
    assert _canonical_table(report.table_md) == _read_golden("multi-trade.md")


def test_backtest_reporting_table_md_exposes_expected_numeric_rows_for_multi_trade() -> None:
    """
    Verify key deterministic numeric rows are rendered as expected for multi-trade golden input.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Metric formatting follows EPIC-06 fixed precision and ordering contracts.
    Raises:
        AssertionError: If key row values drift from expected canonical literals.
    Side Effects:
        None.
    """
    report = _build_report_for_scenario(
        closes=(100.0, 110.0, 120.0, 130.0, 90.0, 80.0),
        signal_values=("LONG", "NEUTRAL", "SHORT", "NEUTRAL", "LONG", "NEUTRAL"),
    )
    metrics_by_name = {row.metric: row.value for row in report.rows}

    assert metrics_by_name["Total Return [%]"] == "-10.3704"
    assert metrics_by_name["Num. Trades"] == "3"
    assert report.table_md is not None
    assert "|Total Return [%]|-10.3704|" in report.table_md
    assert "|Num. Trades|3|" in report.table_md


def _build_report_for_scenario(
    *,
    closes: tuple[float, ...],
    signal_values: tuple[str, ...],
) -> BacktestReportV1:
    """
    Run deterministic engine and reporting services for one explicit close/signal scenario.

    Args:
        closes: Fixed close-price sequence on daily bars.
        signal_values: Precomputed final-signal sequence (`LONG|SHORT|NEUTRAL`).
    Returns:
        BacktestReportV1: Full reporting payload with markdown table enabled.
    Assumptions:
        Input signal vector length equals candles bars count.
    Raises:
        ValueError: If one scenario invariant is invalid.
    Side Effects:
        None.
    """
    if len(closes) != len(signal_values):
        raise ValueError("signal_values length must match closes length")

    candles = _build_daily_candles(closes=closes)
    signals = np.asarray(signal_values, dtype="U7")
    target_slice = slice(0, len(closes))

    engine = BacktestExecutionEngineV1()
    reporting = BacktestReportingServiceV1()
    execution_params = _execution_params()
    outcome = engine.run(
        candles=candles,
        target_slice=target_slice,
        final_signal=signals,
        execution_params=execution_params,
        risk_params=RiskParamsV1(
            sl_enabled=False,
            sl_pct=None,
            tp_enabled=False,
            tp_pct=None,
        ),
    )
    return reporting.build_report(
        requested_time_range=candles.time_range,
        candles=candles,
        target_slice=target_slice,
        execution_params=execution_params,
        execution_outcome=outcome,
        include_table_md=True,
        include_trades=True,
    )


def _execution_params() -> ExecutionParamsV1:
    """
    Build deterministic execution params fixture for golden reporting scenarios.

    Args:
        None.
    Returns:
        ExecutionParamsV1: Stable execution settings for scenario replay.
    Assumptions:
        Fee/slippage are disabled to keep golden values easy to reason about.
    Raises:
        ValueError: If execution settings violate value-object invariants.
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


def _build_daily_candles(*, closes: tuple[float, ...]) -> CandleArrays:
    """
    Build deterministic daily `CandleArrays` used by golden-table scenarios.

    Args:
        closes: Ordered daily close values.
    Returns:
        CandleArrays: Dense finite daily candles fixture.
    Assumptions:
        Open/high/low equal close values for deterministic fixture simplicity.
    Raises:
        ValueError: If generated arrays violate candle DTO invariants.
    Side Effects:
        Allocates numpy arrays.
    """
    bars = len(closes)
    day_ms = int(timedelta(days=1) // timedelta(milliseconds=1))
    ts_open = np.arange(bars, dtype=np.int64) * np.int64(day_ms)
    close_array = np.asarray(closes, dtype=np.float32)
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(
            start=UtcTimestamp(_EPOCH_UTC),
            end=UtcTimestamp(_EPOCH_UTC + timedelta(days=bars)),
        ),
        timeframe=Timeframe("1d"),
        ts_open=ts_open,
        open=close_array.copy(),
        high=close_array.copy(),
        low=close_array.copy(),
        close=close_array,
        volume=np.ones(bars, dtype=np.float32),
    )


def _read_golden(filename: str) -> str:
    """
    Read one golden fixture file and normalize newlines to deterministic `\\n`.

    Args:
        filename: Golden fixture filename under backtest golden directory.
    Returns:
        str: Canonicalized golden markdown table.
    Assumptions:
        Golden fixtures are ASCII markdown files committed in repository.
    Raises:
        FileNotFoundError: If requested fixture file does not exist.
    Side Effects:
        Reads file from repository.
    """
    return _canonical_table((_GOLDEN_DIR / filename).read_text(encoding="ascii"))


def _canonical_table(value: str) -> str:
    """
    Convert markdown table text into canonical representation for cross-platform comparison.

    Args:
        value: Raw markdown table text.
    Returns:
        str: Canonical table string with normalized line endings and no trailing newline.
    Assumptions:
        Table text is UTF-8/ASCII compatible and does not require locale-specific handling.
    Raises:
        None.
    Side Effects:
        None.
    """
    return value.replace("\r\n", "\n").rstrip("\n")
