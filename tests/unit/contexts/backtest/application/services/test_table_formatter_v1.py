from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading.contexts.backtest.application.services import (
    BACKTEST_METRIC_ORDER_V1,
    BacktestMetricsTableFormatterV1,
)
from trading.contexts.backtest.application.services.metrics_calculator_v1 import (
    BacktestMetricValueV1,
)


def test_backtest_metrics_table_formatter_v1_formats_values_deterministically() -> None:
    """
    Verify formatter keeps fixed metric ordering and deterministic numeric/date string formats.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Formatter receives one complete raw metric payload.
    Raises:
        AssertionError: If ordering, trimming, precision, or markdown output differs.
    Side Effects:
        None.
    """
    formatter = BacktestMetricsTableFormatterV1()
    rows = formatter.rows_from_metrics(metrics=_metric_payload())

    assert tuple(row.metric for row in rows) == BACKTEST_METRIC_ORDER_V1
    assert rows[0].value == "2026-02-16 00:00:00.000+00:00"
    assert rows[1].value == "2026-02-18 00:00:00.000+00:00"
    assert rows[2].value == "2 days, 0:00:00"
    assert _row_value(rows=rows, metric="Init. Cash") == "10000"
    assert _row_value(rows=rows, metric="Total Profit") == "12.34"
    assert _row_value(rows=rows, metric="Total Return [%]") == "1.2346"
    assert _row_value(rows=rows, metric="Benchmark Return [%]") == "0"
    assert _row_value(rows=rows, metric="Sharpe Ratio") == "1.23457"
    assert _row_value(rows=rows, metric="SQN") == "N/A"

    table_md = formatter.markdown_table(rows=rows)
    assert table_md.startswith("|Metric|Value|")
    assert "|Total Profit|12.34|" in table_md


def _row_value(*, rows: tuple[object, ...], metric: str) -> str:
    """
    Return one formatted value from rows by metric literal.

    Args:
        rows: Reporting rows tuple.
        metric: Metric literal to find.
    Returns:
        str: Formatted metric value.
    Assumptions:
        Requested metric exists in rows tuple.
    Raises:
        KeyError: If metric is absent.
    Side Effects:
        None.
    """
    for row in rows:
        if getattr(row, "metric") == metric:
            return str(getattr(row, "value"))
    raise KeyError(metric)


def _metric_payload() -> dict[str, BacktestMetricValueV1]:
    """
    Build complete raw metric mapping fixture for deterministic formatter tests.

    Args:
        None.
    Returns:
        dict[str, BacktestMetricValueV1]: Full raw metric payload.
    Assumptions:
        Missing metrics default to `None` and should render as `N/A`.
    Raises:
        None.
    Side Effects:
        None.
    """
    start = datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 2, 18, 0, 0, tzinfo=timezone.utc)
    payload: dict[str, BacktestMetricValueV1] = {name: None for name in BACKTEST_METRIC_ORDER_V1}
    payload.update(
        {
            "Start": start,
            "End": end,
            "Duration": timedelta(days=2),
            "Init. Cash": 10000.0,
            "Total Profit": 12.34,
            "Total Return [%]": 1.23456,
            "Benchmark Return [%]": 0.0,
            "Position Coverage [%]": 50.0,
            "Max. Drawdown [%]": 10.0,
            "Avg. Drawdown [%]": 2.5,
            "Max. Drawdown Duration": timedelta(hours=5),
            "Avg. Drawdown Duration": timedelta(hours=2, minutes=30),
            "Num. Trades": 4,
            "Win Rate [%]": 50.0,
            "Best Trade [%]": 1.5,
            "Worst Trade [%]": -2.0,
            "Avg. Trade [%]": 0.25,
            "Max. Trade Duration": timedelta(hours=6),
            "Avg. Trade Duration": timedelta(hours=3),
            "Expectancy": 0.125,
            "SQN": None,
            "Gross Exposure": 0.42,
            "Sharpe Ratio": 1.234567,
            "Sortino Ratio": 2.345678,
            "Calmar Ratio": 3.456789,
        }
    )
    return payload
