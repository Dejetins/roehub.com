from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping

import numpy as np

from trading.contexts.backtest.application.dto import BacktestMetricRowV1

from .metrics_calculator_v1 import BACKTEST_METRIC_ORDER_V1, BacktestMetricValueV1

_PERCENT_PRECISION = 4
_RATIO_PRECISION = 5
_QUOTE_PRECISION = 2

_DATE_METRICS = {"Start", "End"}
_DURATION_METRICS = {
    "Duration",
    "Max. Drawdown Duration",
    "Avg. Drawdown Duration",
    "Max. Trade Duration",
    "Avg. Trade Duration",
}
_INTEGER_METRICS = {"Num. Trades"}
_QUOTE_METRICS = {"Init. Cash", "Total Profit"}
_PERCENT_METRICS = {
    "Total Return [%]",
    "Benchmark Return [%]",
    "Position Coverage [%]",
    "Max. Drawdown [%]",
    "Avg. Drawdown [%]",
    "Win Rate [%]",
    "Best Trade [%]",
    "Worst Trade [%]",
    "Avg. Trade [%]",
    "Expectancy",
    "Gross Exposure",
}
_RATIO_METRICS = {"SQN", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"}


class BacktestMetricsTableFormatterV1:
    """
    Format deterministic EPIC-06 metrics into stable `rows` and markdown table payload.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/metrics_calculator_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/services/reporting_service_v1.py
    """

    def rows_from_metrics(
        self,
        *,
        metrics: Mapping[str, BacktestMetricValueV1],
    ) -> tuple[BacktestMetricRowV1, ...]:
        """
        Build fixed-order metric rows with deterministic string formatting per metric kind.

        Args:
            metrics: Raw metric mapping produced by metrics calculator.
        Returns:
            tuple[BacktestMetricRowV1, ...]: Deterministic fixed-order table rows.
        Assumptions:
            Mapping contains all required keys from `BACKTEST_METRIC_ORDER_V1`.
        Raises:
            ValueError: If required metric is missing or has incompatible value type.
        Side Effects:
            None.
        """
        rows: list[BacktestMetricRowV1] = []
        for metric_name in BACKTEST_METRIC_ORDER_V1:
            if metric_name not in metrics:
                raise ValueError(f"missing required metric '{metric_name}'")
            rows.append(
                BacktestMetricRowV1(
                    metric=metric_name,
                    value=_format_metric_value(metric=metric_name, value=metrics[metric_name]),
                )
            )
        return tuple(rows)

    def markdown_table(self, *, rows: tuple[BacktestMetricRowV1, ...]) -> str:
        """
        Build canonical markdown representation from deterministic metric rows.

        Args:
            rows: Deterministic metric rows in fixed order.
        Returns:
            str: Canonical markdown table string with `|Metric|Value|` header.
        Assumptions:
            Row values are already formatted as stable strings.
        Raises:
            None.
        Side Effects:
            None.
        """
        lines = ["|Metric|Value|", "|---|---|"]
        for row in rows:
            lines.append(f"|{row.metric}|{row.value}|")
        return "\n".join(lines)


def _format_metric_value(*, metric: str, value: BacktestMetricValueV1) -> str:
    """
    Format one metric value into deterministic string representation by metric category.

    Args:
        metric: Metric name literal.
        value: Raw metric value.
    Returns:
        str: Deterministic formatted value string or `N/A`.
    Assumptions:
        `metric` belongs to fixed EPIC-06 metrics list.
    Raises:
        ValueError: If value type is incompatible with metric category.
    Side Effects:
        None.
    """
    if value is None:
        return "N/A"

    if metric in _DATE_METRICS:
        if not isinstance(value, datetime):
            raise ValueError(f"metric '{metric}' must be datetime")
        return value.astimezone(timezone.utc).isoformat(sep=" ", timespec="milliseconds")

    if metric in _DURATION_METRICS:
        if not isinstance(value, timedelta):
            raise ValueError(f"metric '{metric}' must be timedelta")
        return str(value)

    if metric in _INTEGER_METRICS:
        return _format_int(value=value, metric=metric)
    if metric in _QUOTE_METRICS:
        return _format_float(value=value, metric=metric, precision=_QUOTE_PRECISION)
    if metric in _RATIO_METRICS:
        return _format_float(value=value, metric=metric, precision=_RATIO_PRECISION)
    if metric in _PERCENT_METRICS:
        return _format_float(value=value, metric=metric, precision=_PERCENT_PRECISION)
    return _format_float(value=value, metric=metric, precision=_PERCENT_PRECISION)


def _format_int(*, value: BacktestMetricValueV1, metric: str) -> str:
    """
    Format integer metric without fractional suffix while preserving deterministic checks.

    Args:
        value: Raw numeric value.
        metric: Metric name for deterministic errors.
    Returns:
        str: Integer literal without `.0` suffix.
    Assumptions:
        Integer metrics are finite and integral.
    Raises:
        ValueError: If value is non-numeric or not integral.
    Side Effects:
        None.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"metric '{metric}' must be numeric")
    numeric = float(value)
    if not np.isfinite(numeric):
        return "N/A"
    rounded = int(round(numeric))
    if abs(numeric - rounded) > 1e-9:
        raise ValueError(f"metric '{metric}' must be integral")
    return str(rounded)


def _format_float(*, value: BacktestMetricValueV1, metric: str, precision: int) -> str:
    """
    Format floating metric using fixed precision and trailing-zero trimming rules.

    Args:
        value: Raw numeric value.
        metric: Metric name for deterministic errors.
        precision: Decimal precision to apply before trimming.
    Returns:
        str: Deterministic decimal string or `N/A` when value is non-finite.
    Assumptions:
        Precision is non-negative integer and metric expects decimal formatting.
    Raises:
        ValueError: If value is not numeric.
    Side Effects:
        None.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"metric '{metric}' must be numeric")
    numeric = float(value)
    if not np.isfinite(numeric):
        return "N/A"

    rounded = round(numeric, precision)
    if abs(rounded) < (10.0 ** (-precision)):
        rounded = 0.0

    text = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
    if text in {"-0", "-0.0", ""}:
        return "0"
    return text


__all__ = [
    "BacktestMetricsTableFormatterV1",
]
