from __future__ import annotations

from dataclasses import dataclass

from trading.contexts.backtest.application.dto import BacktestReportV1
from trading.contexts.backtest.application.ports import BacktestVariantScoreDetailsV1
from trading.contexts.backtest.domain.entities import ExecutionOutcomeV1
from trading.contexts.backtest.domain.value_objects import ExecutionParamsV1
from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import TimeRange

from .equity_curve_builder_v1 import BacktestEquityCurveBuilderV1
from .metrics_calculator_v1 import BacktestMetricsCalculatorV1
from .table_formatter_v1 import BacktestMetricsTableFormatterV1


@dataclass(frozen=True, slots=True)
class BacktestReportingServiceV1:
    """
    Orchestrate deterministic EPIC-06 report assembly for one Stage-B variant outcome.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/equity_curve_builder_v1.py
      - src/trading/contexts/backtest/application/services/metrics_calculator_v1.py
      - src/trading/contexts/backtest/application/services/table_formatter_v1.py
    """

    equity_curve_builder: BacktestEquityCurveBuilderV1 = BacktestEquityCurveBuilderV1()
    metrics_calculator: BacktestMetricsCalculatorV1 = BacktestMetricsCalculatorV1()
    table_formatter: BacktestMetricsTableFormatterV1 = BacktestMetricsTableFormatterV1()

    def build_report(
        self,
        *,
        requested_time_range: TimeRange,
        candles: CandleArrays,
        target_slice: slice,
        execution_params: ExecutionParamsV1,
        execution_outcome: ExecutionOutcomeV1,
        include_table_md: bool = True,
        include_trades: bool = False,
    ) -> BacktestReportV1:
        """
        Build deterministic report payload from engine outcome and candle timeline inputs.

        Args:
            requested_time_range: User request range for Start/End/Duration metrics.
            candles: Warmup-inclusive candle arrays.
            target_slice: Stage-B target bars for deterministic reporting calculations.
            execution_params: Execution settings used by engine run.
            execution_outcome: Engine output with closed trades and final equity summary.
            include_table_md: Whether to include markdown `|Metric|Value|` representation.
            include_trades: Whether to include full trade list in response payload.
        Returns:
            BacktestReportV1: Deterministic report rows with optional markdown/trades payload.
        Assumptions:
            Reporting calculations are performed only on `target_slice` bars.
        Raises:
            ValueError: Propagated from builder/calculator/formatter invariants.
        Side Effects:
            None.
        """
        ordered_trades = tuple(
            sorted(execution_outcome.trades, key=lambda item: (item.trade_id, item.entry_bar_index))
        )
        equity_curve = self.equity_curve_builder.build(
            candles=candles,
            target_slice=target_slice,
            trades=ordered_trades,
            execution_params=execution_params,
        )
        metrics = self.metrics_calculator.calculate(
            requested_time_range=requested_time_range,
            candles=candles,
            target_slice=target_slice,
            execution_params=execution_params,
            trades=ordered_trades,
            equity_curve=equity_curve,
        )
        rows = self.table_formatter.rows_from_metrics(metrics=metrics)
        table_md = self.table_formatter.markdown_table(rows=rows) if include_table_md else None
        trades_payload = ordered_trades if include_trades else None
        return BacktestReportV1(
            rows=rows,
            table_md=table_md,
            trades=trades_payload,
        )

    def build_report_from_details(
        self,
        *,
        requested_time_range: TimeRange,
        candles: CandleArrays,
        details: BacktestVariantScoreDetailsV1,
        include_table_md: bool = True,
        include_trades: bool = False,
    ) -> BacktestReportV1:
        """
        Build report payload from Stage-B details scorer output for one explicit variant.

        Docs:
          - docs/architecture/backtest/
            backtest-staged-ranking-reporting-perf-optimization-plan-v1.md
          - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
        Related:
          - src/trading/contexts/backtest/application/ports/staged_runner.py
          - src/trading/contexts/backtest/application/use_cases/run_backtest.py
          - apps/api/routes/backtests.py

        Args:
            requested_time_range: User request range for Start/End/Duration metrics.
            candles: Warmup-inclusive candle arrays.
            details: Deterministic Stage-B details payload from scorer.
            include_table_md: Whether to include markdown `|Metric|Value|` representation.
            include_trades: Whether to include full trade list in response payload.
        Returns:
            BacktestReportV1: Deterministic report rows with optional markdown/trades payload.
        Assumptions:
            Details payload corresponds to same candle timeline as `candles`.
        Raises:
            ValueError: Propagated from report builders and details invariants.
        Side Effects:
            None.
        """
        return self.build_report(
            requested_time_range=requested_time_range,
            candles=candles,
            target_slice=details.target_slice,
            execution_params=details.execution_params,
            execution_outcome=details.execution_outcome,
            include_table_md=include_table_md,
            include_trades=include_trades,
        )


__all__ = [
    "BacktestReportingServiceV1",
]
