"""Compose bounded execution reads and canonical market candles for one owned strategy."""

from datetime import UTC, datetime, timedelta

from apps.api.dto.strategy_operations import StrategyOperationsResponse
from apps.api.dto.ui_strategies_dashboard import StrategyChartCandleResponse, StrategyChartResponse
from apps.api.wiring.modules.strategy_execution_projection import project_fills


class StrategyOperationReads:
    def __init__(self, reader, candle_reader=None):
        self.reader = reader
        self.candle_reader = candle_reader

    def read(self, *, organization_id, user_id, strategy, run, profile):
        operations = StrategyOperationsResponse(reason="no_run")
        chart = StrategyChartResponse(
            source="canonical_candles",
            state="unavailable",
            symbol=str(strategy.spec.instrument_id.symbol),
            range="1m",
            max_points=500,
            candles=[],
            markers=[],
        )
        if run is None:
            return operations, chart
        try:
            rows = self.reader.read(
                organization_id=organization_id,
                owner_user_id=user_id,
                strategy_id=strategy.strategy_id,
                run_id=run.run_id,
                paper=profile.mode == "paper",
            )
            facts = []
            for raw in rows[:5000]:
                row = dict(raw)
                explicit = row.get("exit_reason") or row.get("signal_reason")
                row["reason"] = (
                    explicit
                    if explicit in {"stop_loss", "take_profit", "trailing_stop"}
                    else "manual"
                    if row.get("source_type") == "manual_request"
                    else "signal"
                    if row.get("source_type") == "strategy_signal"
                    else "unknown_reason"
                )
                # Costs in another asset cannot be subtracted from quote P&L without conversion.
                symbol = str(strategy.spec.instrument_id.symbol)
                if not row.get("fee_asset") or not symbol.endswith(str(row["fee_asset"])):
                    row["fee"] = None
                row["costs_complete"] = strategy.spec.market_type == "spot"
                facts.append(row)
            operations = project_fills(
                facts,
                initial_cash=float(rows[0]["initial_cash"])
                if rows and rows[0].get("initial_cash") is not None
                else None,
                partial=len(rows) > 5000,
            )
        except Exception:
            operations = StrategyOperationsResponse(reason="execution_read_unavailable")
        if self.candle_reader is not None:
            try:
                from trading.shared_kernel.primitives import TimeRange, UtcTimestamp

                end = datetime.now(UTC).replace(second=0, microsecond=0)
                start = end - timedelta(days=3)
                candles = list(
                    self.candle_reader.read_1m(
                        strategy.spec.instrument_id,
                        TimeRange(start=UtcTimestamp(start), end=UtcTimestamp(end)),
                    )
                )
                # Only complete consecutive bars, never manufactured OHLC from fills.
                seconds = int(strategy.spec.timeframe.duration().total_seconds())
                buckets = {}
                for item in candles:
                    c = item.candle
                    bucket = int(c.ts_open.value.timestamp()) // seconds * seconds
                    buckets.setdefault(bucket, []).append(c)
                for ts, values in sorted(buckets.items()):
                    if len(values) != seconds // 60:
                        continue
                    chart.candles.append(
                        StrategyChartCandleResponse(
                            timestamp=datetime.fromtimestamp(ts, UTC),
                            open=values[0].open,
                            close=values[-1].close,
                            high=max(c.high for c in values),
                            low=min(c.low for c in values),
                        )
                    )
                chart.candles = chart.candles[-500:]
                chart.state = "ready" if chart.candles else "empty"
                if chart.candles:
                    operations.current_price = chart.candles[-1].close
                    operations.price_at = chart.candles[-1].timestamp
            except Exception:
                chart.state = "unavailable"
        return operations, chart
