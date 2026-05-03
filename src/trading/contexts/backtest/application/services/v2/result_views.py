from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

from trading.contexts.backtest.application.dto import (
    BacktestJobReadModel,
    BacktestJobTopVariantReadModel,
    BacktestLazyTradesDetailReadModel,
    BacktestResultMonthlyStatsReadModel,
    BacktestResultSeriesReadModel,
    BacktestResultSummaryReadModel,
    BacktestResultSymbolStatsReadModel,
    BacktestResultTradesPageReadModel,
    BacktestTradesCsvReadModel,
)
from trading.contexts.backtest.domain.entities import BacktestJob

DEFAULT_RESULT_CHART_POINTS = 1200
MAX_RESULT_CHART_POINTS = 1500
DEFAULT_RESULT_TRADES_PAGE_SIZE = 50
MAX_RESULT_TRADES_PAGE_SIZE = 100

_CSV_FIELDS = (
    "trade_index",
    "side",
    "entry_timestamp",
    "exit_timestamp",
    "entry_bar_index",
    "exit_bar_index",
    "entry_price",
    "exit_price",
    "quantity",
    "notional_quote",
    "gross_pnl_quote",
    "net_pnl_quote",
    "return_pct",
    "fee_quote",
    "slippage_quote",
    "exit_reason",
    "equity_after",
    "safe_quote_after",
    "timeframe",
)


@dataclass(frozen=True, slots=True)
class BacktestResultViewService:
    max_chart_points: int = MAX_RESULT_CHART_POINTS
    max_trades_page_size: int = MAX_RESULT_TRADES_PAGE_SIZE

    def __post_init__(self) -> None:
        if self.max_chart_points <= 0:
            raise ValueError("max_chart_points must be > 0")
        if self.max_trades_page_size <= 0:
            raise ValueError("max_trades_page_size must be > 0")

    def summary(
        self,
        *,
        job: BacktestJobReadModel,
        variants: Sequence[BacktestJobTopVariantReadModel],
    ) -> BacktestResultSummaryReadModel:
        selected = variants[0].variant_key if variants else None
        links: dict[str, Any] = {
            "self": f"/backtests/jobs/{job.job_id}/summary",
            "job": f"/backtests/jobs/{job.job_id}",
            "top": f"/backtests/jobs/{job.job_id}/top",
        }
        if selected is not None:
            links["selected_variant"] = (
                f"/backtests/jobs/{job.job_id}/variants/{selected}"
            )
        return BacktestResultSummaryReadModel(
            job=job,
            variants=tuple(variants),
            selected_variant_key=selected,
            links=links,
        )

    def equity(
        self,
        *,
        detail: BacktestLazyTradesDetailReadModel,
        points: int,
    ) -> BacktestResultSeriesReadModel:
        raw_points = _equity_points(detail=detail)
        limited = _downsample(points=raw_points, limit=self._chart_limit(points=points))
        return self._series(
            detail=detail,
            series="equity",
            requested_points=points,
            raw_points=raw_points,
            limited_points=limited,
        )

    def drawdown(
        self,
        *,
        detail: BacktestLazyTradesDetailReadModel,
        points: int,
    ) -> BacktestResultSeriesReadModel:
        equity_points = _equity_points(detail=detail)
        running_peak: float | None = None
        raw_points: list[dict[str, Any]] = []
        for point in equity_points:
            equity = _optional_float(point.get("value"))
            if equity is None:
                continue
            running_peak = equity if running_peak is None else max(running_peak, equity)
            drawdown_pct = 0.0
            if running_peak and running_peak > 0.0:
                drawdown_pct = ((equity - running_peak) / running_peak) * 100.0
            raw_points.append(
                {
                    "x": point.get("x"),
                    "value": drawdown_pct,
                    "equity": equity,
                    "trade_index": point.get("trade_index"),
                }
            )
        limited = _downsample(points=raw_points, limit=self._chart_limit(points=points))
        return self._series(
            detail=detail,
            series="drawdown",
            requested_points=points,
            raw_points=raw_points,
            limited_points=limited,
        )

    def monthly_stats(
        self,
        *,
        detail: BacktestLazyTradesDetailReadModel,
    ) -> BacktestResultMonthlyStatsReadModel:
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for trade in _sorted_trades(detail.trades):
            groups.setdefault(_trade_month(trade), []).append(trade)
        items = tuple(
            _stats_row(name=month, trades=tuple(groups[month]), key_name="month")
            for month in sorted(groups)
        )
        totals = _stats_totals(trades=detail.trades, summary_metrics=detail.summary_metrics)
        return BacktestResultMonthlyStatsReadModel(
            job_id=detail.job_id,
            variant_key=detail.variant_key,
            variant_hash=detail.variant_hash,
            items=items,
            totals=totals,
        )

    def symbol_stats(
        self,
        *,
        job: BacktestJob,
        detail: BacktestLazyTradesDetailReadModel,
    ) -> BacktestResultSymbolStatsReadModel:
        symbol = _symbol_from_job(job=job)
        trades = _sorted_trades(detail.trades)
        items = (
            {
                **_stats_row(name=symbol, trades=trades, key_name="symbol"),
                "long_count": sum(1 for trade in trades if str(trade.get("side")) == "long"),
                "short_count": sum(1 for trade in trades if str(trade.get("side")) == "short"),
            },
        )
        totals = _stats_totals(trades=trades, summary_metrics=detail.summary_metrics)
        return BacktestResultSymbolStatsReadModel(
            job_id=detail.job_id,
            variant_key=detail.variant_key,
            variant_hash=detail.variant_hash,
            items=items,
            totals=totals,
        )

    def trades_page(
        self,
        *,
        detail: BacktestLazyTradesDetailReadModel,
        page: int,
        page_size: int,
    ) -> BacktestResultTradesPageReadModel:
        resolved_page = max(1, int(page))
        resolved_page_size = min(
            max(1, int(page_size)),
            self.max_trades_page_size,
        )
        trades = _sorted_trades(detail.trades)
        total = len(trades)
        offset = (resolved_page - 1) * resolved_page_size
        items = tuple(dict(item) for item in trades[offset : offset + resolved_page_size])
        total_pages = (total + resolved_page_size - 1) // resolved_page_size if total else 0
        pagination = {
            "page": resolved_page,
            "page_size": resolved_page_size,
            "max_page_size": self.max_trades_page_size,
            "total": total,
            "total_pages": total_pages,
            "has_previous": resolved_page > 1 and total > 0,
            "has_next": total_pages > 0 and resolved_page < total_pages,
        }
        return BacktestResultTradesPageReadModel(
            job_id=detail.job_id,
            variant_key=detail.variant_key,
            variant_hash=detail.variant_hash,
            items=items,
            pagination=pagination,
            summary={
                "summary_metrics": dict(detail.summary_metrics),
                "cache": dict(detail.cache),
                "timing": dict(detail.timing),
            },
            links={
                "self": (
                    f"/backtests/jobs/{detail.job_id}/variants/"
                    f"{detail.variant_key}/trades?page={resolved_page}"
                    f"&page_size={resolved_page_size}"
                ),
                "csv": (
                    f"/backtests/jobs/{detail.job_id}/variants/"
                    f"{detail.variant_key}/trades.csv"
                ),
            },
        )

    def trades_csv(
        self,
        *,
        detail: BacktestLazyTradesDetailReadModel,
    ) -> BacktestTradesCsvReadModel:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for trade in _sorted_trades(detail.trades):
            writer.writerow({field: _csv_value(trade.get(field)) for field in _CSV_FIELDS})
        return BacktestTradesCsvReadModel(
            filename=_csv_filename(job_id=detail.job_id, variant_key=detail.variant_key),
            content=buffer.getvalue(),
        )

    def _series(
        self,
        *,
        detail: BacktestLazyTradesDetailReadModel,
        series: str,
        requested_points: int,
        raw_points: Sequence[Mapping[str, Any]],
        limited_points: Sequence[Mapping[str, Any]],
    ) -> BacktestResultSeriesReadModel:
        return BacktestResultSeriesReadModel(
            job_id=detail.job_id,
            variant_key=detail.variant_key,
            variant_hash=detail.variant_hash,
            series=series,
            requested_points=requested_points,
            point_limit=self._chart_limit(points=requested_points),
            total_points=len(raw_points),
            downsampled=len(limited_points) < len(raw_points),
            points=tuple(dict(point) for point in limited_points),
            summary={
                "summary_metrics": dict(detail.summary_metrics),
                "cache": dict(detail.cache),
                "timing": dict(detail.timing),
            },
        )

    def _chart_limit(self, *, points: int) -> int:
        return min(max(1, int(points)), self.max_chart_points)


def _equity_points(*, detail: BacktestLazyTradesDetailReadModel) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    running_equity: float | None = None
    for trade in _sorted_trades(detail.trades):
        equity = _optional_float(trade.get("equity_after"))
        if equity is None:
            pnl = _optional_float(trade.get("net_pnl_quote"))
            if running_equity is not None and pnl is not None:
                equity = running_equity + pnl
        if equity is None:
            continue
        running_equity = equity
        output.append(
            {
                "x": trade.get("exit_timestamp") or trade.get("entry_timestamp"),
                "value": equity,
                "trade_index": trade.get("trade_index"),
                "return_pct": trade.get("return_pct"),
            }
        )
    return output


def _downsample(
    *,
    points: Sequence[Mapping[str, Any]],
    limit: int,
) -> tuple[Mapping[str, Any], ...]:
    total = len(points)
    if total <= limit:
        return tuple(dict(point) for point in points)
    if limit <= 1:
        return (dict(points[-1]),)
    last = total - 1
    indexes = [round((last * index) / (limit - 1)) for index in range(limit)]
    deduped: list[int] = []
    for index in indexes:
        normalized = min(last, max(0, int(index)))
        if not deduped or deduped[-1] != normalized:
            deduped.append(normalized)
    if deduped[-1] != last:
        deduped[-1] = last
    return tuple(dict(points[index]) for index in deduped[:limit])


def _sorted_trades(trades: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    indexed = tuple(enumerate(trades))
    return tuple(
        dict(trade)
        for _, trade in sorted(
            indexed,
            key=lambda item: (_int_or_default(item[1].get("trade_index"), item[0]), item[0]),
        )
    )


def _trade_month(trade: Mapping[str, Any]) -> str:
    timestamp = _parse_datetime(trade.get("exit_timestamp") or trade.get("entry_timestamp"))
    if timestamp is None:
        return "unknown"
    return f"{timestamp.year:04d}-{timestamp.month:02d}"


def _stats_row(
    *,
    name: str,
    trades: Sequence[Mapping[str, Any]],
    key_name: str,
) -> dict[str, Any]:
    count = len(trades)
    returns = tuple(
        value
        for value in (_optional_float(trade.get("return_pct")) for trade in trades)
        if value is not None
    )
    net_pnl = sum(
        value
        for value in (_optional_float(trade.get("net_pnl_quote")) for trade in trades)
        if value is not None
    )
    winners = sum(1 for value in returns if value > 0.0)
    return {
        key_name: name,
        "trade_count": count,
        "win_count": winners,
        "loss_count": sum(1 for value in returns if value < 0.0),
        "win_rate_pct": (winners / count) * 100.0 if count else 0.0,
        "avg_return_pct": sum(returns) / len(returns) if returns else 0.0,
        "net_pnl_quote": net_pnl,
    }


def _stats_totals(
    *,
    trades: Sequence[Mapping[str, Any]],
    summary_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    base = _stats_row(name="total", trades=trades, key_name="scope")
    total_return = _optional_float(summary_metrics.get("total_return_pct"))
    if total_return is not None:
        base["total_return_pct"] = total_return
    base["profit_factor"] = summary_metrics.get("profit_factor")
    base["max_drawdown_pct"] = summary_metrics.get("max_drawdown_pct")
    return base


def _symbol_from_job(*, job: BacktestJob) -> str:
    request = dict(job.request_json)
    coordinates = request.get("coordinates")
    if isinstance(coordinates, Mapping):
        symbol = coordinates.get("symbol")
        if isinstance(symbol, str) and symbol.strip():
            return symbol.strip().upper()
    return job.symbol or "UNKNOWN"


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _csv_filename(*, job_id: str, variant_key: str) -> str:
    safe_variant = re.sub(r"[^A-Za-z0-9_.-]+", "-", variant_key).strip("-")
    if not safe_variant:
        safe_variant = "variant"
    return f"backtest-{job_id[:8]}-{safe_variant[:80]}-trades.csv"


__all__ = [
    "DEFAULT_RESULT_CHART_POINTS",
    "DEFAULT_RESULT_TRADES_PAGE_SIZE",
    "MAX_RESULT_CHART_POINTS",
    "MAX_RESULT_TRADES_PAGE_SIZE",
    "BacktestResultViewService",
]
