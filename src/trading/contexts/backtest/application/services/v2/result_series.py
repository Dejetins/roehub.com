from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from trading.contexts.backtest.application.dto import (
    BacktestJobReadModel,
    BacktestJobTopResult,
    BacktestLazyTradesDetailReadModel,
)

DEFAULT_BACKTEST_RESULT_POINTS = 600
MAX_BACKTEST_RESULT_POINTS = 1500
DEFAULT_BACKTEST_TRADES_PAGE_SIZE = 50
MAX_BACKTEST_TRADES_PAGE_SIZE = 100
DEFAULT_BACKTEST_TRADES_CSV_MAX_ROWS = 10_000
MAX_BACKTEST_TRADES_CSV_MAX_ROWS = 100_000
MAX_BACKTEST_MONTHLY_STATS_ITEMS = 600
MAX_BACKTEST_SYMBOL_STATS_ITEMS = 1

BacktestResultSeriesKind = Literal["equity", "drawdown"]


@dataclass(frozen=True, slots=True)
class BacktestResultSummaryReadModel:
    job: BacktestJobReadModel
    top_variants: BacktestJobTopResult
    selected_variant_key: str | None
    refresh_status: str
    retry_after_seconds: int
    links: Mapping[str, Any]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job": self.job.as_mapping(),
            "top_variants": self.top_variants.as_mapping(),
            "selected_variant_key": self.selected_variant_key,
            "refresh_status": self.refresh_status,
            "retry_after_seconds": self.retry_after_seconds,
            "links": dict(self.links),
        }


@dataclass(frozen=True, slots=True)
class BacktestResultSeriesReadModel:
    job_id: str
    variant_key: str
    variant_hash: str
    kind: BacktestResultSeriesKind
    points: tuple[Mapping[str, Any], ...]
    requested_points: int
    max_points: int
    returned_points: int
    source_points: int
    downsampled: bool
    cache: Mapping[str, Any]
    timing: Mapping[str, Any]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "variant_key": self.variant_key,
            "variant_hash": self.variant_hash,
            "kind": self.kind,
            "points": [dict(point) for point in self.points],
            "requested_points": self.requested_points,
            "max_points": self.max_points,
            "returned_points": self.returned_points,
            "source_points": self.source_points,
            "downsampled": self.downsampled,
            "cache": dict(self.cache),
            "timing": dict(self.timing),
        }


@dataclass(frozen=True, slots=True)
class BacktestResultStatsReadModel:
    job_id: str
    variant_key: str
    variant_hash: str
    kind: Literal["monthly", "symbol"]
    items: tuple[Mapping[str, Any], ...]
    bounds: Mapping[str, Any]
    cache: Mapping[str, Any]
    timing: Mapping[str, Any]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "variant_key": self.variant_key,
            "variant_hash": self.variant_hash,
            "kind": self.kind,
            "items": [dict(item) for item in self.items],
            "bounds": dict(self.bounds),
            "cache": dict(self.cache),
            "timing": dict(self.timing),
        }


@dataclass(frozen=True, slots=True)
class BacktestPaginatedTradesReadModel:
    job_id: str
    variant_key: str
    variant_hash: str
    items: tuple[Mapping[str, Any], ...]
    pagination: Mapping[str, Any]
    summary_metrics: Mapping[str, Any]
    cache: Mapping[str, Any]
    timing: Mapping[str, Any]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "variant_key": self.variant_key,
            "variant_hash": self.variant_hash,
            "items": [dict(item) for item in self.items],
            "pagination": dict(self.pagination),
            "summary_metrics": dict(self.summary_metrics),
            "cache": dict(self.cache),
            "timing": dict(self.timing),
        }


@dataclass(frozen=True, slots=True)
class BacktestTradesCsvReadModel:
    content: str
    row_count: int
    total_rows: int
    max_rows: int
    truncated: bool
    sort: str
    cache: Mapping[str, Any]
    timing: Mapping[str, Any]


def build_result_summary_read_model(
    *,
    job: BacktestJobReadModel,
    top_variants: BacktestJobTopResult,
) -> BacktestResultSummaryReadModel:
    selected = top_variants.items[0].variant_key if top_variants.items else None
    job_id = job.job_id
    links: dict[str, Any] = {
        "self": f"/backtests/jobs/{job_id}/summary",
        "job": f"/backtests/jobs/{job_id}",
        "top": f"/backtests/jobs/{job_id}/top",
    }
    if selected is not None:
        links["selected_variant"] = f"/backtests/jobs/{job_id}/variants/{selected}"
    return BacktestResultSummaryReadModel(
        job=job,
        top_variants=top_variants,
        selected_variant_key=selected,
        refresh_status=job.refresh_status,
        retry_after_seconds=job.retry_after_seconds,
        links=links,
    )


def build_result_series_read_model(
    *,
    detail: BacktestLazyTradesDetailReadModel,
    kind: BacktestResultSeriesKind,
    requested_points: int,
) -> BacktestResultSeriesReadModel:
    limit = normalize_chart_points(requested_points)
    full_points = _equity_points(detail=detail)
    if kind == "drawdown":
        full_points = _drawdown_points(points=full_points)
    sampled = _downsample(points=full_points, limit=limit)
    return BacktestResultSeriesReadModel(
        job_id=detail.job_id,
        variant_key=detail.variant_key,
        variant_hash=detail.variant_hash,
        kind=kind,
        points=tuple(sampled),
        requested_points=requested_points,
        max_points=limit,
        returned_points=len(sampled),
        source_points=len(full_points),
        downsampled=len(sampled) < len(full_points),
        cache=detail.cache,
        timing=detail.timing,
    )


def build_monthly_stats_read_model(
    *,
    detail: BacktestLazyTradesDetailReadModel,
) -> BacktestResultStatsReadModel:
    buckets: dict[str, dict[str, Any]] = {}
    for trade in detail.trades:
        month = _month_key(trade.get("exit_timestamp"))
        bucket = buckets.setdefault(
            month,
            {
                "month": month,
                "trades_count": 0,
                "net_pnl_quote": 0.0,
                "return_pct": 0.0,
                "wins": 0,
                "losses": 0,
            },
        )
        pnl = _float(trade.get("net_pnl_quote"))
        bucket["trades_count"] += 1
        bucket["net_pnl_quote"] += pnl
        bucket["return_pct"] += _float(trade.get("return_pct"))
        if pnl > 0:
            bucket["wins"] += 1
        elif pnl < 0:
            bucket["losses"] += 1
    source_items = tuple(_with_win_rate(item) for _, item in sorted(buckets.items()))
    items = source_items[:MAX_BACKTEST_MONTHLY_STATS_ITEMS]
    return BacktestResultStatsReadModel(
        job_id=detail.job_id,
        variant_key=detail.variant_key,
        variant_hash=detail.variant_hash,
        kind="monthly",
        items=items,
        bounds={
            "max_items": MAX_BACKTEST_MONTHLY_STATS_ITEMS,
            "returned_items": len(items),
            "source_items": len(source_items),
            "truncated": len(items) < len(source_items),
            "sort": "month_asc",
        },
        cache=detail.cache,
        timing=detail.timing,
    )


def build_symbol_stats_read_model(
    *,
    detail: BacktestLazyTradesDetailReadModel,
    symbol: str | None,
) -> BacktestResultStatsReadModel:
    item: dict[str, Any] = {
        "symbol": symbol or "unknown",
        "trades_count": len(detail.trades),
        "net_pnl_quote": sum(_float(trade.get("net_pnl_quote")) for trade in detail.trades),
        "return_pct": sum(_float(trade.get("return_pct")) for trade in detail.trades),
        "wins": sum(1 for trade in detail.trades if _float(trade.get("net_pnl_quote")) > 0),
        "losses": sum(1 for trade in detail.trades if _float(trade.get("net_pnl_quote")) < 0),
    }
    return BacktestResultStatsReadModel(
        job_id=detail.job_id,
        variant_key=detail.variant_key,
        variant_hash=detail.variant_hash,
        kind="symbol",
        items=(_with_win_rate(item),),
        bounds={
            "max_items": MAX_BACKTEST_SYMBOL_STATS_ITEMS,
            "returned_items": 1,
            "source_items": 1,
            "truncated": False,
            "sort": "symbol_asc",
        },
        cache=detail.cache,
        timing=detail.timing,
    )


def build_paginated_trades_read_model(
    *,
    detail: BacktestLazyTradesDetailReadModel,
    page: int,
    page_size: int,
) -> BacktestPaginatedTradesReadModel:
    effective_page = max(1, page)
    effective_page_size = min(max(1, page_size), MAX_BACKTEST_TRADES_PAGE_SIZE)
    ordered_trades = _ordered_trades(trades=detail.trades)
    total = len(ordered_trades)
    offset = (effective_page - 1) * effective_page_size
    items = tuple(dict(item) for item in ordered_trades[offset : offset + effective_page_size])
    has_next = offset + effective_page_size < total
    pagination = {
        "mode": "page",
        "page": effective_page,
        "page_size": effective_page_size,
        "max_page_size": MAX_BACKTEST_TRADES_PAGE_SIZE,
        "total": total,
        "has_next": has_next,
        "has_previous": effective_page > 1,
        "next_page": effective_page + 1 if has_next else None,
        "previous_page": effective_page - 1 if effective_page > 1 else None,
        "sort": "trade_index_asc",
    }
    return BacktestPaginatedTradesReadModel(
        job_id=detail.job_id,
        variant_key=detail.variant_key,
        variant_hash=detail.variant_hash,
        items=items,
        pagination=pagination,
        summary_metrics=detail.summary_metrics,
        cache=detail.cache,
        timing=detail.timing,
    )


def build_trades_csv(
    *,
    detail: BacktestLazyTradesDetailReadModel,
    max_rows: int | None = None,
) -> BacktestTradesCsvReadModel:
    effective_max_rows = normalize_csv_max_rows(max_rows)
    fields = (
        "trade_index",
        "entry_timestamp",
        "exit_timestamp",
        "side",
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
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    ordered_trades = _ordered_trades(trades=detail.trades)
    for trade in ordered_trades[:effective_max_rows]:
        writer.writerow({field: trade.get(field) for field in fields})
    return BacktestTradesCsvReadModel(
        content=output.getvalue(),
        row_count=min(len(ordered_trades), effective_max_rows),
        total_rows=len(ordered_trades),
        max_rows=effective_max_rows,
        truncated=len(ordered_trades) > effective_max_rows,
        sort="trade_index_asc",
        cache=detail.cache,
        timing=detail.timing,
    )


def normalize_chart_points(value: int | None) -> int:
    if value is None:
        return DEFAULT_BACKTEST_RESULT_POINTS
    return min(max(10, int(value)), MAX_BACKTEST_RESULT_POINTS)


def normalize_csv_max_rows(value: int | None) -> int:
    if value is None:
        return DEFAULT_BACKTEST_TRADES_CSV_MAX_ROWS
    return min(max(1, int(value)), MAX_BACKTEST_TRADES_CSV_MAX_ROWS)


def _equity_points(*, detail: BacktestLazyTradesDetailReadModel) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for trade in _ordered_trades(trades=detail.trades):
        equity = _float_or_none(trade.get("equity_after"))
        if equity is None:
            continue
        points.append(
            {
                "x": trade.get("exit_timestamp") or trade.get("trade_index"),
                "trade_index": trade.get("trade_index"),
                "value": equity,
                "net_pnl_quote": _float(trade.get("net_pnl_quote")),
            }
        )
    return points


def _drawdown_points(*, points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    peak: float | None = None
    result: list[dict[str, Any]] = []
    for point in points:
        value = _float(point.get("value"))
        peak = value if peak is None else max(peak, value)
        drawdown = 0.0 if peak <= 0 else ((value - peak) / peak) * 100.0
        result.append(
            {
                "x": point.get("x"),
                "trade_index": point.get("trade_index"),
                "value": drawdown,
                "equity": value,
            }
        )
    return result


def _ordered_trades(*, trades: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        trade
        for _, trade in sorted(
            enumerate(trades),
            key=lambda item: _trade_sort_key(index=item[0], trade=item[1]),
        )
    )


def _trade_sort_key(*, index: int, trade: Mapping[str, Any]) -> tuple[int, float, str, str, int]:
    trade_index = _float_or_none(trade.get("trade_index"))
    trade_index_missing = 1 if trade_index is None else 0
    return (
        trade_index_missing,
        trade_index if trade_index is not None else float(index),
        str(trade.get("exit_timestamp") or ""),
        str(trade.get("entry_timestamp") or ""),
        index,
    )


def _downsample(*, points: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(points) <= limit:
        return [dict(point) for point in points]
    if limit <= 1:
        return [dict(points[-1])]
    step = (len(points) - 1) / (limit - 1)
    sampled: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item_index in range(limit):
        source_index = int(round(item_index * step))
        if source_index in seen:
            continue
        seen.add(source_index)
        sampled.append(dict(points[source_index]))
    return sampled


def _month_key(value: Any) -> str:
    if isinstance(value, str) and len(value) >= 7:
        return value[:7]
    return "unknown"


def _with_win_rate(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    trades = int(payload.get("trades_count") or 0)
    wins = int(payload.get("wins") or 0)
    payload["win_rate_pct"] = (wins / trades) * 100.0 if trades > 0 else None
    return payload


def _float(value: Any) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None else 0.0


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def symbol_from_job_request(request: Mapping[str, Any]) -> str | None:
    coordinates = request.get("coordinates")
    if not isinstance(coordinates, Mapping):
        return None
    symbol = coordinates.get("symbol")
    return str(symbol) if symbol is not None else None


__all__ = [
    "DEFAULT_BACKTEST_RESULT_POINTS",
    "DEFAULT_BACKTEST_TRADES_CSV_MAX_ROWS",
    "DEFAULT_BACKTEST_TRADES_PAGE_SIZE",
    "MAX_BACKTEST_MONTHLY_STATS_ITEMS",
    "MAX_BACKTEST_RESULT_POINTS",
    "MAX_BACKTEST_SYMBOL_STATS_ITEMS",
    "MAX_BACKTEST_TRADES_CSV_MAX_ROWS",
    "MAX_BACKTEST_TRADES_PAGE_SIZE",
    "BacktestPaginatedTradesReadModel",
    "BacktestResultSeriesKind",
    "BacktestResultSeriesReadModel",
    "BacktestResultStatsReadModel",
    "BacktestResultSummaryReadModel",
    "BacktestTradesCsvReadModel",
    "build_monthly_stats_read_model",
    "build_paginated_trades_read_model",
    "build_result_series_read_model",
    "build_result_summary_read_model",
    "build_symbol_stats_read_model",
    "build_trades_csv",
    "normalize_chart_points",
    "normalize_csv_max_rows",
    "symbol_from_job_request",
]
