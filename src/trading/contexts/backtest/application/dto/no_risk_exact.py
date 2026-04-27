from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

NO_RISK_SUMMARY_METRIC_NAMES: tuple[str, ...] = (
    "total_return_pct",
    "max_drawdown_pct",
    "return_over_max_drawdown",
    "profit_factor",
    "trade_count",
    "sharpe_trades",
    "win_rate_pct",
    "avg_trade_ret_pct",
    "avg_trade_exec_bars",
    "exposure_pct",
)


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExecutionPrices:
    """
    1m execution prices consumed by the no-risk scorer.
    """

    open_1m: np.ndarray
    close_1m: np.ndarray

    def __post_init__(self) -> None:
        open_1m = np.ascontiguousarray(np.asarray(self.open_1m, dtype=np.float32))
        close_1m = np.ascontiguousarray(np.asarray(self.close_1m, dtype=np.float32))
        if open_1m.ndim != 1 or close_1m.ndim != 1:
            raise ValueError("execution open/close arrays must be one-dimensional")
        if int(open_1m.shape[0]) != int(close_1m.shape[0]):
            raise ValueError("execution open/close arrays must share length")
        if int(open_1m.shape[0]) == 0:
            raise ValueError("execution prices must not be empty")
        object.__setattr__(self, "open_1m", open_1m)
        object.__setattr__(self, "close_1m", close_1m)


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExecutionConfig:
    """
    Scalar no-risk execution settings normalized for kernel dispatch.
    """

    direction_mode: str
    sizing_mode: str
    initial_cash_quote: float
    fixed_quote: float
    fee_rate: float
    slippage_rate: float
    safe_profit_percent: float
    use_fixed_quote: bool
    use_profit_lock: bool
    bars_per_year_exec: float
    close_on_end: bool

    def as_mapping(self) -> dict[str, Any]:
        return {
            "direction_mode": self.direction_mode,
            "sizing_mode": self.sizing_mode,
            "initial_cash_quote": self.initial_cash_quote,
            "fixed_quote": self.fixed_quote,
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
            "safe_profit_percent": self.safe_profit_percent,
            "use_fixed_quote": self.use_fixed_quote,
            "use_profit_lock": self.use_profit_lock,
            "bars_per_year_exec": self.bars_per_year_exec,
            "close_on_end": self.close_on_end,
        }


@dataclass(frozen=True, slots=True)
class BacktestNoRiskExactScoringConfig:
    """
    Internal knobs for Iteration 4 no-risk exact scoring and top-N.
    """

    top_n: int = 100
    ranking_metric: str = "total_return_pct"
    ranking_direction: str = "desc"
    self_check_n: int = 2
    self_check_return_tolerance: float = 1e-4
    combo_chunk_size: int = 4096
    bars_per_year_exec: float = 365.0 * 24.0 * 60.0

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be > 0")
        if self.ranking_metric not in NO_RISK_SUMMARY_METRIC_NAMES:
            raise ValueError(f"unsupported ranking_metric={self.ranking_metric!r}")
        if self.ranking_direction not in ("asc", "desc"):
            raise ValueError("ranking_direction must be 'asc' or 'desc'")
        if self.self_check_n < 0:
            raise ValueError("self_check_n must be >= 0")
        if self.self_check_return_tolerance < 0.0:
            raise ValueError("self_check_return_tolerance must be >= 0")
        if self.combo_chunk_size <= 0:
            raise ValueError("combo_chunk_size must be > 0")
        if self.bars_per_year_exec <= 0.0:
            raise ValueError("bars_per_year_exec must be > 0")


@dataclass(frozen=True, slots=True)
class BacktestNoRiskSummaryMetrics:
    total_return_pct: float
    max_drawdown_pct: float
    return_over_max_drawdown: float
    profit_factor: float
    trade_count: int
    sharpe_trades: float
    win_rate_pct: float
    avg_trade_ret_pct: float
    avg_trade_exec_bars: float
    exposure_pct: float

    def as_mapping(self) -> dict[str, float | int]:
        return {
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "return_over_max_drawdown": self.return_over_max_drawdown,
            "profit_factor": self.profit_factor,
            "trade_count": self.trade_count,
            "sharpe_trades": self.sharpe_trades,
            "win_rate_pct": self.win_rate_pct,
            "avg_trade_ret_pct": self.avg_trade_ret_pct,
            "avg_trade_exec_bars": self.avg_trade_exec_bars,
            "exposure_pct": self.exposure_pct,
        }

    def metric_value(self, metric_name: str) -> float:
        if metric_name == "trade_count":
            return float(self.trade_count)
        value = getattr(self, metric_name)
        return float(value)


@dataclass(frozen=True, slots=True)
class BacktestNoRiskChunkScores:
    """
    Columnar exact-scoring result for one selected combo chunk.
    """

    total_return_pct: np.ndarray
    max_drawdown_pct: np.ndarray
    return_over_max_drawdown: np.ndarray
    profit_factor: np.ndarray
    trade_count: np.ndarray
    sharpe_trades: np.ndarray
    win_rate_pct: np.ndarray
    avg_trade_ret_pct: np.ndarray
    avg_trade_exec_bars: np.ndarray
    exposure_pct: np.ndarray

    @property
    def size(self) -> int:
        return int(self.total_return_pct.shape[0])

    def metrics_at(self, index: int) -> BacktestNoRiskSummaryMetrics:
        return BacktestNoRiskSummaryMetrics(
            total_return_pct=float(self.total_return_pct[index]),
            max_drawdown_pct=float(self.max_drawdown_pct[index]),
            return_over_max_drawdown=float(self.return_over_max_drawdown[index]),
            profit_factor=float(self.profit_factor[index]),
            trade_count=int(self.trade_count[index]),
            sharpe_trades=float(self.sharpe_trades[index]),
            win_rate_pct=float(self.win_rate_pct[index]),
            avg_trade_ret_pct=float(self.avg_trade_ret_pct[index]),
            avg_trade_exec_bars=float(self.avg_trade_exec_bars[index]),
            exposure_pct=float(self.exposure_pct[index]),
        )


@dataclass(frozen=True, slots=True)
class BacktestNoRiskSelfCheckResult:
    checked: int
    passed: bool
    exact_backend: str
    direction_mode: str
    return_tolerance: float
    max_abs_exact_backend_ret_diff: float
    trade_count_equal: bool

    def as_mapping(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "passed": self.passed,
            "exact_backend": self.exact_backend,
            "direction_mode": self.direction_mode,
            "return_tolerance": self.return_tolerance,
            "max_abs_exact_backend_ret_diff": self.max_abs_exact_backend_ret_diff,
            "trade_count_equal": self.trade_count_equal,
        }


@dataclass(frozen=True, slots=True)
class BacktestNoRiskTopRow:
    """
    Final public top-N summary row plus explicit storage identity mapping.
    """

    rank: int
    variant_index: int
    public_variant_key: str
    variant_hash: str
    indicator_variant_hash: str
    row_ids_by_indicator: Mapping[str, int]
    local_rows_by_indicator: Mapping[str, int]
    summary_metrics: BacktestNoRiskSummaryMetrics
    ranking_metric: str
    ranking_score: float
    confirm_count: int
    proxy_score: float
    variant_params: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "row_ids_by_indicator",
            MappingProxyType(
                {str(key): int(value) for key, value in self.row_ids_by_indicator.items()}
            ),
        )
        object.__setattr__(
            self,
            "local_rows_by_indicator",
            MappingProxyType(
                {str(key): int(value) for key, value in self.local_rows_by_indicator.items()}
            ),
        )
        object.__setattr__(self, "variant_params", MappingProxyType(dict(self.variant_params)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "variant_index": self.variant_index,
            "variant_key": self.public_variant_key,
            "public_variant_key": self.public_variant_key,
            "variant_hash": self.variant_hash,
            "indicator_variant_hash": self.indicator_variant_hash,
            "row_ids_by_indicator": dict(self.row_ids_by_indicator),
            "local_rows_by_indicator": dict(self.local_rows_by_indicator),
            "summary_metrics": self.summary_metrics.as_mapping(),
            "ranking_metric": self.ranking_metric,
            "ranking_score": self.ranking_score,
            "confirm_count": self.confirm_count,
            "proxy_score": self.proxy_score,
            "variant_params": dict(self.variant_params),
        }


@dataclass(frozen=True, slots=True)
class BacktestNoRiskScoringTelemetry:
    stage_timings: Mapping[str, float]
    cartesian_combinations: int
    combo_chunks_processed: int
    exact_candidates_evaluated: int
    heap_candidates_seen: int
    top_result_proxy_filled: int
    self_check: BacktestNoRiskSelfCheckResult

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_timings",
            MappingProxyType({str(key): float(value) for key, value in self.stage_timings.items()}),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "stage_timings": dict(self.stage_timings),
            "cartesian_combinations": self.cartesian_combinations,
            "combo_chunks_processed": self.combo_chunks_processed,
            "exact_candidates_evaluated": self.exact_candidates_evaluated,
            "heap_candidates_seen": self.heap_candidates_seen,
            "top_result_proxy_filled": self.top_result_proxy_filled,
            "self_check": self.self_check.as_mapping(),
        }


@dataclass(frozen=True, slots=True)
class BacktestNoRiskScoringResult:
    top_rows: tuple[BacktestNoRiskTopRow, ...]
    telemetry: BacktestNoRiskScoringTelemetry

    def as_mapping(self) -> dict[str, Any]:
        return {
            "top_rows": [row.as_mapping() for row in self.top_rows],
            "telemetry": self.telemetry.as_mapping(),
        }


__all__ = [
    "NO_RISK_SUMMARY_METRIC_NAMES",
    "BacktestNoRiskChunkScores",
    "BacktestNoRiskExactScoringConfig",
    "BacktestNoRiskExecutionConfig",
    "BacktestNoRiskExecutionPrices",
    "BacktestNoRiskScoringResult",
    "BacktestNoRiskScoringTelemetry",
    "BacktestNoRiskSelfCheckResult",
    "BacktestNoRiskSummaryMetrics",
    "BacktestNoRiskTopRow",
]
