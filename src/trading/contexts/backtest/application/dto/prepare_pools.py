from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class BacktestPreparePoolsConfig:
    """
    Internal runtime knobs for the measured `prepare_pools` stage.
    """

    row_prefilter_top_fraction: float = 142.0 / 588.0
    row_prefilter_min_nonzero: int = 1
    time_chunk: int = 4096

    def __post_init__(self) -> None:
        if not (0.0 < self.row_prefilter_top_fraction <= 1.0):
            raise ValueError("row_prefilter_top_fraction must be in (0, 1]")
        if self.row_prefilter_min_nonzero < 0:
            raise ValueError("row_prefilter_min_nonzero must be >= 0")
        if self.time_chunk <= 0:
            raise ValueError("time_chunk must be > 0")


@dataclass(frozen=True, slots=True)
class PreparedSignalSegments:
    """
    Padded per-row signal change-point segments.
    """

    starts: np.ndarray
    ends: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    change_count: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedIndicatorRowMetadata:
    """
    Stable row identity metadata aligned to a prepared indicator pool row.
    """

    indicator_id: str
    row_id: int
    source: str | None
    window: int

    def as_mapping(self) -> dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "row_id": self.row_id,
            "source": self.source,
            "window": self.window,
        }


@dataclass(frozen=True, slots=True)
class PreparedIndicatorPool:
    """
    Prepared per-indicator signal pool consumed by later exact/proxy compute stages.
    """

    indicator_id: str
    row_ids: np.ndarray
    filtered_row_ids: np.ndarray
    trade_T: np.ndarray
    eval_T: np.ndarray
    segments: PreparedSignalSegments
    row_score: np.ndarray
    score_adj: np.ndarray
    nonzero: np.ndarray
    proxy: np.ndarray
    change_count: np.ndarray
    metadata: tuple[PreparedIndicatorRowMetadata, ...]

    @property
    def trade_T_length(self) -> int:
        return int(self.trade_T.shape[1])

    @property
    def eval_T_length(self) -> int:
        return int(self.eval_T.shape[1])


@dataclass(frozen=True, slots=True)
class PreparedExecutionMapping:
    """
    15m signal-bar to 1m execution mapping for no-risk execution.
    """

    signal_entry_exec_idx_15m: np.ndarray
    run_bar_open_1m_idx_15m: np.ndarray
    run_bar_close_1m_idx_15m: np.ndarray
    t_exec_limit_1m: int


@dataclass(frozen=True, slots=True)
class PreparePoolsTiming:
    """
    Measured wall timing for prepare-pools stages.

    `prepare_pools_core` is the notebook-compatible timing scope. `prepare_pools_total`
    is aggregate service telemetry that may include artifact context resolution, mmap
    opening, and request slicing overhead.
    """

    stage_name: str
    wall_time_s: float
    subsegments: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subsegments",
            MappingProxyType({str(key): float(value) for key, value in self.subsegments.items()}),
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "wall_time_s": self.wall_time_s,
            "subsegments": dict(self.subsegments),
        }

    @property
    def prepare_pools_core_s(self) -> float | None:
        value = self.subsegments.get("prepare_pools_core")
        return None if value is None else float(value)

    @property
    def prepare_pools_total_s(self) -> float | None:
        if self.stage_name == "prepare_pools_total":
            return float(self.wall_time_s)
        value = self.subsegments.get("prepare_pools_total")
        return None if value is None else float(value)


@dataclass(frozen=True, slots=True)
class BacktestPreparePoolsResult:
    """
    Internal result of the measured `prepare_pools` stage.
    """

    timeframe: str
    indicator_ids: tuple[str, ...]
    indicator_pools: tuple[PreparedIndicatorPool, ...]
    signal_returns_15m: np.ndarray
    execution_mapping: PreparedExecutionMapping
    time_slice_start_15m: int
    time_slice_stop_15m: int
    trade_T_length: int
    eval_T_length: int
    row_metadata_order_hash: str
    timing: PreparePoolsTiming
    execution_open_1m: np.ndarray | None = None
    execution_close_1m: np.ndarray | None = None
    execution_open_time_1m: np.ndarray | None = None
    execution_close_time_1m: np.ndarray | None = None


__all__ = [
    "BacktestPreparePoolsConfig",
    "BacktestPreparePoolsResult",
    "PreparedExecutionMapping",
    "PreparedIndicatorPool",
    "PreparedIndicatorRowMetadata",
    "PreparedSignalSegments",
    "PreparePoolsTiming",
]
