from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class BacktestComboPlanningConfig:
    """
    Internal runtime knobs for combo planning before exact scoring.
    """

    combo_top_frac: float = 1.0
    combo_min_confirm: int = 1
    combo_chunk_size: int = 4096
    fee_penalty_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if not (0.0 < self.combo_top_frac <= 1.0):
            raise ValueError("combo_top_frac must be in (0, 1]")
        if self.combo_min_confirm < 1:
            raise ValueError("combo_min_confirm must be >= 1")
        if self.combo_chunk_size <= 0:
            raise ValueError("combo_chunk_size must be > 0")
        if self.fee_penalty_multiplier < 0.0:
            raise ValueError("fee_penalty_multiplier must be >= 0")

    @property
    def proxy_filter_active(self) -> bool:
        return self.combo_top_frac < 1.0 or self.combo_min_confirm > 1


@dataclass(frozen=True, slots=True)
class BacktestSelectedBackend:
    """
    Internal backend choice for a normalized request and prepared pool arity.
    """

    backend_id: str
    risk_mode: str
    arity: int
    direction_mode: str
    requires_exact_context: bool
    role: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "risk_mode": self.risk_mode,
            "arity": self.arity,
            "direction_mode": self.direction_mode,
            "requires_exact_context": self.requires_exact_context,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class BacktestExactContext:
    """
    Arity-first packed signal segment context for generic exact kernels.
    """

    indicator_ids: tuple[str, ...]
    required: bool
    starts: np.ndarray | None
    ends: np.ndarray | None
    values: np.ndarray | None
    counts: np.ndarray | None
    row_counts: tuple[int, ...]
    max_rows: int
    max_segments: int

    @property
    def materialized(self) -> bool:
        return self.starts is not None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "indicator_ids": list(self.indicator_ids),
            "required": self.required,
            "materialized": self.materialized,
            "starts_shape": None if self.starts is None else list(self.starts.shape),
            "ends_shape": None if self.ends is None else list(self.ends.shape),
            "values_shape": None if self.values is None else list(self.values.shape),
            "counts_shape": None if self.counts is None else list(self.counts.shape),
            "row_counts": list(self.row_counts),
            "max_rows": self.max_rows,
            "max_segments": self.max_segments,
        }


@dataclass(frozen=True, slots=True)
class BacktestProxyContext:
    """
    Optional combo prefilter context.
    """

    indicator_ids: tuple[str, ...]
    active: bool
    context_type: str
    combo_top_frac: float
    combo_min_confirm: int
    fee_penalty_per_confirm: np.float32
    confirm_matrix: np.ndarray | None = None
    proxy_matrix: np.ndarray | None = None
    eval_stack: np.ndarray | None = None
    ret_15m: np.ndarray | None = None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "indicator_ids": list(self.indicator_ids),
            "active": self.active,
            "context_type": self.context_type,
            "combo_top_frac": self.combo_top_frac,
            "combo_min_confirm": self.combo_min_confirm,
            "fee_penalty_per_confirm": float(self.fee_penalty_per_confirm),
            "confirm_matrix_shape": None
            if self.confirm_matrix is None
            else list(self.confirm_matrix.shape),
            "proxy_matrix_shape": None
            if self.proxy_matrix is None
            else list(self.proxy_matrix.shape),
            "eval_stack_shape": None if self.eval_stack is None else list(self.eval_stack.shape),
            "ret_15m_shape": None if self.ret_15m is None else list(self.ret_15m.shape),
        }


@dataclass(frozen=True, slots=True)
class BacktestComboChunk:
    """
    Bounded deterministic Cartesian chunk over local prepared-pool row indexes.
    """

    indicator_ids: tuple[str, ...]
    rows_by_indicator: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        indicator_ids = tuple(str(indicator_id) for indicator_id in self.indicator_ids)
        rows_by_indicator: dict[str, np.ndarray] = {}
        lengths: set[int] = set()
        for indicator_id in indicator_ids:
            if indicator_id not in self.rows_by_indicator:
                raise ValueError(f"rows_by_indicator missing indicator {indicator_id!r}")
            rows = np.ascontiguousarray(
                np.asarray(self.rows_by_indicator[indicator_id], dtype=np.int32)
            )
            if rows.ndim != 1:
                raise ValueError(f"rows for {indicator_id!r} must be one-dimensional")
            rows_by_indicator[indicator_id] = rows
            lengths.add(int(rows.shape[0]))
        if len(lengths) > 1:
            raise ValueError("all combo chunk arrays must have the same length")
        object.__setattr__(self, "indicator_ids", indicator_ids)
        object.__setattr__(self, "rows_by_indicator", MappingProxyType(rows_by_indicator))

    @property
    def size(self) -> int:
        if not self.indicator_ids:
            return 0
        return int(self.rows_by_indicator[self.indicator_ids[0]].shape[0])

    def as_mapping(self) -> dict[str, Any]:
        return {
            "indicator_ids": list(self.indicator_ids),
            "size": self.size,
            "rows_by_indicator": {
                indicator_id: self.rows_by_indicator[indicator_id].tolist()
                for indicator_id in self.indicator_ids
            },
        }


@dataclass(frozen=True, slots=True)
class BacktestProxyFilterResult:
    """
    Result of applying pass-through or active proxy filtering to one chunk.
    """

    indicator_ids: tuple[str, ...]
    selected_indexes: np.ndarray
    selected_rows_by_indicator: Mapping[str, np.ndarray]
    input_candidate_count: int
    valid_candidate_count: int
    selected_candidate_count: int
    confirm: np.ndarray | None = None
    proxy: np.ndarray | None = None

    def __post_init__(self) -> None:
        indicator_ids = tuple(str(indicator_id) for indicator_id in self.indicator_ids)
        selected_indexes = np.ascontiguousarray(
            np.asarray(self.selected_indexes, dtype=np.int32)
        )
        rows_by_indicator: dict[str, np.ndarray] = {}
        for indicator_id in indicator_ids:
            if indicator_id not in self.selected_rows_by_indicator:
                raise ValueError(f"selected rows missing indicator {indicator_id!r}")
            rows_by_indicator[indicator_id] = np.ascontiguousarray(
                np.asarray(self.selected_rows_by_indicator[indicator_id], dtype=np.int32)
            )
        object.__setattr__(self, "indicator_ids", indicator_ids)
        object.__setattr__(self, "selected_indexes", selected_indexes)
        object.__setattr__(
            self,
            "selected_rows_by_indicator",
            MappingProxyType(rows_by_indicator),
        )
        if self.confirm is not None:
            object.__setattr__(
                self,
                "confirm",
                np.ascontiguousarray(np.asarray(self.confirm, dtype=np.int32)),
            )
        if self.proxy is not None:
            object.__setattr__(
                self,
                "proxy",
                np.ascontiguousarray(np.asarray(self.proxy, dtype=np.float32)),
            )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "indicator_ids": list(self.indicator_ids),
            "selected_indexes": self.selected_indexes.tolist(),
            "input_candidate_count": self.input_candidate_count,
            "valid_candidate_count": self.valid_candidate_count,
            "selected_candidate_count": self.selected_candidate_count,
        }


@dataclass(frozen=True, slots=True)
class BacktestComboPlanningTelemetry:
    """
    Stage timings and candidate-count telemetry before exact scoring.
    """

    stage_timings: Mapping[str, float]
    cartesian_combinations: int
    combo_chunks_processed: int
    exact_candidates_evaluated: int
    proxy_candidates_seen: int
    proxy_candidates_valid: int
    proxy_candidates_selected: int

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
            "proxy_candidates_seen": self.proxy_candidates_seen,
            "proxy_candidates_valid": self.proxy_candidates_valid,
            "proxy_candidates_selected": self.proxy_candidates_selected,
        }


@dataclass(frozen=True, slots=True)
class BacktestComboPlanningResult:
    """
    Runtime combo planning result consumed by later exact scoring iterations.
    """

    backend: BacktestSelectedBackend
    exact_context: BacktestExactContext
    proxy_context: BacktestProxyContext
    telemetry: BacktestComboPlanningTelemetry

    def as_mapping(self) -> dict[str, Any]:
        return {
            "backend": self.backend.as_mapping(),
            "exact_context": self.exact_context.as_mapping(),
            "proxy_context": self.proxy_context.as_mapping(),
            "telemetry": self.telemetry.as_mapping(),
        }


__all__ = [
    "BacktestComboChunk",
    "BacktestComboPlanningConfig",
    "BacktestComboPlanningResult",
    "BacktestComboPlanningTelemetry",
    "BacktestExactContext",
    "BacktestProxyContext",
    "BacktestProxyFilterResult",
    "BacktestSelectedBackend",
]
