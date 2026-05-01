from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
        ArtifactHitTimesManifestDocumentV2,
    )

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class BacktestTpSlHitTimesGridArrays:
    """
    Small `hit_times/15m` grid payload loaded before table materialization.
    """

    manifest: ArtifactHitTimesManifestDocumentV2
    manifest_hash: str
    tp_values: np.ndarray
    sl_values: np.ndarray

    def __post_init__(self) -> None:
        _ensure_sha256(self.manifest_hash, field_name="manifest_hash")
        _ensure_f32_vector(self.tp_values, field_name="tp_values")
        _ensure_f32_vector(self.sl_values, field_name="sl_values")


@dataclass(frozen=True, slots=True)
class BacktestTpSlHitTimesTableArrays:
    """
    Heavy `hit_times/15m` table mmaps loaded only after grid coverage passes.
    """

    manifest: ArtifactHitTimesManifestDocumentV2
    manifest_hash: str
    long_tp: np.ndarray
    long_sl: np.ndarray
    short_tp: np.ndarray
    short_sl: np.ndarray

    def __post_init__(self) -> None:
        _ensure_sha256(self.manifest_hash, field_name="manifest_hash")
        _ensure_u32_matrix(self.long_tp, field_name="long_tp")
        _ensure_u32_matrix(self.long_sl, field_name="long_sl")
        _ensure_u32_matrix(self.short_tp, field_name="short_tp")
        _ensure_u32_matrix(self.short_sl, field_name="short_sl")


@dataclass(frozen=True, slots=True)
class BacktestTpSlRequestedGrid:
    tp_levels_pct: tuple[float, ...]
    sl_levels_pct: tuple[float, ...]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "tp_levels_pct": list(self.tp_levels_pct),
            "sl_levels_pct": list(self.sl_levels_pct),
            "tp_count": len(self.tp_levels_pct),
            "sl_count": len(self.sl_levels_pct),
            "cells": len(self.tp_levels_pct) * len(self.sl_levels_pct),
        }


@dataclass(frozen=True, slots=True)
class BacktestTpSlGridEvidence:
    artifact_path: str
    timeframe: str
    target_grid: Mapping[str, Any]
    artifact_grid: Mapping[str, Any]
    requested_grid: BacktestTpSlRequestedGrid
    resolved_tp_indexes: tuple[int, ...]
    resolved_sl_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_grid", MappingProxyType(dict(self.target_grid)))
        object.__setattr__(self, "artifact_grid", MappingProxyType(dict(self.artifact_grid)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "timeframe": self.timeframe,
            "target_grid": dict(self.target_grid),
            "artifact_grid": dict(self.artifact_grid),
            "requested_grid": self.requested_grid.as_mapping(),
            "resolved_tp_indexes": list(self.resolved_tp_indexes),
            "resolved_sl_indexes": list(self.resolved_sl_indexes),
        }


@dataclass(frozen=True, slots=True)
class BacktestTpSlGridResolution:
    requested_grid: BacktestTpSlRequestedGrid
    tp_indexes: np.ndarray
    sl_indexes: np.ndarray
    tp_values: np.ndarray
    sl_values: np.ndarray
    evidence: BacktestTpSlGridEvidence

    def __post_init__(self) -> None:
        _ensure_i32_vector(self.tp_indexes, field_name="tp_indexes")
        _ensure_i32_vector(self.sl_indexes, field_name="sl_indexes")
        _ensure_f32_vector(self.tp_values, field_name="tp_values")
        _ensure_f32_vector(self.sl_values, field_name="sl_values")


@dataclass(frozen=True, slots=True)
class BacktestTpSlHitTimesSubset:
    """
    Per-request contiguous hit-time table subset for Iteration 6 kernels.
    """

    tp_values: np.ndarray
    sl_values: np.ndarray
    long_tp: np.ndarray
    long_sl: np.ndarray
    short_tp: np.ndarray
    short_sl: np.ndarray
    sentinel_index: int

    def __post_init__(self) -> None:
        if self.sentinel_index < 0:
            raise ValueError("sentinel_index must be >= 0")
        _ensure_f32_vector(self.tp_values, field_name="tp_values")
        _ensure_f32_vector(self.sl_values, field_name="sl_values")
        for name, array in (
            ("long_tp", self.long_tp),
            ("long_sl", self.long_sl),
            ("short_tp", self.short_tp),
            ("short_sl", self.short_sl),
        ):
            _ensure_u32_matrix(array, field_name=name)
            if not array.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
            if int(array.shape[1]) != self.sentinel_index:
                raise ValueError(f"{name} width must match sentinel_index")
            if int(array.size) and int(np.max(array)) > self.sentinel_index:
                raise ValueError(f"{name} values must be <= sentinel_index")
        if self.long_tp.shape[0] != self.tp_values.shape[0]:
            raise ValueError("long_tp rows must match tp_values")
        if self.short_tp.shape[0] != self.tp_values.shape[0]:
            raise ValueError("short_tp rows must match tp_values")
        if self.long_sl.shape[0] != self.sl_values.shape[0]:
            raise ValueError("long_sl rows must match sl_values")
        if self.short_sl.shape[0] != self.sl_values.shape[0]:
            raise ValueError("short_sl rows must match sl_values")

    def compact_mapping(self) -> dict[str, Any]:
        return {
            "tp_values_shape": list(self.tp_values.shape),
            "sl_values_shape": list(self.sl_values.shape),
            "long_tp_shape": list(self.long_tp.shape),
            "long_sl_shape": list(self.long_sl.shape),
            "short_tp_shape": list(self.short_tp.shape),
            "short_sl_shape": list(self.short_sl.shape),
            "dtype": "uint32",
            "grid_dtype": "float32",
            "sentinel_index": self.sentinel_index,
            "contiguous": True,
        }


@dataclass(frozen=True, slots=True)
class BacktestTpSlHitTimesTiming:
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
            "wall_time_s": float(self.wall_time_s),
            "subsegments": dict(self.subsegments),
        }


@dataclass(frozen=True, slots=True)
class BacktestTpSlHitTimesCleanupEvidence:
    status: str
    retained_hit_times_grid_arrays: bool
    retained_hit_times_table_arrays: bool
    retained_materialized_subset: bool

    def as_mapping(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "retained_hit_times_grid_arrays": self.retained_hit_times_grid_arrays,
            "retained_hit_times_table_arrays": self.retained_hit_times_table_arrays,
            "retained_materialized_subset": self.retained_materialized_subset,
        }


@dataclass(frozen=True, slots=True)
class BacktestTpSlHitTimesResult:
    hit_times_manifest_hash: str
    resolution: BacktestTpSlGridResolution
    hit_times: BacktestTpSlHitTimesSubset
    timing: BacktestTpSlHitTimesTiming
    cleanup_evidence: BacktestTpSlHitTimesCleanupEvidence

    def __post_init__(self) -> None:
        _ensure_sha256(self.hit_times_manifest_hash, field_name="hit_times_manifest_hash")

    def compact_mapping(self) -> dict[str, Any]:
        return {
            "hit_times_manifest_hash": self.hit_times_manifest_hash,
            "grid_evidence": self.resolution.evidence.as_mapping(),
            "hit_times_subset": self.hit_times.compact_mapping(),
            "timing": self.timing.as_mapping(),
            "cleanup_evidence": self.cleanup_evidence.as_mapping(),
        }


def _ensure_sha256(value: str, *, field_name: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be 64 lowercase hex chars")


def _ensure_f32_vector(array: np.ndarray, *, field_name: str) -> None:
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"{field_name} dtype must be float32")
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")


def _ensure_u32_matrix(array: np.ndarray, *, field_name: str) -> None:
    if array.dtype != np.dtype(np.uint32):
        raise ValueError(f"{field_name} dtype must be uint32")
    if array.ndim != 2:
        raise ValueError(f"{field_name} must be two-dimensional")


def _ensure_i32_vector(array: np.ndarray, *, field_name: str) -> None:
    if array.dtype != np.dtype(np.int32):
        raise ValueError(f"{field_name} dtype must be int32")
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")


__all__ = [
    "BacktestTpSlGridEvidence",
    "BacktestTpSlGridResolution",
    "BacktestTpSlHitTimesCleanupEvidence",
    "BacktestTpSlHitTimesGridArrays",
    "BacktestTpSlHitTimesResult",
    "BacktestTpSlHitTimesSubset",
    "BacktestTpSlHitTimesTableArrays",
    "BacktestTpSlHitTimesTiming",
    "BacktestTpSlRequestedGrid",
]
