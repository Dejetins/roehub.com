from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

import numpy as np

from trading.contexts.backtest.application.dto import (
    BacktestTpSlGridEvidence,
    BacktestTpSlGridResolution,
    BacktestTpSlHitTimesCleanupEvidence,
    BacktestTpSlHitTimesGridArrays,
    BacktestTpSlHitTimesResult,
    BacktestTpSlHitTimesSubset,
    BacktestTpSlHitTimesTableArrays,
    BacktestTpSlHitTimesTiming,
    BacktestTpSlRequestedGrid,
    BacktestValidationIssue,
)
from trading.contexts.backtest.application.ports.artifact_arrays import (
    BacktestArtifactArrayLoader,
)
from trading.contexts.backtest.application.services.v2.preflight import (
    BACKTEST_ERROR_ARTIFACTS_UNAVAILABLE,
    BACKTEST_ERROR_INVALID_REQUEST,
    BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    HIT_TIMES_TIMEFRAME_LITERAL_V2,
    ArtifactSlotPinnedRuntimeContextV2,
)

LOAD_HIT_TIMES_STAGE_NAME = "load_hit_times"
TP_SL_GRID_VALIDATION_STAGE_NAME = "tp_sl_grid_validation"
HIT_TIMES_ARTIFACT_PATH_V2 = f"hit_times/{HIT_TIMES_TIMEFRAME_LITERAL_V2}"
TARGET_TP_SL_GRID_START_PCT = 2.0
TARGET_TP_SL_GRID_STOP_PCT = 25.0
TARGET_TP_SL_GRID_STEP_PCT = 0.5
DEFAULT_TP_SL_GRID_MATCH_ATOL = 1e-7


class BacktestTpSlHitTimesRejected(ValueError):
    """
    Deterministic rejection for Iteration 5 TP/SL hit-times validation/loading.
    """

    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        issues: Sequence[BacktestValidationIssue],
        cleanup_evidence: BacktestTpSlHitTimesCleanupEvidence,
        timing: BacktestTpSlHitTimesTiming,
        grid_evidence: BacktestTpSlGridEvidence | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.issues = tuple(
            sorted(issues, key=lambda issue: (issue.path, issue.code, issue.message))
        )
        self.cleanup_evidence = cleanup_evidence
        self.timing = timing
        self.grid_evidence = grid_evidence
        self.retryable = retryable

    def details(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "errors": [issue.as_mapping() for issue in self.issues],
            "retryable": self.retryable,
            "cleanup_evidence": self.cleanup_evidence.as_mapping(),
            "timing": self.timing.as_mapping(),
        }
        if self.grid_evidence is not None:
            payload["grid_evidence"] = self.grid_evidence.as_mapping()
        return payload


@dataclass(frozen=True, slots=True)
class BacktestTpSlHitTimesService:
    """
    Artifact-backed TP/SL grid validation and hit-times subset materialization.
    """

    artifact_array_loader: BacktestArtifactArrayLoader
    match_atol: float = DEFAULT_TP_SL_GRID_MATCH_ATOL

    def execute(
        self,
        *,
        normalized_request: Mapping[str, Any],
        context: ArtifactSlotPinnedRuntimeContextV2,
    ) -> BacktestTpSlHitTimesResult:
        subsegments: dict[str, float] = {}
        load_wall_s = 0.0
        total_start = time.perf_counter()

        segment_start = time.perf_counter()
        try:
            grid_arrays = self.artifact_array_loader.load_hit_times_grid_arrays(
                context=context,
            )
        except Exception as error:
            load_wall_s += time.perf_counter() - segment_start
            subsegments[LOAD_HIT_TIMES_STAGE_NAME] = load_wall_s
            raise self._load_rejected(
                message="Required hit_times/15m grid artifacts are unavailable",
                original_error=error,
                timing=BacktestTpSlHitTimesTiming(
                    wall_time_s=time.perf_counter() - total_start,
                    subsegments=subsegments,
                ),
            ) from error
        load_wall_s += time.perf_counter() - segment_start

        validation_start = time.perf_counter()
        try:
            resolution = self.validate_grid(
                normalized_request=normalized_request,
                grid_arrays=grid_arrays,
            )
        except BacktestTpSlHitTimesRejected as error:
            subsegments[TP_SL_GRID_VALIDATION_STAGE_NAME] = (
                time.perf_counter() - validation_start
            )
            subsegments[LOAD_HIT_TIMES_STAGE_NAME] = load_wall_s
            raise BacktestTpSlHitTimesRejected(
                error_code=error.error_code,
                message=error.message,
                issues=error.issues,
                cleanup_evidence=error.cleanup_evidence,
                timing=BacktestTpSlHitTimesTiming(
                    wall_time_s=time.perf_counter() - total_start,
                    subsegments=subsegments,
                ),
                grid_evidence=error.grid_evidence,
                retryable=error.retryable,
            ) from error
        subsegments[TP_SL_GRID_VALIDATION_STAGE_NAME] = time.perf_counter() - validation_start

        segment_start = time.perf_counter()
        try:
            table_arrays = self.artifact_array_loader.load_hit_times_table_arrays(
                context=context,
                manifest=grid_arrays.manifest,
            )
            hit_times = self.materialize_subset(
                grid_arrays=grid_arrays,
                table_arrays=table_arrays,
                resolution=resolution,
            )
        except Exception as error:
            load_wall_s += time.perf_counter() - segment_start
            subsegments[LOAD_HIT_TIMES_STAGE_NAME] = load_wall_s
            raise self._load_rejected(
                message="Required hit_times/15m table artifacts are unavailable or invalid",
                original_error=error,
                timing=BacktestTpSlHitTimesTiming(
                    wall_time_s=time.perf_counter() - total_start,
                    subsegments=subsegments,
                ),
                grid_evidence=resolution.evidence,
            ) from error
        load_wall_s += time.perf_counter() - segment_start
        subsegments[LOAD_HIT_TIMES_STAGE_NAME] = load_wall_s

        return BacktestTpSlHitTimesResult(
            hit_times_manifest_hash=grid_arrays.manifest_hash,
            resolution=resolution,
            hit_times=hit_times,
            timing=BacktestTpSlHitTimesTiming(
                wall_time_s=time.perf_counter() - total_start,
                subsegments=subsegments,
            ),
            cleanup_evidence=BacktestTpSlHitTimesCleanupEvidence(
                status="success",
                retained_hit_times_grid_arrays=False,
                retained_hit_times_table_arrays=False,
                retained_materialized_subset=True,
            ),
        )

    def validate_grid(
        self,
        *,
        normalized_request: Mapping[str, Any],
        grid_arrays: BacktestTpSlHitTimesGridArrays,
    ) -> BacktestTpSlGridResolution:
        requested_grid = _requested_grid_from_normalized(normalized_request)
        tp_match = _resolve_level_indexes(
            requested_pct=requested_grid.tp_levels_pct,
            artifact_levels=grid_arrays.tp_values,
            axis="tp",
            tolerance=self.match_atol,
        )
        sl_match = _resolve_level_indexes(
            requested_pct=requested_grid.sl_levels_pct,
            artifact_levels=grid_arrays.sl_values,
            axis="sl",
            tolerance=self.match_atol,
        )
        evidence = _grid_evidence(
            grid_arrays=grid_arrays,
            requested_grid=requested_grid,
            tp_indexes=tp_match.indexes,
            sl_indexes=sl_match.indexes,
            tolerance=self.match_atol,
        )
        issues = (*tp_match.issues, *sl_match.issues)
        if issues:
            raise BacktestTpSlHitTimesRejected(
                error_code=BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED,
                message="Requested TP/SL grid is not covered by published hit_times/15m",
                issues=issues,
                cleanup_evidence=BacktestTpSlHitTimesCleanupEvidence(
                    status="failed_validation",
                    retained_hit_times_grid_arrays=False,
                    retained_hit_times_table_arrays=False,
                    retained_materialized_subset=False,
                ),
                timing=BacktestTpSlHitTimesTiming(wall_time_s=0.0, subsegments={}),
                grid_evidence=evidence,
                retryable=False,
            )

        tp_indexes = np.ascontiguousarray(np.asarray(tp_match.indexes, dtype=np.int32))
        sl_indexes = np.ascontiguousarray(np.asarray(sl_match.indexes, dtype=np.int32))
        tp_values = np.ascontiguousarray(grid_arrays.tp_values[tp_indexes], dtype=np.float32)
        sl_values = np.ascontiguousarray(grid_arrays.sl_values[sl_indexes], dtype=np.float32)
        return BacktestTpSlGridResolution(
            requested_grid=requested_grid,
            tp_indexes=tp_indexes,
            sl_indexes=sl_indexes,
            tp_values=tp_values,
            sl_values=sl_values,
            evidence=evidence,
        )

    def materialize_subset(
        self,
        *,
        grid_arrays: BacktestTpSlHitTimesGridArrays,
        table_arrays: BacktestTpSlHitTimesTableArrays,
        resolution: BacktestTpSlGridResolution,
    ) -> BacktestTpSlHitTimesSubset:
        if table_arrays.manifest_hash != grid_arrays.manifest_hash:
            raise ValueError("hit-times table manifest hash does not match grid manifest hash")
        if table_arrays.manifest.path != grid_arrays.manifest.path:
            raise ValueError("hit-times table manifest path does not match grid manifest path")
        _validate_table_shapes(grid_arrays=grid_arrays, table_arrays=table_arrays)
        tp_indexes = resolution.tp_indexes
        sl_indexes = resolution.sl_indexes
        return BacktestTpSlHitTimesSubset(
            tp_values=np.ascontiguousarray(resolution.tp_values, dtype=np.float32),
            sl_values=np.ascontiguousarray(resolution.sl_values, dtype=np.float32),
            long_tp=np.ascontiguousarray(table_arrays.long_tp[tp_indexes, :], dtype=np.uint32),
            long_sl=np.ascontiguousarray(table_arrays.long_sl[sl_indexes, :], dtype=np.uint32),
            short_tp=np.ascontiguousarray(table_arrays.short_tp[tp_indexes, :], dtype=np.uint32),
            short_sl=np.ascontiguousarray(table_arrays.short_sl[sl_indexes, :], dtype=np.uint32),
            sentinel_index=int(grid_arrays.manifest.sentinel_index),
        )

    def _load_rejected(
        self,
        *,
        message: str,
        original_error: Exception,
        timing: BacktestTpSlHitTimesTiming,
        grid_evidence: BacktestTpSlGridEvidence | None = None,
    ) -> BacktestTpSlHitTimesRejected:
        return BacktestTpSlHitTimesRejected(
            error_code=BACKTEST_ERROR_ARTIFACTS_UNAVAILABLE,
            message=message,
            issues=(
                BacktestValidationIssue(
                    path="artifacts.hit_times",
                    code="hit_times_unavailable",
                    message=str(original_error),
                ),
            ),
            cleanup_evidence=BacktestTpSlHitTimesCleanupEvidence(
                status="failed_load",
                retained_hit_times_grid_arrays=False,
                retained_hit_times_table_arrays=False,
                retained_materialized_subset=False,
            ),
            timing=timing,
            grid_evidence=grid_evidence,
            retryable=True,
        )


@dataclass(frozen=True, slots=True)
class _LevelMatchResult:
    indexes: tuple[int, ...]
    issues: tuple[BacktestValidationIssue, ...]


def _requested_grid_from_normalized(
    normalized_request: Mapping[str, Any],
) -> BacktestTpSlRequestedGrid:
    risk = normalized_request.get("risk")
    if not isinstance(risk, Mapping):
        raise _invalid_request("risk", "risk must be a normalized mapping")
    if risk.get("mode") != "tp_sl_grid":
        raise _invalid_request("risk.mode", "risk.mode must be tp_sl_grid")
    return BacktestTpSlRequestedGrid(
        tp_levels_pct=_percent_levels_from_range(risk.get("tp"), path="risk.tp"),
        sl_levels_pct=_percent_levels_from_range(risk.get("sl"), path="risk.sl"),
    )


def _percent_levels_from_range(value: Any, *, path: str) -> tuple[float, ...]:
    if not isinstance(value, Mapping):
        raise _invalid_request(path, f"{path} must be a normalized range mapping")
    start = _positive_decimal(value.get("start_pct"), path=f"{path}.start_pct")
    stop = _positive_decimal(value.get("stop_pct"), path=f"{path}.stop_pct")
    step = _positive_decimal(value.get("step_pct"), path=f"{path}.step_pct")
    if start > stop:
        raise _invalid_request(path, f"{path}.start_pct must be <= {path}.stop_pct")
    levels: list[float] = []
    current = start
    while current <= stop:
        levels.append(float(current))
        current += step
    if not levels:
        raise _invalid_request(path, f"{path} must materialize at least one level")
    return tuple(levels)


def _positive_decimal(value: Any, *, path: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise _invalid_request(path, f"{path} must be numeric")
    try:
        numeric = Decimal(str(value))
    except Exception as error:
        raise _invalid_request(path, f"{path} is invalid") from error
    if numeric <= 0:
        raise _invalid_request(path, f"{path} must be > 0")
    return numeric


def _invalid_request(path: str, message: str) -> BacktestTpSlHitTimesRejected:
    return BacktestTpSlHitTimesRejected(
        error_code=BACKTEST_ERROR_INVALID_REQUEST,
        message=message,
        issues=(BacktestValidationIssue(path=path, code="invalid_request", message=message),),
        cleanup_evidence=BacktestTpSlHitTimesCleanupEvidence(
            status="failed_validation",
            retained_hit_times_grid_arrays=False,
            retained_hit_times_table_arrays=False,
            retained_materialized_subset=False,
        ),
        timing=BacktestTpSlHitTimesTiming(wall_time_s=0.0, subsegments={}),
    )


def _resolve_level_indexes(
    *,
    requested_pct: Sequence[float],
    artifact_levels: np.ndarray,
    axis: str,
    tolerance: float,
) -> _LevelMatchResult:
    issues: list[BacktestValidationIssue] = []
    indexes: list[int] = []
    for level_pct in requested_pct:
        level_decimal = float(Decimal(str(level_pct)) / Decimal("100"))
        matches = np.flatnonzero(
            np.isclose(
                np.asarray(artifact_levels, dtype=np.float32),
                np.float32(level_decimal),
                rtol=0.0,
                atol=tolerance,
            )
        )
        if int(matches.size) == 1:
            indexes.append(int(matches[0]))
            continue
        path = f"risk.{axis}"
        if int(matches.size) == 0:
            issues.append(
                BacktestValidationIssue(
                    path=path,
                    code="tp_sl_grid_not_covered",
                    message=(
                        f"Requested {axis.upper()} level {level_pct:g}% is not covered "
                        f"by {HIT_TIMES_ARTIFACT_PATH_V2}"
                    ),
                )
            )
            continue
        issues.append(
            BacktestValidationIssue(
                path=path,
                code="tp_sl_grid_not_covered",
                message=(
                    f"Requested {axis.upper()} level {level_pct:g}% matches multiple "
                    f"{HIT_TIMES_ARTIFACT_PATH_V2} artifact levels"
                ),
            )
        )
    return _LevelMatchResult(indexes=tuple(indexes), issues=tuple(issues))


def _grid_evidence(
    *,
    grid_arrays: BacktestTpSlHitTimesGridArrays,
    requested_grid: BacktestTpSlRequestedGrid,
    tp_indexes: Sequence[int],
    sl_indexes: Sequence[int],
    tolerance: float,
) -> BacktestTpSlGridEvidence:
    return BacktestTpSlGridEvidence(
        artifact_path=HIT_TIMES_ARTIFACT_PATH_V2,
        timeframe=grid_arrays.manifest.timeframe,
        target_grid={
            "start_pct": TARGET_TP_SL_GRID_START_PCT,
            "stop_pct": TARGET_TP_SL_GRID_STOP_PCT,
            "step_pct": TARGET_TP_SL_GRID_STEP_PCT,
            "covered_by_artifact": _target_grid_covered(
                tp_values=grid_arrays.tp_values,
                sl_values=grid_arrays.sl_values,
                tolerance=tolerance,
            ),
        },
        artifact_grid={
            "tp": _artifact_axis_evidence(grid_arrays.tp_values),
            "sl": _artifact_axis_evidence(grid_arrays.sl_values),
            "match_tolerance_decimal": tolerance,
        },
        requested_grid=requested_grid,
        resolved_tp_indexes=tuple(int(value) for value in tp_indexes),
        resolved_sl_indexes=tuple(int(value) for value in sl_indexes),
    )


def _target_grid_covered(
    *,
    tp_values: np.ndarray,
    sl_values: np.ndarray,
    tolerance: float,
) -> bool:
    target_pct = _materialize_target_pct()
    return all(
        _has_exact_one_match(artifact_levels=tp_values, pct=value, tolerance=tolerance)
        for value in target_pct
    ) and all(
        _has_exact_one_match(artifact_levels=sl_values, pct=value, tolerance=tolerance)
        for value in target_pct
    )


def _materialize_target_pct() -> tuple[float, ...]:
    values: list[float] = []
    current = Decimal(str(TARGET_TP_SL_GRID_START_PCT))
    stop = Decimal(str(TARGET_TP_SL_GRID_STOP_PCT))
    step = Decimal(str(TARGET_TP_SL_GRID_STEP_PCT))
    while current <= stop:
        values.append(float(current))
        current += step
    return tuple(values)


def _has_exact_one_match(
    *,
    artifact_levels: np.ndarray,
    pct: float,
    tolerance: float,
) -> bool:
    decimal_level = np.float32(float(Decimal(str(pct)) / Decimal("100")))
    return int(
        np.flatnonzero(
            np.isclose(
                np.asarray(artifact_levels, dtype=np.float32),
                decimal_level,
                rtol=0.0,
                atol=tolerance,
            )
        ).size
    ) == 1


def _artifact_axis_evidence(values: np.ndarray) -> dict[str, Any]:
    values_pct = np.asarray(values, dtype=np.float32) * np.float32(100.0)
    if int(values_pct.size) == 0:
        return {"count": 0}
    diffs = np.diff(values_pct)
    step_pct = None
    if int(diffs.size) > 0 and np.allclose(diffs, diffs[0], rtol=0.0, atol=1e-6):
        step_pct = round(float(diffs[0]), 6)
    return {
        "count": int(values_pct.size),
        "min_pct": round(float(np.min(values_pct)), 6),
        "max_pct": round(float(np.max(values_pct)), 6),
        "step_pct": step_pct,
    }


def _validate_table_shapes(
    *,
    grid_arrays: BacktestTpSlHitTimesGridArrays,
    table_arrays: BacktestTpSlHitTimesTableArrays,
) -> None:
    sentinel = int(grid_arrays.manifest.sentinel_index)
    expected_tp = (int(grid_arrays.tp_values.shape[0]), sentinel)
    expected_sl = (int(grid_arrays.sl_values.shape[0]), sentinel)
    if tuple(int(value) for value in table_arrays.long_tp.shape) != expected_tp:
        raise ValueError("long_tp shape must match tp_values and sentinel")
    if tuple(int(value) for value in table_arrays.short_tp.shape) != expected_tp:
        raise ValueError("short_tp shape must match tp_values and sentinel")
    if tuple(int(value) for value in table_arrays.long_sl.shape) != expected_sl:
        raise ValueError("long_sl shape must match sl_values and sentinel")
    if tuple(int(value) for value in table_arrays.short_sl.shape) != expected_sl:
        raise ValueError("short_sl shape must match sl_values and sentinel")


__all__ = [
    "DEFAULT_TP_SL_GRID_MATCH_ATOL",
    "HIT_TIMES_ARTIFACT_PATH_V2",
    "LOAD_HIT_TIMES_STAGE_NAME",
    "TARGET_TP_SL_GRID_START_PCT",
    "TARGET_TP_SL_GRID_STEP_PCT",
    "TARGET_TP_SL_GRID_STOP_PCT",
    "TP_SL_GRID_VALIDATION_STAGE_NAME",
    "BacktestTpSlHitTimesRejected",
    "BacktestTpSlHitTimesService",
]
