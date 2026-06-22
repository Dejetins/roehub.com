from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestTpSlHitTimesGridArrays,
    BacktestTpSlHitTimesTableArrays,
)
from trading.contexts.backtest.application.ports.artifact_arrays import (
    BacktestArtifactArrayLoader,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ARTIFACT_FUNDING_DATA_QUALITY_DTYPE_LITERAL_V2,
    ARTIFACT_FUNDING_INTERVAL_MINUTES_DTYPE_LITERAL_V2,
    ARTIFACT_FUNDING_MARK_PRICE_DTYPE_LITERAL_V2,
    ARTIFACT_FUNDING_RATE_DTYPE_LITERAL_V2,
    ARTIFACT_FUNDING_TIME_DTYPE_LITERAL_V2,
    ArtifactArrayMetadataV2,
    ArtifactCoordinatesV2,
    ArtifactFundingArraysV2,
    ArtifactHitTimesManifestDocumentV2,
    ArtifactMappingArraysV2,
    ArtifactMappingTimeframeManifestV2,
    ArtifactPriceArraysV2,
    ArtifactPriceTimeframeManifestV2,
    ArtifactSignalMatrixV2,
    ArtifactSlotLiteralV2,
    ArtifactSlotPinnedRuntimeContextV2,
    BacktestArtifactLoaderV2,
)


@dataclass(frozen=True, slots=True)
class FilesystemBacktestArtifactArrayLoader(BacktestArtifactArrayLoader):
    """
    Strict filesystem mmap loader for Iteration 2 runtime arrays.

    Service timing records these mmap handle opens under `artifact_array_open`
    while keeping `np.load(..., mmap_mode="r")` validation inside this adapter.
    """

    artifact_loader: BacktestArtifactLoaderV2

    def resolve_context(
        self,
        *,
        coordinates: BacktestCoordinates,
        artifact_metadata: BacktestArtifactMetadata,
    ) -> ArtifactSlotPinnedRuntimeContextV2:
        artifact_coordinates = ArtifactCoordinatesV2(
            exchange=coordinates.exchange,
            market_type=coordinates.market_type,
            symbol=coordinates.symbol,
        )
        slot = cast(ArtifactSlotLiteralV2, artifact_metadata.artifact_slot)
        slot_manifest_path = self.artifact_loader.resolve_slot_manifest_path(
            artifact_coordinates,
            slot,
        )
        manifest_sha256 = _file_sha256_hex(slot_manifest_path)
        if manifest_sha256 != artifact_metadata.artifact_manifest_hash:
            raise ValueError(
                "slot manifest hash does not match preflight artifact metadata; "
                f"got {manifest_sha256!r}, expected "
                f"{artifact_metadata.artifact_manifest_hash!r}"
            )
        slot_manifest = self.artifact_loader.load_manifest_from_path(
            slot_manifest_path,
            slot=slot,
        )
        return ArtifactSlotPinnedRuntimeContextV2(
            coordinates=artifact_coordinates,
            artifact_slot=slot,
            slot_generation=artifact_metadata.artifact_slot_generation,
            artifact_asof_date=artifact_metadata.artifact_asof_date,
            artifact_manifest_hash=artifact_metadata.artifact_manifest_hash,
            slot_root_path=slot_manifest_path.parent,
            slot_manifest_path=slot_manifest_path,
            slot_manifest=slot_manifest,
        )

    def load_price_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
    ) -> ArtifactPriceArraysV2:
        manifest = _price_manifest(context=context, timeframe=timeframe)
        paths = self.artifact_loader.resolve_price_paths(
            context.coordinates,
            context.artifact_slot,
            timeframe,
        )
        open_time = _load_npy_mmap(
            paths.open_time,
            metadata=manifest.open_time,
            expected_dtype=np.dtype(np.int64),
            expected_ndim=1,
        )
        close_time = _load_npy_mmap(
            paths.close_time,
            metadata=manifest.close_time,
            expected_dtype=np.dtype(np.int64),
            expected_ndim=1,
        )
        ohlcv = _load_npy_mmap(
            paths.ohlcv,
            metadata=manifest.ohlcv,
            expected_dtype=np.dtype(np.float32),
            expected_ndim=2,
        )
        if int(ohlcv.shape[1]) != 5:
            raise ValueError(f"{paths.ohlcv} must have five OHLCV fields")
        return ArtifactPriceArraysV2(
            timeframe=manifest.timeframe,
            manifest=manifest,
            open_time=open_time,
            close_time=close_time,
            ohlcv=ohlcv,
        )

    def load_mapping_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
    ) -> ArtifactMappingArraysV2:
        manifest = _mapping_manifest(context=context, timeframe=timeframe)
        paths = self.artifact_loader.resolve_mapping_paths(
            context.coordinates,
            context.artifact_slot,
            timeframe,
        )
        bar_open_1m_idx = _load_npy_mmap(
            paths.bar_open_1m_idx,
            metadata=manifest.bar_open_1m_idx,
            expected_dtype=np.dtype(np.uint32),
            expected_ndim=1,
        )
        bar_close_1m_idx = _load_npy_mmap(
            paths.bar_close_1m_idx,
            metadata=manifest.bar_close_1m_idx,
            expected_dtype=np.dtype(np.uint32),
            expected_ndim=1,
        )
        return ArtifactMappingArraysV2(
            timeframe=manifest.timeframe,
            manifest=manifest,
            bar_open_1m_idx=bar_open_1m_idx,
            bar_close_1m_idx=bar_close_1m_idx,
        )

    def load_funding_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
    ) -> ArtifactFundingArraysV2:
        manifest = context.slot_manifest.funding
        if manifest is None:
            raise ValueError("slot manifest does not declare funding artifacts")
        if manifest.coverage_status not in ("ready", "degraded"):
            raise ValueError(
                "funding arrays are available only for ready/degraded coverage; "
                f"got {manifest.coverage_status!r}"
            )
        paths = self.artifact_loader.resolve_funding_paths(
            context.coordinates,
            context.artifact_slot,
        )
        if (
            manifest.funding_time is None
            or manifest.funding_rate is None
            or manifest.mark_price is None
            or manifest.funding_interval_minutes is None
            or manifest.data_quality is None
        ):
            raise ValueError("funding manifest missing array metadata")
        return ArtifactFundingArraysV2(
            manifest=manifest,
            funding_manifest_hash=manifest.funding_manifest_hash,
            coverage_status=manifest.coverage_status,
            funding_time=_load_npy_mmap(
                paths.funding_time,
                metadata=manifest.funding_time,
                expected_dtype=np.dtype(ARTIFACT_FUNDING_TIME_DTYPE_LITERAL_V2),
                expected_ndim=1,
            ),
            funding_rate=_load_npy_mmap(
                paths.funding_rate,
                metadata=manifest.funding_rate,
                expected_dtype=np.dtype(ARTIFACT_FUNDING_RATE_DTYPE_LITERAL_V2),
                expected_ndim=1,
            ),
            mark_price=_load_npy_mmap(
                paths.mark_price,
                metadata=manifest.mark_price,
                expected_dtype=np.dtype(ARTIFACT_FUNDING_MARK_PRICE_DTYPE_LITERAL_V2),
                expected_ndim=1,
            ),
            funding_interval_minutes=_load_npy_mmap(
                paths.funding_interval_minutes,
                metadata=manifest.funding_interval_minutes,
                expected_dtype=np.dtype(ARTIFACT_FUNDING_INTERVAL_MINUTES_DTYPE_LITERAL_V2),
                expected_ndim=1,
            ),
            data_quality=_load_npy_mmap(
                paths.data_quality,
                metadata=manifest.data_quality,
                expected_dtype=np.dtype(ARTIFACT_FUNDING_DATA_QUALITY_DTYPE_LITERAL_V2),
                expected_ndim=1,
            ),
        )

    def load_signal_matrix(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalMatrixV2:
        manifest = self.artifact_loader.load_signal_manifest(
            context.coordinates,
            context.artifact_slot,
            timeframe,
            indicator_id,
        )
        paths = self.artifact_loader.resolve_signal_paths(
            context.coordinates,
            context.artifact_slot,
            timeframe,
            indicator_id,
        )
        matrix = _load_npy_mmap(
            paths.signals,
            metadata=manifest.signals,
            expected_dtype=np.dtype(np.int8),
            expected_ndim=2,
        )
        if int(matrix.shape[0]) != int(manifest.rows_count):
            raise ValueError(
                f"{paths.signals} row count must match signal manifest rows_count; "
                f"got {matrix.shape[0]!r}, expected {manifest.rows_count!r}"
            )
        return ArtifactSignalMatrixV2(
            timeframe=manifest.timeframe,
            indicator_id=manifest.indicator_id,
            manifest=manifest,
            matrix=matrix,
        )

    def load_signal_rows(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        timeframe: str,
        indicator_id: str,
        row_ids: np.ndarray,
        time_slice: slice,
    ) -> np.ndarray:
        signal_matrix = self.load_signal_matrix(
            context=context,
            timeframe=timeframe,
            indicator_id=indicator_id,
        )
        return copy_signal_rows_i8(
            signal_matrix.matrix,
            row_ids=row_ids,
            time_slice=time_slice,
        )

    def load_hit_times_grid_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
    ) -> BacktestTpSlHitTimesGridArrays:
        manifest = self.artifact_loader.load_hit_times_manifest(
            context.coordinates,
            context.artifact_slot,
        )
        manifest_hash = _verified_hit_times_manifest_hash(context=context, manifest=manifest)
        paths = self.artifact_loader.resolve_hit_times_paths(
            context.coordinates,
            context.artifact_slot,
        )
        if manifest.path != paths.manifest:
            raise ValueError(
                "hit-times manifest path does not match resolved hit_times/15m path; "
                f"got {manifest.path!r}, expected {paths.manifest!r}"
            )
        return BacktestTpSlHitTimesGridArrays(
            manifest=manifest,
            manifest_hash=manifest_hash,
            tp_values=_load_npy_mmap(
                paths.tp_values,
                metadata=manifest.tp_values,
                expected_dtype=np.dtype(np.float32),
                expected_ndim=1,
            ),
            sl_values=_load_npy_mmap(
                paths.sl_values,
                metadata=manifest.sl_values,
                expected_dtype=np.dtype(np.float32),
                expected_ndim=1,
            ),
        )

    def load_hit_times_table_arrays(
        self,
        *,
        context: ArtifactSlotPinnedRuntimeContextV2,
        manifest: ArtifactHitTimesManifestDocumentV2,
    ) -> BacktestTpSlHitTimesTableArrays:
        manifest_hash = _verified_hit_times_manifest_hash(context=context, manifest=manifest)
        paths = self.artifact_loader.resolve_hit_times_paths(
            context.coordinates,
            context.artifact_slot,
        )
        if manifest.path != paths.manifest:
            raise ValueError(
                "hit-times manifest path does not match resolved hit_times/15m path; "
                f"got {manifest.path!r}, expected {paths.manifest!r}"
            )
        return BacktestTpSlHitTimesTableArrays(
            manifest=manifest,
            manifest_hash=manifest_hash,
            long_tp=_load_npy_mmap(
                paths.long_tp,
                metadata=manifest.long_tp.array,
                expected_dtype=np.dtype(np.uint32),
                expected_ndim=2,
            ),
            long_sl=_load_npy_mmap(
                paths.long_sl,
                metadata=manifest.long_sl.array,
                expected_dtype=np.dtype(np.uint32),
                expected_ndim=2,
            ),
            short_tp=_load_npy_mmap(
                paths.short_tp,
                metadata=manifest.short_tp.array,
                expected_dtype=np.dtype(np.uint32),
                expected_ndim=2,
            ),
            short_sl=_load_npy_mmap(
                paths.short_sl,
                metadata=manifest.short_sl.array,
                expected_dtype=np.dtype(np.uint32),
                expected_ndim=2,
            ),
        )


def copy_signal_rows_i8(
    matrix: np.ndarray,
    *,
    row_ids: np.ndarray,
    time_slice: slice,
) -> np.ndarray:
    row_ids_i32 = np.asarray(row_ids, dtype=np.int32)
    if row_ids_i32.ndim != 1 or int(row_ids_i32.size) == 0:
        raise ValueError("row_ids must be a non-empty one-dimensional array")
    if int(row_ids_i32.min()) < 0 or int(row_ids_i32.max()) >= int(matrix.shape[0]):
        raise ValueError(
            "row_ids must be within signal matrix row bounds; "
            f"got min={int(row_ids_i32.min())}, max={int(row_ids_i32.max())}, "
            f"rows={int(matrix.shape[0])}"
        )

    row_selector = _contiguous_row_selector(row_ids_i32)
    selected = matrix[row_selector, time_slice]
    return np.ascontiguousarray(np.asarray(selected, dtype=np.int8))


def _contiguous_row_selector(row_ids: np.ndarray) -> slice | np.ndarray:
    if int(row_ids.size) == 1:
        start = int(row_ids[0])
        return slice(start, start + 1)
    expected = np.arange(int(row_ids[0]), int(row_ids[-1]) + 1, dtype=np.int32)
    if int(expected.size) == int(row_ids.size) and np.array_equal(row_ids, expected):
        return slice(int(row_ids[0]), int(row_ids[-1]) + 1)
    return row_ids


def _price_manifest(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    timeframe: str,
) -> ArtifactPriceTimeframeManifestV2:
    for manifest in context.slot_manifest.prices:
        if manifest.timeframe == timeframe:
            return manifest
    raise FileNotFoundError(f"prices/{timeframe} is not present in slot manifest")


def _mapping_manifest(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    timeframe: str,
) -> ArtifactMappingTimeframeManifestV2:
    for manifest in context.slot_manifest.mappings:
        if manifest.timeframe == timeframe:
            return manifest
    raise FileNotFoundError(f"mappings/{timeframe} is not present in slot manifest")


def _load_npy_mmap(
    path: Path,
    *,
    metadata: ArtifactArrayMetadataV2,
    expected_dtype: np.dtype,
    expected_ndim: int,
) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != expected_dtype:
        raise ValueError(f"{path} dtype must be {expected_dtype.name}; got {array.dtype.name}")
    if array.ndim != expected_ndim:
        raise ValueError(f"{path} ndim must be {expected_ndim}; got {array.ndim}")
    if tuple(int(value) for value in array.shape) != tuple(metadata.shape):
        raise ValueError(
            f"{path} shape must match manifest metadata; got {array.shape!r}, "
            f"expected {metadata.shape!r}"
        )
    return array


def _verified_hit_times_manifest_hash(
    *,
    context: ArtifactSlotPinnedRuntimeContextV2,
    manifest: ArtifactHitTimesManifestDocumentV2,
) -> str:
    expected_hash = context.slot_manifest.hit_times.manifest_sha256
    actual_hash = _file_sha256_hex(manifest.path)
    if actual_hash != expected_hash:
        raise ValueError(
            "hit_times/15m manifest hash does not match root manifest reference; "
            f"got {actual_hash!r}, expected {expected_hash!r}"
        )
    return actual_hash


def _file_sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FilesystemBacktestArtifactArrayLoader",
    "copy_signal_rows_i8",
]
