from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
)
from trading.contexts.backtest.application.ports.artifact_arrays import (
    BacktestArtifactArrayLoader,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactArrayMetadataV2,
    ArtifactCoordinatesV2,
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
