from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
    FilesystemBacktestArtifactContextResolver,
)
from trading.contexts.backtest.application.dto import BacktestCoordinates
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
)


def test_filesystem_artifact_array_loader_mmaps_prices_mappings_and_signals(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    price_arrays_15m = loader.load_price_arrays(context=context, timeframe="15m")
    price_arrays_1m = loader.load_price_arrays(context=context, timeframe="1m")
    mapping_arrays = loader.load_mapping_arrays(context=context, timeframe="15m")
    signal_matrix = loader.load_signal_matrix(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
    )

    assert isinstance(price_arrays_15m.open_time, np.memmap)
    assert isinstance(price_arrays_1m.ohlcv, np.memmap)
    assert isinstance(mapping_arrays.bar_open_1m_idx, np.memmap)
    assert isinstance(signal_matrix.matrix, np.memmap)
    assert price_arrays_15m.open_time.dtype == np.int64
    assert price_arrays_15m.ohlcv.dtype == np.float32
    assert mapping_arrays.bar_open_1m_idx.dtype == np.uint32
    assert signal_matrix.matrix.dtype == np.int8


def test_filesystem_artifact_array_loader_mmaps_funding_arrays(tmp_path: Path) -> None:
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        coordinates=ArtifactCoordinatesV2(
            exchange="binance",
            market_type="futures",
            symbol="BTCUSDT",
        ),
        include_funding=True,
        funding_coverage_status="degraded",
        funding_reason_codes=("funding_interval_gap",),
    )
    context = _resolve_context(
        store=store,
        coordinates=BacktestCoordinates(
            exchange="binance",
            market_type="futures",
            symbol="BTCUSDT",
        ),
    )
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    funding_arrays = loader.load_funding_arrays(context=context)

    assert funding_arrays.coverage_status == "degraded"
    assert funding_arrays.manifest.coverage_policy == "degraded_with_warning"
    assert isinstance(funding_arrays.funding_time, np.memmap)
    assert funding_arrays.funding_time.dtype == np.int64
    assert funding_arrays.funding_rate.dtype == np.float64
    assert funding_arrays.mark_price.dtype == np.float64
    assert funding_arrays.funding_interval_minutes.dtype == np.uint16
    assert funding_arrays.data_quality.dtype == np.uint8


def test_filesystem_artifact_array_loader_copies_contiguous_and_non_contiguous_rows(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    contiguous = loader.load_signal_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_ids=np.asarray([0, 1], dtype=np.int32),
        time_slice=slice(0, 2),
    )
    non_contiguous = loader.load_signal_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_ids=np.asarray([1, 0], dtype=np.int32),
        time_slice=slice(0, 2),
    )

    assert contiguous.flags.c_contiguous
    assert non_contiguous.flags.c_contiguous
    assert contiguous.tolist() == [[-1, 0], [1, 0]]
    assert non_contiguous.tolist() == [[1, 0], [-1, 0]]


def test_filesystem_artifact_array_loader_reports_missing_price_artifact(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    store.builder.price_paths(store.coordinates, store.active_slot, "15m").ohlcv.unlink()
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    with pytest.raises(FileNotFoundError):
        loader.load_price_arrays(context=context, timeframe="15m")


def test_filesystem_artifact_array_loader_reports_missing_mapping_artifact(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    store.builder.mapping_paths(
        store.coordinates,
        store.active_slot,
        "15m",
    ).bar_open_1m_idx.unlink()
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    with pytest.raises(FileNotFoundError):
        loader.load_mapping_arrays(context=context, timeframe="15m")


def test_filesystem_artifact_array_loader_reports_missing_signal_artifact(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    store.builder.signal_paths(
        store.coordinates,
        store.active_slot,
        "15m",
        "ma.ema",
    ).signals.unlink()
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    with pytest.raises(FileNotFoundError):
        loader.load_signal_matrix(
            context=context,
            timeframe="15m",
            indicator_id="ma.ema",
        )


def _resolve_context(
    *,
    store: Any,
    coordinates: BacktestCoordinates | None = None,
):
    effective_coordinates = (
        BacktestCoordinates(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        )
        if coordinates is None
        else coordinates
    )
    resolver = FilesystemBacktestArtifactContextResolver(artifact_loader=store.loader)
    metadata = resolver.resolve_context(
        coordinates=effective_coordinates
    )
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    return loader.resolve_context(
        coordinates=effective_coordinates,
        artifact_metadata=metadata,
    )
