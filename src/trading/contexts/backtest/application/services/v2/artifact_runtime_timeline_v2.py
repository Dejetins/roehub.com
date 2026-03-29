"""Artifact-backed request-timeframe timeline builder for sync, worker, and lazy detail."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from trading.contexts.indicators.application.dto import CandleArrays
from trading.shared_kernel.primitives import (
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)

from .contracts import ArtifactSlotPinnedRuntimeContextV2, BacktestPriceArraysLoaderV2
from .stage_a_shortlist_builder_v2 import compute_target_slice_by_close_time_v2


@dataclass(frozen=True, slots=True)
class BacktestArtifactRuntimeTimelineV2:
    """
    Warmup-inclusive request-timeframe candle timeline derived from pinned artifact prices.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-runs-history-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_runtime_timeline_v2.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """

    candles: CandleArrays
    target_slice: slice
    full_target_slice: slice


class BacktestArtifactTimelineBuilderV2:
    """
    Build warmup-aware request-timeframe candles directly from pinned `prices/<tf>` artifacts.

    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-runs-history-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
    """

    def __init__(self, *, price_arrays_loader: BacktestPriceArraysLoaderV2) -> None:
        """
        Initialize the artifact timeline builder with an explicit price loader.

        Args:
            price_arrays_loader: Loader used to memory-map pinned `prices/<tf>` arrays.
        Returns:
            None.
        Assumptions:
            Constructor wires collaborators only and does not touch artifacts.
        Raises:
            ValueError: If the loader dependency is missing.
        Side Effects:
            None.
        """
        if price_arrays_loader is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "BacktestArtifactTimelineBuilderV2 requires price_arrays_loader"
            )
        self._price_arrays_loader = price_arrays_loader

    def build(
        self,
        *,
        artifact_context: ArtifactSlotPinnedRuntimeContextV2,
        market_id: MarketId,
        symbol: Symbol,
        timeframe: Timeframe,
        requested_time_range: TimeRange,
        warmup_bars: int,
    ) -> BacktestArtifactRuntimeTimelineV2:
        """
        Build warmup-aware request-timeframe candles from pinned artifact prices only.

        Args:
            artifact_context: Slot-pinned runtime context for the current run.
            market_id: Stable market identifier for the run template.
            symbol: Canonical instrument symbol for the run template.
            timeframe: Request timeframe literal.
            requested_time_range: Requested trading window `[Start, End)`.
            warmup_bars: Number of request-timeframe warmup bars retained before target slice.
        Returns:
            BacktestArtifactRuntimeTimelineV2: Warmup-inclusive candles plus local target slice.
        Assumptions:
            Artifact price coverage is already validated by strict loader contracts.
        Raises:
            ValueError: If warmup or artifact coverage contracts are violated.
        Side Effects:
            Memory-maps one `prices/<tf>` artifact family and slices arrays for the run.
        """
        if warmup_bars <= 0:
            raise ValueError("BacktestArtifactTimelineBuilderV2 warmup_bars must be > 0")
        artifact_prices = self._price_arrays_loader.load_price_arrays(
            context=artifact_context,
            timeframe=timeframe.code,
        )
        full_target_slice = compute_target_slice_by_close_time_v2(
            close_time=artifact_prices.close_time,
            target_time_range=requested_time_range,
        )
        if full_target_slice.start is None or full_target_slice.stop is None:
            raise ValueError("artifact target slice must define explicit start/stop bounds")
        if full_target_slice.stop <= full_target_slice.start:
            raise ValueError("artifact-backed runtime found no candles for requested time_range")
        warmup_start = max(0, int(full_target_slice.start) - warmup_bars)
        local_slice = slice(warmup_start, int(full_target_slice.stop))
        local_target_slice = slice(
            int(full_target_slice.start) - warmup_start,
            int(full_target_slice.stop) - warmup_start,
        )
        candles = _artifact_candles_from_price_arrays_v2(
            market_id=market_id,
            symbol=symbol,
            timeframe=timeframe,
            price_open_time=artifact_prices.open_time[local_slice],
            price_close_time=artifact_prices.close_time[local_slice],
            price_ohlcv=artifact_prices.ohlcv[local_slice],
        )
        return BacktestArtifactRuntimeTimelineV2(
            candles=candles,
            target_slice=local_target_slice,
            full_target_slice=full_target_slice,
        )


def _artifact_candles_from_price_arrays_v2(
    *,
    market_id: MarketId,
    symbol: Symbol,
    timeframe: Timeframe,
    price_open_time: np.ndarray,
    price_close_time: np.ndarray,
    price_ohlcv: np.ndarray,
) -> CandleArrays:
    """
    Convert sliced `prices/<tf>` artifact arrays into deterministic `CandleArrays`.

    Args:
        market_id: Stable market identifier for the run template.
        symbol: Canonical instrument symbol for the run template.
        timeframe: Request timeframe of the artifact price family.
        price_open_time: Sliced artifact `open_time` vector.
        price_close_time: Sliced artifact `close_time` vector.
        price_ohlcv: Sliced artifact `ohlcv` matrix.
    Returns:
        CandleArrays: Dense request-timeframe candles aligned to the pinned artifact slice.
    Assumptions:
        `price_ohlcv` columns follow the shipped `[open, high, low, close, volume]` contract.
    Raises:
        ValueError: If one sliced artifact array violates shape expectations.
    Side Effects:
        None.
    """
    normalized_open_time = np.asarray(price_open_time, dtype=np.int64)
    normalized_close_time = np.asarray(price_close_time, dtype=np.int64)
    normalized_ohlcv = np.asarray(price_ohlcv, dtype=np.float64)
    if normalized_open_time.ndim != 1:
        raise ValueError("artifact open_time must be a 1D array")
    if normalized_close_time.ndim != 1:
        raise ValueError("artifact close_time must be a 1D array")
    if normalized_ohlcv.ndim != 2 or normalized_ohlcv.shape[1] != 5:
        raise ValueError("artifact ohlcv must be a 2D array with five OHLCV columns")
    if normalized_open_time.shape[0] != normalized_close_time.shape[0]:
        raise ValueError("artifact open_time and close_time lengths must match")
    if normalized_open_time.shape[0] != normalized_ohlcv.shape[0]:
        raise ValueError("artifact time vectors must match ohlcv row count")
    return CandleArrays(
        market_id=market_id,
        symbol=symbol,
        time_range=TimeRange(
            start=_utc_timestamp_from_epoch_millis_v2(int(normalized_open_time[0])),
            end=_utc_timestamp_from_epoch_millis_v2(int(normalized_close_time[-1])),
        ),
        timeframe=timeframe,
        ts_open=np.ascontiguousarray(normalized_open_time, dtype=np.int64),
        open=np.ascontiguousarray(normalized_ohlcv[:, 0], dtype=np.float32),
        high=np.ascontiguousarray(normalized_ohlcv[:, 1], dtype=np.float32),
        low=np.ascontiguousarray(normalized_ohlcv[:, 2], dtype=np.float32),
        close=np.ascontiguousarray(normalized_ohlcv[:, 3], dtype=np.float32),
        volume=np.ascontiguousarray(normalized_ohlcv[:, 4], dtype=np.float32),
    )


def _utc_timestamp_from_epoch_millis_v2(value: int) -> UtcTimestamp:
    """
    Convert epoch-millis integer into timezone-aware UTC timestamp primitive.

    Args:
        value: Epoch milliseconds literal from artifact price arrays.
    Returns:
        UtcTimestamp: UTC timestamp wrapper used by `TimeRange`.
    Assumptions:
        Artifact timelines are stored in UTC epoch milliseconds.
    Raises:
        None.
    Side Effects:
        None.
    """
    return UtcTimestamp(datetime.fromtimestamp(value / 1000.0, tz=timezone.utc))


__all__ = [
    "BacktestArtifactRuntimeTimelineV2",
    "BacktestArtifactTimelineBuilderV2",
]
