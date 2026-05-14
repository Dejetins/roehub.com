from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
    FilesystemBacktestArtifactContextResolver,
)
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestPreparePoolsConfig,
)
from trading.contexts.backtest.application.services.v2 import (
    ARTIFACT_ARRAY_MMAP_LOAD_SEGMENT,
    ARTIFACT_ARRAY_OPEN_SEGMENT,
    ARTIFACT_CONTEXT_RESOLVE_SEGMENT,
    ARTIFACT_MANIFEST_LOAD_SEGMENT,
    PREPARE_POOLS_CORE_STAGE_NAME,
    PREPARE_POOLS_TOTAL_STAGE_NAME,
    REQUEST_SLICE_PREPARE_SEGMENT,
    ROW_PREFILTER_SEGMENT,
    SEGMENT_BUILD_SEGMENT,
    SIGNAL_ROW_SELECTION_SEGMENT,
    TIME_RANGE_SLICE_SEGMENT,
    BacktestPreparePoolsRejected,
    BacktestPreparePoolsService,
    build_signal_segments,
    notebook_compatible_prepare_pools_core_s,
    time_range_slice,
)


def test_prepare_pools_prepares_indicator_pool_from_normalized_request(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    service = _service(store=store, top_fraction=1.0)
    metadata = _artifact_metadata(store=store)

    first = service.execute(
        normalized_request=_normalized_request(),
        artifact_metadata=metadata,
    )
    second = service.execute(
        normalized_request=_normalized_request(),
        artifact_metadata=metadata,
    )
    pool = first.indicator_pools[0]

    assert first.timeframe == "15m"
    assert first.indicator_ids == ("ma.ema",)
    assert first.time_slice_start_15m == 0
    assert first.time_slice_stop_15m == 2
    assert first.trade_T_length == 2
    assert first.eval_T_length == 1
    assert first.signal_returns_15m.tolist() == pytest.approx([(1.2 / 1.1) - 1.0])
    assert first.execution_mapping.signal_entry_exec_idx_15m.tolist() == [2, 4]
    assert first.execution_mapping.t_exec_limit_1m == 4

    assert pool.row_ids.tolist() == [0, 1]
    assert pool.trade_T.flags.c_contiguous
    assert pool.trade_T.tolist() == [[-1, 0], [1, 0]]
    assert pool.eval_T.tolist() == [[-1], [1]]
    assert [item.as_mapping() for item in pool.metadata] == [
        {"indicator_id": "ma.ema", "row_id": 0, "source": "close", "window": 5},
        {"indicator_id": "ma.ema", "row_id": 1, "source": "close", "window": 6},
    ]
    assert pool.nonzero.tolist() == [1, 1]
    assert pool.change_count.tolist() == [1, 1]
    assert pool.segments.starts.tolist() == [[0, 1], [0, 1]]
    assert pool.segments.ends.tolist() == [[1, 2], [1, 2]]
    assert pool.segments.values.tolist() == [[-1, 0], [1, 0]]
    assert pool.segments.counts.tolist() == [2, 2]
    assert len(first.row_metadata_order_hash) == 64
    assert first.row_metadata_order_hash == second.row_metadata_order_hash
    assert first.timing.stage_name == PREPARE_POOLS_TOTAL_STAGE_NAME
    assert first.timing.prepare_pools_core_s == first.timing.subsegments[
        PREPARE_POOLS_CORE_STAGE_NAME
    ]
    assert first.timing.prepare_pools_total_s == first.timing.wall_time_s
    assert notebook_compatible_prepare_pools_core_s(first.timing) == first.timing.subsegments[
        PREPARE_POOLS_CORE_STAGE_NAME
    ]
    assert set(first.timing.subsegments) == {
        ARTIFACT_CONTEXT_RESOLVE_SEGMENT,
        ARTIFACT_ARRAY_OPEN_SEGMENT,
        REQUEST_SLICE_PREPARE_SEGMENT,
        ARTIFACT_MANIFEST_LOAD_SEGMENT,
        ARTIFACT_ARRAY_MMAP_LOAD_SEGMENT,
        TIME_RANGE_SLICE_SEGMENT,
        SIGNAL_ROW_SELECTION_SEGMENT,
        ROW_PREFILTER_SEGMENT,
        SEGMENT_BUILD_SEGMENT,
        PREPARE_POOLS_CORE_STAGE_NAME,
        PREPARE_POOLS_TOTAL_STAGE_NAME,
    }


def test_prepare_pools_core_excludes_context_open_and_slice_overhead(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    spy_loader = _CountingArtifactArrayLoader(
        FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    )
    service = _service(store=store, top_fraction=1.0, loader=spy_loader)
    request = _normalized_request()
    coordinates = BacktestCoordinates(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )

    context = service.resolve_artifact_context(
        coordinates=coordinates,
        artifact_metadata=_artifact_metadata(store=store),
    )
    runtime_arrays = service.open_artifact_arrays(
        normalized_request=request,
        context=context,
    )
    request_slice = service.prepare_request_slice(
        normalized_request=request,
        runtime_arrays=runtime_arrays,
    )
    counts_before_core = dict(spy_loader.counts)

    result = service.prepare_pools_core(
        normalized_request=request,
        runtime_arrays=runtime_arrays,
        request_slice=request_slice,
    )

    assert spy_loader.counts == counts_before_core
    assert result.timing.stage_name == PREPARE_POOLS_CORE_STAGE_NAME
    assert result.timing.prepare_pools_core_s == result.timing.wall_time_s
    assert ARTIFACT_CONTEXT_RESOLVE_SEGMENT not in result.timing.subsegments
    assert ARTIFACT_ARRAY_OPEN_SEGMENT not in result.timing.subsegments
    assert REQUEST_SLICE_PREPARE_SEGMENT not in result.timing.subsegments
    assert ARTIFACT_MANIFEST_LOAD_SEGMENT not in result.timing.subsegments
    assert ARTIFACT_ARRAY_MMAP_LOAD_SEGMENT not in result.timing.subsegments
    assert TIME_RANGE_SLICE_SEGMENT not in result.timing.subsegments
    assert set(result.timing.subsegments) == {
        SIGNAL_ROW_SELECTION_SEGMENT,
        ROW_PREFILTER_SEGMENT,
        SEGMENT_BUILD_SEGMENT,
        PREPARE_POOLS_CORE_STAGE_NAME,
    }
    assert result.indicator_pools[0].trade_T.tolist() == [[-1, 0], [1, 0]]


def test_prepare_pools_row_prefilter_keeps_top_adjusted_row(tmp_path: Path) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    service = _service(store=store, top_fraction=0.5)

    result = service.execute(
        normalized_request=_normalized_request(fee_rate=0.0),
        artifact_metadata=_artifact_metadata(store=store),
    )
    pool = result.indicator_pools[0]

    assert pool.row_ids.tolist() == [1]
    assert pool.proxy.tolist() == pytest.approx([(1.2 / 1.1) - 1.0])
    assert pool.row_score.tolist() == pytest.approx([(1.2 / 1.1) - 1.0])
    assert [item.row_id for item in pool.metadata] == [1]


def test_prepare_pools_core_retains_required_variant_rows_for_lazy_detail(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    service = _service(store=store, top_fraction=0.5)
    request = _normalized_request(fee_rate=0.0)
    context = service.resolve_artifact_context(
        coordinates=BacktestCoordinates(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        ),
        artifact_metadata=_artifact_metadata(store=store),
    )
    runtime_arrays = service.open_artifact_arrays(
        normalized_request=request,
        context=context,
    )
    request_slice = service.prepare_request_slice(
        normalized_request=request,
        runtime_arrays=runtime_arrays,
    )

    result = service.prepare_pools_core(
        normalized_request=request,
        runtime_arrays=runtime_arrays,
        request_slice=request_slice,
        required_row_ids_by_indicator={"ma.ema": (0,)},
    )

    pool = result.indicator_pools[0]
    assert pool.row_ids.tolist() == [0, 1]
    assert [item.row_id for item in pool.metadata] == [0, 1]


def test_prepare_pools_rejects_time_range_outside_artifact_coverage(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    request = _normalized_request()
    request["time_range"] = {
        "start": "1970-01-01T00:00:10Z",
        "end": "1970-01-01T00:00:11Z",
    }

    with pytest.raises(BacktestPreparePoolsRejected):
        _service(store=store, top_fraction=1.0).execute(
            normalized_request=request,
            artifact_metadata=_artifact_metadata(store=store),
        )


def test_prepare_pools_rejects_mapping_close_index_outside_1m_coverage(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    mapping_paths = store.builder.mapping_paths(store.coordinates, store.active_slot, "15m")
    with mapping_paths.bar_close_1m_idx.open("wb") as file_handle:
        np.save(file_handle, np.asarray([1, 99], dtype=np.uint32), allow_pickle=False)

    with pytest.raises(BacktestPreparePoolsRejected):
        _service(store=store, top_fraction=1.0).execute(
            normalized_request=_normalized_request(),
            artifact_metadata=_artifact_metadata(store=store),
        )


def test_time_range_slice_uses_half_open_15m_open_time() -> None:
    result = time_range_slice(
        open_time_15m=np.asarray([1000, 3000, 5000], dtype=np.int64),
        close_time_15m=np.asarray([2999, 4999, 6999], dtype=np.int64),
        time_range={
            "start": "1970-01-01T00:00:01Z",
            "end": "1970-01-01T00:00:05Z",
        },
    )

    assert result == slice(0, 2)


def test_build_signal_segments_compresses_change_points() -> None:
    segments = build_signal_segments(np.asarray([[1, 1, 0, -1]], dtype=np.int8))

    assert segments.starts.tolist() == [[0, 2, 3]]
    assert segments.ends.tolist() == [[2, 3, 4]]
    assert segments.values.tolist() == [[1, 0, -1]]
    assert segments.counts.tolist() == [3]
    assert segments.change_count.tolist() == [2]


def _service(
    *,
    store: Any,
    top_fraction: float,
    loader: Any | None = None,
) -> BacktestPreparePoolsService:
    return BacktestPreparePoolsService(
        artifact_array_loader=loader
        or FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader),
        defaults_provider=YamlBacktestGridDefaultsProvider.from_yaml(
            config_path="configs/prod/indicators.yaml",
        ),
        config=BacktestPreparePoolsConfig(row_prefilter_top_fraction=top_fraction),
    )


def _artifact_metadata(*, store: Any) -> BacktestArtifactMetadata:
    resolver = FilesystemBacktestArtifactContextResolver(artifact_loader=store.loader)
    return resolver.resolve_context(
        coordinates=BacktestCoordinates(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        )
    )


def _normalized_request(*, fee_rate: float = 0.00075) -> dict[str, Any]:
    return {
        "coordinates": {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTCUSDT",
        },
        "timeframe": "15m",
        "time_range": {
            "start": "1970-01-01T00:00:01Z",
            "end": "1970-01-01T00:00:04Z",
        },
        "indicators": [
            {
                "indicator_id": "ma.ema",
                "sources": ["close"],
                "window": {"start": 5, "stop": 6, "step": 1},
            }
        ],
        "risk": {"mode": "none"},
        "execution": {
            "direction_mode": "long_short_reversal",
            "fee_rate": fee_rate,
            "slippage_rate": 0.0001,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": "fixed_equity_pct", "equity_pct": 10.0},
            "profit_lock": {"enabled": False},
            "close_on_end": True,
        },
        "ranking": {
            "primary_metric": "total_return_pct",
            "direction": "desc",
        },
        "top_n": 100,
    }


class _CountingArtifactArrayLoader:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.counts = {
            "resolve_context": 0,
            "load_price_arrays": 0,
            "load_mapping_arrays": 0,
            "load_signal_matrix": 0,
            "load_signal_rows": 0,
        }

    def resolve_context(self, **kwargs: Any) -> Any:
        self.counts["resolve_context"] += 1
        return self._inner.resolve_context(**kwargs)

    def load_price_arrays(self, **kwargs: Any) -> Any:
        self.counts["load_price_arrays"] += 1
        return self._inner.load_price_arrays(**kwargs)

    def load_mapping_arrays(self, **kwargs: Any) -> Any:
        self.counts["load_mapping_arrays"] += 1
        return self._inner.load_mapping_arrays(**kwargs)

    def load_signal_matrix(self, **kwargs: Any) -> Any:
        self.counts["load_signal_matrix"] += 1
        return self._inner.load_signal_matrix(**kwargs)

    def load_signal_rows(self, **kwargs: Any) -> Any:
        self.counts["load_signal_rows"] += 1
        return self._inner.load_signal_rows(**kwargs)
