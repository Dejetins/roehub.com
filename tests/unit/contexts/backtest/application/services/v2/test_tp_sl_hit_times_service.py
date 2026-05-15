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
from trading.contexts.backtest.application.dto import (
    BacktestCoordinates,
    BacktestTpSlHitTimesGridArrays,
)
from trading.contexts.backtest.application.services.v2 import (
    BACKTEST_ERROR_ARTIFACTS_UNAVAILABLE,
    BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED,
    HIT_TIMES_ARTIFACT_PATH_V2,
    LOAD_HIT_TIMES_STAGE_NAME,
    TP_SL_GRID_VALIDATION_STAGE_NAME,
    BacktestTpSlHitTimesRejected,
    BacktestTpSlHitTimesService,
)


def test_tp_sl_hit_times_service_selects_requested_contiguous_subset(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    service = BacktestTpSlHitTimesService(artifact_array_loader=loader)
    context = _context(store=store, loader=loader)
    metadata = _artifact_metadata(store=store)

    result = service.execute(
        normalized_request=_normalized_request(
            tp_start=2.0,
            tp_stop=2.0,
            sl_start=1.0,
            sl_stop=1.0,
        ),
        context=context,
    )

    assert result.hit_times_manifest_hash == metadata.hit_times_manifest_hash
    assert result.resolution.tp_indexes.tolist() == [1]
    assert result.resolution.sl_indexes.tolist() == [0]
    assert result.hit_times.tp_values.tolist() == pytest.approx([0.02])
    assert result.hit_times.sl_values.tolist() == pytest.approx([0.01])
    assert result.hit_times.long_tp.tolist() == [[1, 2]]
    assert result.hit_times.long_sl.tolist() == [[1, 2]]
    assert result.hit_times.short_tp.tolist() == [[2, 2]]
    assert result.hit_times.short_sl.tolist() == [[1, 2]]
    assert result.hit_times.long_tp.flags.c_contiguous
    assert result.hit_times.long_sl.flags.c_contiguous
    assert result.hit_times.short_tp.flags.c_contiguous
    assert result.hit_times.short_sl.flags.c_contiguous
    assert set(result.timing.subsegments) == {
        LOAD_HIT_TIMES_STAGE_NAME,
        TP_SL_GRID_VALIDATION_STAGE_NAME,
    }
    assert result.resolution.evidence.artifact_path == HIT_TIMES_ARTIFACT_PATH_V2
    compact = result.compact_mapping()
    assert compact["hit_times_subset"]["long_tp_shape"] == [1, 2]
    assert compact["cleanup_evidence"] == {
        "status": "success",
        "retained_hit_times_grid_arrays": False,
        "retained_hit_times_table_arrays": False,
        "retained_materialized_subset": True,
    }


def test_tp_sl_hit_times_service_materializes_disabled_tp_as_never_hit(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    service = BacktestTpSlHitTimesService(artifact_array_loader=loader)

    result = service.execute(
        normalized_request={
            "risk": {
                "mode": "tp_sl_grid",
                "tp": {"enabled": False},
                "sl": {"enabled": True, "start_pct": 1.0, "stop_pct": 1.0, "step_pct": 1.0},
            }
        },
        context=_context(store=store, loader=loader),
    )

    assert result.resolution.requested_grid.tp_enabled is False
    assert result.resolution.tp_indexes.tolist() == []
    assert result.hit_times.tp_values.tolist() == pytest.approx([0.0])
    assert result.hit_times.long_tp.tolist() == [[2, 2]]
    assert result.hit_times.short_tp.tolist() == [[2, 2]]
    assert result.hit_times.sl_values.tolist() == pytest.approx([0.01])


def test_tp_sl_hit_times_service_rejects_missing_tp_before_table_load(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    loader = _CountingHitTimesLoader(
        FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    )
    service = BacktestTpSlHitTimesService(artifact_array_loader=loader)

    with pytest.raises(BacktestTpSlHitTimesRejected) as exc_info:
        service.execute(
            normalized_request=_normalized_request(
                tp_start=3.0,
                tp_stop=3.0,
                sl_start=1.0,
                sl_stop=1.0,
            ),
            context=_context(store=store, loader=loader),
        )

    assert exc_info.value.error_code == BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED
    assert exc_info.value.issues[0].path == "risk.tp"
    assert "3%" in exc_info.value.issues[0].message
    assert loader.counts["load_hit_times_grid_arrays"] == 1
    assert loader.counts["load_hit_times_table_arrays"] == 0
    assert exc_info.value.cleanup_evidence.as_mapping() == {
        "status": "failed_validation",
        "retained_hit_times_grid_arrays": False,
        "retained_hit_times_table_arrays": False,
        "retained_materialized_subset": False,
    }


def test_tp_sl_hit_times_service_rejects_missing_sl_before_table_load(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    loader = _CountingHitTimesLoader(
        FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    )
    service = BacktestTpSlHitTimesService(artifact_array_loader=loader)

    with pytest.raises(BacktestTpSlHitTimesRejected) as exc_info:
        service.execute(
            normalized_request=_normalized_request(
                tp_start=1.0,
                tp_stop=1.0,
                sl_start=3.0,
                sl_stop=3.0,
            ),
            context=_context(store=store, loader=loader),
        )

    assert exc_info.value.error_code == BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED
    assert exc_info.value.issues[0].path == "risk.sl"
    assert "3%" in exc_info.value.issues[0].message
    assert loader.counts["load_hit_times_grid_arrays"] == 1
    assert loader.counts["load_hit_times_table_arrays"] == 0


def test_tp_sl_hit_times_service_rejects_ambiguous_tolerance_match(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    service = BacktestTpSlHitTimesService(artifact_array_loader=loader)
    grid_arrays = loader.load_hit_times_grid_arrays(context=_context(store=store, loader=loader))
    ambiguous_grid = BacktestTpSlHitTimesGridArrays(
        manifest=grid_arrays.manifest,
        manifest_hash=grid_arrays.manifest_hash,
        tp_values=np.asarray([0.02, 0.020000001], dtype=np.float32),
        sl_values=grid_arrays.sl_values,
    )

    with pytest.raises(BacktestTpSlHitTimesRejected) as exc_info:
        service.validate_grid(
            normalized_request=_normalized_request(
                tp_start=2.0,
                tp_stop=2.0,
                sl_start=1.0,
                sl_stop=1.0,
            ),
            grid_arrays=ambiguous_grid,
        )

    assert exc_info.value.error_code == BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED
    assert exc_info.value.issues[0].path == "risk.tp"
    assert "matches multiple" in exc_info.value.issues[0].message
    assert exc_info.value.cleanup_evidence.retained_materialized_subset is False


def test_tp_sl_hit_times_service_wraps_failed_table_load_with_compact_cleanup(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    loader = _FailingTableHitTimesLoader(
        FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    )
    service = BacktestTpSlHitTimesService(artifact_array_loader=loader)

    with pytest.raises(BacktestTpSlHitTimesRejected) as exc_info:
        service.execute(
            normalized_request=_normalized_request(
                tp_start=1.0,
                tp_stop=1.0,
                sl_start=1.0,
                sl_stop=1.0,
            ),
            context=_context(store=store, loader=loader),
        )

    assert exc_info.value.error_code == BACKTEST_ERROR_ARTIFACTS_UNAVAILABLE
    assert exc_info.value.cleanup_evidence.as_mapping() == {
        "status": "failed_load",
        "retained_hit_times_grid_arrays": False,
        "retained_hit_times_table_arrays": False,
        "retained_materialized_subset": False,
    }
    details = exc_info.value.details()
    assert "hit_times_subset" not in details
    assert details["grid_evidence"]["requested_grid"]["cells"] == 1


def _context(*, store: Any, loader: Any) -> Any:
    return loader.resolve_context(
        coordinates=BacktestCoordinates(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        ),
        artifact_metadata=_artifact_metadata(store=store),
    )


def _artifact_metadata(*, store: Any) -> Any:
    return FilesystemBacktestArtifactContextResolver(
        artifact_loader=store.loader,
    ).resolve_context(
        coordinates=BacktestCoordinates(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        )
    )


def _normalized_request(
    *,
    tp_start: float,
    tp_stop: float,
    sl_start: float,
    sl_stop: float,
) -> dict[str, Any]:
    return {
        "risk": {
            "mode": "tp_sl_grid",
            "tp": {"start_pct": tp_start, "stop_pct": tp_stop, "step_pct": 1.0},
            "sl": {"start_pct": sl_start, "stop_pct": sl_stop, "step_pct": 1.0},
        }
    }


class _CountingHitTimesLoader:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.counts = {
            "load_hit_times_grid_arrays": 0,
            "load_hit_times_table_arrays": 0,
        }

    def resolve_context(self, **kwargs: Any) -> Any:
        return self._inner.resolve_context(**kwargs)

    def load_price_arrays(self, **kwargs: Any) -> Any:
        return self._inner.load_price_arrays(**kwargs)

    def load_mapping_arrays(self, **kwargs: Any) -> Any:
        return self._inner.load_mapping_arrays(**kwargs)

    def load_signal_matrix(self, **kwargs: Any) -> Any:
        return self._inner.load_signal_matrix(**kwargs)

    def load_signal_rows(self, **kwargs: Any) -> Any:
        return self._inner.load_signal_rows(**kwargs)

    def load_hit_times_grid_arrays(self, **kwargs: Any) -> Any:
        self.counts["load_hit_times_grid_arrays"] += 1
        return self._inner.load_hit_times_grid_arrays(**kwargs)

    def load_hit_times_table_arrays(self, **kwargs: Any) -> Any:
        self.counts["load_hit_times_table_arrays"] += 1
        return self._inner.load_hit_times_table_arrays(**kwargs)


class _FailingTableHitTimesLoader(_CountingHitTimesLoader):
    def load_hit_times_table_arrays(self, **kwargs: Any) -> Any:
        self.counts["load_hit_times_table_arrays"] += 1
        raise FileNotFoundError("synthetic missing long_tp.u32.npy")
