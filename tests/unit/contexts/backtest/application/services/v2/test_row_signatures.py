from __future__ import annotations

import numpy as np

from trading.contexts.backtest.application.dto import (
    PreparedIndicatorPool,
    PreparedIndicatorRowMetadata,
)
from trading.contexts.backtest.application.services.v2 import build_signal_segments
from trading.contexts.backtest.application.services.v2.matrix_backend.row_signatures import (
    build_row_signature_telemetry,
)


def test_row_signature_telemetry_reports_duplicate_rows_without_pruning() -> None:
    alpha_trade_t = np.asarray(
        [
            [1, 0, -1],
            [1, 0, -1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )
    beta_trade_t = np.asarray(
        [
            [1, 1, 0],
            [-1, -1, 0],
        ],
        dtype=np.int8,
    )

    telemetry = build_row_signature_telemetry(
        (
            _pool("alpha", alpha_trade_t, row_ids=(10, 11, 12)),
            _pool("beta", beta_trade_t, row_ids=(20, 21)),
        )
    )

    assert telemetry.rows_after_prefilter == 5
    assert telemetry.unique_rows_after_dedup == 4
    assert telemetry.duplicate_row_count == 1
    assert telemetry.duplicate_signal_row_ids == {"alpha": (11,), "beta": ()}
    assert telemetry.unique_signal_row_ids == {"alpha": (10, 12), "beta": (20, 21)}
    assert telemetry.row_signature_collision_count == 0
    assert telemetry.candidate_upper_bound_after_row_dedup == 4
    assert telemetry.consensus_signature_mode == "exact_consensus_enumerated"
    assert telemetry.consensus_signature_count == 3

    alpha_pool = telemetry.indicators[0]
    assert alpha_pool.rows_after_prefilter == 3
    assert alpha_pool.unique_rows_after_dedup == 2


def test_row_signature_telemetry_uses_upper_bound_for_large_consensus_space() -> None:
    pools = tuple(
        _pool(
            f"indicator_{indicator_index}",
            np.eye(3, dtype=np.int8),
            row_ids=(0, 1, 2),
        )
        for indicator_index in range(4)
    )

    telemetry = build_row_signature_telemetry(
        pools,
        consensus_signature_enumeration_limit=10,
    )

    assert telemetry.unique_rows_after_dedup == 12
    assert telemetry.candidate_upper_bound_after_row_dedup == 81
    assert telemetry.consensus_signature_count == 81
    assert telemetry.consensus_signature_mode == "upper_bound_unique_row_product"


def _pool(
    indicator_id: str,
    trade_t: np.ndarray,
    *,
    row_ids: tuple[int, ...],
) -> PreparedIndicatorPool:
    row_ids_array = np.asarray(row_ids, dtype=np.int32)
    change_count = np.count_nonzero(np.diff(trade_t, axis=1), axis=1).astype(np.int32)
    return PreparedIndicatorPool(
        indicator_id=indicator_id,
        row_ids=row_ids_array,
        filtered_row_ids=row_ids_array.copy(),
        trade_T=trade_t,
        eval_T=trade_t.copy(),
        segments=build_signal_segments(trade_t, change_count=change_count),
        row_score=np.ones(len(row_ids), dtype=np.float32),
        score_adj=np.ones(len(row_ids), dtype=np.float32),
        nonzero=np.count_nonzero(trade_t, axis=1).astype(np.int32),
        proxy=np.ones(len(row_ids), dtype=np.float32),
        change_count=change_count,
        metadata=tuple(
            PreparedIndicatorRowMetadata(
                indicator_id=indicator_id,
                row_id=row_id,
                source="close",
                window=index + 5,
            )
            for index, row_id in enumerate(row_ids)
        ),
    )
