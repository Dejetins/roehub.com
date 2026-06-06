from __future__ import annotations

import numpy as np

from trading.contexts.backtest.application.dto import (
    PreparedIndicatorPool,
    PreparedIndicatorRowMetadata,
)
from trading.contexts.backtest.application.services.v2 import build_signal_segments
from trading.contexts.backtest.application.services.v2.matrix_backend.bitsets import (
    bitset_consensus_row,
    build_runtime_bitset_pack_telemetry,
    pack_signal_matrix,
    unpack_signal_bitsets,
    word_count_for_signal_length,
)


def test_pack_signal_matrix_round_trips_positive_neutral_negative_and_padding() -> None:
    trade_t = np.zeros((2, 65), dtype=np.int8)
    trade_t[0, 0] = 1
    trade_t[0, 63] = -1
    trade_t[0, 64] = 1
    trade_t[1, 1] = -1
    trade_t[1, 64] = -1

    packed = pack_signal_matrix(trade_t)

    assert packed.word_count == 2
    assert packed.pos_bits.dtype == np.dtype("uint64")
    assert packed.neg_bits.dtype == np.dtype("uint64")
    assert int(packed.pos_bits[0, 0]) & 1 == 1
    assert int(packed.neg_bits[0, 0]) & (1 << 63) == 1 << 63
    assert int(packed.pos_bits[0, 1]) & 1 == 1
    assert int(packed.neg_bits[1, 1]) & 1 == 1
    assert np.array_equal(unpack_signal_bitsets(packed), trade_t)

    padding_mask = ((1 << 64) - 1) ^ 1
    assert int(packed.pos_bits[0, 1]) & padding_mask == 0
    assert int(packed.neg_bits[1, 1]) & padding_mask == 0


def test_bitset_consensus_matches_reference_for_long_only_and_reversal_masks() -> None:
    alpha = np.asarray(
        [
            [1, 1, 0, -1, -1],
            [1, 0, 1, -1, 0],
        ],
        dtype=np.int8,
    )
    beta = np.asarray(
        [
            [1, 1, 1, -1, 0],
            [0, 1, 1, -1, -1],
        ],
        dtype=np.int8,
    )

    packed = (pack_signal_matrix(alpha), pack_signal_matrix(beta))

    assert bitset_consensus_row(packed, (0, 0)).tolist() == [1, 1, 0, -1, 0]
    assert bitset_consensus_row(packed, (1, 1)).tolist() == [0, 0, 1, -1, 0]


def test_runtime_bitset_pack_telemetry_is_shadow_compact_and_checks_samples() -> None:
    alpha = np.asarray(
        [
            [1, 0, -1, 1, 0],
            [1, 1, -1, 0, 0],
        ],
        dtype=np.int8,
    )
    beta = np.asarray(
        [
            [1, 0, -1, 0, 0],
            [0, 1, -1, 1, 0],
        ],
        dtype=np.int8,
    )

    telemetry = build_runtime_bitset_pack_telemetry(
        (
            _pool("alpha", alpha, row_ids=(10, 11)),
            _pool("beta", beta, row_ids=(20, 21)),
        )
    )
    mapping = telemetry.as_mapping()

    assert telemetry.rows_after_prefilter == 4
    assert telemetry.word_count == word_count_for_signal_length(5)
    assert telemetry.padding_bits == 59
    assert telemetry.padding_valid is True
    assert telemetry.consensus_sample_count > 0
    assert telemetry.consensus_sample_mismatches == 0
    assert telemetry.consensus_sample_parity is True
    assert telemetry.arrays_released_before_return is True
    assert mapping["word_count_formula"] == "W = ceil(T / 64)"
    assert mapping["arrays"]["pos_bits"]["dtype"] == "uint64"
    assert mapping["arrays"]["neg_bits"]["dtype"] == "uint64"
    assert not hasattr(telemetry, "pos_bits")
    assert not hasattr(telemetry, "neg_bits")


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
