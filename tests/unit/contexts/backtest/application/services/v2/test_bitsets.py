from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trading.contexts.backtest.application.dto import (
    PreparedIndicatorPool,
    PreparedIndicatorRowMetadata,
)
from trading.contexts.backtest.application.services.v2 import build_signal_segments
from trading.contexts.backtest.application.services.v2.matrix_backend.bitsets import (
    SIDECAR_DUPLICATE_ROW_IDS_FILENAME,
    SIDECAR_MANIFEST_FILENAME,
    SIDECAR_NEG_BITS_FILENAME,
    SIDECAR_POS_BITS_FILENAME,
    SIDECAR_ROW_HASHES_FILENAME,
    SIDECAR_UNIQUE_ROW_IDS_FILENAME,
    bitset_consensus_row,
    build_matrix_sidecar_artifacts,
    build_runtime_bitset_pack_telemetry,
    file_sha256_hex,
    load_matrix_sidecar_artifacts,
    load_or_pack_signal_bitsets,
    pack_signal_matrix,
    signal_row_hashes_u64,
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


def test_matrix_sidecar_artifacts_generate_validate_and_record_duplicate_maps(
    tmp_path: Path,
) -> None:
    signal_matrix = np.asarray(
        [
            [1, 0, -1, 1, 0],
            [1, 0, -1, 1, 0],
            [0, 1, 0, -1, 1],
        ],
        dtype=np.int8,
    )
    source_manifest = tmp_path / "manifest.yaml"
    source_signals = tmp_path / "signals.i8.npy"
    np.save(source_signals, signal_matrix)
    _write_signal_manifest(source_manifest, source_signals=source_signals)

    sidecar = build_matrix_sidecar_artifacts(
        signal_matrix=signal_matrix,
        source_manifest_path=source_manifest,
        source_signals_path=source_signals,
        output_dir=tmp_path / "sidecar" / "ma.ema",
        identity={"symbol": "BTCUSDT", "timeframe": "15m", "indicator_id": "ma.ema"},
    )
    loaded = load_matrix_sidecar_artifacts(tmp_path / "sidecar" / "ma.ema")

    assert (tmp_path / "sidecar" / "ma.ema" / SIDECAR_MANIFEST_FILENAME).is_file()
    assert (tmp_path / "sidecar" / "ma.ema" / SIDECAR_POS_BITS_FILENAME).is_file()
    assert (tmp_path / "sidecar" / "ma.ema" / SIDECAR_NEG_BITS_FILENAME).is_file()
    assert (tmp_path / "sidecar" / "ma.ema" / SIDECAR_ROW_HASHES_FILENAME).is_file()
    assert (tmp_path / "sidecar" / "ma.ema" / SIDECAR_UNIQUE_ROW_IDS_FILENAME).is_file()
    assert (tmp_path / "sidecar" / "ma.ema" / SIDECAR_DUPLICATE_ROW_IDS_FILENAME).is_file()
    assert sidecar.manifest["source_signal_shape"] == [3, 5]
    assert loaded.unique_signal_row_ids.tolist() == [0, 2]
    assert loaded.duplicate_signal_row_ids.tolist() == [1]
    assert loaded.duplicate_unique_signal_row_ids.tolist() == [0]
    assert np.array_equal(loaded.signal_row_hashes, signal_row_hashes_u64(signal_matrix))
    assert np.array_equal(unpack_signal_bitsets(loaded.packed), signal_matrix)


def test_matrix_sidecar_hash_validation_rejects_wrong_source_hash(tmp_path: Path) -> None:
    source_manifest = tmp_path / "manifest.yaml"
    source_signals = tmp_path / "signals.i8.npy"
    np.save(source_signals, np.asarray([[1, 0, -1]], dtype=np.int8))
    _write_signal_manifest(source_manifest, source_signals=source_signals)
    build_matrix_sidecar_artifacts(
        signal_matrix=np.asarray([[1, 0, -1]], dtype=np.int8),
        source_manifest_path=source_manifest,
        source_signals_path=source_signals,
        output_dir=tmp_path / "sidecar" / "ma.ema",
        identity={"indicator_id": "ma.ema"},
    )

    with pytest.raises(ValueError, match="source signals hash mismatch"):
        load_matrix_sidecar_artifacts(
            tmp_path / "sidecar" / "ma.ema",
            expected_source_signals_sha256="0" * 64,
        )


def test_runtime_bitset_telemetry_uses_valid_sidecar_and_reports_load_ms(
    tmp_path: Path,
) -> None:
    signal_matrix = np.asarray(
        [
            [1, 0, -1, 1, 0],
            [0, 1, 0, -1, 1],
            [-1, 0, 1, 0, -1],
        ],
        dtype=np.int8,
    )
    source_manifest = tmp_path / "manifest.yaml"
    source_signals = tmp_path / "signals.i8.npy"
    np.save(source_signals, signal_matrix)
    _write_signal_manifest(source_manifest, source_signals=source_signals)
    build_matrix_sidecar_artifacts(
        signal_matrix=signal_matrix,
        source_manifest_path=source_manifest,
        source_signals_path=source_signals,
        output_dir=tmp_path / "sidecar" / "ma.ema",
        identity={"indicator_id": "ma.ema"},
    )

    telemetry = build_runtime_bitset_pack_telemetry(
        (_pool("ma.ema", signal_matrix[[2, 0]], row_ids=(2, 0)),),
        sidecar_artifact_dir=tmp_path / "sidecar",
        time_slice_start=0,
        time_slice_stop=5,
    )
    mapping = telemetry.as_mapping()

    assert mapping["source"] == "sidecar"
    assert mapping["sidecar_used"] is True
    assert mapping["sidecar_available"] is True
    assert mapping["sidecar_load_ms"] >= 0.0
    assert mapping["sidecar_fallback_reason"] is None
    assert mapping["packed_bytes"] == 32
    assert mapping["consensus_sample_parity"] is True


def test_runtime_bitset_sidecar_fallback_keeps_runtime_pack_when_invalid(
    tmp_path: Path,
) -> None:
    result = load_or_pack_signal_bitsets(
        (_pool("missing", np.asarray([[1, 0, -1]], dtype=np.int8), row_ids=(0,)),),
        sidecar_artifact_dir=tmp_path / "sidecar",
    )

    assert result.sidecar_used is False
    assert result.sidecar_available is True
    assert result.sidecar_load_ms is not None
    assert result.sidecar_fallback_reason is not None
    assert np.array_equal(
        unpack_signal_bitsets(result.packed_by_indicator[0]),
        np.asarray([[1, 0, -1]], dtype=np.int8),
    )


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


def _write_signal_manifest(path: Path, *, source_signals: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "manifest_kind: signal",
                "signals:",
                "  path: signals.i8.npy",
                f"  sha256: {file_sha256_hex(source_signals)}",
                "",
            )
        ),
        encoding="utf-8",
    )
