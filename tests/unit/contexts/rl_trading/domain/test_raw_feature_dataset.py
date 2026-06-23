from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from scripts.rl_trading.stage05_roehub_dataset_builder import _existing_slab_entry_if_resumable
from trading.contexts.rl_trading.domain import (
    BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
    FEATURE_CONTRACT_HASH_V1,
    FEATURE_NAMES_V1,
    RawFeatureCandleBatch,
    RawFeatureDatasetError,
    RawFeatureSourceWindow,
    assert_trainable_source_v1,
    build_golden_feature_parity_fixture_v1,
    build_raw_feature_dataset_manifest_v1,
    build_raw_feature_slab_manifest_entry_v1,
    build_raw_feature_slab_v1,
    feature_stats_payload_v1,
    hash_raw_feature_slab_payload_v1,
    raw_feature_source_windows_from_stage04c_v1,
    render_raw_feature_json_payload_v1,
    training_source_gate_payload_v1,
)


def test_training_source_gate_blocks_non_v1_training_sources() -> None:
    assert training_source_gate_payload_v1(exchange="binance", market_type="futures")[
        "status"
    ] == "trainable"

    for exchange, market_type in (
        ("binance", "spot"),
        ("bybit", "spot"),
        ("bybit", "futures"),
    ):
        gate = training_source_gate_payload_v1(exchange=exchange, market_type=market_type)
        assert gate["status"] == BLOCKED_NOT_TRAINING_SOURCE_REASON_V1
        with pytest.raises(RawFeatureDatasetError) as exc_info:
            assert_trainable_source_v1(exchange=exchange, market_type=market_type)
        assert exc_info.value.reason == BLOCKED_NOT_TRAINING_SOURCE_REASON_V1


def test_stage04c_manifest_yields_only_accepted_included_feature_windows() -> None:
    windows = raw_feature_source_windows_from_stage04c_v1(
        manifest=_stage04c_manifest(),
        dataset_versions=["post_hf_extension_current_trading"],
        symbols=["BTCUSDT"],
    )

    assert windows == (
        RawFeatureSourceWindow(
            dataset_version="post_hf_extension_current_trading",
            symbol="BTCUSDT",
            market_id=2,
            source_start_utc="2025-05-31T22:30:00Z",
            source_end_utc="2025-05-31T22:33:00Z",
            expected_minutes=3,
        ),
    )


def test_raw_feature_slab_uses_article_order_stats_and_deterministic_hash() -> None:
    batch = _batch()
    slab = build_raw_feature_slab_v1(batch)

    assert slab.features_f32.dtype == np.float32
    assert slab.features_f32.shape == (3, 7)
    assert FEATURE_NAMES_V1 == (
        "open",
        "high",
        "volume_weighted_average",
        "low",
        "close",
        "volume",
        "num_trades",
    )
    np.testing.assert_allclose(
        slab.features_f32,
        np.array(
            [
                [100.0, 105.0, 102.0, 99.0, 101.0, 10.0, 12.0],
                [101.0, 104.0, 101.5, 100.0, 101.5, 0.0, 0.0],
                [102.0, 106.0, float(np.float32(1000.0 / 3.0)), 101.0, 104.0, 3.0, 7.0],
            ],
            dtype=np.float32,
        ),
    )

    first_hash = hash_raw_feature_slab_payload_v1(slab)
    second_hash = hash_raw_feature_slab_payload_v1(build_raw_feature_slab_v1(batch))
    assert first_hash == second_hash
    assert len(first_hash) == 64

    stats = feature_stats_payload_v1(slab.features_f32)
    assert [row["feature"] for row in stats] == list(FEATURE_NAMES_V1)
    assert stats[0]["min"] == 100.0
    assert stats[6]["max"] == 12.0


def test_manifest_entry_and_parity_fixture_are_deterministic_and_shared_builder_based() -> None:
    window = RawFeatureSourceWindow(
        dataset_version="post_hf_extension_current_trading",
        symbol="BTCUSDT",
        market_id=2,
        source_start_utc="2025-05-31T22:30:00Z",
        source_end_utc="2025-05-31T22:33:00Z",
        expected_minutes=3,
    )
    batch = _batch()
    slab = build_raw_feature_slab_v1(batch)
    entry = build_raw_feature_slab_manifest_entry_v1(
        source_window=window,
        slab=slab,
        feature_stats=feature_stats_payload_v1(slab.features_f32),
        artifact_files={
            "features": {"path": "features.f32.npy", "sha256": "a" * 64},
            "open_time_ms": {"path": "open_time_ms.i64.npy", "sha256": "b" * 64},
            "close_time_ms": {"path": "close_time_ms.i64.npy", "sha256": "c" * 64},
        },
    )
    fixture = build_golden_feature_parity_fixture_v1(
        source_window=window,
        batch=batch,
        offline_features_f32=slab.features_f32,
        row_indices=(0, 1, 2),
    )
    manifest = build_raw_feature_dataset_manifest_v1(
        generated_at_utc=datetime(2026, 6, 23, 9, 0, tzinfo=UTC),
        stage04c_manifest_path="/opt/roehub/state/rl_trading/stage04c/manifest.json",
        stage04c_manifest_sha256="d" * 64,
        output_root="/opt/roehub/state/rl_trading/datasets/stage05",
        slab_entries=[entry],
        parity_fixture=fixture,
        build_scope={"scope": "unit_test"},
    )

    assert manifest["status"] == "accepted"
    assert manifest["total_rows"] == 3
    assert manifest["feature_contract_dependency"]["feature_contract_hash"] == (
        FEATURE_CONTRACT_HASH_V1
    )
    assert fixture["max_abs_diff"] == 0.0
    assert fixture["sample_count"] == 3
    assert fixture["samples"][0]["offline_vector"] == fixture["samples"][0][
        "live_equivalent_vector"
    ]
    assert manifest["blocked_training_sources"] == [
        {
            "exchange": "binance",
            "market_type": "spot",
            "reason": "spot branch is product/execution inventory only for this cycle",
            "status": BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
        },
        {
            "exchange": "bybit",
            "market_type": "spot",
            "reason": (
                "not a v1 training source; no Bybit trades_count enrich or feature-mask branch"
            ),
            "status": BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
        },
        {
            "exchange": "bybit",
            "market_type": "futures",
            "reason": (
                "not a v1 training source; no Bybit trades_count enrich or feature-mask branch"
            ),
            "status": BLOCKED_NOT_TRAINING_SOURCE_REASON_V1,
        },
    ]


def test_missing_feature_fields_fail_closed_before_slab_materialization() -> None:
    with pytest.raises(RawFeatureDatasetError) as exc_info:
        RawFeatureCandleBatch(
            open_time_ms=np.array([1], dtype=np.int64),
            close_time_ms=np.array([2], dtype=np.int64),
            open_f32=np.array([1.0], dtype=np.float32),
            high_f32=np.array([1.0], dtype=np.float32),
            low_f32=np.array([1.0], dtype=np.float32),
            close_f32=np.array([1.0], dtype=np.float32),
            volume_base_f32=np.array([1.0], dtype=np.float32),
            volume_quote_f32=np.array([np.nan], dtype=np.float32),
            trades_count_i64=np.array([1], dtype=np.int64),
        )

    assert exc_info.value.reason == "non_finite_array"
    assert exc_info.value.field == "volume_quote_f32"


def test_existing_slab_manifest_resume_validation_fails_closed(tmp_path: Path) -> None:
    window = RawFeatureSourceWindow(
        dataset_version="post_hf_extension_current_trading",
        symbol="BTCUSDT",
        market_id=2,
        source_start_utc="2025-05-31T22:30:00Z",
        source_end_utc="2025-05-31T22:33:00Z",
        expected_minutes=3,
    )
    batch = _batch()
    slab = build_raw_feature_slab_v1(batch)
    slab_root = tmp_path / "post_hf_extension_current_trading" / "BTCUSDT"
    slab_root.mkdir(parents=True)
    artifact_files = {
        "features": {"path": str(slab_root / "features.f32.npy"), "sha256": "a" * 64},
        "open_time_ms": {"path": str(slab_root / "open_time_ms.i64.npy"), "sha256": "b" * 64},
        "close_time_ms": {"path": str(slab_root / "close_time_ms.i64.npy"), "sha256": "c" * 64},
    }
    for payload in artifact_files.values():
        Path(str(payload["path"])).write_bytes(b"existing")
    entry = build_raw_feature_slab_manifest_entry_v1(
        source_window=window,
        slab=slab,
        feature_stats=feature_stats_payload_v1(slab.features_f32),
        artifact_files=artifact_files,
    )
    slab_manifest_path = slab_root / "manifest.json"
    entry["manifest_path"] = str(slab_manifest_path)
    slab_manifest_path.write_text(render_raw_feature_json_payload_v1(entry), encoding="utf-8")

    assert _existing_slab_entry_if_resumable(
        slab_manifest_path=slab_manifest_path,
        source_window=window,
    ) == entry

    stale_entry = dict(entry)
    stale_entry["row_count"] = 2
    slab_manifest_path.write_text(
        render_raw_feature_json_payload_v1(stale_entry),
        encoding="utf-8",
    )

    with pytest.raises(RawFeatureDatasetError) as exc_info:
        _existing_slab_entry_if_resumable(
            slab_manifest_path=slab_manifest_path,
            source_window=window,
        )

    assert exc_info.value.reason == "resume_slab_manifest_mismatch"


def _batch() -> RawFeatureCandleBatch:
    return RawFeatureCandleBatch(
        open_time_ms=np.array([0, 60_000, 120_000], dtype=np.int64),
        close_time_ms=np.array([60_000, 120_000, 180_000], dtype=np.int64),
        open_f32=np.array([100.0, 101.0, 102.0], dtype=np.float32),
        high_f32=np.array([105.0, 104.0, 106.0], dtype=np.float32),
        low_f32=np.array([99.0, 100.0, 101.0], dtype=np.float32),
        close_f32=np.array([101.0, 101.5, 104.0], dtype=np.float32),
        volume_base_f32=np.array([10.0, 0.0, 3.0], dtype=np.float32),
        volume_quote_f32=np.array([1020.0, 0.0, 1000.0], dtype=np.float32),
        trades_count_i64=np.array([12, 0, 7], dtype=np.int64),
    )


def _stage04c_manifest() -> dict[str, object]:
    return {
        "acceptance_status": "accepted",
        "feature_contract_dependency": {
            "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        },
        "market": "binance:futures",
        "market_id": 2,
        "stage": "04C",
        "dataset_versions": [
            {
                "dataset_version": "hf_period_rebuild_current_trading",
                "included_symbols": ["BTCUSDT"],
                "status": "accepted",
                "symbol_source_windows": [
                    {
                        "expected_minutes": 3,
                        "safe_source_end_utc": "2020-01-13T22:33:00Z",
                        "safe_source_start_utc": "2020-01-13T22:30:00Z",
                        "status": "accepted",
                        "symbol": "BTCUSDT",
                    },
                    {
                        "expected_minutes": 0,
                        "safe_source_end_utc": "2025-06-01T01:00:00Z",
                        "safe_source_start_utc": "2025-09-17T15:45:00Z",
                        "status": "excluded",
                        "symbol": "NEWUSDT",
                    },
                ],
            },
            {
                "dataset_version": "post_hf_extension_current_trading",
                "included_symbols": ["BTCUSDT", "NEWUSDT"],
                "status": "accepted",
                "symbol_source_windows": [
                    {
                        "expected_minutes": 3,
                        "safe_source_end_utc": "2025-05-31T22:33:00Z",
                        "safe_source_start_utc": "2025-05-31T22:30:00Z",
                        "status": "accepted",
                        "symbol": "BTCUSDT",
                    },
                    {
                        "expected_minutes": 2,
                        "safe_source_end_utc": "2025-09-17T15:47:00Z",
                        "safe_source_start_utc": "2025-09-17T15:45:00Z",
                        "status": "accepted",
                        "symbol": "NEWUSDT",
                    },
                ],
            },
        ],
    }
