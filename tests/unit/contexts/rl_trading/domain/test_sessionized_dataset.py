from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import numpy as np
import pytest

from trading.contexts.rl_trading.domain import (
    FEATURE_CONTRACT_HASH_V1,
    FEATURE_NAMES_V1,
    SESSIONIZED_ARTICLE_POLICY_ID_V1,
    RawFeatureSlab,
    SessionExtractionPolicy,
    SessionizedDatasetError,
    SessionSplitWindow,
    apply_session_split_embargo_v1,
    article_session_extraction_policy_v1,
    assert_sessionized_trainable_source_v1,
    build_gap_report_v1,
    build_leakage_report_v1,
    build_sessionized_dataset_manifest_v1,
    build_split_artifact_entry_v1,
    materialize_session_features_v1,
    select_article_future_impulse_session_candidates_v1,
    select_high_volatility_session_candidates_v1,
    session_signal_time_array_v1,
    session_split_windows_from_stage04c_v1,
)


def test_stage04c_manifest_yields_split_windows_for_hf_and_post_hf_versions() -> None:
    windows = session_split_windows_from_stage04c_v1(manifest=_stage04c_manifest())

    assert [window.split for window in windows] == ["train", "validation", "post_hf_extension"]
    assert windows[0].dataset_version == "hf_period_rebuild_current_trading"
    assert windows[2].dataset_version == "post_hf_extension_current_trading"


def test_non_binance_futures_sources_are_blocked_for_sessionized_dataset() -> None:
    assert_sessionized_trainable_source_v1(exchange="binance", market_type="futures")

    with pytest.raises(SessionizedDatasetError) as exc_info:
        assert_sessionized_trainable_source_v1(exchange="bybit", market_type="spot")

    assert exc_info.value.reason == "blocked_not_training_source_v1"
    assert exc_info.value.field == "bybit:spot"


def test_high_volatility_selection_uses_only_pre_signal_rows() -> None:
    slab = _slab_with_known_pre_signal_volatility()
    policy = SessionExtractionPolicy(
        signal_stride_minutes=60,
        high_volatility_top_fraction=1.0,
        max_sessions_per_symbol_split=1,
    )
    split_window = SessionSplitWindow(
        dataset_version="hf_period_rebuild_current_trading",
        split="train",
        signal_start_utc="1970-01-01T02:00:00Z",
        signal_end_utc="1970-01-01T03:30:00Z",
        source_start_utc="1970-01-01T00:00:00Z",
        source_end_utc="1970-01-01T04:20:00Z",
    )

    candidates = select_high_volatility_session_candidates_v1(
        slab=slab,
        split_window=split_window,
        symbol="BTCUSDT",
        policy=policy,
    )

    assert len(candidates) == 1
    assert candidates[0].signal_ts_open_ms == 120 * 60_000
    assert candidates[0].score_window_end_ms == candidates[0].signal_ts_open_ms
    assert candidates[0].as_payload()["score_window_end_utc"] == "1970-01-01T02:00:00Z"


def test_session_materialization_shape_metadata_and_gap_report_are_deterministic() -> None:
    slab = _slab_with_known_pre_signal_volatility()
    policy = SessionExtractionPolicy(
        signal_stride_minutes=60,
        high_volatility_top_fraction=1.0,
        max_sessions_per_symbol_split=2,
    )
    split_window = SessionSplitWindow(
        dataset_version="hf_period_rebuild_current_trading",
        split="train",
        signal_start_utc="1970-01-01T02:00:00Z",
        signal_end_utc="1970-01-01T03:30:00Z",
        source_start_utc="1970-01-01T00:00:00Z",
        source_end_utc="1970-01-01T04:20:00Z",
    )

    candidates = select_high_volatility_session_candidates_v1(
        slab=slab,
        split_window=split_window,
        symbol="BTCUSDT",
        policy=policy,
    )
    features = materialize_session_features_v1(slab=slab, candidates=candidates, policy=policy)
    signal_times = session_signal_time_array_v1(candidates)

    assert features.dtype == np.float32
    assert features.shape == (2, 150, len(FEATURE_NAMES_V1))
    assert signal_times.tolist() == [120 * 60_000, 180 * 60_000]
    assert build_gap_report_v1(slab=slab)["status"] == "accepted"
    assert candidates[0].session_key().startswith("binance|futures|BTCUSDT|")


def test_leakage_report_allows_within_split_overlap_and_blocks_short_embargo() -> None:
    slab = _slab_with_known_pre_signal_volatility()
    policy = SessionExtractionPolicy(
        signal_stride_minutes=30,
        high_volatility_top_fraction=1.0,
        max_sessions_per_symbol_split=3,
    )
    split_window = SessionSplitWindow(
        dataset_version="hf_period_rebuild_current_trading",
        split="train",
        signal_start_utc="1970-01-01T02:00:00Z",
        signal_end_utc="1970-01-01T03:31:00Z",
        source_start_utc="1970-01-01T00:00:00Z",
        source_end_utc="1970-01-01T04:20:00Z",
    )
    candidates = select_high_volatility_session_candidates_v1(
        slab=slab,
        split_window=split_window,
        symbol="BTCUSDT",
        policy=policy,
    )

    accepted_report = build_leakage_report_v1(
        candidates=candidates,
        split_windows=[split_window],
        policy=policy,
    )
    assert accepted_report["status"] == "accepted"
    assert int(cast(int, accepted_report["within_split_overlap_pairs"])) > 0
    assert accepted_report["cross_split_overlap_violations_count"] == 0

    blocked_report = build_leakage_report_v1(
        candidates=candidates,
        split_windows=[
            split_window,
            SessionSplitWindow(
                dataset_version="hf_period_rebuild_current_trading",
                split="validation",
                signal_start_utc="1970-01-01T04:00:00Z",
                signal_end_utc="1970-01-01T05:00:00Z",
                source_start_utc="1970-01-01T02:30:00Z",
                source_end_utc="1970-01-01T06:00:00Z",
            ),
        ],
        policy=policy,
    )
    assert blocked_report["status"] == "blocked"
    assert blocked_report["embargo_violations_count"] == 1


def test_split_embargo_shifts_right_split_signal_start() -> None:
    policy = SessionExtractionPolicy()
    windows = [
        SessionSplitWindow(
            dataset_version="hf_period_rebuild_current_trading",
            split="validation",
            signal_start_utc="2024-09-01T00:00:00Z",
            signal_end_utc="2024-12-01T00:00:00Z",
            source_start_utc="2024-08-31T22:30:00Z",
            source_end_utc="2024-12-01T01:00:00Z",
        ),
        SessionSplitWindow(
            dataset_version="hf_period_rebuild_current_trading",
            split="test",
            signal_start_utc="2024-12-01T00:00:00Z",
            signal_end_utc="2025-03-01T00:00:00Z",
            source_start_utc="2024-11-30T22:30:00Z",
            source_end_utc="2025-03-01T01:00:00Z",
        ),
    ]

    adjusted = apply_session_split_embargo_v1(windows, policy=policy)
    report = build_leakage_report_v1(candidates=[], split_windows=adjusted, policy=policy)

    assert adjusted[1].signal_start_utc == "2024-12-01T02:30:00Z"
    assert report["embargo_violations_count"] == 0


def test_article_selector_uses_future_10m_event_end_as_signal() -> None:
    slab = _slab_with_article_impulse()
    policy = article_session_extraction_policy_v1()
    split_window = SessionSplitWindow(
        dataset_version="hf_period_rebuild_current_trading",
        split="train",
        signal_start_utc="1970-01-01T02:00:00Z",
        signal_end_utc="1970-01-01T02:01:00Z",
        source_start_utc="1970-01-01T00:00:00Z",
        source_end_utc="1970-01-01T04:20:00Z",
    )

    candidates = select_article_future_impulse_session_candidates_v1(
        slab=slab,
        split_window=split_window,
        symbol="BTCUSDT",
        policy=policy,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    payload = candidate.as_payload()
    assert candidate.signal_ts_open_ms == 120 * 60_000
    assert candidate.score_window_end_ms == candidate.signal_ts_open_ms
    assert payload["selector_id"] == SESSIONIZED_ARTICLE_POLICY_ID_V1
    assert payload["article_event_start_utc"] == "1970-01-01T01:50:00Z"
    assert payload["article_event_end_utc"] == "1970-01-01T02:00:00Z"
    assert float(cast(float, payload["article_event_abs_return"])) >= 0.05


def test_article_selector_contrast_blocks_repeated_prior_impulse() -> None:
    slab = _slab_with_article_impulse(prior_impulse=True)
    policy = article_session_extraction_policy_v1()
    split_window = SessionSplitWindow(
        dataset_version="hf_period_rebuild_current_trading",
        split="train",
        signal_start_utc="1970-01-01T02:00:00Z",
        signal_end_utc="1970-01-01T02:01:00Z",
        source_start_utc="1970-01-01T00:00:00Z",
        source_end_utc="1970-01-01T04:20:00Z",
    )

    candidates = select_article_future_impulse_session_candidates_v1(
        slab=slab,
        split_window=split_window,
        symbol="BTCUSDT",
        policy=policy,
    )

    assert candidates == ()


def test_split_entry_and_dataset_manifest_capture_hashes_policy_and_safety() -> None:
    slab = _slab_with_known_pre_signal_volatility()
    policy = SessionExtractionPolicy(
        signal_stride_minutes=60,
        high_volatility_top_fraction=1.0,
        max_sessions_per_symbol_split=1,
    )
    split_window = SessionSplitWindow(
        dataset_version="hf_period_rebuild_current_trading",
        split="train",
        signal_start_utc="1970-01-01T02:00:00Z",
        signal_end_utc="1970-01-01T03:30:00Z",
        source_start_utc="1970-01-01T00:00:00Z",
        source_end_utc="1970-01-01T04:20:00Z",
    )
    candidates = select_high_volatility_session_candidates_v1(
        slab=slab,
        split_window=split_window,
        symbol="BTCUSDT",
        policy=policy,
    )
    entry = build_split_artifact_entry_v1(
        dataset_version=split_window.dataset_version,
        split=split_window.split,
        symbol="BTCUSDT",
        candidates=candidates,
        artifact_files={
            "features": {"path": "features.f32.npy", "sha256": "a" * 64},
            "signal_time_ms": {"path": "signal_time_ms.i64.npy", "sha256": "b" * 64},
            "metadata": {"path": "metadata.json", "sha256": "c" * 64},
        },
        gap_report=build_gap_report_v1(slab=slab),
        policy=policy,
    )
    leakage_report = build_leakage_report_v1(
        candidates=candidates,
        split_windows=[split_window],
        policy=policy,
    )
    manifest = build_sessionized_dataset_manifest_v1(
        generated_at_utc=datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
        stage04c_manifest_path="/opt/roehub/state/rl_trading/stage04c/manifest.json",
        stage04c_manifest_sha256="d" * 64,
        output_root="/opt/roehub/state/rl_trading/datasets/stage06",
        split_entries=[entry],
        leakage_report=leakage_report,
        build_scope={"scope": "unit_test"},
        policy=policy,
    )

    assert entry["feature_contract_hash"] == FEATURE_CONTRACT_HASH_V1
    assert manifest["status"] == "accepted"
    assert manifest["total_sessions"] == 1
    safety = cast(dict[str, object], manifest["safety"])
    assert safety["score_uses_post_signal_rows"] is False


def _slab_with_known_pre_signal_volatility() -> RawFeatureSlab:
    minutes = 260
    open_time_ms = np.arange(minutes, dtype=np.int64) * 60_000
    close = np.full(minutes, 100.0, dtype=np.float32)
    close[40:61:2] = 108.0
    close[41:62:2] = 92.0
    close[180:240] = np.linspace(100.0, 160.0, 60, dtype=np.float32)
    open_ = close.copy()
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.full(minutes, 10.0, dtype=np.float32)
    trades = np.full(minutes, 5.0, dtype=np.float32)
    features = np.column_stack((open_, high, close, low, close, volume, trades)).astype(np.float32)
    return RawFeatureSlab(
        open_time_ms=open_time_ms,
        close_time_ms=open_time_ms + 60_000,
        features_f32=features,
    )


def _slab_with_article_impulse(*, prior_impulse: bool = False) -> RawFeatureSlab:
    minutes = 260
    open_time_ms = np.arange(minutes, dtype=np.int64) * 60_000
    close = np.full(minutes, 100.0, dtype=np.float32)
    close[110:119] = np.linspace(100.0, 104.5, 9, dtype=np.float32)
    close[119] = 106.0
    if prior_impulse:
        close[30:39] = np.linspace(100.0, 104.5, 9, dtype=np.float32)
        close[39] = 106.0
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.full(minutes, 10.0, dtype=np.float32)
    trades = np.full(minutes, 5.0, dtype=np.float32)
    features = np.column_stack((open_, high, close, low, close, volume, trades)).astype(np.float32)
    return RawFeatureSlab(
        open_time_ms=open_time_ms,
        close_time_ms=open_time_ms + 60_000,
        features_f32=features,
    )


def _stage04c_manifest() -> dict[str, object]:
    return {
        "acceptance_status": "accepted",
        "feature_contract_dependency": {
            "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        },
        "market": "binance:futures",
        "stage": "04C",
        "dataset_versions": [
            {
                "dataset_version": "hf_period_rebuild_current_trading",
                "signal_windows": [
                    {
                        "range_semantics": "half-open UTC [start, end)",
                        "signal_end_utc": "2024-08-31T00:00:00Z",
                        "signal_start_utc": "2020-01-14T00:00:00Z",
                        "source_end_utc": "2024-08-31T01:00:00Z",
                        "source_start_utc": "2020-01-13T22:30:00Z",
                        "split": "train",
                    },
                    {
                        "range_semantics": "half-open UTC [start, end)",
                        "signal_end_utc": "2024-12-01T00:00:00Z",
                        "signal_start_utc": "2024-09-01T00:00:00Z",
                        "source_end_utc": "2024-12-01T01:00:00Z",
                        "source_start_utc": "2024-08-31T22:30:00Z",
                        "split": "validation",
                    },
                ],
                "status": "accepted",
            },
            {
                "dataset_version": "post_hf_extension_current_trading",
                "signal_windows": [
                    {
                        "range_semantics": "half-open UTC [start, end)",
                        "signal_end_utc": "2026-06-21T14:10:00Z",
                        "signal_start_utc": "2025-06-01T00:00:00Z",
                        "source_end_utc": "2026-06-21T15:10:00Z",
                        "source_start_utc": "2025-05-31T22:30:00Z",
                        "split": "post_hf_extension",
                    },
                ],
                "status": "accepted",
            },
        ],
    }
