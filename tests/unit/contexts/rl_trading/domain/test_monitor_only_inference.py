from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading.contexts.live_execution.domain import validate_source_event_fields
from trading.contexts.rl_trading.domain import (
    FEATURE_NAMES_V1,
    STAGE13_SOURCE_TYPE_V1,
    STAGE13_STAGE08M_FEATURE_NAMES_V1,
    FeatureContractViolation,
    RlFeatureCandle,
    Stage13DecisionContext,
    Stage13LatencyObservation,
    build_stage13_feature_matrix_v1,
    build_stage13_source_event_payload_v1,
    compare_stage13_train_live_feature_parity_v1,
    feature_window_from_redis_payloads_v1,
    offline_feature_window_from_candles_v1,
    preload_stage13_policy_from_candidate_manifest_v1,
    summarize_stage13_latency_observations_v1,
)
from trading.contexts.rl_trading.domain import monitor_only_inference as mi


def test_redis_and_offline_feature_windows_have_identical_hashes() -> None:
    live = feature_window_from_redis_payloads_v1(
        payloads=_redis_payloads(),
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )
    offline = offline_feature_window_from_candles_v1(
        candles=_feature_candles(),
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
        ts_open_utc=datetime(2026, 7, 3, 12, 0, tzinfo=UTC),
        ts_close_utc=datetime(2026, 7, 3, 12, 2, tzinfo=UTC),
    )

    parity = compare_stage13_train_live_feature_parity_v1(
        live_window=live,
        offline_window=offline,
    )

    assert parity["status"] == "accepted"
    assert parity["max_abs_diff"] == 0.0
    assert parity["live_feature_hash"] == parity["offline_feature_hash"]


def test_missing_redis_required_feature_fails_closed() -> None:
    payload = dict(_redis_payloads()[0])
    payload["trades_count"] = ""
    window = feature_window_from_redis_payloads_v1(
        payloads=[payload, _redis_payloads()[1]],
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )

    with pytest.raises(FeatureContractViolation, match="missing_trades_count"):
        build_stage13_feature_matrix_v1(window)


def test_preloaded_policy_decides_without_reloading_model_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = feature_window_from_redis_payloads_v1(
        payloads=_redis_payloads(),
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )
    feature_matrix, feature_hash = build_stage13_feature_matrix_v1(window)
    manifest_hash = "b" * 64
    monkeypatch.setattr(mi, "STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1", manifest_hash)
    policy = preload_stage13_policy_from_candidate_manifest_v1(
        candidate_manifest=_candidate_manifest(feature_count=feature_matrix.size),
        candidate_manifest_sha256=manifest_hash,
        loaded_at_utc=datetime(2026, 7, 3, 12, 3, tzinfo=UTC),
    )

    first = policy.decide(
        feature_matrix=feature_matrix,
        feature_hash=feature_hash,
        window_ts_close_utc=window.ts_close_utc,
    )
    second = policy.decide(
        feature_matrix=feature_matrix,
        feature_hash=feature_hash,
        window_ts_close_utc=window.ts_close_utc,
    )

    assert policy.feature_count == len(STAGE13_STAGE08M_FEATURE_NAMES_V1)
    assert first == second
    assert first.action_name == "hold"
    assert first.window_ts_close_utc == window.ts_close_utc


def test_preloaded_stage08m_policy_supports_intercept_weight_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = feature_window_from_redis_payloads_v1(
        payloads=_redis_payloads(),
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )
    feature_matrix, feature_hash = build_stage13_feature_matrix_v1(window)
    manifest_hash = "d" * 64
    monkeypatch.setattr(mi, "STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1", manifest_hash)
    manifest = _candidate_manifest(feature_count=feature_matrix.size)
    model_state = manifest["model_state"]
    assert isinstance(model_state, dict)
    model_state["weights"] = [[0.0, 1000.0, 0.0], *model_state["weights"]]  # intercept row
    policy = preload_stage13_policy_from_candidate_manifest_v1(
        candidate_manifest=manifest,
        candidate_manifest_sha256=manifest_hash,
        loaded_at_utc=datetime(2026, 7, 3, 12, 3, tzinfo=UTC),
    )

    decision = policy.decide(
        feature_matrix=feature_matrix,
        feature_hash=feature_hash,
        window_ts_close_utc=window.ts_close_utc,
    )

    assert policy.uses_intercept is True
    assert decision.action_name == "open_long"


def test_source_event_payload_is_bounded_and_live_execution_compatible() -> None:
    context = Stage13DecisionContext(
        owner_user_id="00000000-0000-0000-0000-000000013001",
        strategy_id="00000000-0000-0000-0000-000000013101",
        strategy_run_id="00000000-0000-0000-0000-000000013201",
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )
    window = feature_window_from_redis_payloads_v1(
        payloads=_redis_payloads(),
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )
    feature_matrix, feature_hash = build_stage13_feature_matrix_v1(window)
    manifest_hash = "c" * 64
    old_hash = mi.STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1
    mi.STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1 = manifest_hash
    try:
        policy = preload_stage13_policy_from_candidate_manifest_v1(
            candidate_manifest=_candidate_manifest(feature_count=feature_matrix.size),
            candidate_manifest_sha256=manifest_hash,
            loaded_at_utc=datetime(2026, 7, 3, 12, 3, tzinfo=UTC),
        )
    finally:
        mi.STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1 = old_hash
    decision = policy.decide(
        feature_matrix=feature_matrix,
        feature_hash=feature_hash,
        window_ts_close_utc=window.ts_close_utc,
    )

    payload = build_stage13_source_event_payload_v1(context=context, decision=decision)

    assert len(payload.source_ref_json) <= 12
    assert payload.outcome == "no_intent"
    assert payload.outcome_reason == "monitor_only_no_intent"
    assert validate_source_event_fields(
        source_type=STAGE13_SOURCE_TYPE_V1,
        source_event_ref=payload.source_event_ref,
        source_ref_json=payload.source_ref_json,
        strategy_signal_id=None,
    ) == STAGE13_SOURCE_TYPE_V1


def test_latency_summary_reports_segment_p95() -> None:
    summary = summarize_stage13_latency_observations_v1(
        [
            Stage13LatencyObservation(
                candle_close_to_feature_ready_s=0.10,
                feature_to_decision_s=0.01,
                decision_to_source_event_s=0.02,
            ),
            Stage13LatencyObservation(
                candle_close_to_feature_ready_s=0.20,
                feature_to_decision_s=0.02,
                decision_to_source_event_s=0.03,
            ),
        ]
    )

    assert summary["observations"] == 2
    assert summary["p95_seconds"] == {
        "candle_close_to_feature_ready": 0.2,
        "decision_to_source_event": 0.03,
        "feature_to_decision": 0.02,
    }


def test_stage13_feature_matrix_matches_stage08m_aggregate_shape() -> None:
    window = feature_window_from_redis_payloads_v1(
        payloads=_redis_payloads(),
        exchange="binance",
        market_type="futures",
        symbol="BTCUSDT",
        instrument_key="binance:futures:BTCUSDT",
    )

    feature_matrix, _feature_hash = build_stage13_feature_matrix_v1(window)

    assert feature_matrix.shape == (1, len(STAGE13_STAGE08M_FEATURE_NAMES_V1))
    assert len(FEATURE_NAMES_V1) == 7


def _redis_payloads() -> list[dict[str, str]]:
    return [
        {
            "schema_version": "1",
            "instrument_key": "binance:futures:BTCUSDT",
            "ts_open": "2026-07-03T12:00:00Z",
            "ts_close": "2026-07-03T12:01:00Z",
            "open": "100.0",
            "high": "103.0",
            "low": "99.0",
            "close": "102.0",
            "volume_base": "10.0",
            "volume_quote": "1010.0",
            "trades_count": "42",
        },
        {
            "schema_version": "1",
            "instrument_key": "binance:futures:BTCUSDT",
            "ts_open": "2026-07-03T12:01:00Z",
            "ts_close": "2026-07-03T12:02:00Z",
            "open": "102.0",
            "high": "104.0",
            "low": "101.0",
            "close": "103.0",
            "volume_base": "8.0",
            "volume_quote": "824.0",
            "trades_count": "37",
        },
    ]


def _feature_candles() -> tuple[RlFeatureCandle, ...]:
    return (
        RlFeatureCandle(
            open=100.0,
            high=103.0,
            low=99.0,
            close=102.0,
            volume_base=10.0,
            volume_quote=1010.0,
            trades_count=42,
        ),
        RlFeatureCandle(
            open=102.0,
            high=104.0,
            low=101.0,
            close=103.0,
            volume_base=8.0,
            volume_quote=824.0,
            trades_count=37,
        ),
    )


def _candidate_manifest(*, feature_count: int) -> dict[str, object]:
    return {
        "candidate_id": mi.STAGE09_ACCEPTED_CANDIDATE_ID_V1,
        "model_state_hash": "a" * 64,
        "model_state": {
            "feature_count": feature_count,
            "label_order": {"0": "hold", "1": "open_long", "2": "open_short"},
            "scaler_mean": [0.0] * feature_count,
            "scaler_std": [1.0] * feature_count,
            "weights": [[0.0, -0.1, -0.2] for _index in range(feature_count)],
        },
        "policy_name": mi.STAGE09_ACCEPTED_CANDIDATE_POLICY_V1,
        "stage": "08M",
        "stage09_allowed": True,
    }
