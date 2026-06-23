from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from trading.contexts.rl_trading.domain import (
    FEATURE_NAMES_V1,
    STAGE08_RESEARCH_FUNDING_MODEL_V1,
    Stage08EvaluationConfig,
    Stage08FixedActionPolicy,
    Stage08SimpleThresholdPolicy,
    build_stage08_evaluation_artifact_v1,
    evaluate_stage08_policy_v1,
    stage08_accounting_parity_fixture_v1,
)


def test_fixed_hold_scorecard_is_deterministic_and_flat_after_costs() -> None:
    features = _session_features(session_count=2, trend=0.01)
    config = Stage08EvaluationConfig()
    scorecard = evaluate_stage08_policy_v1(
        session_features=features,
        symbols=("BTCUSDT", "ETHUSDT"),
        signal_times_utc=("2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z"),
        policy=Stage08FixedActionPolicy(policy_name="hold", action_id=0),
        config=config,
    )
    repeated = evaluate_stage08_policy_v1(
        session_features=features,
        symbols=("BTCUSDT", "ETHUSDT"),
        signal_times_utc=("2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z"),
        policy=Stage08FixedActionPolicy(policy_name="hold", action_id=0),
        config=config,
    )

    assert scorecard["net_pnl_after_costs_quote"] == repeated["net_pnl_after_costs_quote"]
    assert scorecard["action_counts"] == repeated["action_counts"]
    assert scorecard["stability_summary"] == repeated["stability_summary"]
    assert scorecard["net_pnl_after_costs_quote"] == 0.0
    assert scorecard["closed_trades"] == 0
    assert scorecard["action_counts"]["hold"] == 20
    assert scorecard["costs"]["funding_model"] == STAGE08_RESEARCH_FUNDING_MODEL_V1
    assert scorecard["costs"]["funding_policy_status"] == "research_only_approximation"
    assert scorecard["out_of_sample_period"]["start_utc"] == "2026-06-01T00:00:00Z"
    assert scorecard["out_of_sample_period"]["end_utc"] == "2026-06-02T00:00:00Z"
    assert scorecard["stability_summary"]["ticker_count"] == 2


def test_simple_threshold_uses_existing_accounting_and_closes_on_last_step() -> None:
    scorecard = evaluate_stage08_policy_v1(
        session_features=_session_features(session_count=1, trend=0.2),
        symbols=("BTCUSDT",),
        signal_times_utc=("2026-06-01T00:00:00Z",),
        policy=Stage08SimpleThresholdPolicy(threshold_return=0.0001),
        config=Stage08EvaluationConfig(transaction_fee=0.001, slippage=0.0),
    )

    assert scorecard["closed_trades"] == 1
    assert scorecard["profitable_trades"] == 1
    assert scorecard["net_pnl_after_costs_quote"] > 0.0
    assert scorecard["audit_reason_counts"]["last_step_forced_close"] == 1


def test_accounting_parity_fixture_matches_stage02c_reward_contract() -> None:
    fixture = stage08_accounting_parity_fixture_v1()

    assert fixture["passed"] is True
    assert fixture["open_pnl_change"] == -0.1
    assert fixture["closed_pnl_change"] == 9.88011
    assert fixture["observed_total_net_pnl"] == 9.78011


def test_evaluation_artifact_marks_research_save_only_for_positive_candidate() -> None:
    candidate = evaluate_stage08_policy_v1(
        session_features=_session_features(session_count=1, trend=0.2),
        symbols=("BTCUSDT",),
        signal_times_utc=("2026-06-01T00:00:00Z",),
        policy=Stage08SimpleThresholdPolicy(
            policy_name="stage07b_candidate",
            policy_kind="candidate",
            threshold_return=0.0001,
        ),
        config=Stage08EvaluationConfig(),
    )
    baseline = evaluate_stage08_policy_v1(
        session_features=_session_features(session_count=1, trend=0.2),
        symbols=("BTCUSDT",),
        signal_times_utc=("2026-06-01T00:00:00Z",),
        policy=Stage08FixedActionPolicy(policy_name="no_trade", action_id=0),
        config=Stage08EvaluationConfig(),
    )
    artifact = build_stage08_evaluation_artifact_v1(
        generated_at_utc=datetime(2026, 6, 24, 12, 0, tzinfo=UTC),
        candidate_manifest_path="/opt/roehub/state/rl_trading/candidate_manifest.json",
        candidate_manifest_sha256="a" * 64,
        sessionized_manifest_path="/opt/roehub/state/rl_trading/stage06_sessionized_manifest.json",
        sessionized_manifest_sha256="b" * 64,
        selection={"selected_session_count": 1, "selected_symbols": ["BTCUSDT"]},
        scorecards=(candidate, baseline),
        candidate_report={
            "metrics": {
                "train_curve": [{"loss_window_mean": 0.2}],
                "validation_curve": [{"td_mse": 0.3}],
            }
        },
        parity_fixture=stage08_accounting_parity_fixture_v1(),
        config=Stage08EvaluationConfig(),
        code_version={"git_head": "test"},
        artifact_hashes={"scorecards": {"sha256": "c" * 64}},
    )

    assert artifact["research_candidate_save_allowed"] is True
    assert artifact["next_stage_handoff"]["stage09_allowed"] is True
    assert artifact["status"] == "accepted_for_research"
    assert artifact["safety"]["promotion_or_activation"] is False
    assert "evaluation_hash" in artifact


def _session_features(*, session_count: int, trend: float) -> np.ndarray:
    features = np.zeros((session_count, 150, len(FEATURE_NAMES_V1)), dtype=np.float32)
    minute = np.arange(150, dtype=np.float32)
    for session_idx in range(session_count):
        base = 100.0 + float(session_idx)
        close = base + minute * np.float32(trend)
        values = {
            "close": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "num_trades": np.full_like(close, 11.0 + session_idx),
            "open": close - 0.01,
            "volume": np.full_like(close, 25.0),
            "volume_weighted_average": close,
        }
        for name, value in values.items():
            features[session_idx, :, FEATURE_NAMES_V1.index(name)] = value
    return features
