from __future__ import annotations

import pytest

from trading.contexts.rl_trading.domain import (
    FEATURE_CONTRACT_HASH_V1,
    FEATURE_DTYPE_V1,
    FEATURE_NAMES_V1,
    FeatureContractViolation,
    RlFeatureCandle,
    build_article_feature_vector_v1,
    derive_volume_weighted_average_v1,
    feature_contract_canonical_payload_v1,
    futures_metadata_gate_payload_v1,
    training_source_matrix_payload_v1,
)


def test_feature_contract_hash_and_channel_order_are_stable() -> None:
    """
    Freeze article-compatible feature order, dtype and hash for Stage 02B.
    """
    assert FEATURE_NAMES_V1 == (
        "open",
        "high",
        "volume_weighted_average",
        "low",
        "close",
        "volume",
        "num_trades",
    )
    assert FEATURE_DTYPE_V1 == "float32"
    assert FEATURE_CONTRACT_HASH_V1 == (
        "d2e99786b68482d730494c6aeec72a1e9f40ac225729019fac5c82f96f900be9"
    )
    assert feature_contract_canonical_payload_v1()["live_feed_policy"] == {
        "hot_path_clickhouse_scan": "forbidden",
        "missing_trades_count_behavior": "fail_closed_degraded_or_blocked",
        "redis_required_for_rl": ["volume_quote", "trades_count"],
        "repair_scope": "gap_or_degraded_path_only",
    }


def test_build_article_feature_vector_uses_article_channel_order() -> None:
    """
    Ensure vector construction follows the frozen HF/article-compatible order.
    """
    candle = _candle(volume_base=10.0, volume_quote=1020.0, trades_count=12)

    assert build_article_feature_vector_v1(candle) == (
        100.0,
        105.0,
        102.0,
        99.0,
        101.0,
        10.0,
        12.0,
    )


def test_vwap_zero_volume_policy_uses_close_when_quote_volume_is_zero() -> None:
    """
    Freeze deterministic zero-volume VWAP behavior without introducing a feature mask.
    """
    candle = _candle(volume_base=0.0, volume_quote=0.0, close=101.25)

    assert derive_volume_weighted_average_v1(candle) == 101.25
    assert build_article_feature_vector_v1(candle)[2] == 101.25


@pytest.mark.parametrize(
    ("overrides", "reason", "field"),
    [
        ({"volume_quote": None}, "missing_volume_quote", "volume_quote"),
        (
            {"volume_base": 0.0, "volume_quote": 1.0},
            "inconsistent_zero_base_positive_quote_volume",
            "volume_quote",
        ),
        ({"trades_count": None}, "missing_trades_count", "trades_count"),
        ({"trades_count": -1}, "negative_trades_count", "trades_count"),
    ],
)
def test_feature_contract_fails_closed_for_missing_or_inconsistent_fields(
    *,
    overrides: dict[str, object],
    reason: str,
    field: str,
) -> None:
    """
    Ensure missing critical fields block vector construction instead of masking features.
    """
    with pytest.raises(FeatureContractViolation) as exc_info:
        build_article_feature_vector_v1(_candle(**overrides))

    assert exc_info.value.reason == reason
    assert exc_info.value.field == field


def test_training_source_matrix_blocks_non_v1_training_branches() -> None:
    """
    Freeze Binance Futures as the only v1 training source.
    """
    matrix = {
        (row["exchange"], row["market_type"]): row["status"]
        for row in training_source_matrix_payload_v1()
    }

    assert matrix == {
        ("binance", "futures"): "trainable",
        ("binance", "spot"): "blocked_not_training_source_v1",
        ("bybit", "spot"): "blocked_not_training_source_v1",
        ("bybit", "futures"): "blocked_not_training_source_v1",
    }


def test_futures_metadata_gate_blocks_production_grade_evaluation_until_resolved() -> None:
    """
    Ensure futures metadata gaps remain explicit fail-closed gates.
    """
    gate = futures_metadata_gate_payload_v1()
    requirements = {item["name"]: item["status"] for item in gate["requirements"]}  # type: ignore[index]

    assert gate["activation_behavior"] == (
        "fail_closed_for_production_grade_futures_evaluation_until_resolved"
    )
    assert requirements["funding_rate_history"] == "missing_required_source"
    assert requirements["mark_price_history"] == "missing_required_source"
    assert requirements["index_price_history"] == "missing_required_source"
    assert requirements["leverage_tiers"] == "missing_required_source"
    assert requirements["point_in_time_filters"] == "available_current_snapshot_only"
    assert requirements["fee_policy"] == "assumption_required"
    assert requirements["slippage_policy"] == "assumption_required"
    assert requirements["liquidation_policy"] == "assumption_required"


def _candle(**overrides: object) -> RlFeatureCandle:
    values = {
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 101.0,
        "volume_base": 10.0,
        "volume_quote": 1020.0,
        "trades_count": 12,
    }
    values.update(overrides)
    return RlFeatureCandle(**values)  # type: ignore[arg-type]
