from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from trading.contexts.rl_trading.domain.risk_sizing_policy import (
    RL_RISK_POLICY_MONITOR_ONLY_NO_INTENT,
    RlPolicyDecisionInput,
    RlRiskSizingPolicyConfig,
    RlRiskSizingPolicyKey,
    evaluate_rl_decision_policy,
    validate_rl_risk_sizing_policy,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


def test_stage14_policy_validates_balance_between_risk_inputs_and_synthetic_exits() -> None:
    validation = validate_rl_risk_sizing_policy(config=_valid_config())

    assert validation.ready is True
    assert validation.reasons == ("rl_risk_policy_ready",)
    assert [rule.rule_type for rule in validation.synthetic_exit_rules] == [
        "take_profit",
        "stop_loss",
        "trailing_stop",
    ]
    assert all(rule.platform_side for rule in validation.synthetic_exit_rules)
    assert {rule.creates_intent_action for rule in validation.synthetic_exit_rules} == {"close"}


def test_stage14_policy_fails_closed_when_required_limits_are_missing() -> None:
    validation = validate_rl_risk_sizing_policy(config=RlRiskSizingPolicyConfig())

    assert validation.ready is False
    assert "rl_risk_policy_base_quote_notional_required" in validation.reasons
    assert "rl_risk_policy_stop_loss_required" in validation.reasons
    assert validation.synthetic_exit_rules == ()


def test_stage14_monitor_only_decision_remains_no_intent_even_with_valid_policy() -> None:
    preview = evaluate_rl_decision_policy(
        config=_valid_config(),
        decision=RlPolicyDecisionInput(
            mode="monitor_only",
            action_name="open_long",
            confidence=Decimal("0.95"),
            expected_pnl_pct=Decimal("0.03"),
        ),
    )

    assert preview.status == "no_intent"
    assert preview.reason == RL_RISK_POLICY_MONITOR_ONLY_NO_INTENT
    assert preview.order_type is None
    assert preview.advanced_order_flags == {}


def test_stage14_open_preview_sizes_existing_execution_order_without_native_exit_fields() -> None:
    preview = evaluate_rl_decision_policy(
        config=_valid_config(),
        decision=RlPolicyDecisionInput(
            mode="paper",
            action_name="open_short",
            confidence=Decimal("0.91"),
            expected_pnl_pct=Decimal("0.02"),
        ),
    )

    assert preview.status == "would_create_open_intent"
    assert preview.side == "sell"
    assert preview.order_type == "market"
    assert preview.quote_notional == Decimal("25")
    assert preview.advanced_order_flags == {}
    assert [rule.as_payload()["rule_type"] for rule in preview.synthetic_exit_rules] == [
        "take_profit",
        "stop_loss",
        "trailing_stop",
    ]


def test_stage14_policy_key_is_owner_strategy_ticker_market_scoped() -> None:
    key = RlRiskSizingPolicyKey(
        organization_id=OrganizationId(
            UUID("00000000-0000-4000-8000-000000001400")
        ),
        owner_user_id=UserId(UUID("00000000-0000-0000-0000-000000001401")),
        strategy_id=UUID("00000000-0000-0000-0000-000000001402"),
        exchange_name=" Binance ",
        market_type=" Futures ",
        symbol="btcusdt",
    )

    assert key.persistence_key == (
        "00000000-0000-4000-8000-000000001400",
        "00000000-0000-0000-0000-000000001401",
        "00000000-0000-0000-0000-000000001402",
        "binance",
        "futures",
        "BTCUSDT",
    )


def _valid_config() -> RlRiskSizingPolicyConfig:
    return RlRiskSizingPolicyConfig(
        sizing_method="fixed_quote",
        base_quote_notional=Decimal("25"),
        max_position_notional=Decimal("100"),
        max_daily_loss_notional=Decimal("50"),
        max_drawdown_pct=Decimal("0.10"),
        max_turnover_notional=Decimal("500"),
        max_exposure_notional=Decimal("250"),
        min_expected_pnl_pct=Decimal("0.01"),
        min_confidence=Decimal("0.80"),
        take_profit_pct=Decimal("0.05"),
        stop_loss_pct=Decimal("0.02"),
        trailing_stop_pct=Decimal("0.03"),
    )
