from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from trading.contexts.rl_trading.adapters.outbound.persistence import (
    InMemoryRlRiskSizingPolicyRepository,
)
from trading.contexts.rl_trading.domain.risk_sizing_policy import (
    RlRiskSizingPolicyConfig,
    RlRiskSizingPolicyKey,
    RlRiskSizingPolicyService,
)
from trading.shared_kernel.primitives import OrganizationId, UserId


def test_stage14_in_memory_policy_repository_persists_validation_and_audit() -> None:
    repository = InMemoryRlRiskSizingPolicyRepository()
    service = RlRiskSizingPolicyService(repository=repository)
    key = _key()

    record = service.upsert_policy(
        key=key,
        config=_valid_config(),
        observed_at=datetime(2026, 7, 3, 12, 0, tzinfo=UTC),
    )

    assert record.policy_id is not None
    assert record.validation.ready is True
    assert repository.audit_event_count(policy_id=record.policy_id) == 1

    loaded = service.get_policy(key=key)
    assert loaded.policy_id == record.policy_id
    assert loaded.validation.reasons == ("rl_risk_policy_ready",)


def test_stage14_service_returns_blocked_default_when_policy_missing() -> None:
    service = RlRiskSizingPolicyService(repository=InMemoryRlRiskSizingPolicyRepository())

    record = service.get_policy(key=_key())

    assert record.policy_id is None
    assert record.validation.ready is False
    assert record.validation.reasons == ("rl_risk_policy_not_configured",)


def _key() -> RlRiskSizingPolicyKey:
    return RlRiskSizingPolicyKey(
        organization_id=OrganizationId(
            UUID("00000000-0000-4000-8000-000000001430")
        ),
        owner_user_id=UserId(UUID("00000000-0000-0000-0000-000000001431")),
        strategy_id=UUID("00000000-0000-0000-0000-000000001432"),
        exchange_name="binance",
        market_type="futures",
        symbol="BTCUSDT",
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
    )
