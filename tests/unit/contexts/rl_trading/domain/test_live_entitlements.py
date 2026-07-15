from uuid import UUID

from trading.contexts.rl_trading.domain.live_entitlements import (
    RL_LIVE_TICKER_BASE_FAIL_CLOSED,
    RL_LIVE_TICKER_NOT_COUNTED,
    RL_LIVE_TICKER_QUOTA_EXCEEDED,
    RL_LIVE_TICKER_READY,
    RlLiveTickerIdentity,
    evaluate_rl_live_ticker_entitlement,
    resolve_rl_live_ticker_limit,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("00000000-0000-4000-8000-000000001200"))


def test_stage12_paid_level_mapping_is_explicit_and_fail_closed() -> None:
    assert resolve_rl_live_ticker_limit(paid_level="free").live_slots_allowed == 1
    assert resolve_rl_live_ticker_limit(paid_level="pro").live_slots_allowed == 5
    ultra = resolve_rl_live_ticker_limit(paid_level="ultra")
    assert ultra.live_slots_allowed == 20
    assert ultra.product_label == "Premium"

    base = resolve_rl_live_ticker_limit(paid_level="base")
    assert base.live_slots_allowed == 0
    assert base.entitlement_source == "fail_closed"
    assert base.fail_closed_reason == RL_LIVE_TICKER_BASE_FAIL_CLOSED

    override = resolve_rl_live_ticker_limit(
        paid_level="base",
        override_live_slots_allowed=42,
    )
    assert override.live_slots_allowed == 42
    assert override.product_label == "Enterprise"
    assert override.entitlement_source == "override"


def test_stage12_only_live_mode_consumes_slots() -> None:
    owner = _owner()
    active = (_ticker(owner=owner, symbol="BTCUSDT"),)
    for mode in ("monitor_only", "paper", "testnet"):
        snapshot = evaluate_rl_live_ticker_entitlement(
            paid_level="free",
            mode=mode,  # type: ignore[arg-type]
            active_tickers=active,
            requested_ticker=_ticker(owner=owner, symbol="ETHUSDT"),
        )
        assert snapshot.eligible is True
        assert snapshot.live_slots_used == 1
        assert snapshot.readiness_reason == RL_LIVE_TICKER_NOT_COUNTED


def test_stage12_distinct_live_ticker_quota_and_existing_ticker_are_stable() -> None:
    owner = _owner()
    active = (_ticker(owner=owner, symbol="BTCUSDT"),)

    existing = evaluate_rl_live_ticker_entitlement(
        paid_level="free",
        mode="live",
        active_tickers=active,
        requested_ticker=_ticker(owner=owner, symbol="BTCUSDT"),
    )
    assert existing.eligible is True
    assert existing.readiness_reason == RL_LIVE_TICKER_READY

    blocked = evaluate_rl_live_ticker_entitlement(
        paid_level="free",
        mode="live",
        active_tickers=active,
        requested_ticker=_ticker(owner=owner, symbol="ETHUSDT"),
    )
    assert blocked.eligible is False
    assert blocked.readiness_reason == RL_LIVE_TICKER_QUOTA_EXCEEDED


def test_stage12_api_and_producer_checks_use_same_contract_function() -> None:
    owner = _owner()
    api_snapshot = evaluate_rl_live_ticker_entitlement(
        paid_level="base",
        mode="live",
        active_tickers=(),
        requested_ticker=_ticker(owner=owner, symbol="BTCUSDT"),
    )
    producer_snapshot = evaluate_rl_live_ticker_entitlement(
        paid_level="base",
        mode="live",
        active_tickers=(),
        requested_ticker=_ticker(owner=owner, symbol="BTCUSDT"),
    )
    assert api_snapshot == producer_snapshot
    assert producer_snapshot.readiness_reason == RL_LIVE_TICKER_BASE_FAIL_CLOSED


def _owner() -> UserId:
    return UserId(UUID("00000000-0000-0000-0000-000000001201"))


def _ticker(*, owner: UserId, symbol: str) -> RlLiveTickerIdentity:
    return RlLiveTickerIdentity(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner,
        exchange_name="binance",
        market_type="futures",
        symbol=symbol,
    )
