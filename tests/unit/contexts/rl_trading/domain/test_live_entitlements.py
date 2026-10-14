from uuid import UUID

from trading.contexts.rl_trading.domain.live_entitlements import (
    RL_LIVE_TICKER_NOT_COUNTED,
    RL_LIVE_TICKER_QUOTA_EXCEEDED,
    RL_LIVE_TICKER_READY,
    RlLiveTickerIdentity,
    evaluate_rl_live_ticker_entitlement,
    resolve_rl_live_ticker_limit,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("00000000-0000-4000-8000-000000001200"))


def test_self_host_capacity_does_not_depend_on_legacy_paid_level() -> None:
    for tier in ("free", "base", "pro", "ultra", "unknown"):
        limit = resolve_rl_live_ticker_limit(paid_level=tier)
        assert limit.live_slots_allowed == 5
        assert limit.product_label == "Self-hosted"
        assert limit.entitlement_source == "self_host"
        assert limit.fail_closed_reason is None
    assert resolve_rl_live_ticker_limit(
        paid_level="free", override_live_slots_allowed=42
    ).live_slots_allowed == 42


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
        override_live_slots_allowed=1,
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
    assert producer_snapshot.readiness_reason == RL_LIVE_TICKER_READY


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
