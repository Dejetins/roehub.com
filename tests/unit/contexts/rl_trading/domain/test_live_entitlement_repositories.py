from datetime import UTC, datetime, timedelta
from uuid import UUID

from trading.contexts.rl_trading.adapters.outbound.persistence import (
    InMemoryRlLiveTickerEntitlementRepository,
)
from trading.contexts.rl_trading.domain.live_entitlements import (
    RL_LIVE_TICKER_QUOTA_EXCEEDED,
    RlLiveTickerIdentity,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("00000000-0000-4000-8000-000000001200"))


def test_stage12_in_memory_repository_records_active_live_tickers_and_releases_slots() -> None:
    repository = InMemoryRlLiveTickerEntitlementRepository()
    owner = UserId(UUID("00000000-0000-0000-0000-000000001202"))
    now = datetime(2026, 7, 3, 8, 0, tzinfo=UTC)

    first = repository.sync_profile(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner,
        paid_level="free",
        strategy_id=UUID("00000000-0000-0000-0000-000000001301"),
        live_profile_id=UUID("00000000-0000-0000-0000-000000001401"),
        mode="live",
        requested_ticker=_ticker(owner=owner, symbol="BTCUSDT"),
        profile_ready=True,
        observed_at=now,
    )
    assert first.eligible is True
    assert first.live_slots_used == 1

    blocked = repository.sync_profile(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner,
        paid_level="free",
        strategy_id=UUID("00000000-0000-0000-0000-000000001302"),
        live_profile_id=UUID("00000000-0000-0000-0000-000000001402"),
        mode="live",
        requested_ticker=_ticker(owner=owner, symbol="ETHUSDT"),
        profile_ready=True,
        observed_at=now + timedelta(minutes=1),
    )
    assert blocked.eligible is False
    assert blocked.readiness_reason == RL_LIVE_TICKER_QUOTA_EXCEEDED

    released = repository.sync_profile(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner,
        paid_level="free",
        strategy_id=UUID("00000000-0000-0000-0000-000000001301"),
        live_profile_id=UUID("00000000-0000-0000-0000-000000001401"),
        mode="monitor_only",
        requested_ticker=_ticker(owner=owner, symbol="BTCUSDT"),
        profile_ready=True,
        observed_at=now + timedelta(minutes=2),
    )
    assert released.live_slots_used == 0

    second = repository.sync_profile(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner,
        paid_level="free",
        strategy_id=UUID("00000000-0000-0000-0000-000000001302"),
        live_profile_id=UUID("00000000-0000-0000-0000-000000001402"),
        mode="live",
        requested_ticker=_ticker(owner=owner, symbol="ETHUSDT"),
        profile_ready=True,
        observed_at=now + timedelta(minutes=3),
    )
    assert second.eligible is True
    assert second.live_slots_used == 1


def test_stage12_in_memory_repository_uses_enterprise_override() -> None:
    repository = InMemoryRlLiveTickerEntitlementRepository()
    owner = UserId(UUID("00000000-0000-0000-0000-000000001203"))
    repository.set_override(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner,
        live_slots_allowed=2,
    )

    snapshot = repository.snapshot(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner,
        paid_level="base",
        mode="live",
        requested_ticker=_ticker(owner=owner, symbol="BTCUSDT"),
    )

    assert snapshot.eligible is True
    assert snapshot.live_slots_allowed == 2
    assert snapshot.product_label == "Enterprise"
    assert snapshot.entitlement_source == "override"


def test_stage10_live_ticker_state_isolated_for_same_user_across_organizations() -> None:
    repository = InMemoryRlLiveTickerEntitlementRepository()
    owner = UserId(UUID("00000000-0000-0000-0000-000000001204"))
    other_organization = OrganizationId(
        UUID("00000000-0000-4000-8000-000000001299")
    )
    now = datetime(2026, 7, 3, 9, 0, tzinfo=UTC)

    repository.sync_profile(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner,
        paid_level="free",
        strategy_id=UUID("00000000-0000-0000-0000-000000001305"),
        live_profile_id=UUID("00000000-0000-0000-0000-000000001405"),
        mode="live",
        requested_ticker=_ticker(owner=owner, symbol="BTCUSDT"),
        profile_ready=True,
        observed_at=now,
    )

    other_snapshot = repository.snapshot(
        organization_id=other_organization,
        owner_user_id=owner,
        paid_level="free",
        mode="live",
        requested_ticker=RlLiveTickerIdentity(
            organization_id=other_organization,
            owner_user_id=owner,
            exchange_name="binance",
            market_type="futures",
            symbol="ETHUSDT",
        ),
    )

    assert other_snapshot.live_slots_used == 0
    assert other_snapshot.eligible is True


def _ticker(*, owner: UserId, symbol: str) -> RlLiveTickerIdentity:
    return RlLiveTickerIdentity(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=owner,
        exchange_name="binance",
        market_type="futures",
        symbol=symbol,
    )
