from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExchangeAccountProjectionRepository,
)
from trading.contexts.live_execution.application import ExchangeAccountProjectionService
from trading.contexts.live_execution.domain import (
    AccountConfigGuardResult,
    ExchangeAccountProjection,
    ExchangeBalanceSnapshot,
    ExchangeInstrumentFilterSnapshot,
    ExpectedInstrumentConfig,
)
from trading.shared_kernel.primitives import UserId

_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000007001")
_CONNECTION_ID = UUID("00000000-0000-0000-0000-000000007101")
_ACCOUNT_ID = UUID("00000000-0000-0000-0000-000000007201")


class _StaticClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def now(self) -> datetime:
        return self.value


def test_account_projection_readiness_reports_fresh_and_config_mismatch() -> None:
    repository = InMemoryExchangeAccountProjectionRepository()
    clock = _StaticClock(datetime(2026, 5, 31, 9, 0, tzinfo=UTC))
    service = ExchangeAccountProjectionService(repository=repository, clock=clock)
    projection = _projection(observed_at=clock.now() - timedelta(seconds=30))
    repository.record_projection(projection=projection)
    ok_requirement = ExpectedInstrumentConfig(
        instrument_key="binance:spot:BTCUSDT",
        market_type="spot",
        min_notional=Decimal("5"),
    )

    fresh = service.get_readiness(
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        requirement=ok_requirement,
    )

    assert fresh.status == "fresh"
    assert fresh.ready_for_risk is True
    assert fresh.reason_codes == ("account_projection_fresh",)
    assert fresh.age_seconds == 30
    assert repository.config_results[-1].status == "verified"

    mismatch_requirement = ExpectedInstrumentConfig(
        instrument_key="binance:spot:BTCUSDT",
        market_type="spot",
        min_notional=Decimal("20"),
    )
    mismatch_result = service.verify_config(
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        requirement=mismatch_requirement,
        projection=projection,
    )
    repository.record_config_guard_result(result=mismatch_result)

    mismatch = service.get_readiness(
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        requirement=mismatch_requirement,
    )

    assert mismatch.status == "config_mismatch"
    assert mismatch.ready_for_risk is False
    assert mismatch.reason_codes == ("min_notional_below_requirement",)


def test_account_projection_readiness_reports_stale_and_missing_projection() -> None:
    repository = InMemoryExchangeAccountProjectionRepository()
    clock = _StaticClock(datetime(2026, 5, 31, 9, 0, tzinfo=UTC))
    service = ExchangeAccountProjectionService(
        repository=repository,
        clock=clock,
        max_projection_age=timedelta(minutes=2),
    )
    requirement = ExpectedInstrumentConfig(
        instrument_key="binance:spot:BTCUSDT",
        market_type="spot",
    )

    missing = service.get_readiness(
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        requirement=requirement,
    )

    assert missing.status == "degraded"
    assert missing.reason_codes == ("account_projection_missing",)

    projection = _projection(observed_at=clock.now() - timedelta(minutes=5))
    repository.record_projection(projection=projection)
    repository.record_config_guard_result(
        result=AccountConfigGuardResult(
            config_guard_result_id=UUID("00000000-0000-0000-0000-000000007301"),
            account_snapshot_id=projection.account_snapshot_id,
            owner_user_id=_USER_ID,
            exchange_connection_id=_CONNECTION_ID,
            instrument_key=requirement.instrument_key,
            market_type=requirement.market_type,
            status="verified",
            reason_codes=("verify_only_config_ok",),
            checked_at=clock.now(),
            requirement=requirement,
        )
    )

    stale = service.get_readiness(
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        requirement=requirement,
    )

    assert stale.status == "stale"
    assert stale.reason_codes == ("account_projection_stale",)
    assert stale.age_seconds == 300


def _projection(*, observed_at: datetime) -> ExchangeAccountProjection:
    return ExchangeAccountProjection(
        account_snapshot_id=_ACCOUNT_ID,
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        exchange_name="binance",
        market_type="spot",
        environment="testnet",
        account_mode="spot",
        balances=(
            ExchangeBalanceSnapshot(
                asset="USDT",
                free=Decimal("100"),
                locked=Decimal("0"),
                total=Decimal("100"),
            ),
        ),
        positions=(),
        open_orders=(),
        instrument_filters=(
            ExchangeInstrumentFilterSnapshot(
                instrument_key="binance:spot:BTCUSDT",
                tick_size=Decimal("0.01"),
                step_size=Decimal("0.00001"),
                min_notional=Decimal("10"),
            ),
        ),
        source_hash="a" * 64,
        observed_at=observed_at,
        synced_at=observed_at,
    )
