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
    ExchangePositionSnapshot,
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


def test_futures_short_guard_accepts_only_isolated_1x_with_balance_and_notional() -> None:
    repository = InMemoryExchangeAccountProjectionRepository()
    clock = _StaticClock(datetime(2026, 5, 31, 9, 0, tzinfo=UTC))
    service = ExchangeAccountProjectionService(repository=repository, clock=clock)
    requirement = _futures_short_requirement()
    repository.record_projection(
        projection=_futures_projection(observed_at=clock.now() - timedelta(seconds=15))
    )

    readiness = service.get_readiness(
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        requirement=requirement,
    )

    assert readiness.status == "fresh"
    assert readiness.ready_for_risk is True
    assert readiness.reason_codes == ("account_projection_fresh",)
    assert repository.config_results[-1].reason_codes == ("verify_only_config_ok",)


def test_futures_short_guard_blocks_missing_position_config() -> None:
    repository = InMemoryExchangeAccountProjectionRepository()
    clock = _StaticClock(datetime(2026, 5, 31, 9, 0, tzinfo=UTC))
    service = ExchangeAccountProjectionService(repository=repository, clock=clock)
    requirement = _futures_short_requirement()
    repository.record_projection(
        projection=_futures_projection(
            observed_at=clock.now() - timedelta(seconds=15),
            positions=(),
        )
    )

    readiness = service.get_readiness(
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        requirement=requirement,
    )

    assert readiness.status == "config_mismatch"
    assert readiness.ready_for_risk is False
    assert readiness.reason_codes == ("unsafe_futures_short",)


def test_futures_short_guard_blocks_margin_balance_and_min_notional_mismatch() -> None:
    repository = InMemoryExchangeAccountProjectionRepository()
    clock = _StaticClock(datetime(2026, 5, 31, 9, 0, tzinfo=UTC))
    service = ExchangeAccountProjectionService(repository=repository, clock=clock)
    requirement = _futures_short_requirement(order_notional=Decimal("50"))
    projection = _futures_projection(
        observed_at=clock.now() - timedelta(seconds=15),
        balances=(),
        positions=(
            ExchangePositionSnapshot(
                instrument_key="binance:futures:BTCUSDT",
                side="net",
                quantity=Decimal("0"),
                leverage=Decimal("5"),
                margin_mode="cross",
                position_mode="one_way",
            ),
        ),
        min_notional=Decimal("55"),
    )
    repository.record_projection(projection=projection)

    readiness = service.get_readiness(
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        requirement=requirement,
    )

    assert readiness.status == "config_mismatch"
    assert readiness.reason_codes == (
        "min_notional_issue",
        "missing_balance",
        "margin_mode_mismatch",
        "leverage_mismatch",
    )


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


def _futures_short_requirement(
    *, order_notional: Decimal = Decimal("50")
) -> ExpectedInstrumentConfig:
    return ExpectedInstrumentConfig(
        instrument_key="binance:futures:BTCUSDT",
        market_type="futures",
        side="short",
        expected_margin_mode="isolated",
        expected_position_mode="one_way",
        required_leverage=Decimal("1"),
        order_notional=order_notional,
        required_balance_asset="USDT",
    )


def _futures_projection(
    *,
    observed_at: datetime,
    balances: tuple[ExchangeBalanceSnapshot, ...] | None = None,
    positions: tuple[ExchangePositionSnapshot, ...] | None = None,
    min_notional: Decimal = Decimal("5"),
) -> ExchangeAccountProjection:
    return ExchangeAccountProjection(
        account_snapshot_id=_ACCOUNT_ID,
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        exchange_name="binance",
        market_type="futures",
        environment="testnet",
        account_mode="futures",
        balances=(
            balances
            if balances is not None
            else (
                ExchangeBalanceSnapshot(
                    asset="USDT",
                    free=Decimal("100"),
                    locked=Decimal("0"),
                    total=Decimal("100"),
                ),
            )
        ),
        positions=(
            positions
            if positions is not None
            else (
                ExchangePositionSnapshot(
                    instrument_key="binance:futures:BTCUSDT",
                    side="net",
                    quantity=Decimal("0"),
                    leverage=Decimal("1"),
                    margin_mode="isolated",
                    position_mode="one_way",
                ),
            )
        ),
        open_orders=(),
        instrument_filters=(
            ExchangeInstrumentFilterSnapshot(
                instrument_key="binance:futures:BTCUSDT",
                tick_size=Decimal("0.1"),
                step_size=Decimal("0.001"),
                min_notional=min_notional,
            ),
        ),
        source_hash="b" * 64,
        observed_at=observed_at,
        synced_at=observed_at,
    )
