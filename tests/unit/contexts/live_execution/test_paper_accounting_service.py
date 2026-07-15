from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryExchangeAccountProjectionRepository,
    InMemoryPaperAccountingRepository,
)
from trading.contexts.live_execution.application import CapitalReservationPaperAccountingService
from trading.contexts.live_execution.domain import (
    CapitalReservationBlockedError,
    ExchangeAccountProjection,
    ExchangeBalanceSnapshot,
    ExchangeInstrumentFilterSnapshot,
)
from trading.contexts.strategy.domain.entities import StrategySignal
from trading.shared_kernel.primitives import OrganizationId, UserId

_ORGANIZATION_ID = OrganizationId(UUID("00000000-0000-4000-8000-000000009000"))
_SECOND_ORGANIZATION_ID = OrganizationId(UUID("00000000-0000-4000-8000-000000009999"))
_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000009001")
_CONNECTION_ID = UUID("00000000-0000-0000-0000-000000009101")
_STRATEGY_ID = UUID("00000000-0000-0000-0000-000000009201")
_RUN_ID = UUID("00000000-0000-0000-0000-000000009301")
_PROFILE_ID = UUID("00000000-0000-0000-0000-000000009401")
_NOW = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)


class _Clock:
    def now(self) -> datetime:
        return _NOW


def test_capital_reservation_uses_fresh_projection_and_releases() -> None:
    projection_repository = InMemoryExchangeAccountProjectionRepository()
    projection_repository.record_projection(
        projection=_projection(free=Decimal("100"), observed_at=_NOW - timedelta(seconds=30))
    )
    accounting_repository = InMemoryPaperAccountingRepository()
    service = CapitalReservationPaperAccountingService(
        repository=accounting_repository,
        account_projection_repository=projection_repository,
        clock=_Clock(),
    )

    reservation = service.reserve_for_strategy_run(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        strategy_id=_STRATEGY_ID,
        live_profile_id=_PROFILE_ID,
        strategy_run_id=_RUN_ID,
        requested_amount=Decimal("25"),
        now=_NOW,
    )
    released = service.release_for_strategy_run(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        strategy_run_id=_RUN_ID,
        now=_NOW + timedelta(minutes=1),
        reason="run_stopped",
    )

    assert reservation.state == "reserved"
    assert reservation.reserved_amount == Decimal("25")
    assert released is not None
    assert released.state == "released"
    assert accounting_repository.sum_active_reserved(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        asset="USDT",
    ) == Decimal("0")


def test_virtual_paper_capital_reservation_does_not_require_projection() -> None:
    accounting_repository = InMemoryPaperAccountingRepository()
    service = CapitalReservationPaperAccountingService(
        repository=accounting_repository,
        account_projection_repository=None,
        clock=_Clock(),
    )

    reservation = service.reserve_virtual_for_strategy_run(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        strategy_id=_STRATEGY_ID,
        live_profile_id=_PROFILE_ID,
        strategy_run_id=_RUN_ID,
        requested_amount=Decimal("50"),
        now=_NOW,
    )

    assert reservation.state == "reserved"
    assert reservation.reserved_amount == Decimal("50")
    assert reservation.reason == "paper_virtual_capital_reserved"
    assert reservation.source_account_snapshot_id is None


def test_manual_paper_execution_records_idempotent_order_fill_and_accounting() -> None:
    accounting_repository = InMemoryPaperAccountingRepository()
    service = CapitalReservationPaperAccountingService(
        repository=accounting_repository,
        account_projection_repository=None,
        clock=_Clock(),
    )
    source_event_id = UUID("00000000-0000-0000-0000-000000009501")

    first = service.record_manual_paper_execution(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        strategy_id=_STRATEGY_ID,
        live_profile_id=_PROFILE_ID,
        strategy_run_id=_RUN_ID,
        source_event_id=source_event_id,
        instrument_key="binance:spot:BTCUSDT",
        market_type="spot",
        side="buy",
        quote_notional=Decimal("50"),
        reference_price=Decimal("50000"),
        now=_NOW,
    )
    replay = service.record_manual_paper_execution(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        strategy_id=_STRATEGY_ID,
        live_profile_id=_PROFILE_ID,
        strategy_run_id=_RUN_ID,
        source_event_id=source_event_id,
        instrument_key="binance:spot:BTCUSDT",
        market_type="spot",
        side="buy",
        quote_notional=Decimal("50"),
        reference_price=Decimal("50000"),
        now=_NOW,
    )

    assert replay.accounting_id == first.accounting_id
    assert len(accounting_repository.reservations) == 1
    assert len(accounting_repository.orders) == 1
    assert accounting_repository.orders[0].source_event_id == source_event_id
    assert len(accounting_repository.fills) == 1
    assert len(accounting_repository.accounting) == 1


def test_paper_execution_identity_and_reads_are_isolated_by_organization() -> None:
    repository = InMemoryPaperAccountingRepository()
    service = CapitalReservationPaperAccountingService(
        repository=repository,
        account_projection_repository=None,
        clock=_Clock(),
    )
    source_event_id = UUID("00000000-0000-0000-0000-000000009599")

    snapshots = tuple(
        service.record_manual_paper_execution(
            organization_id=organization_id,
            owner_user_id=_USER_ID,
            strategy_id=_STRATEGY_ID,
            live_profile_id=_PROFILE_ID,
            strategy_run_id=_RUN_ID,
            source_event_id=source_event_id,
            instrument_key="binance:spot:BTCUSDT",
            market_type="spot",
            side="buy",
            quote_notional=Decimal("50"),
            reference_price=Decimal("50000"),
            now=_NOW,
        )
        for organization_id in (_ORGANIZATION_ID, _SECOND_ORGANIZATION_ID)
    )

    assert snapshots[0].accounting_id != snapshots[1].accounting_id
    assert len(repository.orders) == 2
    assert len(repository.fills) == 2
    assert len(repository.accounting) == 2
    assert (
        repository.get_latest_accounting_for_strategy(
            organization_id=_ORGANIZATION_ID,
            owner_user_id=_USER_ID,
            strategy_id=_STRATEGY_ID,
        )
        == snapshots[0]
    )
    assert (
        repository.get_latest_accounting_for_strategy(
            organization_id=_SECOND_ORGANIZATION_ID,
            owner_user_id=_USER_ID,
            strategy_id=_STRATEGY_ID,
        )
        == snapshots[1]
    )


def test_rl_paper_execution_records_idempotent_order_fill_and_accounting_parity() -> None:
    accounting_repository = InMemoryPaperAccountingRepository()
    service = CapitalReservationPaperAccountingService(
        repository=accounting_repository,
        account_projection_repository=None,
        clock=_Clock(),
    )
    source_event_id = UUID("00000000-0000-0000-0000-000000009511")

    first = service.record_rl_paper_execution(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        strategy_id=_STRATEGY_ID,
        live_profile_id=_PROFILE_ID,
        strategy_run_id=_RUN_ID,
        source_event_id=source_event_id,
        instrument_key="binance:futures:BTCUSDT",
        market_type="futures",
        side="buy",
        quote_notional=Decimal("50"),
        reference_price=Decimal("10000"),
        now=_NOW,
    )
    replay = service.record_rl_paper_execution(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        strategy_id=_STRATEGY_ID,
        live_profile_id=_PROFILE_ID,
        strategy_run_id=_RUN_ID,
        source_event_id=source_event_id,
        instrument_key="binance:futures:BTCUSDT",
        market_type="futures",
        side="buy",
        quote_notional=Decimal("50"),
        reference_price=Decimal("10000"),
        now=_NOW,
    )

    simulator_expected = {
        "position_quantity": Decimal("0.00500000"),
        "fee_total": Decimal("0.05000000"),
        "equity": Decimal("49.95000000"),
    }
    parity_abs_diff = {
        "position_quantity": abs(first.position_quantity - simulator_expected["position_quantity"]),
        "fee_total": abs(first.fee_total - simulator_expected["fee_total"]),
        "equity": abs(first.equity - simulator_expected["equity"]),
    }

    assert replay.accounting_id == first.accounting_id
    assert first.reserved_budget == Decimal("50")
    assert first.position_quantity == simulator_expected["position_quantity"]
    assert first.average_entry_price == Decimal("10000")
    assert first.fee_model == "paper_fixed_bps_10"
    assert first.funding_model == "funding_unknown"
    assert first.pnl_complete is False
    assert first.completeness_reason == "paper_funding_unknown"
    assert parity_abs_diff == {
        "position_quantity": Decimal("0E-8"),
        "fee_total": Decimal("0E-8"),
        "equity": Decimal("0E-8"),
    }
    assert len(accounting_repository.reservations) == 1
    assert len(accounting_repository.orders) == 1
    assert accounting_repository.orders[0].source_event_id == source_event_id
    assert accounting_repository.orders[0].reason == "paper_market_fill_from_ml_agent_decision"
    assert len(accounting_repository.fills) == 1
    assert len(accounting_repository.accounting) == 1


@pytest.mark.parametrize(
    ("free", "observed_at", "reason"),
    (
        (Decimal("10"), _NOW - timedelta(seconds=30), "capital_insufficient_available_balance"),
        (Decimal("100"), _NOW - timedelta(minutes=5), "capital_projection_stale"),
    ),
)
def test_capital_reservation_rejects_insufficient_or_stale_projection(
    free: Decimal, observed_at: datetime, reason: str
) -> None:
    projection_repository = InMemoryExchangeAccountProjectionRepository()
    projection_repository.record_projection(
        projection=_projection(free=free, observed_at=observed_at)
    )
    accounting_repository = InMemoryPaperAccountingRepository()
    service = CapitalReservationPaperAccountingService(
        repository=accounting_repository,
        account_projection_repository=projection_repository,
        clock=_Clock(),
    )

    with pytest.raises(CapitalReservationBlockedError) as error_info:
        service.reserve_for_strategy_run(
            organization_id=_ORGANIZATION_ID,
            owner_user_id=_USER_ID,
            exchange_connection_id=_CONNECTION_ID,
            strategy_id=_STRATEGY_ID,
            live_profile_id=_PROFILE_ID,
            strategy_run_id=_RUN_ID,
            requested_amount=Decimal("25"),
            now=_NOW,
        )

    assert error_info.value.reason == reason
    assert accounting_repository.reservations[-1].state == "rejected"
    assert accounting_repository.reservations[-1].reason == reason


def test_paper_signal_records_order_fill_accounting_idempotently() -> None:
    projection_repository = InMemoryExchangeAccountProjectionRepository()
    projection_repository.record_projection(
        projection=_projection(free=Decimal("100"), observed_at=_NOW - timedelta(seconds=30))
    )
    accounting_repository = InMemoryPaperAccountingRepository()
    service = CapitalReservationPaperAccountingService(
        repository=accounting_repository,
        account_projection_repository=projection_repository,
        clock=_Clock(),
    )
    service.reserve_for_strategy_run(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        strategy_id=_STRATEGY_ID,
        live_profile_id=_PROFILE_ID,
        strategy_run_id=_RUN_ID,
        requested_amount=Decimal("50"),
        now=_NOW,
    )
    signal = _signal(signal_id=UUID("00000000-0000-0000-0000-000000009501"))

    first = service.record_paper_signal(signal=signal)
    replay = service.record_paper_signal(signal=signal)

    assert first is not None
    assert replay is not None
    assert replay.accounting_id == first.accounting_id
    assert len(accounting_repository.orders) == 1
    assert len(accounting_repository.fills) == 1
    assert len(accounting_repository.accounting) == 1
    assert first.reserved_budget == Decimal("50")
    assert first.position_quantity == Decimal("0.00500000")
    assert first.fee_model == "paper_fixed_bps_10"
    assert first.funding_model == "spot_not_applicable"
    assert first.pnl_complete is True
    assert first.equity == Decimal("49.95000000")


def test_paper_short_records_negative_position_and_incomplete_borrow_model() -> None:
    accounting_repository = InMemoryPaperAccountingRepository()
    service = CapitalReservationPaperAccountingService(
        repository=accounting_repository,
        account_projection_repository=None,
        clock=_Clock(),
    )
    service.reserve_virtual_for_strategy_run(
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        strategy_id=_STRATEGY_ID,
        live_profile_id=_PROFILE_ID,
        strategy_run_id=_RUN_ID,
        requested_amount=Decimal("50"),
        now=_NOW,
    )
    signal = _signal(signal_id=UUID("00000000-0000-0000-0000-000000009502"), side="sell")

    accounting = service.record_paper_signal(signal=signal)

    assert accounting is not None
    assert accounting.position_quantity == Decimal("-0.00500000")
    assert accounting.average_entry_price == Decimal("10000")
    assert accounting.cash_balance == Decimal("49.95000000")
    assert accounting.equity == Decimal("49.95000000")
    assert accounting.fee_model == "paper_fixed_bps_10"
    assert accounting.funding_model == "spot_borrow_not_modeled"
    assert accounting.pnl_complete is False
    assert accounting.completeness_reason == "paper_spot_short_borrow_not_modeled"


def _projection(*, free: Decimal, observed_at: datetime) -> ExchangeAccountProjection:
    return ExchangeAccountProjection(
        account_snapshot_id=uuid4(),
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        exchange_connection_id=_CONNECTION_ID,
        exchange_name="binance",
        market_type="spot",
        environment="testnet",
        account_mode="spot",
        balances=(ExchangeBalanceSnapshot(asset="USDT", free=free, total=free),),
        positions=(),
        open_orders=(),
        instrument_filters=(
            ExchangeInstrumentFilterSnapshot(
                instrument_key="binance:spot:BTCUSDT",
                min_notional=Decimal("10"),
            ),
        ),
        source_hash="b" * 64,
        observed_at=observed_at,
        synced_at=observed_at,
    )


def _signal(*, signal_id: UUID, side: str = "buy") -> StrategySignal:
    return StrategySignal(
        signal_id=signal_id,
        organization_id=_ORGANIZATION_ID,
        owner_user_id=_USER_ID,
        strategy_id=_STRATEGY_ID,
        strategy_run_id=_RUN_ID,
        live_profile_id=_PROFILE_ID,
        mode="paper",
        instrument_key="binance:spot:BTCUSDT",
        market_type="spot",
        timeframe="1m",
        bar_ts_open=_NOW,
        bar_ts_close=_NOW + timedelta(minutes=1),
        signal_action="open",
        side=side,  # type: ignore[arg-type]
        outcome="signal",
        reason_code="ma_fast_crossed_above_slow_paper_no_exchange_submit",
        reference_price=Decimal("10000"),
        confidence=Decimal("1"),
        source_message_id="1-0",
        evaluator_version="ma_cross_close_v1",
        expected_order_json={},
        created_at=_NOW + timedelta(minutes=1),
    )
