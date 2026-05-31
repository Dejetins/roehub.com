from __future__ import annotations

from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from typing import Callable
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from trading.contexts.live_execution.application.ports import (
    ExchangeAccountProjectionRepository,
    LiveExecutionClock,
    PaperAccountingRepository,
)
from trading.contexts.live_execution.domain import (
    CapitalReservation,
    CapitalReservationBlockedError,
    PaperFill,
    PaperOrder,
    StrategyPaperAccountingSnapshot,
)
from trading.contexts.strategy.domain.entities import StrategySignal
from trading.shared_kernel.primitives import UserId

_QUOTE_ASSET = "USDT"
_FEE_BPS = Decimal("10")
_QUANTITY_QUANT = Decimal("0.00000001")
_MONEY_QUANT = Decimal("0.00000001")


class CapitalReservationPaperAccountingService:
    def __init__(
        self,
        *,
        repository: PaperAccountingRepository,
        account_projection_repository: ExchangeAccountProjectionRepository | None,
        clock: LiveExecutionClock,
        max_projection_age_seconds: int = 120,
        on_capital_reservation: Callable[[str, str], None] | None = None,
        on_paper_accounting: Callable[[str, str], None] | None = None,
    ) -> None:
        if repository is None:  # type: ignore[truthy-bool]
            raise ValueError("CapitalReservationPaperAccountingService requires repository")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("CapitalReservationPaperAccountingService requires clock")
        if max_projection_age_seconds <= 0:
            raise ValueError("max_projection_age_seconds must be positive")
        self._repository = repository
        self._account_projection_repository = account_projection_repository
        self._clock = clock
        self._max_projection_age_seconds = max_projection_age_seconds
        self._on_capital_reservation = on_capital_reservation
        self._on_paper_accounting = on_paper_accounting

    def reserve_for_strategy_run(
        self,
        *,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        strategy_id: UUID,
        live_profile_id: UUID | None,
        strategy_run_id: UUID,
        requested_amount: Decimal,
        now: datetime,
    ) -> CapitalReservation:
        if requested_amount <= 0:
            return self._reject(
                owner_user_id=owner_user_id,
                exchange_connection_id=exchange_connection_id,
                strategy_id=strategy_id,
                live_profile_id=live_profile_id,
                strategy_run_id=strategy_run_id,
                requested_amount=requested_amount,
                now=now,
                reason="capital_reservation_invalid_amount",
                source_account_snapshot_id=None,
            )
        projection = (
            self._account_projection_repository.get_latest_projection(
                owner_user_id=owner_user_id,
                exchange_connection_id=exchange_connection_id,
            )
            if self._account_projection_repository is not None
            else None
        )
        if projection is None:
            return self._reject(
                owner_user_id=owner_user_id,
                exchange_connection_id=exchange_connection_id,
                strategy_id=strategy_id,
                live_profile_id=live_profile_id,
                strategy_run_id=strategy_run_id,
                requested_amount=requested_amount,
                now=now,
                reason="capital_projection_missing",
                source_account_snapshot_id=None,
            )
        age_seconds = projection.age_seconds(now=now)
        if projection.sync_status == "degraded" or age_seconds > self._max_projection_age_seconds:
            return self._reject(
                owner_user_id=owner_user_id,
                exchange_connection_id=exchange_connection_id,
                strategy_id=strategy_id,
                live_profile_id=live_profile_id,
                strategy_run_id=strategy_run_id,
                requested_amount=requested_amount,
                now=now,
                reason="capital_projection_stale",
                source_account_snapshot_id=projection.account_snapshot_id,
            )
        quote_balance = next(
            (item for item in projection.balances if item.asset.upper() == _QUOTE_ASSET),
            None,
        )
        free_amount = quote_balance.free if quote_balance is not None else Decimal("0")
        already_reserved = self._repository.sum_active_reserved(
            owner_user_id=owner_user_id,
            exchange_connection_id=exchange_connection_id,
            asset=_QUOTE_ASSET,
        )
        available = free_amount - already_reserved
        if available < requested_amount:
            return self._reject(
                owner_user_id=owner_user_id,
                exchange_connection_id=exchange_connection_id,
                strategy_id=strategy_id,
                live_profile_id=live_profile_id,
                strategy_run_id=strategy_run_id,
                requested_amount=requested_amount,
                now=now,
                reason="capital_insufficient_available_balance",
                source_account_snapshot_id=projection.account_snapshot_id,
            )
        reservation = CapitalReservation(
            reservation_id=uuid4(),
            owner_user_id=owner_user_id,
            exchange_connection_id=exchange_connection_id,
            strategy_id=strategy_id,
            live_profile_id=live_profile_id,
            strategy_run_id=strategy_run_id,
            asset=_QUOTE_ASSET,
            requested_amount=requested_amount,
            reserved_amount=requested_amount,
            state="reserved",
            source_account_snapshot_id=projection.account_snapshot_id,
            acquired_at=now,
            released_at=None,
            reason="capital_reserved",
            fee_model="paper_fixed_bps_10",
            funding_model="spot_not_applicable",
            pnl_complete=True,
        )
        recorded = self._repository.record_reservation(reservation=reservation)
        self._record_capital(result="reserved", reason="capital_reserved")
        return recorded

    def release_for_strategy_run(
        self, *, owner_user_id: UserId, strategy_run_id: UUID, now: datetime, reason: str
    ) -> CapitalReservation | None:
        released = self._repository.release_reservation_for_run(
            owner_user_id=owner_user_id,
            strategy_run_id=strategy_run_id,
            changed_at=now,
            reason=reason,
        )
        if released is not None:
            self._record_capital(result="released", reason=reason)
        return released

    def record_paper_signal(
        self, *, signal: StrategySignal
    ) -> StrategyPaperAccountingSnapshot | None:
        if signal.mode != "paper" or signal.outcome != "signal":
            return None
        if signal.side is None:
            raise CapitalReservationBlockedError(reason="paper_signal_missing_side")
        reservation = self._repository.get_active_reservation_for_run(
            owner_user_id=signal.owner_user_id,
            strategy_run_id=signal.strategy_run_id,
        )
        if reservation is None:
            raise CapitalReservationBlockedError(reason="capital_reservation_missing")
        created_at = signal.created_at or self._clock.now()
        quote_notional = min(reservation.reserved_amount, reservation.requested_amount)
        quantity = (quote_notional / signal.reference_price).quantize(
            _QUANTITY_QUANT,
            rounding=ROUND_DOWN,
        )
        fee_amount = (quote_notional * _FEE_BPS / Decimal("10000")).quantize(_MONEY_QUANT)
        funding_model = "spot_not_applicable" if signal.market_type == "spot" else "funding_unknown"
        pnl_complete = funding_model == "spot_not_applicable"
        completeness_reason = (
            "paper_fee_fixed_bps_funding_not_applicable"
            if pnl_complete
            else "paper_funding_unknown"
        )
        order = PaperOrder(
            paper_order_id=_stable_uuid("paper-order", signal.signal_id),
            owner_user_id=signal.owner_user_id,
            strategy_id=signal.strategy_id,
            strategy_run_id=signal.strategy_run_id,
            reservation_id=reservation.reservation_id,
            source_signal_id=signal.signal_id,
            instrument_key=signal.instrument_key,
            market_type=signal.market_type,
            side=signal.side,
            order_type="market",
            quantity=quantity,
            quote_notional=quote_notional,
            reference_price=signal.reference_price,
            status="filled",
            reason="paper_market_fill_from_strategy_signal",
            created_at=created_at,
        )
        fill = PaperFill(
            paper_fill_id=_stable_uuid("paper-fill", signal.signal_id),
            paper_order_id=order.paper_order_id,
            owner_user_id=signal.owner_user_id,
            strategy_id=signal.strategy_id,
            strategy_run_id=signal.strategy_run_id,
            instrument_key=signal.instrument_key,
            side=signal.side,
            quantity=quantity,
            fill_price=signal.reference_price,
            quote_notional=quote_notional,
            fee_amount=fee_amount,
            fee_asset=_QUOTE_ASSET,
            funding_amount=Decimal("0"),
            funding_asset=_QUOTE_ASSET,
            filled_at=created_at,
        )
        position_quantity = quantity if signal.side == "buy" else Decimal("0")
        cash_balance = reservation.reserved_amount - quote_notional - fee_amount
        accounting = StrategyPaperAccountingSnapshot(
            accounting_id=_stable_uuid("paper-accounting", signal.signal_id),
            owner_user_id=signal.owner_user_id,
            strategy_id=signal.strategy_id,
            strategy_run_id=signal.strategy_run_id,
            reservation_id=reservation.reservation_id,
            paper_fill_id=fill.paper_fill_id,
            instrument_key=signal.instrument_key,
            market_type=signal.market_type,
            position_quantity=position_quantity,
            average_entry_price=signal.reference_price if position_quantity > 0 else None,
            reserved_budget=reservation.reserved_amount,
            cash_balance=cash_balance,
            equity=(cash_balance + quote_notional - fee_amount).quantize(_MONEY_QUANT),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0") - fee_amount,
            fee_total=fee_amount,
            funding_total=Decimal("0"),
            fee_model="paper_fixed_bps_10",
            funding_model=funding_model,
            pnl_complete=pnl_complete,
            completeness_reason=completeness_reason,
            created_at=created_at,
        )
        recorded = self._repository.record_paper_execution(
            order=order,
            fill=fill,
            accounting=accounting,
        )
        self._record_paper(result="filled", reason=accounting.completeness_reason)
        return recorded

    def get_latest_accounting_for_strategy(
        self, *, owner_user_id: UserId, strategy_id: UUID
    ) -> StrategyPaperAccountingSnapshot | None:
        return self._repository.get_latest_accounting_for_strategy(
            owner_user_id=owner_user_id,
            strategy_id=strategy_id,
        )

    def _reject(
        self,
        *,
        owner_user_id: UserId,
        exchange_connection_id: UUID,
        strategy_id: UUID,
        live_profile_id: UUID | None,
        strategy_run_id: UUID,
        requested_amount: Decimal,
        now: datetime,
        reason: str,
        source_account_snapshot_id: UUID | None,
    ) -> CapitalReservation:
        reservation = CapitalReservation(
            reservation_id=uuid4(),
            owner_user_id=owner_user_id,
            exchange_connection_id=exchange_connection_id,
            strategy_id=strategy_id,
            live_profile_id=live_profile_id,
            strategy_run_id=strategy_run_id,
            asset=_QUOTE_ASSET,
            requested_amount=requested_amount,
            reserved_amount=Decimal("0"),
            state="rejected",
            source_account_snapshot_id=source_account_snapshot_id,
            acquired_at=now,
            released_at=now,
            reason=reason,
            fee_model="paper_fixed_bps_10",
            funding_model="spot_not_applicable",
            pnl_complete=False,
        )
        recorded = self._repository.record_reservation(reservation=reservation)
        self._record_capital(result="rejected", reason=reason)
        raise CapitalReservationBlockedError(reason=recorded.reason)

    def _record_capital(self, *, result: str, reason: str) -> None:
        if self._on_capital_reservation is not None:
            self._on_capital_reservation(result, reason)

    def _record_paper(self, *, result: str, reason: str) -> None:
        if self._on_paper_accounting is not None:
            self._on_paper_accounting(result, reason)


def _stable_uuid(prefix: str, source_id: UUID) -> UUID:
    return uuid5(NAMESPACE_URL, f"roehub:{prefix}:{source_id}")
