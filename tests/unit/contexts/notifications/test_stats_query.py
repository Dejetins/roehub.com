from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

from trading.contexts.notifications.adapters import InMemoryNotificationStatsSourceReader
from trading.contexts.notifications.application import (
    NotificationStatsQueryService,
    NotificationStatsSourceRow,
    render_notification_stats_snapshot,
)
from trading.shared_kernel.primitives import UserId


def _now() -> datetime:
    return datetime(2026, 6, 29, 15, 0, tzinfo=timezone.utc)


def _user_id() -> UserId:
    return UserId(UUID("11111111-1111-4111-8111-111111111111"))


def _other_user_id() -> UserId:
    return UserId(UUID("22222222-2222-4222-8222-222222222222"))


def test_stats_query_covers_day_week_month_periods_with_complete_quality() -> None:
    service = NotificationStatsQueryService(
        source_reader=InMemoryNotificationStatsSourceReader(rows=_complete_rows())
    )

    today = service.get_portfolio_stats(
        owner_user_id=_user_id(), period="today", generated_at=_now()
    )
    week = service.get_portfolio_stats(
        owner_user_id=_user_id(), period="week", generated_at=_now()
    )
    month = service.get_portfolio_stats(
        owner_user_id=_user_id(), period="month", generated_at=_now()
    )

    assert today.period_start == datetime(2026, 6, 29, tzinfo=timezone.utc)
    assert week.period_start == datetime(2026, 6, 29, tzinfo=timezone.utc)
    assert month.period_start == datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert today.quality_status == "complete"
    assert today.missing_sources == ()
    assert today.signal_count == 2
    assert today.fill_count == 1
    assert today.realized_pnl == Decimal("12.50")
    assert today.unrealized_pnl == Decimal("2.25")


def test_stats_query_partial_and_unavailable_are_explicit() -> None:
    partial_service = NotificationStatsQueryService(
        source_reader=InMemoryNotificationStatsSourceReader(
            rows=(
                NotificationStatsSourceRow(
                    owner_user_id=_user_id(),
                    source="strategy_signals",
                    observed_at=_now(),
                    signal_count=1,
                ),
            ),
            unavailable_sources=("execution_fills",),
        )
    )
    empty_service = NotificationStatsQueryService(
        source_reader=InMemoryNotificationStatsSourceReader(rows=())
    )

    partial = partial_service.get_portfolio_stats(
        owner_user_id=_user_id(), period="today", generated_at=_now()
    )
    unavailable = empty_service.get_portfolio_stats(
        owner_user_id=_user_id(), period="today", generated_at=_now()
    )

    assert partial.quality_status == "partial"
    assert "strategy_paper_accounting" in partial.missing_sources
    assert "execution_fills" in partial.missing_sources
    assert partial.realized_pnl is None
    assert unavailable.quality_status == "unavailable"
    assert unavailable.missing_sources


def test_strategy_and_exchange_filters_are_owner_scoped() -> None:
    service = NotificationStatsQueryService(
        source_reader=InMemoryNotificationStatsSourceReader(
            rows=(
                *_complete_rows(strategy_ref="owned-strategy", exchange_ref="owned-exchange"),
                *_complete_rows(
                    owner_user_id=_other_user_id(),
                    strategy_ref="foreign-strategy",
                    exchange_ref="foreign-exchange",
                ),
            )
        )
    )

    owned_strategy = service.get_strategy_stats(
        owner_user_id=_user_id(),
        strategy_ref="owned-strategy",
        period="today",
        generated_at=_now(),
    )
    foreign_strategy = service.get_strategy_stats(
        owner_user_id=_user_id(),
        strategy_ref="foreign-strategy",
        period="today",
        generated_at=_now(),
    )
    owned_exchange = service.get_exchange_stats(
        owner_user_id=_user_id(),
        exchange_ref="owned-exchange",
        period="today",
        generated_at=_now(),
    )

    assert owned_strategy.quality_status == "complete"
    assert owned_strategy.signal_count == 2
    assert foreign_strategy.quality_status == "unavailable"
    assert foreign_strategy.signal_count == 0
    assert owned_exchange.quality_status == "complete"
    assert owned_exchange.balance_count == 3


def test_stats_snapshot_renders_without_invented_metrics() -> None:
    service = NotificationStatsQueryService(
        source_reader=InMemoryNotificationStatsSourceReader(rows=_complete_rows())
    )

    response_text = render_notification_stats_snapshot(
        snapshot=service.get_portfolio_stats(
            owner_user_id=_user_id(), period="today", generated_at=_now()
        )
    )

    assert "Portfolio stats for today: complete" in response_text
    assert "signals=2" in response_text
    assert "realized_pnl=12.50" in response_text
    assert "missing_sources" not in response_text


def _complete_rows(
    *,
    owner_user_id: UserId | None = None,
    strategy_ref: str | None = None,
    exchange_ref: str | None = None,
) -> tuple[NotificationStatsSourceRow, ...]:
    effective_owner = owner_user_id or _user_id()
    return (
        NotificationStatsSourceRow(
            owner_user_id=effective_owner,
            source="strategy_signals",
            observed_at=_now(),
            strategy_ref=strategy_ref,
            exchange_ref=exchange_ref,
            signal_count=2,
        ),
        NotificationStatsSourceRow(
            owner_user_id=effective_owner,
            source="strategy_paper_accounting",
            observed_at=_now(),
            strategy_ref=strategy_ref,
            exchange_ref=exchange_ref,
            realized_pnl=Decimal("12.50"),
            unrealized_pnl=Decimal("2.25"),
            equity=Decimal("1012.50"),
            pnl_complete=True,
        ),
        NotificationStatsSourceRow(
            owner_user_id=effective_owner,
            source="execution_fills",
            observed_at=_now(),
            strategy_ref=strategy_ref,
            exchange_ref=exchange_ref,
            fill_count=1,
            order_count=1,
            fee_total=Decimal("0.10"),
            funding_total=Decimal("0"),
        ),
        NotificationStatsSourceRow(
            owner_user_id=effective_owner,
            source="exchange_account_projection",
            observed_at=_now(),
            strategy_ref=strategy_ref,
            exchange_ref=exchange_ref,
            balance_count=3,
            position_count=1,
            open_order_count=0,
        ),
    )
