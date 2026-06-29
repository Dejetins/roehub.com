from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from typing import Literal, Protocol

from trading.shared_kernel.primitives import UserId

NotificationStatsPeriod = Literal["today", "week", "month"]
NotificationStatsQualityStatus = Literal["complete", "partial", "unavailable"]
NotificationStatsScopeKind = Literal["portfolio", "strategy", "exchange"]

_SOURCE_SIGNALS = "strategy_signals"
_SOURCE_PAPER_ACCOUNTING = "strategy_paper_accounting"
_SOURCE_EXECUTION_FILLS = "execution_fills"
_SOURCE_EXCHANGE_ACCOUNT = "exchange_account_projection"
_REQUIRED_SOURCES = (
    _SOURCE_SIGNALS,
    _SOURCE_PAPER_ACCOUNTING,
    _SOURCE_EXECUTION_FILLS,
    _SOURCE_EXCHANGE_ACCOUNT,
)


@dataclass(frozen=True, slots=True)
class NotificationStatsPeriodWindow:
    period: NotificationStatsPeriod
    timezone: str
    start_at: datetime
    end_at: datetime


@dataclass(frozen=True, slots=True)
class NotificationStatsSourceRow:
    owner_user_id: UserId
    source: str
    observed_at: datetime
    strategy_ref: str | None = None
    exchange_ref: str | None = None
    signal_count: int = 0
    fill_count: int = 0
    order_count: int = 0
    balance_count: int = 0
    position_count: int = 0
    open_order_count: int = 0
    realized_pnl: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    fee_total: Decimal | None = None
    funding_total: Decimal | None = None
    equity: Decimal | None = None
    pnl_complete: bool = False


@dataclass(frozen=True, slots=True)
class NotificationStatsSourceResult:
    rows: tuple[NotificationStatsSourceRow, ...]
    unavailable_sources: tuple[str, ...] = ()


class NotificationStatsSourceReader(Protocol):
    def read_stats_rows(
        self,
        *,
        owner_user_id: UserId,
        period_start: datetime,
        period_end: datetime,
        strategy_ref: str | None = None,
        exchange_ref: str | None = None,
    ) -> NotificationStatsSourceResult: ...


@dataclass(frozen=True, slots=True)
class NotificationStatsSnapshot:
    owner_user_id: UserId
    scope_kind: NotificationStatsScopeKind
    scope_ref: str | None
    period: NotificationStatsPeriod
    timezone: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    quality_status: NotificationStatsQualityStatus
    missing_sources: tuple[str, ...]
    latest_source_at: datetime | None
    freshness_seconds: int | None
    signal_count: int
    fill_count: int
    order_count: int
    balance_count: int
    position_count: int
    open_order_count: int
    realized_pnl: Decimal | None
    unrealized_pnl: Decimal | None
    fee_total: Decimal | None
    funding_total: Decimal | None
    equity: Decimal | None


class NotificationStatsQueryService:
    def __init__(self, *, source_reader: NotificationStatsSourceReader) -> None:
        if source_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("NotificationStatsQueryService requires source_reader")
        self._source_reader = source_reader

    def get_portfolio_stats(
        self,
        *,
        owner_user_id: UserId,
        period: NotificationStatsPeriod,
        generated_at: datetime,
        timezone: str = "UTC",
    ) -> NotificationStatsSnapshot:
        window = _period_window(period=period, generated_at=generated_at, timezone=timezone)
        return self._build_snapshot(
            owner_user_id=owner_user_id,
            scope_kind="portfolio",
            scope_ref=None,
            strategy_ref=None,
            exchange_ref=None,
            window=window,
            generated_at=generated_at,
        )

    def get_portfolio_stats_for_window(
        self,
        *,
        owner_user_id: UserId,
        period: NotificationStatsPeriod,
        period_start: datetime,
        period_end: datetime,
        generated_at: datetime,
        timezone: str = "UTC",
    ) -> NotificationStatsSnapshot:
        if period_start >= period_end:
            raise ValueError("stats report window must be non-empty")
        return self._build_snapshot(
            owner_user_id=owner_user_id,
            scope_kind="portfolio",
            scope_ref=None,
            strategy_ref=None,
            exchange_ref=None,
            window=NotificationStatsPeriodWindow(
                period=period,
                timezone=timezone,
                start_at=period_start,
                end_at=period_end,
            ),
            generated_at=generated_at,
        )

    def get_strategy_stats(
        self,
        *,
        owner_user_id: UserId,
        strategy_ref: str,
        period: NotificationStatsPeriod,
        generated_at: datetime,
        timezone: str = "UTC",
    ) -> NotificationStatsSnapshot:
        window = _period_window(period=period, generated_at=generated_at, timezone=timezone)
        return self._build_snapshot(
            owner_user_id=owner_user_id,
            scope_kind="strategy",
            scope_ref=strategy_ref,
            strategy_ref=strategy_ref,
            exchange_ref=None,
            window=window,
            generated_at=generated_at,
        )

    def get_exchange_stats(
        self,
        *,
        owner_user_id: UserId,
        exchange_ref: str,
        period: NotificationStatsPeriod,
        generated_at: datetime,
        timezone: str = "UTC",
    ) -> NotificationStatsSnapshot:
        window = _period_window(period=period, generated_at=generated_at, timezone=timezone)
        return self._build_snapshot(
            owner_user_id=owner_user_id,
            scope_kind="exchange",
            scope_ref=exchange_ref,
            strategy_ref=None,
            exchange_ref=exchange_ref,
            window=window,
            generated_at=generated_at,
        )

    def _build_snapshot(
        self,
        *,
        owner_user_id: UserId,
        scope_kind: NotificationStatsScopeKind,
        scope_ref: str | None,
        strategy_ref: str | None,
        exchange_ref: str | None,
        window: NotificationStatsPeriodWindow,
        generated_at: datetime,
    ) -> NotificationStatsSnapshot:
        source_result = self._source_reader.read_stats_rows(
            owner_user_id=owner_user_id,
            period_start=window.start_at,
            period_end=window.end_at,
            strategy_ref=strategy_ref,
            exchange_ref=exchange_ref,
        )
        rows = source_result.rows
        present_sources = {row.source for row in rows}
        missing_sources = tuple(
            source
            for source in (*_REQUIRED_SOURCES, *source_result.unavailable_sources)
            if source not in present_sources
        )
        latest_source_at = max((row.observed_at for row in rows), default=None)
        accounting_rows = tuple(row for row in rows if row.source == _SOURCE_PAPER_ACCOUNTING)
        pnl_complete = bool(accounting_rows) and all(row.pnl_complete for row in accounting_rows)
        pnl_missing = () if pnl_complete else ("pnl_complete_accounting",)
        quality_status = _quality_status(
            rows=rows,
            missing_sources=(*missing_sources, *pnl_missing),
        )

        return NotificationStatsSnapshot(
            owner_user_id=owner_user_id,
            scope_kind=scope_kind,
            scope_ref=scope_ref,
            period=window.period,
            timezone=window.timezone,
            period_start=window.start_at,
            period_end=window.end_at,
            generated_at=generated_at,
            quality_status=quality_status,
            missing_sources=(*missing_sources, *pnl_missing),
            latest_source_at=latest_source_at,
            freshness_seconds=(
                int((generated_at - latest_source_at).total_seconds())
                if latest_source_at is not None
                else None
            ),
            signal_count=sum(row.signal_count for row in rows),
            fill_count=sum(row.fill_count for row in rows),
            order_count=sum(row.order_count for row in rows),
            balance_count=sum(row.balance_count for row in rows),
            position_count=sum(row.position_count for row in rows),
            open_order_count=sum(row.open_order_count for row in rows),
            realized_pnl=_sum_decimal(rows=accounting_rows, field="realized_pnl")
            if pnl_complete
            else None,
            unrealized_pnl=_sum_decimal(rows=accounting_rows, field="unrealized_pnl")
            if pnl_complete
            else None,
            fee_total=_sum_decimal(rows=rows, field="fee_total"),
            funding_total=_sum_decimal(rows=rows, field="funding_total"),
            equity=_latest_decimal(rows=accounting_rows, field="equity") if pnl_complete else None,
        )


def render_notification_stats_snapshot(*, snapshot: NotificationStatsSnapshot) -> str:
    title = (
        f"{snapshot.scope_kind.title()} stats"
        if snapshot.scope_ref is None
        else f"{snapshot.scope_kind.title()} {snapshot.scope_ref} stats"
    )
    parts = [
        f"{title} for {snapshot.period}: {snapshot.quality_status}",
        f"signals={snapshot.signal_count}",
        f"fills={snapshot.fill_count}",
        f"orders={snapshot.order_count}",
        f"balances={snapshot.balance_count}",
    ]
    if snapshot.realized_pnl is not None:
        parts.append(f"realized_pnl={snapshot.realized_pnl}")
    if snapshot.unrealized_pnl is not None:
        parts.append(f"unrealized_pnl={snapshot.unrealized_pnl}")
    if snapshot.missing_sources:
        parts.append("missing_sources=" + ",".join(snapshot.missing_sources))
    return "; ".join(parts) + "."


def _period_window(
    *, period: NotificationStatsPeriod, generated_at: datetime, timezone: str
) -> NotificationStatsPeriodWindow:
    if timezone != "UTC":
        return NotificationStatsPeriodWindow(
            period=period,
            timezone="UTC",
            start_at=_period_start(period=period, generated_at=generated_at),
            end_at=generated_at,
        )
    return NotificationStatsPeriodWindow(
        period=period,
        timezone=timezone,
        start_at=_period_start(period=period, generated_at=generated_at),
        end_at=generated_at,
    )


def _period_start(*, period: NotificationStatsPeriod, generated_at: datetime) -> datetime:
    normalized = generated_at.astimezone(UTC)
    day_start = datetime.combine(normalized.date(), time.min, tzinfo=UTC)
    if period == "today":
        return day_start
    if period == "week":
        return day_start - timedelta(days=normalized.weekday())
    if period == "month":
        return day_start.replace(day=1)
    raise ValueError(f"unsupported stats period: {period}")


def _quality_status(
    *, rows: tuple[NotificationStatsSourceRow, ...], missing_sources: tuple[str, ...]
) -> NotificationStatsQualityStatus:
    if not rows:
        return "unavailable"
    if missing_sources:
        return "partial"
    return "complete"


def _sum_decimal(
    *, rows: tuple[NotificationStatsSourceRow, ...], field: str
) -> Decimal | None:
    values = [getattr(row, field) for row in rows]
    present = [value for value in values if isinstance(value, Decimal)]
    if not present:
        return None
    return sum(present, Decimal("0"))


def _latest_decimal(
    *, rows: tuple[NotificationStatsSourceRow, ...], field: str
) -> Decimal | None:
    if not rows:
        return None
    latest = max(rows, key=lambda row: row.observed_at)
    value = getattr(latest, field)
    return value if isinstance(value, Decimal) else None
