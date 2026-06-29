from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from trading.contexts.market_data.application.dto import (
    CandleRepairSource,
    CandleRepairSourceAttempt,
    CandleWithMeta,
    ClosedCandleTailResult,
    ClosedCandleTailRow,
)
from trading.contexts.strategy.application import ClosedCandleTailProvider
from trading.shared_kernel.primitives import (
    Candle,
    CandleMeta,
    InstrumentId,
    MarketId,
    Symbol,
    TimeRange,
    UtcTimestamp,
)

_INSTRUMENT_ID = InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT"))
_INSTRUMENT_KEY = "binance:spot:BTCUSDT"


class _FakeClosedCandleTailProvider:
    def __init__(self, *, result: ClosedCandleTailResult) -> None:
        self._result = result
        self.calls: list[tuple[str, UtcTimestamp, UtcTimestamp, str]] = []

    def get_closed_1m_tail(
        self,
        *,
        instrument_id: InstrumentId,
        instrument_key: str,
        start_ts_open: UtcTimestamp,
        end_ts_open: UtcTimestamp,
        correlation_id: str,
    ) -> ClosedCandleTailResult:
        self.calls.append((instrument_key, start_ts_open, end_ts_open, correlation_id))
        assert instrument_id == _INSTRUMENT_ID
        return self._result


def test_closed_candle_tail_provider_can_return_continuous_result() -> None:
    result = ClosedCandleTailResult(
        instrument_id=_INSTRUMENT_ID,
        instrument_key=_INSTRUMENT_KEY,
        time_range=_range(0, 2),
        candles=(_row(0, "redis_hot_cache"), _row(1, "redis_hot_cache")),
        sources_attempted=(
            CandleRepairSourceAttempt(source="redis_hot_cache", status="succeeded"),
        ),
    )
    fake = _FakeClosedCandleTailProvider(result=result)
    provider: ClosedCandleTailProvider = fake

    returned = provider.get_closed_1m_tail(
        instrument_id=_INSTRUMENT_ID,
        instrument_key=_INSTRUMENT_KEY,
        start_ts_open=_ts(0),
        end_ts_open=_ts(2),
        correlation_id="stage01-continuous",
    )

    assert returned.continuous is True
    assert returned.missing_ts_opens == ()
    assert tuple(str(ts) for ts in returned.restored_ts_opens) == (
        "2026-06-29T12:00:00.000Z",
        "2026-06-29T12:01:00.000Z",
    )
    assert fake.calls == [(_INSTRUMENT_KEY, _ts(0), _ts(2), "stage01-continuous")]


def test_closed_candle_tail_provider_can_return_missing_result_deterministically() -> None:
    result = ClosedCandleTailResult(
        instrument_id=_INSTRUMENT_ID,
        instrument_key=_INSTRUMENT_KEY,
        time_range=_range(0, 3),
        candles=(_row(0, "redis_hot_cache"), _row(2, "rest")),
        sources_attempted=(
            CandleRepairSourceAttempt(source="redis_hot_cache", status="miss"),
            CandleRepairSourceAttempt(
                source="clickhouse",
                status="failed",
                error_code="http_connection_reset",
            ),
            CandleRepairSourceAttempt(source="rest", status="miss"),
        ),
        error_code="short_tail_missing",
    )
    fake = _FakeClosedCandleTailProvider(result=result)
    provider: ClosedCandleTailProvider = fake

    returned = provider.get_closed_1m_tail(
        instrument_id=_INSTRUMENT_ID,
        instrument_key=_INSTRUMENT_KEY,
        start_ts_open=_ts(0),
        end_ts_open=_ts(3),
        correlation_id="stage01-missing",
    )

    assert returned.continuous is False
    assert tuple(str(ts) for ts in returned.restored_ts_opens) == (
        "2026-06-29T12:00:00.000Z",
        "2026-06-29T12:02:00.000Z",
    )
    assert tuple(str(ts) for ts in returned.missing_ts_opens) == (
        "2026-06-29T12:01:00.000Z",
    )
    assert tuple(attempt.source for attempt in returned.sources_attempted) == (
        "redis_hot_cache",
        "clickhouse",
        "rest",
    )
    assert returned.error_code == "short_tail_missing"


def _row(minute: int, source: CandleRepairSource) -> ClosedCandleTailRow:
    ts_open = _ts(minute)
    ts_close = UtcTimestamp(ts_open.value + timedelta(minutes=1))
    return ClosedCandleTailRow(
        candle=CandleWithMeta(
            candle=Candle(
                instrument_id=_INSTRUMENT_ID,
                ts_open=ts_open,
                ts_close=ts_close,
                open=100.0 + minute,
                high=101.0 + minute,
                low=99.0 + minute,
                close=100.5 + minute,
                volume_base=1.0,
                volume_quote=100.5,
            ),
            meta=CandleMeta(
                source="ws",
                ingested_at=ts_close,
                ingest_id=UUID(int=minute + 1),
                instrument_key=_INSTRUMENT_KEY,
                trades_count=None,
                taker_buy_volume_base=None,
                taker_buy_volume_quote=None,
            ),
        ),
        source=source,
    )


def _range(start_minute: int, end_minute: int) -> TimeRange:
    return TimeRange(start=_ts(start_minute), end=_ts(end_minute))


def _ts(minute: int) -> UtcTimestamp:
    return UtcTimestamp(datetime(2026, 6, 29, 12, minute, tzinfo=timezone.utc))
