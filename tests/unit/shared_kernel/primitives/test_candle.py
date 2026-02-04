from datetime import datetime, timezone

from trading.shared_kernel.primitives import Candle, InstrumentId, MarketId, Symbol, UtcTimestamp


def test_candle_accepts_valid_values() -> None:
    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    ts_open = UtcTimestamp(datetime(2026, 2, 4, 12, 0, 0, 0, tzinfo=timezone.utc))
    ts_close = UtcTimestamp(datetime(2026, 2, 4, 12, 1, 0, 0, tzinfo=timezone.utc))

    c = Candle(
        instrument_id=instrument,
        ts_open=ts_open,
        ts_close=ts_close,
        open=100.0,
        high=110.0,
        low=90.0,
        close=105.0,
        volume_base=1.23,
        volume_quote=None,
    )

    d = c.as_dict()
    assert d["instrument_id"]["market_id"] == 1
    assert d["instrument_id"]["symbol"] == "BTCUSDT"
    assert d["ts_open"] == "2026-02-04T12:00:00.000Z"
    assert d["ts_close"] == "2026-02-04T12:01:00.000Z"


def test_candle_rejects_invalid_time_order() -> None:
    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    ts = UtcTimestamp(datetime(2026, 2, 4, 12, 0, 0, 0, tzinfo=timezone.utc))

    try:
        Candle(
            instrument_id=instrument,
            ts_open=ts,
            ts_close=ts,  # не может быть равно
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume_base=0.0,
            volume_quote=None,
        )
        assert False, "Expected ValueError for ts_open >= ts_close"
    except ValueError:
        assert True


def test_candle_rejects_invalid_ohlc_invariants() -> None:
    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    ts_open = UtcTimestamp(datetime(2026, 2, 4, 12, 0, 0, 0, tzinfo=timezone.utc))
    ts_close = UtcTimestamp(datetime(2026, 2, 4, 12, 1, 0, 0, tzinfo=timezone.utc))

    # high < max(open, close)
    try:
        Candle(
            instrument_id=instrument,
            ts_open=ts_open,
            ts_close=ts_close,
            open=100.0,
            high=101.0,
            low=90.0,
            close=105.0,
            volume_base=1.0,
            volume_quote=None,
        )
        assert False, "Expected ValueError for high invariant"
    except ValueError:
        assert True

    # low > min(open, close)
    try:
        Candle(
            instrument_id=instrument,
            ts_open=ts_open,
            ts_close=ts_close,
            open=100.0,
            high=110.0,
            low=99.0,
            close=95.0,
            volume_base=1.0,
            volume_quote=None,
        )
        assert False, "Expected ValueError for low invariant"
    except ValueError:
        assert True


def test_candle_rejects_negative_volumes() -> None:
    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))

    ts_open = UtcTimestamp(datetime(2026, 2, 4, 12, 0, 0, 0, tzinfo=timezone.utc))
    ts_close = UtcTimestamp(datetime(2026, 2, 4, 12, 1, 0, 0, tzinfo=timezone.utc))

    try:
        Candle(
            instrument_id=instrument,
            ts_open=ts_open,
            ts_close=ts_close,
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume_base=-1.0,
            volume_quote=None,
        )
        assert False, "Expected ValueError for negative volume_base"
    except ValueError:
        assert True

    try:
        Candle(
            instrument_id=instrument,
            ts_open=ts_open,
            ts_close=ts_close,
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume_base=1.0,
            volume_quote=-0.01,
        )
        assert False, "Expected ValueError for negative volume_quote"
    except ValueError:
        assert True
