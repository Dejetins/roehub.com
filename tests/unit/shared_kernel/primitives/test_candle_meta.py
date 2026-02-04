from datetime import datetime, timezone
from uuid import uuid4

from trading.shared_kernel.primitives import CandleMeta, UtcTimestamp


def test_candle_meta_accepts_ws_rest_file_sources() -> None:
    ts = UtcTimestamp(datetime(2026, 2, 4, 12, 0, 0, 123456, tzinfo=timezone.utc))

    CandleMeta(
        source="ws",
        ingested_at=ts,
        ingest_id=None,
        instrument_key="binance:spot:BTCUSDT",
        trades_count=None,
        taker_buy_volume_base=None,
        taker_buy_volume_quote=None,
    )

    CandleMeta(
        source="rest",
        ingested_at=ts,
        ingest_id=uuid4(),
        instrument_key="binance:futures:BTCUSDT",
        trades_count=0,
        taker_buy_volume_base=0.0,
        taker_buy_volume_quote=0.0,
    )

    CandleMeta(
        source="file",
        ingested_at=ts,
        ingest_id=uuid4(),
        instrument_key="bybit:spot:BTCUSDT",
        trades_count=None,
        taker_buy_volume_base=None,
        taker_buy_volume_quote=None,
    )


def test_candle_meta_normalizes_source_and_instrument_key() -> None:
    ts = UtcTimestamp(datetime(2026, 2, 4, 12, 0, 0, 123456, tzinfo=timezone.utc))

    meta = CandleMeta(
        source="  WS  ",
        ingested_at=ts,
        ingest_id=None,
        instrument_key="  binance:spot:BTCUSDT  ",
        trades_count=None,
        taker_buy_volume_base=None,
        taker_buy_volume_quote=None,
    )

    assert meta.source == "ws"
    assert meta.instrument_key == "binance:spot:BTCUSDT"


def test_candle_meta_rejects_invalid_source() -> None:
    ts = UtcTimestamp(datetime(2026, 2, 4, 12, 0, 0, 123456, tzinfo=timezone.utc))

    try:
        CandleMeta(
            source="manual",  # не входит в допустимый набор
            ingested_at=ts,
            ingest_id=None,
            instrument_key="binance:spot:BTCUSDT",
            trades_count=None,
            taker_buy_volume_base=None,
            taker_buy_volume_quote=None,
        )
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_candle_meta_rejects_negative_values() -> None:
    ts = UtcTimestamp(datetime(2026, 2, 4, 12, 0, 0, 123456, tzinfo=timezone.utc))

    try:
        CandleMeta(
            source="rest",
            ingested_at=ts,
            ingest_id=None,
            instrument_key="binance:spot:BTCUSDT",
            trades_count=-1,  # нельзя
            taker_buy_volume_base=None,
            taker_buy_volume_quote=None,
        )
        assert False, "Expected ValueError"
    except ValueError:
        assert True

    try:
        CandleMeta(
            source="rest",
            ingested_at=ts,
            ingest_id=None,
            instrument_key="binance:spot:BTCUSDT",
            trades_count=None,
            taker_buy_volume_base=-0.1,  # нельзя
            taker_buy_volume_quote=None,
        )
        assert False, "Expected ValueError"
    except ValueError:
        assert True
