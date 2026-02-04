from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol


def test_instrument_id_builds_and_serializes() -> None:
    ins = InstrumentId(MarketId(1), Symbol(" btcusdt "))
    d = ins.as_dict()

    assert d["market_id"] == 1
    assert d["symbol"] == "BTCUSDT"
    assert str(ins) == "1:BTCUSDT"


def test_instrument_id_requires_fields() -> None:
    # Если кто-то попытается создать с None — должен упасть.
    try:
        InstrumentId(None, Symbol("BTCUSDT"))  # type: ignore[arg-type]
        assert False, "Expected ValueError for InstrumentId(None, Symbol(...))"
    except ValueError:
        assert True

    try:
        InstrumentId(MarketId(1), None)  # type: ignore[arg-type]
        assert False, "Expected ValueError for InstrumentId(MarketId(...), None)"
    except ValueError:
        assert True
