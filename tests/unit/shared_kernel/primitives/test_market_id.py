from trading.shared_kernel.primitives import MarketId


def test_market_id_accepts_valid_uint16_range() -> None:
    assert MarketId(1).value == 1
    assert MarketId(65535).value == 65535


def test_market_id_rejects_zero_negative_and_too_large() -> None:
    try:
        MarketId(0)
        assert False, "Expected ValueError for MarketId(0)"
    except ValueError:
        assert True

    try:
        MarketId(-1)
        assert False, "Expected ValueError for MarketId(-1)"
    except ValueError:
        assert True

    try:
        MarketId(65536)
        assert False, "Expected ValueError for MarketId(65536)"
    except ValueError:
        assert True


def test_market_id_rejects_bool() -> None:
    # bool is a subclass of int, но для ID это ошибка.
    try:
        MarketId(True)
        assert False, "Expected ValueError for MarketId(True)"
    except ValueError:
        assert True
