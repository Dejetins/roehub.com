from __future__ import annotations

from dataclasses import dataclass

import pytest

from trading.contexts.market_data.application.dto.reference_api import EnabledMarketReference
from trading.contexts.market_data.application.use_cases import (
    ListEnabledMarketsUseCase,
    SearchEnabledTradableInstrumentsUseCase,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol


@dataclass(frozen=True, slots=True)
class _EnabledMarketReaderStub:
    """
    Stub returning deterministic enabled market rows for use-case tests.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - tests/unit/contexts/market_data/application/use_cases/test_reference_api_use_cases.py
      - src/trading/contexts/market_data/application/use_cases/list_enabled_markets.py
      - src/trading/contexts/market_data/application/ports/stores/enabled_market_reader.py
    """

    rows: tuple[EnabledMarketReference, ...]

    def list_enabled_markets(self) -> tuple[EnabledMarketReference, ...]:
        """
        Return preconfigured enabled market rows.

        Parameters:
        - None.

        Returns:
        - Tuple of enabled market rows.

        Assumptions/Invariants:
        - Test fixture rows are already valid.

        Errors/Exceptions:
        - None.

        Side effects:
        - None.
        """
        return self.rows


class _SearchReaderStub:
    """
    Stub capturing search arguments for use-case behavior assertions.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - tests/unit/contexts/market_data/application/use_cases/test_reference_api_use_cases.py
      - src/trading/contexts/market_data/application/use_cases/
        search_enabled_tradable_instruments.py
      - src/trading/contexts/market_data/application/ports/stores/
        enabled_tradable_instrument_search_reader.py
    """

    def __init__(self, *, rows_by_market: dict[int, tuple[InstrumentId, ...]]) -> None:
        """
        Store deterministic search rows keyed by market id.

        Parameters:
        - rows_by_market: mapping `market_id -> instrument rows`.

        Returns:
        - None.

        Assumptions/Invariants:
        - Keys are valid positive market ids used in tests.

        Errors/Exceptions:
        - None.

        Side effects:
        - Initializes mutable call-capture state.
        """
        self._rows_by_market = rows_by_market
        self.last_market_id: int | None = None
        self.last_symbol_prefix: str | None = None
        self.last_limit: int | None = None

    def search_enabled_tradable_by_market(
        self,
        *,
        market_id: MarketId,
        symbol_prefix: str | None,
        limit: int,
    ) -> tuple[InstrumentId, ...]:
        """
        Capture search arguments and return configured rows for requested market.

        Parameters:
        - market_id: market id filter.
        - symbol_prefix: optional prefix value from use-case normalization.
        - limit: requested row limit.

        Returns:
        - Tuple of preconfigured instrument ids for requested market.

        Assumptions/Invariants:
        - Unknown market ids map to empty tuples.

        Errors/Exceptions:
        - None.

        Side effects:
        - Stores latest call arguments for assertions.
        """
        self.last_market_id = market_id.value
        self.last_symbol_prefix = symbol_prefix
        self.last_limit = limit
        return self._rows_by_market.get(market_id.value, ())


def test_list_enabled_markets_use_case_returns_market_id_sorted_rows() -> None:
    """
    Verify use-case enforces deterministic ordering by `market_id ASC`.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Reader may return rows in arbitrary order.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    use_case = ListEnabledMarketsUseCase(
        reader=_EnabledMarketReaderStub(
            rows=(
                _market(2, "binance", "futures", "binance:futures"),
                _market(1, "binance", "spot", "binance:spot"),
            )
        )
    )

    rows = use_case.execute()

    assert [item.market_id.value for item in rows] == [1, 2]


def test_search_use_case_applies_default_limit_prefix_normalization_and_symbol_sort() -> None:
    """
    Verify default limit, prefix normalization, and deterministic `symbol ASC` ordering.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Reader may return rows in arbitrary symbol order.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    reader = _SearchReaderStub(
        rows_by_market={
            1: (
                _instrument(1, "ETHUSDT"),
                _instrument(1, "BTCUSDT"),
            )
        }
    )
    use_case = SearchEnabledTradableInstrumentsUseCase(reader=reader)

    rows = use_case.execute(
        market_id=MarketId(1),
        q=" bt ",
        limit=None,
    )

    assert reader.last_market_id == 1
    assert reader.last_symbol_prefix == "BT"
    assert reader.last_limit == 50
    assert [str(item.symbol) for item in rows] == ["BTCUSDT", "ETHUSDT"]


def test_search_use_case_treats_blank_prefix_as_no_filter() -> None:
    """
    Verify blank `q` values are normalized to `None` before port call.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Blank strings must not produce `LIKE '%'` style filters in adapter.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    reader = _SearchReaderStub(rows_by_market={1: ()})
    use_case = SearchEnabledTradableInstrumentsUseCase(reader=reader)

    _ = use_case.execute(
        market_id=MarketId(1),
        q="   ",
        limit=10,
    )

    assert reader.last_symbol_prefix is None


def test_search_use_case_accepts_max_limit_value() -> None:
    """
    Verify use-case accepts `limit=200` and forwards it to reader.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Max limit follows reference API v1 contract.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    reader = _SearchReaderStub(rows_by_market={1: ()})
    use_case = SearchEnabledTradableInstrumentsUseCase(reader=reader)

    rows = use_case.execute(
        market_id=MarketId(1),
        q=None,
        limit=200,
    )

    assert rows == ()
    assert reader.last_limit == 200


def test_search_use_case_rejects_limit_above_maximum() -> None:
    """
    Verify limit values above API contract max raise deterministic validation error.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - `max_limit` defaults to 200.

    Errors/Exceptions:
    - Expects `ValueError` for invalid limit.

    Side effects:
    - None.
    """
    use_case = SearchEnabledTradableInstrumentsUseCase(reader=_SearchReaderStub(rows_by_market={}))

    with pytest.raises(ValueError, match="limit must be <= 200"):
        use_case.execute(market_id=MarketId(1), q=None, limit=201)


def test_search_use_case_returns_empty_for_unknown_or_disabled_market() -> None:
    """
    Verify unknown/disabled market is represented as empty tuple without errors.

    Parameters:
    - None.

    Returns:
    - None.

    Assumptions/Invariants:
    - Reader returns empty rows for markets not available in latest enabled state.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    use_case = SearchEnabledTradableInstrumentsUseCase(
        reader=_SearchReaderStub(rows_by_market={1: (_instrument(1, "BTCUSDT"),)})
    )

    rows = use_case.execute(
        market_id=MarketId(999),
        q=None,
        limit=50,
    )

    assert rows == ()


def _market(
    market_id: int,
    exchange_name: str,
    market_type: str,
    market_code: str,
) -> EnabledMarketReference:
    """
    Build enabled market read-model fixture row.

    Parameters:
    - market_id: market identifier.
    - exchange_name: exchange literal.
    - market_type: market type literal.
    - market_code: composed market code literal.

    Returns:
    - `EnabledMarketReference` fixture instance.

    Assumptions/Invariants:
    - Fixture values satisfy read-model invariants.

    Errors/Exceptions:
    - Propagates constructor validation errors for invalid fixture values.

    Side effects:
    - None.
    """
    return EnabledMarketReference(
        market_id=MarketId(market_id),
        exchange_name=exchange_name,
        market_type=market_type,
        market_code=market_code,
    )


def _instrument(market_id: int, symbol: str) -> InstrumentId:
    """
    Build instrument id fixture row.

    Parameters:
    - market_id: market identifier.
    - symbol: instrument symbol value.

    Returns:
    - `InstrumentId` fixture instance.

    Assumptions/Invariants:
    - Symbol normalization is handled by `Symbol` primitive.

    Errors/Exceptions:
    - Propagates constructor validation errors for invalid fixture values.

    Side effects:
    - None.
    """
    return InstrumentId(
        market_id=MarketId(market_id),
        symbol=Symbol(symbol),
    )
