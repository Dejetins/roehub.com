from __future__ import annotations

from dataclasses import dataclass

from trading.shared_kernel.primitives import MarketId


@dataclass(frozen=True, slots=True)
class EnabledMarketReference:
    """
    Read-model row for one enabled market in reference API use-cases.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/enabled_market_reader.py
      - src/trading/contexts/market_data/application/use_cases/list_enabled_markets.py
      - apps/api/dto/market_data_reference.py
    """

    market_id: MarketId
    exchange_name: str
    market_type: str
    market_code: str

    def __post_init__(self) -> None:
        """
        Validate enabled market read-model invariants.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Text fields are non-empty after trimming.

        Errors/Exceptions:
        - Raises `ValueError` when one of fields is blank.

        Side effects:
        - None.
        """
        if not self.exchange_name.strip():
            raise ValueError("EnabledMarketReference.exchange_name must be non-empty")
        if not self.market_type.strip():
            raise ValueError("EnabledMarketReference.market_type must be non-empty")
        if not self.market_code.strip():
            raise ValueError("EnabledMarketReference.market_code must be non-empty")
