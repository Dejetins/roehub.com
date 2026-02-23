from __future__ import annotations

from dataclasses import dataclass

from trading.contexts.market_data.application.ports.stores.enabled_tradable_instrument_search_reader import (  # noqa: E501
    EnabledTradableInstrumentSearchReader,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId

DEFAULT_INSTRUMENT_SEARCH_LIMIT = 50
MAX_INSTRUMENT_SEARCH_LIMIT = 200


@dataclass(frozen=True, slots=True)
class SearchEnabledTradableInstrumentsUseCase:
    """
    Use-case for market-scoped enabled/tradable instrument prefix search.

    Docs:
      - docs/architecture/market_data/market-data-reference-api-v1.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/
        enabled_tradable_instrument_search_reader.py
      - src/trading/contexts/market_data/adapters/outbound/persistence/clickhouse/
        enabled_tradable_instrument_search_reader.py
      - apps/api/routes/market_data_reference.py
    """

    reader: EnabledTradableInstrumentSearchReader
    default_limit: int = DEFAULT_INSTRUMENT_SEARCH_LIMIT
    max_limit: int = MAX_INSTRUMENT_SEARCH_LIMIT

    def __post_init__(self) -> None:
        """
        Validate required collaborators and limit invariants.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Reader is non-null.
        - Limits are positive and `default_limit <= max_limit`.

        Errors/Exceptions:
        - Raises `ValueError` when dependencies or limits are invalid.

        Side effects:
        - None.
        """
        if self.reader is None:  # type: ignore[truthy-bool]
            raise ValueError("SearchEnabledTradableInstrumentsUseCase requires reader")
        if self.default_limit <= 0:
            raise ValueError("SearchEnabledTradableInstrumentsUseCase.default_limit must be > 0")
        if self.max_limit <= 0:
            raise ValueError("SearchEnabledTradableInstrumentsUseCase.max_limit must be > 0")
        if self.default_limit > self.max_limit:
            raise ValueError(
                "SearchEnabledTradableInstrumentsUseCase.default_limit must be <= max_limit"
            )

    def execute(
        self,
        *,
        market_id: MarketId,
        q: str | None = None,
        limit: int | None = None,
    ) -> tuple[InstrumentId, ...]:
        """
        Search instruments with deterministic symbol ordering and validated limit.

        Parameters:
        - market_id: target market identifier.
        - q: optional symbol prefix; blank string means no filter.
        - limit: optional request limit; defaults to use-case `default_limit`.

        Returns:
        - Tuple of matching instrument ids ordered by `symbol ASC`.

        Assumptions/Invariants:
        - Unknown/disabled market is represented as empty result from reader adapter.

        Errors/Exceptions:
        - Raises `ValueError` on invalid limit values.
        - Propagates reader/storage errors.

        Side effects:
        - Executes one read query through search reader port.
        """
        effective_limit = _resolve_limit(
            limit=limit,
            default_limit=self.default_limit,
            max_limit=self.max_limit,
        )
        symbol_prefix = _normalize_prefix(q=q)
        instruments = list(
            self.reader.search_enabled_tradable_by_market(
                market_id=market_id,
                symbol_prefix=symbol_prefix,
                limit=effective_limit,
            )
        )
        instruments.sort(key=_instrument_symbol_sort_key)
        return tuple(instruments[:effective_limit])


def _resolve_limit(*, limit: int | None, default_limit: int, max_limit: int) -> int:
    """
    Resolve effective limit and enforce API contract bounds.

    Parameters:
    - limit: user-provided optional limit.
    - default_limit: fallback value when limit is omitted.
    - max_limit: allowed maximum limit.

    Returns:
    - Effective validated limit.

    Assumptions/Invariants:
    - `default_limit` and `max_limit` are positive integers.

    Errors/Exceptions:
    - Raises `ValueError` when limit is non-positive or greater than `max_limit`.

    Side effects:
    - None.
    """
    if limit is None:
        return default_limit
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if limit > max_limit:
        raise ValueError(f"limit must be <= {max_limit}")
    return limit


def _normalize_prefix(*, q: str | None) -> str | None:
    """
    Normalize symbol prefix filter value.

    Parameters:
    - q: raw query prefix from API layer.

    Returns:
    - Uppercase trimmed prefix or `None` when no filter should be applied.

    Assumptions/Invariants:
    - Prefix comparison is case-insensitive by using uppercase normalization.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if q is None:
        return None
    normalized = q.strip().upper()
    if not normalized:
        return None
    return normalized


def _instrument_symbol_sort_key(instrument_id: InstrumentId) -> str:
    """
    Build deterministic symbol sort key for instrument ids.

    Parameters:
    - instrument_id: instrument identity row.

    Returns:
    - Symbol string used for ascending sort.

    Assumptions/Invariants:
    - `Symbol` primitive is already normalized to uppercase.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    return str(instrument_id.symbol)
