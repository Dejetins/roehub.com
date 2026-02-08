from __future__ import annotations

from trading.shared_kernel.primitives import InstrumentId

from .runtime_config import MarketDataRuntimeConfig


def build_instrument_key(*, cfg: MarketDataRuntimeConfig, instrument_id: InstrumentId) -> str:
    """
    Build a canonical instrument key for ingestion metadata.

    Parameters:
    - cfg: runtime market-data configuration with market_id -> exchange/market_type mapping.
    - instrument_id: domain instrument identity `(market_id, symbol)` for which key is built.

    Returns:
    - Canonical key in the format `"{exchange}:{market_type}:{symbol}"`.

    Assumptions/Invariants:
    - `instrument_id.market_id` exists in `cfg`; otherwise config is considered invalid for runtime.
    - `instrument_id.symbol` is already normalized by `Symbol` primitive.

    Errors/Exceptions:
    - Propagates `KeyError` from `cfg.market_by_id(...)` when market_id is missing.

    Side effects:
    - None.
    """
    market = cfg.market_by_id(instrument_id.market_id)
    return f"{market.exchange}:{market.market_type}:{instrument_id.symbol}"
