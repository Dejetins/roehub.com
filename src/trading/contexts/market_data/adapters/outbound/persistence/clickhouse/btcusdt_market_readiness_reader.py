from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.dto.reference_api import (
    BTCUSDT_MARKET_READINESS_MARKETS,
    BTCUSDT_MARKET_READINESS_SYMBOL,
    BTCUSDTMarketReferenceSnapshot,
)
from trading.contexts.market_data.application.ports.stores import (
    BTCUSDTMarketReadinessReferenceReader,
)
from trading.shared_kernel.primitives import MarketId


@dataclass(frozen=True, slots=True)
class ClickHouseBTCUSDTMarketReadinessReader(BTCUSDTMarketReadinessReferenceReader):
    gateway: ClickHouseGateway
    database: str = "market_data"

    def __post_init__(self) -> None:
        if self.gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseBTCUSDTMarketReadinessReader requires gateway")
        if not self.database.strip():
            raise ValueError("ClickHouseBTCUSDTMarketReadinessReader requires non-empty database")

    def list_btcusdt_reference_rows(self) -> Sequence[BTCUSDTMarketReferenceSnapshot]:
        query, parameters = _build_query(database=self.database)
        rows = self.gateway.select(query, parameters)
        return tuple(_row_to_snapshot(row=row) for row in rows)


def _build_query(*, database: str) -> tuple[str, Mapping[str, Any]]:
    market_codes = [market_code for _, _, market_code in BTCUSDT_MARKET_READINESS_MARKETS]
    query = f"""
    SELECT
        markets.market_id AS market_id,
        markets.exchange_name AS exchange_name,
        markets.market_type AS market_type,
        markets.market_code AS market_code,
        markets.is_enabled AS market_enabled,
        instruments.symbol AS symbol,
        instruments.status AS status,
        instruments.is_tradable AS is_tradable,
        instruments.base_asset AS base_asset,
        instruments.quote_asset AS quote_asset,
        instruments.price_step AS price_step,
        instruments.qty_step AS qty_step,
        instruments.min_notional AS min_notional
    FROM
    (
        SELECT
            market_id,
            exchange_name,
            market_type,
            market_code,
            is_enabled,
            updated_at
        FROM {database}.ref_market
        WHERE market_code IN %(market_codes)s
        ORDER BY updated_at DESC
        LIMIT 1 BY market_id
    ) AS markets
    LEFT JOIN
    (
        SELECT
            market_id,
            symbol,
            status,
            is_tradable,
            base_asset,
            quote_asset,
            price_step,
            qty_step,
            min_notional,
            updated_at
        FROM {database}.ref_instruments
        WHERE symbol = %(symbol)s
        ORDER BY updated_at DESC
        LIMIT 1 BY market_id, symbol
    ) AS instruments ON instruments.market_id = markets.market_id
    ORDER BY markets.market_code ASC
    """
    return query, {
        "market_codes": tuple(market_codes),
        "symbol": BTCUSDT_MARKET_READINESS_SYMBOL,
    }


def _row_to_snapshot(*, row: Mapping[str, Any]) -> BTCUSDTMarketReferenceSnapshot:
    exchange_name = str(row["exchange_name"]).strip()
    market_type = str(row["market_type"]).strip()
    market_code = str(row["market_code"]).strip()
    symbol = str(row.get("symbol") or BTCUSDT_MARKET_READINESS_SYMBOL).strip().upper()
    return BTCUSDTMarketReferenceSnapshot(
        market_id=MarketId(int(row["market_id"])),
        exchange_name=exchange_name,
        market_type=market_type,
        market_code=market_code,
        market_enabled=_as_bool(row.get("market_enabled")),
        symbol=symbol,
        status=_str_or_none(row.get("status")),
        is_tradable=_int_or_none(row.get("is_tradable")),
        base_asset=_str_or_none(row.get("base_asset")),
        quote_asset=_str_or_none(row.get("quote_asset")),
        price_step=_float_or_none(row.get("price_step")),
        qty_step=_float_or_none(row.get("qty_step")),
        min_notional=_float_or_none(row.get("min_notional")),
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return int(str(value)) == 1
    except (TypeError, ValueError):
        return False


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
