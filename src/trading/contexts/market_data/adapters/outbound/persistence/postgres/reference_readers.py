"""Current reference rows, stable ordering and bounded server-side transport."""

from dataclasses import dataclass

from trading.contexts.market_data.application.dto.reference_api import (
    BTCUSDTMarketReferenceSnapshot,
    EnabledMarketReference,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol

from .ingestion_gateway import IngestionGateway
from .query_support import rows

MARKETS = """SELECT market_id,exchange_name,market_type,market_code
FROM market_data.ref_market WHERE is_enabled=1 ORDER BY market_id"""
INSTRUMENTS = """SELECT i.market_id,i.symbol FROM market_data.ref_instruments i
JOIN market_data.ref_market m USING(market_id)
WHERE m.is_enabled=1 AND i.status='ENABLED' AND i.is_tradable=1"""
READINESS = """SELECT m.market_id,m.exchange_name,m.market_type,m.market_code,
m.is_enabled,i.status,i.is_tradable,i.base_asset,i.quote_asset,i.price_step,i.qty_step,i.min_notional
FROM market_data.ref_market m LEFT JOIN market_data.ref_instruments i
ON i.market_id=m.market_id AND i.symbol='BTCUSDT'
WHERE m.market_id IN (1,2,3,4) ORDER BY m.market_code"""


@dataclass(frozen=True)
class PostgresEnabledMarketReader:
    gateway: IngestionGateway

    def list_enabled_markets(self):
        return self.gateway.run(
            lambda c: tuple(
                EnabledMarketReference(
                    MarketId(r["market_id"]), r["exchange_name"], r["market_type"], r["market_code"]
                )
                for r in rows(c, MARKETS)
            )
        )


@dataclass(frozen=True)
class PostgresEnabledInstrumentReader:
    gateway: IngestionGateway

    def list_enabled_tradable(self):
        return self.gateway.run(
            lambda c: tuple(
                InstrumentId(MarketId(r["market_id"]), Symbol(r["symbol"]))
                for r in rows(c, INSTRUMENTS + " ORDER BY i.market_id,i.symbol")
            )
        )


@dataclass(frozen=True)
class PostgresEnabledTradableInstrumentSearchReader:
    gateway: IngestionGateway

    def search_enabled_tradable_by_market(self, *, market_id, symbol_prefix, limit):
        if not 1 <= limit <= 1000:
            raise ValueError("invalid_search_limit")
        prefix = (symbol_prefix or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return self.gateway.run(
            lambda c: tuple(
                InstrumentId(MarketId(r["market_id"]), Symbol(r["symbol"]))
                for r in rows(
                    c,
                    INSTRUMENTS
                    + " AND i.market_id=%s AND i.symbol LIKE %s ORDER BY i.symbol LIMIT %s",
                    (market_id.value, prefix + "%", limit),
                )
            )
        )


@dataclass(frozen=True)
class PostgresBTCUSDTMarketReadinessReader:
    gateway: IngestionGateway

    def list_btcusdt_reference_rows(self):
        return self.gateway.run(
            lambda c: tuple(
                BTCUSDTMarketReferenceSnapshot(
                    market_id=MarketId(r["market_id"]),
                    exchange_name=r["exchange_name"],
                    market_type=r["market_type"],
                    market_code=r["market_code"],
                    market_enabled=r["is_enabled"] == 1,
                    symbol="BTCUSDT",
                    status=r["status"],
                    is_tradable=r["is_tradable"],
                    base_asset=r["base_asset"],
                    quote_asset=r["quote_asset"],
                    price_step=r["price_step"],
                    qty_step=r["qty_step"],
                    min_notional=r["min_notional"],
                )
                for r in rows(c, READINESS)
            )
        )
