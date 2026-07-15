from __future__ import annotations

from dataclasses import dataclass

from trading.contexts.backtest.application.ports import (
    ResearchOrganizationScope,
    ResearchOrganizationScopeResolver,
)
from trading.contexts.market_data.application.dto import CanonicalCandleBatch1m
from trading.contexts.market_data.application.ports.stores import CanonicalCandleReader
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UserId


@dataclass(frozen=True, slots=True)
class OrganizationScopedCanonicalCandleBatch:
    """Installation-shared candle content with its server-derived access scope."""

    scope: ResearchOrganizationScope
    candles: CanonicalCandleBatch1m


@dataclass(frozen=True, slots=True)
class OrganizationScopedCanonicalCandleReader:
    """Authorize a research actor before reading shared canonical candles."""

    scope_resolver: ResearchOrganizationScopeResolver
    canonical_reader: CanonicalCandleReader

    def read_1m_arrays(
        self,
        *,
        user_id: UserId,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> OrganizationScopedCanonicalCandleBatch:
        scope = self.scope_resolver.resolve(user_id=user_id)
        candles = self.canonical_reader.read_1m_arrays(instrument_id, time_range)
        return OrganizationScopedCanonicalCandleBatch(
            scope=scope,
            candles=candles,
        )


__all__ = [
    "OrganizationScopedCanonicalCandleBatch",
    "OrganizationScopedCanonicalCandleReader",
]
