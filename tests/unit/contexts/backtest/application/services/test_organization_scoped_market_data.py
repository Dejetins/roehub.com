from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Iterator

import numpy as np

from trading.contexts.backtest.application.ports import ResearchOrganizationScope
from trading.contexts.backtest.application.services import (
    OrganizationScopedCanonicalCandleReader,
)
from trading.contexts.market_data.application.dto import (
    CandleWithMeta,
    CanonicalCandleBatch1m,
)
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    OrganizationId,
    Symbol,
    TimeRange,
    UserId,
    UtcTimestamp,
)


def test_scoped_reader_derives_organization_and_reuses_shared_candle_batch() -> None:
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000101")
    organization_id = OrganizationId.from_string("00000000-0000-0000-0000-000000000201")
    resolver = _ScopeResolver(organization_id=organization_id)
    canonical_reader = _CanonicalReader()
    service = OrganizationScopedCanonicalCandleReader(
        scope_resolver=resolver,
        canonical_reader=canonical_reader,
    )
    start = datetime(2026, 7, 13, tzinfo=UTC)

    result = service.read_1m_arrays(
        user_id=user_id,
        instrument_id=InstrumentId(MarketId(1), Symbol("BTCUSDT")),
        time_range=TimeRange(
            UtcTimestamp(start),
            UtcTimestamp(start + timedelta(minutes=1)),
        ),
    )

    assert result.scope.organization_id == organization_id
    assert result.scope.user_id == user_id
    assert result.candles is canonical_reader.batch
    assert resolver.calls == (user_id,)
    assert canonical_reader.calls == 1


class _ScopeResolver:
    def __init__(self, *, organization_id: OrganizationId) -> None:
        self.organization_id = organization_id
        self.calls: tuple[UserId, ...] = ()

    def resolve(self, *, user_id: UserId) -> ResearchOrganizationScope:
        self.calls = (*self.calls, user_id)
        return ResearchOrganizationScope(
            organization_id=self.organization_id,
            user_id=user_id,
        )


class _CanonicalReader:
    def __init__(self) -> None:
        self.calls = 0
        self.batch = CanonicalCandleBatch1m(
            open_time_ms=np.asarray([1], dtype=np.int64),
            close_time_ms=np.asarray([2], dtype=np.int64),
            ohlcv_f32=np.asarray([[1, 2, 0.5, 1.5, 10]], dtype=np.float32),
        )

    def read_1m_arrays(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> CanonicalCandleBatch1m:
        _ = instrument_id, time_range
        self.calls += 1
        return self.batch

    def read_1m(
        self,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Iterator[CandleWithMeta]:
        _ = instrument_id, time_range
        return iter(())
