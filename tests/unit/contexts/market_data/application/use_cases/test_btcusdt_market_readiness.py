from __future__ import annotations

from datetime import UTC, datetime
from typing import Sequence

from trading.contexts.market_data.application.dto.reference_api import (
    BTCUSDTMarketReferenceSnapshot,
    BTCUSDTStreamReadinessSnapshot,
)
from trading.contexts.market_data.application.use_cases import BTCUSDTMarketReadinessUseCase
from trading.shared_kernel.primitives import MarketId


class _ReferenceReader:
    def __init__(self, rows: Sequence[BTCUSDTMarketReferenceSnapshot]) -> None:
        self._rows = tuple(rows)

    def list_btcusdt_reference_rows(self) -> Sequence[BTCUSDTMarketReferenceSnapshot]:
        return self._rows


class _StreamReader:
    def check(
        self,
        *,
        instrument_key: str,
        timeframe: str,
        observed_at: datetime,
    ) -> BTCUSDTStreamReadinessSnapshot:
        _ = timeframe
        _ = observed_at
        if instrument_key == "binance:spot:BTCUSDT":
            return BTCUSDTStreamReadinessSnapshot(
                state="ready",
                reason_code="market_data_stream_ready",
                stream_name=f"md.candles.1m.{instrument_key}",
                stream_length=12,
                last_message_id="1800000000000-0",
                last_observed_at=datetime(2027, 1, 15, 8, 0, tzinfo=UTC),
                age_seconds=30,
            )
        return BTCUSDTStreamReadinessSnapshot(
            state="missing",
            reason_code="market_data_stream_missing",
            stream_name=f"md.candles.1m.{instrument_key}",
            stream_length=0,
            last_message_id=None,
            last_observed_at=None,
            age_seconds=None,
        )


def test_btcusdt_market_readiness_combines_reference_and_stream_state() -> None:
    use_case = BTCUSDTMarketReadinessUseCase(
        reference_reader=_ReferenceReader(
            rows=(
                _reference(
                    market_id=1,
                    exchange_name="binance",
                    market_type="spot",
                    market_code="binance:spot",
                    price_step=0.01,
                    qty_step=0.00001,
                    min_notional=10.0,
                ),
                _reference(
                    market_id=2,
                    exchange_name="binance",
                    market_type="futures",
                    market_code="binance:futures",
                    price_step=0.01,
                    qty_step=0.001,
                    min_notional=None,
                ),
            )
        ),
        stream_reader=_StreamReader(),
    )

    report = use_case.execute(observed_at=datetime(2027, 1, 15, 8, 0, 30, tzinfo=UTC))

    rows = {row.market_code: row for row in report.rows}
    assert report.symbol == "BTCUSDT"
    assert report.freshness_threshold_seconds == 180
    assert rows["binance:spot"].readiness_state == "ready"
    assert rows["binance:spot"].reason_codes == ("btcusdt_market_ready",)
    assert rows["binance:spot"].stream_name == "md.candles.1m.binance:spot:BTCUSDT"
    assert rows["binance:futures"].readiness_state == "blocked"
    assert rows["binance:futures"].reference_state == "incomplete"
    assert rows["binance:futures"].reason_codes == ("reference_min_notional_missing",)
    assert rows["bybit:spot"].readiness_state == "blocked"
    assert rows["bybit:spot"].reason_codes == ("reference_market_missing",)
    assert rows["bybit:futures"].instrument_key == "bybit:futures:BTCUSDT"


def _reference(
    *,
    market_id: int,
    exchange_name: str,
    market_type: str,
    market_code: str,
    price_step: float | None,
    qty_step: float | None,
    min_notional: float | None,
) -> BTCUSDTMarketReferenceSnapshot:
    return BTCUSDTMarketReferenceSnapshot(
        market_id=MarketId(market_id),
        exchange_name=exchange_name,
        market_type=market_type,
        market_code=market_code,
        market_enabled=True,
        symbol="BTCUSDT",
        status="ENABLED",
        is_tradable=1,
        base_asset="BTC",
        quote_asset="USDT",
        price_step=price_step,
        qty_step=qty_step,
        min_notional=min_notional,
    )
