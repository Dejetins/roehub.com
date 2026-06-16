from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from trading.contexts.market_data.application.dto.reference_api import (
    BTCUSDT_MARKET_READINESS_MARKETS,
    BTCUSDT_MARKET_READINESS_SYMBOL,
    BTCUSDTMarketReadinessReport,
    BTCUSDTMarketReadinessRow,
    BTCUSDTMarketReadinessState,
    BTCUSDTMarketReferenceSnapshot,
    BTCUSDTReferenceState,
    BTCUSDTStreamReadinessSnapshot,
)
from trading.contexts.market_data.application.ports.stores import (
    BTCUSDTMarketReadinessReferenceReader,
)
from trading.shared_kernel.primitives import MarketId

BTCUSDT_MARKET_READINESS_STALE_AFTER_SECONDS = 180


class BTCUSDTMarketReadinessStreamReader(Protocol):
    def check(
        self,
        *,
        instrument_key: str,
        timeframe: str,
        observed_at: datetime,
    ) -> BTCUSDTStreamReadinessSnapshot: ...


@dataclass(frozen=True, slots=True)
class BTCUSDTMarketReadinessUseCase:
    reference_reader: BTCUSDTMarketReadinessReferenceReader
    stream_reader: BTCUSDTMarketReadinessStreamReader
    stale_after_seconds: int = BTCUSDT_MARKET_READINESS_STALE_AFTER_SECONDS

    def __post_init__(self) -> None:
        if self.reference_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("BTCUSDTMarketReadinessUseCase requires reference_reader")
        if self.stream_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("BTCUSDTMarketReadinessUseCase requires stream_reader")
        if self.stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be > 0")

    def execute(self, *, observed_at: datetime | None = None) -> BTCUSDTMarketReadinessReport:
        checked_at = _ensure_aware_utc(observed_at or datetime.now(UTC))
        references = {
            row.market_code: row
            for row in self.reference_reader.list_btcusdt_reference_rows()
        }
        rows = tuple(
            self._build_row(
                reference=references.get(market_code),
                expected=expected,
                checked_at=checked_at,
            )
            for expected in BTCUSDT_MARKET_READINESS_MARKETS
            for market_code in (expected[2],)
        )
        return BTCUSDTMarketReadinessReport(
            symbol=BTCUSDT_MARKET_READINESS_SYMBOL,
            freshness_threshold_seconds=self.stale_after_seconds,
            rows=rows,
            checked_at=checked_at,
        )

    def _build_row(
        self,
        *,
        reference: BTCUSDTMarketReferenceSnapshot | None,
        expected: tuple[str, str, str],
        checked_at: datetime,
    ) -> BTCUSDTMarketReadinessRow:
        exchange_name, market_type, market_code = expected
        if reference is None:
            reference = BTCUSDTMarketReferenceSnapshot(
                market_id=None,
                exchange_name=exchange_name,
                market_type=market_type,
                market_code=market_code,
                market_enabled=False,
                symbol=BTCUSDT_MARKET_READINESS_SYMBOL,
                status=None,
                is_tradable=None,
                base_asset=None,
                quote_asset=None,
                price_step=None,
                qty_step=None,
                min_notional=None,
            )
        stream = self.stream_reader.check(
            instrument_key=reference.instrument_key,
            timeframe="1m",
            observed_at=checked_at,
        )
        reference_state, reference_reasons = _reference_state(reference=reference)
        readiness_state, reasons = _combined_state(
            reference_state=reference_state,
            reference_reasons=reference_reasons,
            stream=stream,
        )
        return BTCUSDTMarketReadinessRow(
            market_id=reference.market_id,
            exchange_name=reference.exchange_name,
            market_type=reference.market_type,
            market_code=reference.market_code,
            symbol=reference.symbol,
            instrument_key=reference.instrument_key,
            readiness_state=readiness_state,
            reason_codes=reasons,
            reference_state=reference_state,
            reference_reason_codes=reference_reasons,
            market_enabled=reference.market_enabled,
            status=reference.status,
            is_tradable=reference.is_tradable,
            base_asset=reference.base_asset,
            quote_asset=reference.quote_asset,
            price_step=reference.price_step,
            qty_step=reference.qty_step,
            min_notional=reference.min_notional,
            stream_state=stream.state,
            stream_reason_code=stream.reason_code,
            stream_name=stream.stream_name,
            stream_length=stream.stream_length,
            stream_last_message_id=stream.last_message_id,
            stream_last_observed_at=stream.last_observed_at,
            stream_age_seconds=stream.age_seconds,
            checked_at=checked_at,
        )


def _reference_state(
    *, reference: BTCUSDTMarketReferenceSnapshot
) -> tuple[BTCUSDTReferenceState, tuple[str, ...]]:
    if reference.market_id is None:
        return "missing", ("reference_market_missing",)
    if not reference.market_enabled:
        return "disabled", ("reference_market_disabled",)
    if reference.status is None:
        return "missing", ("reference_instrument_missing",)
    if reference.status != "ENABLED":
        return "disabled", ("reference_instrument_disabled",)
    if reference.is_tradable != 1:
        return "disabled", ("reference_instrument_not_tradable",)

    missing_fields = []
    if reference.price_step is None or reference.price_step <= 0:
        missing_fields.append("price_step")
    if reference.qty_step is None or reference.qty_step <= 0:
        missing_fields.append("qty_step")
    if reference.min_notional is None or reference.min_notional <= 0:
        missing_fields.append("min_notional")
    if missing_fields:
        return "incomplete", tuple(f"reference_{field}_missing" for field in missing_fields)
    return "ready", ("reference_ready",)


def _combined_state(
    *,
    reference_state: BTCUSDTReferenceState,
    reference_reasons: tuple[str, ...],
    stream: BTCUSDTStreamReadinessSnapshot,
) -> tuple[BTCUSDTMarketReadinessState, tuple[str, ...]]:
    if reference_state != "ready":
        return "blocked", reference_reasons
    if stream.state != "ready":
        return stream.state, (stream.reason_code,)
    return "ready", ("btcusdt_market_ready",)


def _ensure_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def btcusdt_market_id(value: int | None) -> MarketId | None:
    return MarketId(value) if value is not None else None
