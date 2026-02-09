from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from trading.contexts.market_data.application.dto import (
    ExchangeInstrumentMetadata,
    InstrumentRefEnrichmentSnapshot,
)
from trading.contexts.market_data.application.use_cases import (
    EnrichRefInstrumentsFromExchangeUseCase,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, UtcTimestamp


@dataclass(frozen=True, slots=True)
class _Clock:
    now_value: UtcTimestamp

    def now(self) -> UtcTimestamp:
        """Return fixed timestamp used in enrichment writes."""
        return self.now_value


class _InstrumentReader:
    def __init__(self, instruments: list[InstrumentId]) -> None:
        """Store deterministic instrument list for tests."""
        self._instruments = instruments

    def list_enabled_tradable(self):  # noqa: ANN001
        """Return configured enabled instruments."""
        return list(self._instruments)


class _MetadataSource:
    def __init__(self, by_market: dict[int, list[ExchangeInstrumentMetadata]]) -> None:
        """Store metadata payload keyed by market id."""
        self._by_market = by_market

    def list_for_market(self, market_id):  # noqa: ANN001
        """Return configured metadata rows for requested market id."""
        return list(self._by_market.get(int(market_id.value), []))


class _Writer:
    def __init__(
        self,
        existing_enrichment: dict[tuple[int, str], InstrumentRefEnrichmentSnapshot] | None = None,
    ) -> None:
        """Initialize in-memory captured enrichment rows."""
        self.rows = []
        self._existing_enrichment = (
            dict(existing_enrichment) if existing_enrichment is not None else {}
        )

    def existing_latest(self, *, market_id, symbols):  # noqa: ANN001
        """Unused in this use-case; return empty mapping."""
        _ = market_id
        _ = symbols
        return {}

    def upsert(self, rows):  # noqa: ANN001
        """Unused in this use-case."""
        _ = rows

    def existing_latest_enrichment(self, *, market_id, symbols):  # noqa: ANN001
        """Return configured latest enrichment snapshots for provided symbols."""
        out = {}
        for symbol in symbols:
            key = (int(market_id.value), str(symbol).upper())
            if key in self._existing_enrichment:
                out[str(symbol).upper()] = self._existing_enrichment[key]
        return out

    def upsert_enrichment(self, rows):  # noqa: ANN001
        """Capture enrichment rows for assertions."""
        self.rows.extend(list(rows))


def test_enrich_use_case_updates_only_symbols_present_in_metadata() -> None:
    """Ensure use-case writes enrichment for matched symbols and counts missing metadata."""
    instruments = [
        InstrumentId(MarketId(1), Symbol("BTCUSDT")),
        InstrumentId(MarketId(1), Symbol("ETHUSDT")),
        InstrumentId(MarketId(3), Symbol("ADAUSDT")),
    ]
    metadata = {
        1: [
            ExchangeInstrumentMetadata(
                instrument_id=InstrumentId(MarketId(1), Symbol("BTCUSDT")),
                base_asset="BTC",
                quote_asset="USDT",
                price_step=0.01,
                qty_step=0.0001,
                min_notional=10.0,
            )
        ],
        3: [
            ExchangeInstrumentMetadata(
                instrument_id=InstrumentId(MarketId(3), Symbol("ADAUSDT")),
                base_asset="ADA",
                quote_asset="USDT",
                price_step=0.0001,
                qty_step=0.1,
                min_notional=5.0,
            )
        ],
    }
    writer = _Writer()
    use_case = EnrichRefInstrumentsFromExchangeUseCase(
        instrument_reader=_InstrumentReader(instruments),
        metadata_source=_MetadataSource(metadata),
        writer=writer,
        clock=_Clock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))),
    )

    report = use_case.run()

    assert report.instruments_total == 3
    assert report.markets_total == 2
    assert report.rows_upserted == 2
    assert report.symbols_missing_metadata == 1
    assert len(writer.rows) == 2
    assert {str(row.symbol) for row in writer.rows} == {"BTCUSDT", "ADAUSDT"}


def test_enrich_use_case_returns_zero_report_for_empty_instrument_set() -> None:
    """Ensure empty enabled list produces empty report without writes."""
    writer = _Writer()
    use_case = EnrichRefInstrumentsFromExchangeUseCase(
        instrument_reader=_InstrumentReader([]),
        metadata_source=_MetadataSource({}),
        writer=writer,
        clock=_Clock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))),
    )

    report = use_case.run()

    assert report.instruments_total == 0
    assert report.markets_total == 0
    assert report.rows_upserted == 0
    assert report.symbols_missing_metadata == 0
    assert writer.rows == []


def test_enrich_use_case_skips_unchanged_metadata_rows() -> None:
    """Ensure enrich run does not append duplicate rows when metadata did not change."""
    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    writer = _Writer(
        existing_enrichment={
            (1, "BTCUSDT"): InstrumentRefEnrichmentSnapshot(
                status="ENABLED",
                is_tradable=1,
                base_asset="BTC",
                quote_asset="USDT",
                price_step=0.01,
                qty_step=0.0001,
                min_notional=10.0,
            )
        }
    )
    use_case = EnrichRefInstrumentsFromExchangeUseCase(
        instrument_reader=_InstrumentReader([instrument]),
        metadata_source=_MetadataSource(
            {
                1: [
                    ExchangeInstrumentMetadata(
                        instrument_id=instrument,
                        base_asset="BTC",
                        quote_asset="USDT",
                        price_step=0.01,
                        qty_step=0.0001,
                        min_notional=10.0,
                    )
                ]
            }
        ),
        writer=writer,
        clock=_Clock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))),
    )

    report = use_case.run()

    assert report.instruments_total == 1
    assert report.rows_upserted == 0
    assert writer.rows == []


def test_enrich_use_case_deduplicates_duplicate_instrument_ids() -> None:
    """Ensure duplicate symbols from reader are collapsed before metadata upsert planning."""
    instrument = InstrumentId(MarketId(1), Symbol("BTCUSDT"))
    writer = _Writer()
    use_case = EnrichRefInstrumentsFromExchangeUseCase(
        instrument_reader=_InstrumentReader([instrument, instrument]),
        metadata_source=_MetadataSource(
            {
                1: [
                    ExchangeInstrumentMetadata(
                        instrument_id=instrument,
                        base_asset="BTC",
                        quote_asset="USDT",
                        price_step=0.01,
                        qty_step=0.0001,
                        min_notional=10.0,
                    )
                ]
            }
        ),
        writer=writer,
        clock=_Clock(UtcTimestamp(datetime(2026, 2, 9, 14, 0, tzinfo=timezone.utc))),
    )

    report = use_case.run()

    assert report.instruments_total == 2
    assert report.rows_upserted == 1
    assert len(writer.rows) == 1
