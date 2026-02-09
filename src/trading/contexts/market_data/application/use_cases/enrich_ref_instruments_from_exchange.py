from __future__ import annotations

from dataclasses import dataclass

from trading.contexts.market_data.application.dto import (
    ExchangeInstrumentMetadata,
    InstrumentRefEnrichmentSnapshot,
    InstrumentRefEnrichmentUpsert,
)
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.contexts.market_data.application.ports.sources.instrument_metadata_source import (
    InstrumentMetadataSource,
)
from trading.contexts.market_data.application.ports.stores.enabled_instrument_reader import (
    EnabledInstrumentReader,
)
from trading.contexts.market_data.application.ports.stores.instrument_ref_writer import (
    InstrumentRefWriter,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId


@dataclass(frozen=True, slots=True)
class EnrichRefInstrumentsReport:
    """
    Summary counters for one enrich run.

    Parameters:
    - instruments_total: number of enabled instruments considered.
    - markets_total: number of markets with at least one enabled instrument.
    - rows_upserted: number of enrichment rows written to storage.
    - symbols_missing_metadata: enabled symbols absent in exchange metadata payloads.
    """

    instruments_total: int
    markets_total: int
    rows_upserted: int
    symbols_missing_metadata: int


@dataclass(frozen=True, slots=True)
class EnrichRefInstrumentsFromExchangeUseCase:
    """
    Enrich `ref_instruments` fields from exchange instrument-info endpoints.

    Parameters:
    - instrument_reader: reader of enabled tradable instruments.
    - metadata_source: exchange metadata source port.
    - writer: writer used for enrichment upserts.
    - clock: UTC clock for `updated_at`.

    Assumptions/Invariants:
    - Only enabled tradable instruments are enriched in this use-case.
    - Existing status/tradable flags are preserved as `ENABLED/1` for selected rows.
    """

    instrument_reader: EnabledInstrumentReader
    metadata_source: InstrumentMetadataSource
    writer: InstrumentRefWriter
    clock: Clock

    def __post_init__(self) -> None:
        """
        Validate required collaborators.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Collaborators are non-null object references.

        Errors/Exceptions:
        - Raises `ValueError` when any dependency is missing.

        Side effects:
        - None.
        """
        if self.instrument_reader is None:  # type: ignore[truthy-bool]
            raise ValueError("EnrichRefInstrumentsFromExchangeUseCase requires instrument_reader")
        if self.metadata_source is None:  # type: ignore[truthy-bool]
            raise ValueError("EnrichRefInstrumentsFromExchangeUseCase requires metadata_source")
        if self.writer is None:  # type: ignore[truthy-bool]
            raise ValueError("EnrichRefInstrumentsFromExchangeUseCase requires writer")
        if self.clock is None:  # type: ignore[truthy-bool]
            raise ValueError("EnrichRefInstrumentsFromExchangeUseCase requires clock")

    def run(self) -> EnrichRefInstrumentsReport:
        """
        Execute enrichment pass for all enabled tradable instruments.

        Parameters:
        - None.

        Returns:
        - Run report with processed/upserted counters.

        Assumptions/Invariants:
        - Exchange metadata can include symbols outside whitelist; those are ignored.

        Errors/Exceptions:
        - Propagates metadata source and writer adapter errors.

        Side effects:
        - Writes enrichment rows into `ref_instruments`.
        """
        instruments = list(self.instrument_reader.list_enabled_tradable())
        if not instruments:
            return EnrichRefInstrumentsReport(
                instruments_total=0,
                markets_total=0,
                rows_upserted=0,
                symbols_missing_metadata=0,
            )

        by_market = _group_by_market(instruments)
        now = self.clock.now()
        payload: list[InstrumentRefEnrichmentUpsert] = []
        missing_metadata = 0

        for market_id_int, instrument_rows in by_market.items():
            metadata_rows = self.metadata_source.list_for_market(MarketId(market_id_int))
            metadata_by_symbol = {
                str(item.instrument_id.symbol).upper(): item for item in metadata_rows
            }
            existing_by_symbol = self.writer.existing_latest_enrichment(
                market_id=MarketId(market_id_int),
                symbols=[instrument.symbol for instrument in instrument_rows],
            )

            for instrument in instrument_rows:
                symbol_key = str(instrument.symbol).upper()
                metadata = metadata_by_symbol.get(symbol_key)
                if metadata is None:
                    missing_metadata += 1
                    continue
                current_snapshot = existing_by_symbol.get(symbol_key)
                if not _needs_enrichment_upsert(
                    current_snapshot=current_snapshot,
                    metadata=metadata,
                ):
                    continue
                payload.append(
                    InstrumentRefEnrichmentUpsert(
                        market_id=instrument.market_id,
                        symbol=instrument.symbol,
                        status="ENABLED",
                        is_tradable=1,
                        base_asset=metadata.base_asset,
                        quote_asset=metadata.quote_asset,
                        price_step=metadata.price_step,
                        qty_step=metadata.qty_step,
                        min_notional=metadata.min_notional,
                        updated_at=now,
                    )
                )

        self.writer.upsert_enrichment(payload)
        return EnrichRefInstrumentsReport(
            instruments_total=len(instruments),
            markets_total=len(by_market),
            rows_upserted=len(payload),
            symbols_missing_metadata=missing_metadata,
        )


def _group_by_market(instruments: list[InstrumentId]) -> dict[int, list[InstrumentId]]:
    """
    Group instrument ids by integer market id with per-market symbol deduplication.

    Parameters:
    - instruments: list of instruments to split.

    Returns:
    - Mapping `market_id -> [InstrumentId, ...]`.

    Assumptions/Invariants:
    - Duplicate `(market_id, symbol)` items are collapsed to one entry.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    out: dict[int, list[InstrumentId]] = {}
    seen: set[tuple[int, str]] = set()
    for instrument in instruments:
        market_id_int = int(instrument.market_id.value)
        symbol_key = str(instrument.symbol).upper()
        dedup_key = (market_id_int, symbol_key)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        out.setdefault(market_id_int, []).append(instrument)
    return out


def _needs_enrichment_upsert(
    *,
    current_snapshot: InstrumentRefEnrichmentSnapshot | None,
    metadata: ExchangeInstrumentMetadata,
) -> bool:
    """
    Decide whether enrichment row must be appended based on latest persisted values.

    Parameters:
    - current_snapshot: latest persisted enrichment state for symbol, if any.
    - metadata: freshly fetched metadata from exchange endpoint.

    Returns:
    - `True` when at least one tracked field changed or no snapshot exists, else `False`.

    Assumptions/Invariants:
    - Enrich use-case targets enabled/tradable rows and therefore expects `ENABLED/1`.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if current_snapshot is None:
        return True
    if current_snapshot.status != "ENABLED":
        return True
    if current_snapshot.is_tradable != 1:
        return True
    if current_snapshot.base_asset != metadata.base_asset:
        return True
    if current_snapshot.quote_asset != metadata.quote_asset:
        return True
    if not _optional_float_equal(current_snapshot.price_step, metadata.price_step):
        return True
    if not _optional_float_equal(current_snapshot.qty_step, metadata.qty_step):
        return True
    if not _optional_float_equal(current_snapshot.min_notional, metadata.min_notional):
        return True
    return False


def _optional_float_equal(left: float | None, right: float | None) -> bool:
    """
    Compare optional float values with tiny tolerance for representation drift.

    Parameters:
    - left: first optional float value.
    - right: second optional float value.

    Returns:
    - `True` when both values are missing or numerically close, else `False`.

    Assumptions/Invariants:
    - Absolute tolerance `1e-12` is sufficient for exchange step/notional fields.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if left is None or right is None:
        return left is right
    return abs(left - right) <= 1e-12
