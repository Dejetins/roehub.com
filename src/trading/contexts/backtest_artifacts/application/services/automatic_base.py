"""Finite, source-only base builds; dispatch/cadence remain coordinator responsibilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Callable, Iterator, Mapping

import numpy as np

from trading.contexts.backtest_artifacts.application.services.management import (
    Coordinate,
    ManagementError,
)

if TYPE_CHECKING:
    from trading.contexts.backtest_artifacts.application.services.derivative_profiles import (
        DerivativePlan,
    )
from trading.contexts.backtest_artifacts.application.services.v2 import (
    artifact_precompute_runner as native,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (
    BacktestArtifactSlotPublisherV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCoordinatesV2,
    ArtifactPublishResultV2,
    ArtifactSlotValidationSpecV2,
    artifact_market_id_from_coordinates_v2,
)
from trading.contexts.market_data.application.dto import CandleWithMeta, CanonicalCandleBatch1m
from trading.contexts.market_data.application.ports.stores.canonical_candle_reader import (
    CanonicalCandleReader,
)
from trading.contexts.market_data.application.ports.stores.funding_rate_coverage_reader import (
    FundingRateCoverageSnapshot,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


class BasePrepublicationError(ManagementError):
    """Proven rejection before calling the physical publisher; no ambiguous file outcome."""


@dataclass(frozen=True)
class FrozenFunding:
    instrument: InstrumentId
    snapshot: FundingRateCoverageSnapshot

    def read_funding_coverage(self, *, instrument_id: InstrumentId, time_range: TimeRange):
        if instrument_id != self.instrument or time_range != self.snapshot.requested_range:
            raise ValueError("frozen_funding_domain_mismatch")
        return self.snapshot


@dataclass(frozen=True)
class FrozenBaseSource:
    """An attempt owns copies of candle payload; later storage changes cannot change its plan."""

    instrument: InstrumentId
    time_range: TimeRange
    batch: CanonicalCandleBatch1m
    sha256: str

    def read_1m_arrays(self, instrument_id: InstrumentId, time_range: TimeRange):
        if instrument_id != self.instrument or time_range != self.time_range:
            raise ValueError("frozen_source_domain_mismatch")
        return self.batch

    def read_1m(
        self, instrument_id: InstrumentId, time_range: TimeRange
    ) -> Iterator[CandleWithMeta]:
        raise NotImplementedError("base builds use frozen columnar source")


def freeze_source(
    *,
    reader: CanonicalCandleReader,
    coordinates: ArtifactCoordinatesV2,
    time_range: TimeRange,
    max_minutes: int = 5_000_000,
    chunk_minutes: int = 20_000,
    checkpoint: Callable[[int, int], None] = lambda completed, total: None,
) -> FrozenBaseSource:
    """Bound source scan and total allocation; never truncate an admitted history to fit.

    One chunk in flight, bounded by time (hence also rows). A second bounded pass checks
    the consumed content for concurrent corrections before freezing. This is not a DB
    transaction snapshot: the resulting immutable payload/digest is the attempt identity.
    """
    if not 1 <= chunk_minutes <= 20_000 or not 1 <= max_minutes <= 5_000_000:
        raise ValueError("invalid_source_budget")
    start, end = time_range.start.value, time_range.end.value
    if any(t.second or t.microsecond for t in (start, end)):
        raise ValueError("unaligned_source_range")
    minutes = int((end - start).total_seconds() // 60)
    if minutes > max_minutes:
        raise ManagementError("resource_conflict")
    instrument = InstrumentId(
        MarketId(artifact_market_id_from_coordinates_v2(coordinates)), Symbol(coordinates.symbol)
    )
    # The span ceiling bounds allocation even with sparse data. No filesystem write here.
    opens = np.empty(minutes, dtype=np.int64)
    closes = np.empty(minutes, dtype=np.int64)
    values = np.empty((minutes, 5), dtype=np.float32)
    count = 0
    hashes: list[bytes] = []
    for verification in (False, True):
        cursor = start
        position = 0
        while cursor < end:
            stop = min(end, cursor + timedelta(minutes=chunk_minutes))
            window = TimeRange(UtcTimestamp(cursor), UtcTimestamp(stop))
            batch = reader.read_1m_arrays(instrument_id=instrument, time_range=window)
            n = batch.row_count()
            if n > int((stop - cursor).total_seconds() // 60):
                raise ValueError("source_chunk_overflow")
            if n:
                arrays = native._canonical_price_arrays_from_batch_v2(
                    batch=batch, source_time_range=window
                )
                native._validate_rollup_source_one_minute_arrays_v2(arrays=arrays, label="snapshot")
                if arrays.open_time[0] < int(cursor.timestamp() * 1000) or (
                    arrays.close_time[-1] > int(stop.timestamp() * 1000)
                ):
                    raise ValueError("source_outside_frozen_range")
            digest = hashlib.sha256()
            for array in (batch.open_time_ms, batch.close_time_ms, batch.ohlcv_f32):
                digest.update(array.tobytes())
            if verification:
                if digest.digest() != hashes[position]:
                    raise ManagementError("source_changed")
            else:
                hashes.append(digest.digest())
                opens[count : count + n] = batch.open_time_ms
                closes[count : count + n] = batch.close_time_ms
                values[count : count + n] = batch.ohlcv_f32
                count += n
            position += 1
            cursor = stop
            checkpoint(
                (minutes if verification else 0) + int((cursor - start).total_seconds() // 60),
                2 * minutes,
            )
    arrays = (opens[:count], closes[:count], values[:count])
    digest = hashlib.sha256()
    for array in arrays:
        for offset in range(0, len(array), 4096):
            digest.update(array[offset : offset + 4096].tobytes())
        array.flags.writeable = False
    return FrozenBaseSource(
        instrument, time_range, CanonicalCandleBatch1m(*arrays), digest.hexdigest()
    )


@dataclass(frozen=True)
class BasePublication:
    state: str
    source_sha256: str
    timeframes: Mapping[str, Mapping[str, object]]
    publication: ArtifactPublishResultV2 | None


@dataclass(frozen=True)
class AutomaticBasePublisher:
    """Use native numerics and 0028 reservations; no on-demand derivatives or downloads."""

    runner: native.BacktestArtifactPrecomputeRunnerV2
    publisher: BacktestArtifactSlotPublisherV2
    on_publication: Callable[[BasePublication], None] | None = None

    def publish(
        self,
        *,
        coordinates: ArtifactCoordinatesV2,
        source: FrozenBaseSource,
        generated_at: datetime,
        derivatives: DerivativePlan | None = None,
    ) -> BasePublication:
        expected = InstrumentId(
            MarketId(artifact_market_id_from_coordinates_v2(coordinates)),
            Symbol(coordinates.symbol),
        )
        if source.instrument != expected or generated_at.tzinfo is None:
            raise ValueError("source_identity_mismatch")
        if source.time_range.end.value > generated_at.astimezone(UTC).replace(
            second=0, microsecond=0
        ):
            raise ValueError("source_not_closed")
        inventory: dict[str, Mapping[str, object]] = {}
        batch = source.batch
        if batch.row_count() == 0:
            for tf in ("1m", *ARTIFACT_MAPPING_TIMEFRAMES_V2):
                inventory[tf] = {"state": "waiting_for_source", "closed_bars": 0}
            return BasePublication("waiting_for_source", source.sha256, inventory, None)
        funding = None
        source_digest = source.sha256
        if coordinates.market_type == "futures":
            reader = self.runner.funding_rate_coverage_reader
            if reader is None:
                raise BasePrepublicationError("funding_unavailable")
            snapshot = reader.read_funding_coverage(
                instrument_id=expected, time_range=source.time_range
            )
            if snapshot.status not in ("ready", "degraded") or (
                snapshot.requested_range != source.time_range
            ):
                raise BasePrepublicationError("funding_unavailable")
            if not snapshot.records:
                raise BasePrepublicationError("funding_insufficient_history")
            if (
                len(snapshot.records)
                > int(
                    (source.time_range.end.value - source.time_range.start.value).total_seconds()
                    // 60
                )
                + 1
            ):
                raise BasePrepublicationError("funding_resource_conflict")
            funding = FrozenFunding(expected, snapshot)
            source_digest = hashlib.sha256(
                (
                    source.sha256
                    + json.dumps(asdict(snapshot), sort_keys=True, default=str, allow_nan=False)
                ).encode()
            ).hexdigest()
        arrays = native._canonical_price_arrays_from_batch_v2(
            batch=batch, source_time_range=source.time_range
        )
        eligible = []
        for tf in ("1m", *ARTIFACT_MAPPING_TIMEFRAMES_V2):
            rolled = (
                arrays
                if tf == "1m"
                else native._rollup_price_arrays_from_one_minute_v2(
                    source_arrays=arrays, timeframe=tf, allow_empty=True
                )
            )
            duration = int(native._timeframe_duration_millis_v2(native.Timeframe(tf)))
            end_ms = int(source.time_range.end.value.timestamp() * 1000)
            if rolled is None:
                inventory[tf] = {
                    "state": "insufficient_history",
                    "closed_bars": 0,
                    "retry_after_source_end_ms": (end_ms // duration + 1) * duration,
                }
            else:
                if tf != "1m":
                    eligible.append(tf)
                inventory[tf] = {
                    "state": "ready",
                    "closed_bars": len(rolled.open_time),
                    "start_ms": int(rolled.open_time[0]),
                    "end_ms": int(rolled.close_time[-1]),
                    "missing_bars": int(
                        (rolled.close_time[-1] - rolled.open_time[0]) // duration
                        - len(rolled.open_time)
                    ),
                }
        mappings = tuple(eligible)
        prices = ("1m", *mappings)
        settings = replace(
            self.runner.runtime_settings,
            price_timeframes=prices,
            mapping_timeframes=mappings,
            signal_artifacts=(),
            precompute_hit_times=False,
            partial_publication=True,
        )
        spec = ArtifactSlotValidationSpecV2(
            price_timeframes=prices, mapping_timeframes=mappings, require_hit_times_manifest=False
        )
        configured_runner = replace(self.runner, runtime_settings=settings)
        if derivatives is not None:
            derivatives.validate_source(
                coordinate=Coordinate(
                    exchange=coordinates.exchange,
                    market_type=coordinates.market_type,
                    symbol=coordinates.symbol,
                ),
                source_sha256=source_digest,
                start=source.time_range.start.value,
                end=source.time_range.end.value,
            )
            configured_runner, spec = derivatives.configure(self.runner, mappings)
        # Recovery validates the inventory actually recorded in the journal/slot, not a new
        # source-dependent expected subset. Recovery runs before this call at composition.
        publisher = replace(self.publisher, recover_declared_inventory=True)
        publisher.recover_publications(coordinates=coordinates, validation_spec=spec)
        precheck = publisher.precheck_publish(coordinates)
        if not precheck.ready:
            raise BasePrepublicationError(precheck.failure_code or "publisher_busy")
        request = ArtifactCanonicalPriceExportRequestV2(
            coordinates=coordinates,
            time_range=source.time_range,
            asof_date=(source.time_range.end.value - timedelta(minutes=1)).date().isoformat(),
            generated_at_utc=generated_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            target_slot=precheck.inactive_slot,
            target_slot_generation=precheck.target_slot_generation,
            force_full_rebuild=True,
        )
        # Full finite rebuild intentionally includes historical extensions/corrections. Never
        # reattach derivative inventory from a different source prefix.
        state = "ready" if len(mappings) == len(ARTIFACT_MAPPING_TIMEFRAMES_V2) else "partial"
        if self.on_publication is not None:
            callback = self.on_publication
            publisher = replace(
                publisher,
                on_publication=lambda published: callback(
                    BasePublication(state, source_digest, inventory, published)
                ),
            )
        _, result = publisher.build_and_publish(
            request=request,
            precheck=precheck,
            precompute_runner=replace(
                configured_runner,
                canonical_candle_reader=source,
                funding_rate_coverage_reader=funding,
            ),
            validation_spec=spec,
        )
        state = "ready" if len(mappings) == len(ARTIFACT_MAPPING_TIMEFRAMES_V2) else "partial"
        return BasePublication(state, source_digest, inventory, result)
