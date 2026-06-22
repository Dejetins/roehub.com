from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp

type FundingCoverageStatus = Literal["ready", "degraded", "unavailable", "not_applicable"]


@dataclass(frozen=True, slots=True)
class FundingRateArtifactRecord:
    instrument_id: InstrumentId
    instrument_key: str
    funding_time: UtcTimestamp
    funding_rate: float
    mark_price: float | None
    funding_interval_minutes: int
    data_quality: int

    def __post_init__(self) -> None:
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("FundingRateArtifactRecord.instrument_id is required")
        if not self.instrument_key.strip():
            raise ValueError("FundingRateArtifactRecord.instrument_key must be non-empty")
        if self.funding_time is None:  # type: ignore[truthy-bool]
            raise ValueError("FundingRateArtifactRecord.funding_time is required")
        if self.funding_interval_minutes <= 0:
            raise ValueError(
                "FundingRateArtifactRecord.funding_interval_minutes must be positive"
            )
        if self.data_quality < 0:
            raise ValueError("FundingRateArtifactRecord.data_quality must be non-negative")


@dataclass(frozen=True, slots=True)
class FundingRateCoverageSnapshot:
    status: FundingCoverageStatus
    coverage_policy: str
    requested_range: TimeRange
    available_start: UtcTimestamp | None
    available_end: UtcTimestamp | None
    expected_event_count: int
    observed_event_count: int
    missing_event_count: int
    reason_codes: tuple[str, ...]
    records: tuple[FundingRateArtifactRecord, ...]

    def __post_init__(self) -> None:
        if self.requested_range is None:  # type: ignore[truthy-bool]
            raise ValueError("FundingRateCoverageSnapshot.requested_range is required")
        if self.expected_event_count < 0:
            raise ValueError("FundingRateCoverageSnapshot.expected_event_count must be >= 0")
        if self.observed_event_count < 0:
            raise ValueError("FundingRateCoverageSnapshot.observed_event_count must be >= 0")
        if self.missing_event_count < 0:
            raise ValueError("FundingRateCoverageSnapshot.missing_event_count must be >= 0")
        if self.observed_event_count != len(self.records):
            raise ValueError(
                "FundingRateCoverageSnapshot.observed_event_count must match records length"
            )
        if self.status == "degraded" and self.coverage_policy != "degraded_with_warning":
            raise ValueError(
                "FundingRateCoverageSnapshot degraded status requires degraded_with_warning"
            )
        if self.status in ("ready", "not_applicable") and self.reason_codes:
            raise ValueError(
                "FundingRateCoverageSnapshot ready/not_applicable status cannot carry reasons"
            )


class FundingRateCoverageReader(Protocol):
    def read_funding_coverage(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> FundingRateCoverageSnapshot:
        """
        Read scheduler-maintained canonical funding rows and coverage metadata for one window.
        """
        ...


__all__ = [
    "FundingCoverageStatus",
    "FundingRateArtifactRecord",
    "FundingRateCoverageReader",
    "FundingRateCoverageSnapshot",
]
