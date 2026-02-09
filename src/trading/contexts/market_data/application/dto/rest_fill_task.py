from __future__ import annotations

from dataclasses import dataclass

from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp


@dataclass(frozen=True, slots=True)
class RestFillTask:
    """
    One asynchronous REST fill command for one instrument.

    Parameters:
    - instrument_id: target instrument `(market_id, symbol)`.
    - time_range: UTC half-open range `[start, end)` to load from REST.
    - reason: operational reason tag (`gap`, `reconnect_tail`, `bootstrap`, etc.).

    Assumptions/Invariants:
    - `time_range.start < time_range.end` is validated by `TimeRange`.
    - `reason` is non-empty and used only for metrics/log labels.
    """

    instrument_id: InstrumentId
    time_range: TimeRange
    reason: str

    def __post_init__(self) -> None:
        """
        Validate non-empty reason text.

        Parameters:
        - None.

        Returns:
        - None.

        Assumptions/Invariants:
        - Domain identity and time-range invariants are already enforced by primitives.

        Errors/Exceptions:
        - Raises `ValueError` when reason is empty.

        Side effects:
        - None.
        """
        if not self.reason.strip():
            raise ValueError("RestFillTask requires non-empty reason")


@dataclass(frozen=True, slots=True)
class RestFillResult:
    """
    Execution outcome for one REST fill task.

    Parameters:
    - task: original fill task.
    - rows_read: source rows observed in range.
    - rows_written: rows persisted into raw tables.
    - batches_written: number of raw insert batches.
    - started_at: execution start in UTC.
    - finished_at: execution finish in UTC.

    Assumptions/Invariants:
    - Counters are non-negative.
    - `started_at <= finished_at`.
    """

    task: RestFillTask
    rows_read: int
    rows_written: int
    batches_written: int
    started_at: UtcTimestamp
    finished_at: UtcTimestamp

