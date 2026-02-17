from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading.shared_kernel.primitives import Timeframe, UtcTimestamp


@dataclass(frozen=True, slots=True)
class TimeframeRollupProgress:
    """
    TimeframeRollupProgress — persisted partial-bucket state for Strategy timeframe rollup.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/strategy/domain/entities/strategy_run.py
    """

    bucket_open_ts: datetime | None
    bucket_count_1m: int

    def __post_init__(self) -> None:
        """
        Validate rollup progress invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `bucket_count_1m` stores count of already accepted contiguous 1m candles.
        Raises:
            ValueError: If state values are invalid.
        Side Effects:
            None.
        """
        if self.bucket_count_1m < 0:
            raise ValueError("TimeframeRollupProgress.bucket_count_1m must be >= 0")
        if self.bucket_open_ts is None and self.bucket_count_1m != 0:
            raise ValueError(
                "TimeframeRollupProgress.bucket_open_ts is required when bucket_count_1m > 0"
            )


@dataclass(frozen=True, slots=True)
class TimeframeRollupStep:
    """
    TimeframeRollupStep — result of one accepted base 1m candle applied to rollup state.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/timeframe_rollup.py
      - src/trading/contexts/strategy/application/services/live_runner.py
      - src/trading/shared_kernel/primitives/timeframe.py
    """

    progress: TimeframeRollupProgress
    bucket_closed: bool


class TimeframeRollupPolicy:
    """
    TimeframeRollupPolicy — deterministic closure policy for Strategy timeframe rollup buckets.

    Contract:
    - `1m` timeframe is pass-through and closes on every accepted candle.
    - Derived timeframe bucket closes only when all contained 1m candles exist.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - src/trading/shared_kernel/primitives/timeframe.py
      - src/trading/contexts/strategy/application/services/live_runner.py
      - tests/unit/contexts/strategy/application/test_strategy_live_runner.py
    """

    def advance(
        self,
        *,
        timeframe: Timeframe,
        progress: TimeframeRollupProgress,
        candle_ts_open: datetime,
    ) -> TimeframeRollupStep:
        """
        Apply one accepted 1m candle timestamp to rollup progress and closure logic.

        Args:
            timeframe: Strategy timeframe.
            progress: Current persisted rollup progress.
            candle_ts_open: Accepted candle open timestamp in UTC.
        Returns:
            TimeframeRollupStep: Updated progress and closure marker.
        Assumptions:
            Runner calls this method only for strictly monotonic contiguous base 1m candles.
        Raises:
            ValueError: If timestamps are invalid or inconsistent with progress invariants.
        Side Effects:
            None.
        """
        if timeframe.code == "1m":
            return TimeframeRollupStep(
                progress=TimeframeRollupProgress(bucket_open_ts=None, bucket_count_1m=0),
                bucket_closed=True,
            )

        bucket_open_ts = timeframe.bucket_open(UtcTimestamp(candle_ts_open)).value
        required_count = int(timeframe.duration().total_seconds() // 60)

        current_count = progress.bucket_count_1m
        if progress.bucket_open_ts != bucket_open_ts:
            current_count = 0

        next_count = current_count + 1
        if next_count > required_count:
            raise ValueError(
                "TimeframeRollupPolicy observed more candles in bucket than timeframe requires"
            )

        if next_count == required_count:
            return TimeframeRollupStep(
                progress=TimeframeRollupProgress(bucket_open_ts=None, bucket_count_1m=0),
                bucket_closed=True,
            )

        return TimeframeRollupStep(
            progress=TimeframeRollupProgress(
                bucket_open_ts=bucket_open_ts,
                bucket_count_1m=next_count,
            ),
            bucket_closed=False,
        )
