from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import cast
from uuid import UUID

import numpy as np

from trading.contexts.backtest.application.dto import RunBacktestRequest, RunBacktestTemplate
from trading.contexts.backtest.application.ports import BacktestStrategySnapshot, CurrentUser
from trading.contexts.backtest.application.use_cases import RunBacktestUseCase
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    EstimateResult,
    IndicatorTensor,
    IndicatorVariantSelection,
)
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UserId,
    UtcTimestamp,
)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_MINUTE = timedelta(minutes=1)


class _AlignedOnlyCandleFeed:
    """
    CandleFeed stub that accepts only minute-aligned ranges to verify use-case wiring.

    Docs:
      - docs/architecture/backtest/backtest-candle-timeline-rollup-warmup-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/services/candle_timeline_builder.py
      - src/trading/contexts/indicators/application/ports/feeds/candle_feed.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic call recorder for stub assertions.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Stub has no external dependencies and stores calls in-memory.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.calls: list[TimeRange] = []

    def load_1m_dense(
        self,
        market_id: MarketId,
        symbol: Symbol,
        time_range: TimeRange,
    ) -> CandleArrays:
        """
        Reject non-minute-aligned calls and return deterministic dense `1m` arrays.

        Args:
            market_id: Requested market identifier.
            symbol: Requested symbol.
            time_range: Requested feed range.
        Returns:
            CandleArrays: Dense `1m` candles for supplied range.
        Assumptions:
            Range is expected to be normalized by backtest timeline builder.
        Raises:
            ValueError: If range bounds are not aligned to minute boundaries.
        Side Effects:
            Appends requested range to in-memory calls list.
        """
        _ = market_id, symbol
        if (
            time_range.start.value.second != 0
            or time_range.start.value.microsecond != 0
            or time_range.end.value.second != 0
            or time_range.end.value.microsecond != 0
        ):
            raise ValueError("time_range must be minute-aligned")
        self.calls.append(time_range)
        return _build_dense_1m_from_time_range(time_range=time_range)


class _NoOpIndicatorCompute:
    """
    IndicatorCompute stub that records compute invocations.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic call counter.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Estimate and warmup are not used in this wiring test.
        Raises:
            None.
        Side Effects:
            None.
        """
        self.compute_calls = 0
        self.compute_indicator_ids: list[str] = []

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Return placeholder estimate result for protocol compatibility in tests.

        Args:
            grid: Indicator grid payload.
            max_variants_guard: Variants guard value.
        Returns:
            EstimateResult: Placeholder casted object.
        Assumptions:
            Method is not used in this test scenario.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = grid, max_variants_guard
        return cast(EstimateResult, object())

    def compute(self, req: ComputeRequest) -> IndicatorTensor:
        """
        Record compute call and return placeholder tensor.

        Args:
            req: Compute request payload.
        Returns:
            IndicatorTensor: Placeholder casted object.
        Assumptions:
            RunBacktestUseCase ignores returned tensor in current skeleton.
        Raises:
            None.
        Side Effects:
            Increments compute call counter.
        """
        self.compute_indicator_ids.append(req.grid.indicator_id.value)
        self.compute_calls += 1
        return cast(IndicatorTensor, object())

    def warmup(self) -> None:
        """
        No-op warmup implementation for protocol compatibility.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup is irrelevant for this test.
        Raises:
            None.
        Side Effects:
            None.
        """
        return None


class _UnusedStrategyReader:
    """
    Backtest strategy reader stub for template-mode tests.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    def load_any(self, *, strategy_id: UUID) -> BacktestStrategySnapshot | None:
        """
        Return no snapshot because template mode does not need saved strategy lookup.

        Args:
            strategy_id: Requested saved strategy id.
        Returns:
            BacktestStrategySnapshot | None: Always `None`.
        Assumptions:
            Caller runs in template mode and never consumes snapshot.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = strategy_id
        return None


def test_run_backtest_use_case_normalizes_non_aligned_range_via_timeline_builder() -> None:
    """
    Verify use-case succeeds for non-aligned request range by delegating candle loading to builder.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Timeline builder is responsible for minute normalization and warmup lookback.
    Raises:
        AssertionError: If use-case calls feed with non-aligned range or misses compute call.
    Side Effects:
        None.
    """
    candle_feed = _AlignedOnlyCandleFeed()
    indicator_compute = _NoOpIndicatorCompute()
    use_case = RunBacktestUseCase(
        candle_feed=candle_feed,
        indicator_compute=indicator_compute,
        strategy_reader=_UnusedStrategyReader(),
    )
    request = RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 12, 0, 45, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 12, 10, 5, tzinfo=timezone.utc)),
        ),
        template=_build_template(),
        warmup_bars=2,
    )

    response = use_case.execute(
        request=request,
        current_user=CurrentUser(user_id=UserId(UUID("00000000-0000-0000-0000-000000000111"))),
    )

    assert len(candle_feed.calls) == 1
    normalized_range = candle_feed.calls[0]
    assert normalized_range.start == UtcTimestamp(
        datetime(2026, 2, 16, 11, 58, tzinfo=timezone.utc)
    )
    assert normalized_range.end == UtcTimestamp(datetime(2026, 2, 16, 12, 11, tzinfo=timezone.utc))
    assert indicator_compute.compute_calls == 1
    assert response.total_indicator_compute_calls == 1
    assert response.warmup_bars == 2


def test_run_backtest_use_case_expands_pivot_signal_dependencies() -> None:
    """
    Verify use-case compute plan includes pivot wrapper dependencies for signal rules v1.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `structure.pivots` requires `structure.pivot_high` and `structure.pivot_low`.
    Raises:
        AssertionError: If dependency ids are missing or compute call count is incorrect.
    Side Effects:
        None.
    """
    candle_feed = _AlignedOnlyCandleFeed()
    indicator_compute = _NoOpIndicatorCompute()
    use_case = RunBacktestUseCase(
        candle_feed=candle_feed,
        indicator_compute=indicator_compute,
        strategy_reader=_UnusedStrategyReader(),
    )
    request = RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 2, 16, 12, 5, tzinfo=timezone.utc)),
        ),
        template=_build_pivots_template(),
    )

    response = use_case.execute(
        request=request,
        current_user=CurrentUser(user_id=UserId(UUID("00000000-0000-0000-0000-000000000111"))),
    )

    assert response.total_indicator_compute_calls == 3
    assert indicator_compute.compute_calls == 3
    assert indicator_compute.compute_indicator_ids == [
        "structure.pivots",
        "structure.pivot_high",
        "structure.pivot_low",
    ]


def _build_template() -> RunBacktestTemplate:
    """
    Build deterministic minimal template payload for run use-case tests.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Valid template-mode request payload.
    Assumptions:
        One indicator grid/selection is enough for skeleton compute wiring.
    Raises:
        ValueError: If any primitive/grid invariant fails.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={
                    "window": ExplicitValuesSpec(name="window", values=(20,)),
                },
            ),
        ),
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ma.sma",
                inputs={"source": "close"},
                params={"window": 20},
            ),
        ),
    )


def _build_pivots_template() -> RunBacktestTemplate:
    """
    Build template payload that includes `structure.pivots` for dependency expansion tests.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Valid template with one pivots grid and selection.
    Assumptions:
        Pivots params use explicit integer values for left/right windows.
    Raises:
        ValueError: If primitive/grid invariants fail.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("structure.pivots"),
                params={
                    "left": ExplicitValuesSpec(name="left", values=(3,)),
                    "right": ExplicitValuesSpec(name="right", values=(2,)),
                },
            ),
        ),
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="structure.pivots",
                inputs={},
                params={"left": 3, "right": 2},
            ),
        ),
    )


def _build_dense_1m_from_time_range(*, time_range: TimeRange) -> CandleArrays:
    """
    Build deterministic dense `1m` candles for supplied aligned range.

    Args:
        time_range: Requested aligned time range.
    Returns:
        CandleArrays: Dense finite `1m` arrays covering entire range.
    Assumptions:
        Duration is divisible by one minute.
    Raises:
        ValueError: If duration is not divisible by one minute.
    Side Effects:
        Allocates numpy arrays.
    """
    duration = time_range.duration()
    if duration % _ONE_MINUTE != timedelta(0):
        raise ValueError("time_range duration must be divisible by one minute")

    count = int(duration // _ONE_MINUTE)
    start_ms = _to_epoch_millis(time_range.start.value)
    ts_open = np.arange(count, dtype=np.int64) * np.int64(60_000) + np.int64(start_ms)
    values = np.arange(1, count + 1, dtype=np.float32)
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=time_range,
        timeframe=Timeframe("1m"),
        ts_open=np.ascontiguousarray(ts_open, dtype=np.int64),
        open=np.ascontiguousarray(values, dtype=np.float32),
        high=np.ascontiguousarray(values, dtype=np.float32),
        low=np.ascontiguousarray(values, dtype=np.float32),
        close=np.ascontiguousarray(values, dtype=np.float32),
        volume=np.ascontiguousarray(values, dtype=np.float32),
    )


def _to_epoch_millis(dt: datetime) -> int:
    """
    Convert timezone-aware datetime to epoch milliseconds.

    Args:
        dt: Timezone-aware datetime.
    Returns:
        int: Epoch milliseconds.
    Assumptions:
        Input datetime uses timezone information.
    Raises:
        ValueError: If datetime is naive.
    Side Effects:
        None.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    delta = dt.astimezone(timezone.utc) - _EPOCH_UTC
    return int(delta // timedelta(milliseconds=1))
