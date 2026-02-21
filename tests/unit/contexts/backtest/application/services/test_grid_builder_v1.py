from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Mapping, cast

import numpy as np
import pytest

from trading.contexts.backtest.application.dto import BacktestRiskGridSpec, RunBacktestTemplate
from trading.contexts.backtest.application.services import BacktestGridBuilderV1
from trading.contexts.indicators.application.dto import (
    CandleArrays,
    ComputeRequest,
    EstimateResult,
    IndicatorTensor,
)
from trading.contexts.indicators.domain.entities import AxisDef, IndicatorId
from trading.contexts.indicators.domain.specifications import (
    ExplicitValuesSpec,
    GridParamSpec,
    GridSpec,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UtcTimestamp,
)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_ONE_MINUTE = timedelta(minutes=1)


class _EstimateOnlyIndicatorCompute:
    """
    IndicatorCompute fake that materializes estimate axes directly from grid specs.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/indicators/application/ports/compute/indicator_compute.py
      - tests/unit/contexts/backtest/application/services/test_grid_builder_v1.py
    """

    def estimate(self, grid: GridSpec, *, max_variants_guard: int) -> EstimateResult:
        """
        Materialize deterministic estimate result from explicit request/default axis values.

        Args:
            grid: Indicator grid payload.
            max_variants_guard: Variants guard threshold.
        Returns:
            EstimateResult: Materialized estimate for staged grid builder tests.
        Assumptions:
            Fixtures use explicit axis values only.
        Raises:
            ValueError: If variants exceed provided guard.
        Side Effects:
            None.
        """
        axes: list[AxisDef] = []
        variants = 1

        if grid.source is not None:
            source_values = tuple(str(value) for value in grid.source.materialize())
            axes.append(AxisDef(name="source", values_enum=source_values))
            variants *= len(source_values)

        for param_name in sorted(grid.params.keys()):
            values = tuple(grid.params[param_name].materialize())
            axes.append(_axis_def(name=param_name, values=values))
            variants *= len(values)

        if variants > max_variants_guard:
            raise ValueError("variants exceed max_variants_guard")

        return EstimateResult(
            indicator_id=grid.indicator_id,
            axes=tuple(axes),
            variants=variants,
            max_variants_guard=max_variants_guard,
        )

    def compute(self, req: ComputeRequest) -> IndicatorTensor:
        """
        Return placeholder value because compute path is not used in these tests.

        Args:
            req: Compute request payload.
        Returns:
            object: Placeholder object.
        Assumptions:
            Staged grid builder depends only on `estimate`.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = req
        return cast(IndicatorTensor, object())

    def warmup(self) -> None:
        """
        Provide no-op warmup implementation for protocol compatibility.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Warmup is out of scope for staged grid builder tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        return None


class _DefaultsProvider:
    """
    Defaults provider fake exposing deterministic compute/signal defaults by indicator id.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/staged_runner.py
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - tests/unit/contexts/backtest/application/services/test_grid_builder_v1.py
    """

    def __init__(self) -> None:
        """
        Initialize deterministic defaults maps used by test scenarios.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Defaults are static in-memory fixtures.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._compute_defaults = {
            "ma.sma": GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                source=ExplicitValuesSpec(name="source", values=("close", "high")),
                params={
                    "window": ExplicitValuesSpec(name="window", values=(10, 20)),
                },
            )
        }
        self._signal_defaults: dict[str, dict[str, GridParamSpec]] = {
            "ma.sma": {
                "long_threshold": ExplicitValuesSpec(name="long_threshold", values=(10, 20)),
            }
        }

    def compute_defaults(self, *, indicator_id: str) -> GridSpec | None:
        """
        Resolve deterministic compute defaults for requested indicator id.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            GridSpec | None: Defaults grid when available.
        Assumptions:
            Missing indicators return `None`.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._compute_defaults.get(indicator_id)

    def signal_param_defaults(self, *, indicator_id: str) -> Mapping[str, GridParamSpec]:
        """
        Resolve deterministic signal parameter defaults by indicator id.

        Args:
            indicator_id: Indicator identifier.
        Returns:
            Mapping[str, GridParamSpec]: Deterministic signal defaults map.
        Assumptions:
            Missing indicator returns empty mapping.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._signal_defaults.get(indicator_id, {})


def test_grid_builder_v1_enumerates_stage_a_deterministically() -> None:
    """
    Verify Stage A variants are deterministic with request/default grid merge semantics.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Request overrides defaults while missing axes are filled from defaults provider.
    Raises:
        AssertionError: If stage totals or enumeration ordering are non-deterministic.
    Side Effects:
        None.
    """
    builder = BacktestGridBuilderV1()
    context_a = builder.build(
        template=_template_without_source_with_request_signal_values(),
        candles=_build_candles(bars=30),
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        preselect=3,
        defaults_provider=_DefaultsProvider(),
    )
    context_b = builder.build(
        template=_template_without_source_with_request_signal_values(),
        candles=_build_candles(bars=30),
        indicator_compute=_EstimateOnlyIndicatorCompute(),
        preselect=3,
        defaults_provider=_DefaultsProvider(),
    )

    keys_a = tuple(variant.base_variant_key for variant in context_a.iter_stage_a_variants())
    keys_b = tuple(variant.base_variant_key for variant in context_b.iter_stage_a_variants())

    assert context_a.stage_a_variants_total == 24
    assert context_a.stage_b_variants_total == 3
    assert keys_a == keys_b


def test_grid_builder_v1_raises_stage_a_variants_guard() -> None:
    """
    Verify Stage A variants guard raises deterministic 422 payload with `stage_a` details.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage A guard is checked before Stage B expansion.
    Raises:
        AssertionError: If raised payload does not match deterministic guard contract.
    Side Effects:
        None.
    """
    builder = BacktestGridBuilderV1()

    with pytest.raises(RoehubError) as exc_info:
        builder.build(
            template=_template_without_source_with_request_signal_values(),
            candles=_build_candles(bars=10),
            indicator_compute=_EstimateOnlyIndicatorCompute(),
            preselect=3,
            defaults_provider=_DefaultsProvider(),
            max_variants_per_compute=10,
        )

    error = exc_info.value
    assert error.code == "validation_error"
    assert error.details == {
        "error": "max_variants_per_compute_exceeded",
        "max_variants_per_compute": 10,
        "stage": "stage_a",
        "total_variants": 24,
    }


def test_grid_builder_v1_raises_stage_b_variants_guard() -> None:
    """
    Verify Stage B variants guard uses `shortlist_len * sl_variants * tp_variants` formula.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage B guard is checked after Stage A totals and before scoring.
    Raises:
        AssertionError: If raised payload does not include `stage_b` deterministic details.
    Side Effects:
        None.
    """
    template = RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={
                    "window": ExplicitValuesSpec(name="window", values=(5, 10, 15, 20, 25)),
                },
                source=ExplicitValuesSpec(name="source", values=("close",)),
            ),
        ),
        risk_grid=BacktestRiskGridSpec(
            sl_enabled=True,
            tp_enabled=True,
            sl=ExplicitValuesSpec(name="sl", values=(1.0, 2.0, 3.0)),
            tp=ExplicitValuesSpec(name="tp", values=(2.0, 4.0)),
        ),
    )
    builder = BacktestGridBuilderV1()

    with pytest.raises(RoehubError) as exc_info:
        builder.build(
            template=template,
            candles=_build_candles(bars=10),
            indicator_compute=_EstimateOnlyIndicatorCompute(),
            preselect=4,
            max_variants_per_compute=20,
        )

    error = exc_info.value
    assert error.code == "validation_error"
    assert error.details == {
        "error": "max_variants_per_compute_exceeded",
        "max_variants_per_compute": 20,
        "stage": "stage_b",
        "total_variants": 24,
    }


def test_grid_builder_v1_raises_memory_guard() -> None:
    """
    Verify memory guard uses indicators estimator policy and raises deterministic 422 payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Memory estimate includes candles bytes, indicator tensors, and reserve.
    Raises:
        AssertionError: If payload does not match deterministic memory guard contract.
    Side Effects:
        None.
    """
    builder = BacktestGridBuilderV1()

    with pytest.raises(RoehubError) as exc_info:
        builder.build(
            template=_template_without_source_with_request_signal_values(),
            candles=_build_candles(bars=1000),
            indicator_compute=_EstimateOnlyIndicatorCompute(),
            preselect=3,
            defaults_provider=_DefaultsProvider(),
            max_compute_bytes_total=1,
        )

    error = exc_info.value
    assert error.code == "validation_error"
    assert error.details is not None
    assert error.details["error"] == "max_compute_bytes_total_exceeded"
    assert error.details["stage"] == "stage_a"
    assert error.details["max_compute_bytes_total"] == 1
    assert isinstance(error.details["estimated_memory_bytes"], int)


def _template_without_source_with_request_signal_values() -> RunBacktestTemplate:
    """
    Build template where compute source is provided by defaults and signal grid overrides defaults.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Deterministic template fixture for defaults-merge tests.
    Assumptions:
        Request sets `window` and one signal axis while defaults provide `source` and the
        second signal axis.
    Raises:
        ValueError: If template payload violates DTO invariants.
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
                    "window": ExplicitValuesSpec(name="window", values=(50, 100)),
                },
            ),
        ),
        signal_grids={
            "ma.sma": {
                "short_threshold": ExplicitValuesSpec(name="short_threshold", values=(60, 70, 80))
            }
        },
    )


def _build_candles(*, bars: int) -> CandleArrays:
    """
    Build deterministic dense candle arrays fixture with specified number of bars.

    Args:
        bars: Number of `1m` bars in fixture.
    Returns:
        CandleArrays: Dense finite candle arrays.
    Assumptions:
        Bars count is positive.
    Raises:
        ValueError: If bars count is not positive.
    Side Effects:
        Allocates numpy arrays.
    """
    if bars <= 0:
        raise ValueError("bars must be > 0")

    start = datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc)
    end = start + (bars * _ONE_MINUTE)
    start_ms = _to_epoch_millis(start)

    ts_open = np.arange(bars, dtype=np.int64) * np.int64(60_000) + np.int64(start_ms)
    values = np.arange(1, bars + 1, dtype=np.float32)
    return CandleArrays(
        market_id=MarketId(1),
        symbol=Symbol("BTCUSDT"),
        time_range=TimeRange(start=UtcTimestamp(start), end=UtcTimestamp(end)),
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
        Datetime is timezone-aware.
    Raises:
        ValueError: If datetime is naive.
    Side Effects:
        None.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    delta = dt.astimezone(timezone.utc) - _EPOCH_UTC
    return int(delta // timedelta(milliseconds=1))


def _axis_def(name: str, values: tuple[int | float | str, ...]) -> AxisDef:
    """
    Build `AxisDef` with value family inferred from axis materialized values.

    Args:
        name: Axis name.
        values: Materialized scalar values.
    Returns:
        AxisDef: Deterministic axis definition.
    Assumptions:
        Axis values are homogeneous and non-empty.
    Raises:
        ValueError: If values are empty or unsupported scalar type is encountered.
    Side Effects:
        None.
    """
    if len(values) == 0:
        raise ValueError("axis values must be non-empty")

    first = values[0]
    if isinstance(first, str):
        return AxisDef(name=name, values_enum=tuple(str(value) for value in values))
    if isinstance(first, int):
        return AxisDef(name=name, values_int=tuple(int(value) for value in values))
    if isinstance(first, float):
        return AxisDef(name=name, values_float=tuple(float(value) for value in values))
    raise ValueError(f"unsupported axis value type: {type(first).__name__}")
