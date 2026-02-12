from __future__ import annotations

from datetime import datetime, timezone

import pytest

from trading.contexts.indicators.application.dto import (
    ExplicitValuesSpec,
    GridSpec,
    MergedIndicatorView,
    RangeValuesSpec,
)
from trading.contexts.indicators.application.ports.registry import IndicatorRegistry
from trading.contexts.indicators.application.services import BatchEstimator, GridBuilder
from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.indicators.domain.entities import IndicatorDef, IndicatorId
from trading.contexts.indicators.domain.errors import GridValidationError, UnknownIndicatorError
from trading.shared_kernel.primitives import Timeframe, TimeRange, UtcTimestamp


class _RegistryStub(IndicatorRegistry):
    """
    Minimal deterministic registry stub for grid-builder unit tests.
    """

    def __init__(self, *, defs: tuple[IndicatorDef, ...]) -> None:
        """
        Build lookup map for deterministic indicator definition resolution.

        Args:
            defs: Hard indicator definitions.
        Returns:
            None.
        Assumptions:
            Indicator ids are unique in hard definitions.
        Raises:
            ValueError: If duplicate indicator id is found.
        Side Effects:
            None.
        """
        defs_by_id: dict[str, IndicatorDef] = {}
        for definition in defs:
            key = definition.indicator_id.value
            if key in defs_by_id:
                raise ValueError(f"duplicate indicator_id: {key}")
            defs_by_id[key] = definition
        self._defs_by_id = defs_by_id
        self._defs = defs

    def list_defs(self) -> tuple[IndicatorDef, ...]:
        """
        Return stable tuple of indicator definitions.

        Args:
            None.
        Returns:
            tuple[IndicatorDef, ...]: Hard definitions.
        Assumptions:
            Snapshot is immutable for test runtime.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._defs

    def get_def(self, indicator_id: IndicatorId) -> IndicatorDef:
        """
        Resolve one indicator definition by id.

        Args:
            indicator_id: Target indicator id.
        Returns:
            IndicatorDef: Matching hard definition.
        Assumptions:
            Indicator id normalization is handled by value object constructor.
        Raises:
            UnknownIndicatorError: If id is missing.
        Side Effects:
            None.
        """
        found = self._defs_by_id.get(indicator_id.value)
        if found is None:
            raise UnknownIndicatorError(f"unknown indicator_id: {indicator_id.value}")
        return found

    def list_merged(self) -> tuple[MergedIndicatorView, ...]:
        """
        Return empty merged payload for protocol completeness in tests.

        Args:
            None.
        Returns:
            tuple[MergedIndicatorView, ...]: Empty placeholder.
        Assumptions:
            Grid-builder tests do not use merged views.
        Raises:
            None.
        Side Effects:
            None.
        """
        return ()

    def get_merged(self, indicator_id: IndicatorId) -> MergedIndicatorView:
        """
        Raise lookup error because merged views are not used in these tests.

        Args:
            indicator_id: Target indicator id.
        Returns:
            MergedIndicatorView: Never returns.
        Assumptions:
            Merged-view contract is irrelevant for grid-builder unit tests.
        Raises:
            UnknownIndicatorError: Always.
        Side Effects:
            None.
        """
        raise UnknownIndicatorError(f"unknown indicator_id: {indicator_id.value}")


def _builder() -> GridBuilder:
    """
    Build GridBuilder with deterministic hard-definition registry stub.

    Args:
        None.
    Returns:
        GridBuilder: Ready service instance for tests.
    Assumptions:
        `all_defs()` ordering is deterministic.
    Raises:
        ValueError: If hard definitions contain duplicates.
    Side Effects:
        None.
    """
    registry = _RegistryStub(defs=all_defs())
    return GridBuilder(registry=registry)


def _time_range_1h() -> TimeRange:
    """
    Build deterministic one-hour UTC half-open range `[10:00, 11:00)`.

    Args:
        None.
    Returns:
        TimeRange: One-hour range for memory estimate tests.
    Assumptions:
        Timestamp normalization is delegated to `UtcTimestamp`.
    Raises:
        ValueError: If timestamps violate `TimeRange` invariants.
    Side Effects:
        None.
    """
    start = UtcTimestamp(datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc))
    end = UtcTimestamp(datetime(2026, 2, 11, 11, 0, 0, tzinfo=timezone.utc))
    return TimeRange(start=start, end=end)


def test_grid_builder_preserves_explicit_value_order_for_source_and_params() -> None:
    """
    Verify explicit ordering is preserved exactly for source and numeric params.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Indicator `ma.sma` defines axes order (`source`, `window`).
    Raises:
        AssertionError: If values or variants are materialized in unexpected order.
    Side Effects:
        None.
    """
    builder = _builder()
    grid = GridSpec(
        indicator_id=IndicatorId("ma.sma"),
        params={
            "window": ExplicitValuesSpec(name="window", values=(20, 10, 30)),
        },
        source=ExplicitValuesSpec(name="source", values=("open", "close", "hlc3")),
    )

    materialized = builder.materialize_indicator(grid=grid)

    assert [axis.name for axis in materialized.axes] == ["source", "window"]
    assert materialized.axes[0].values == ("open", "close", "hlc3")
    assert materialized.axes[1].values == (20, 10, 30)
    assert materialized.variants == 9


def test_grid_builder_materializes_float_range_with_inclusive_formula() -> None:
    """
    Verify float range uses inclusive index-based materialization.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `trend.chandelier_exit.mult` supports float range with hard step `0.01`.
    Raises:
        AssertionError: If materialized range does not match deterministic sequence.
    Side Effects:
        None.
    """
    builder = _builder()
    grid = GridSpec(
        indicator_id=IndicatorId("trend.chandelier_exit"),
        params={
            "window": ExplicitValuesSpec(name="window", values=(22,)),
            "mult": RangeValuesSpec(name="mult", start=1.0, stop_inclusive=1.04, step=0.02),
        },
    )

    materialized = builder.materialize_indicator(grid=grid)
    mult_axis = next(axis for axis in materialized.axes if axis.name == "mult")

    assert tuple(float(value) for value in mult_axis.values) == pytest.approx((1.0, 1.02, 1.04))
    assert materialized.variants == 3


def test_grid_builder_requires_source_axis_when_indicator_is_parametrized() -> None:
    """
    Verify missing `source` axis fails for source-parameterized indicators.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `ma.sma` defines `source` in hard axes list.
    Raises:
        AssertionError: If expected validation error is not raised.
    Side Effects:
        None.
    """
    builder = _builder()
    grid = GridSpec(
        indicator_id=IndicatorId("ma.sma"),
        params={"window": ExplicitValuesSpec(name="window", values=(10,))},
    )

    with pytest.raises(GridValidationError, match="source axis is required"):
        builder.materialize_indicator(grid=grid)


def test_grid_builder_rejects_param_values_outside_hard_bounds() -> None:
    """
    Verify hard bounds validation rejects out-of-range explicit param values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `ma.sma.window` hard minimum is `2`.
    Raises:
        AssertionError: If expected validation error is not raised.
    Side Effects:
        None.
    """
    builder = _builder()
    grid = GridSpec(
        indicator_id=IndicatorId("ma.sma"),
        params={"window": ExplicitValuesSpec(name="window", values=(1,))},
        source=ExplicitValuesSpec(name="source", values=("close",)),
    )

    with pytest.raises(GridValidationError, match="below hard_min"):
        builder.materialize_indicator(grid=grid)


def test_batch_estimator_counts_total_variants_with_sl_tp_and_source_axes() -> None:
    """
    Verify total variants include indicators product multiplied by SL and TP variants.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Memory formula uses float32 policy and reserve `max(64MiB, 20%)`.
    Raises:
        AssertionError: If totals mismatch deterministic policy formulas.
    Side Effects:
        None.
    """
    builder = _builder()
    estimator = BatchEstimator(grid_builder=builder)
    indicator_grids = (
        GridSpec(
            indicator_id=IndicatorId("ma.sma"),
            params={
                "window": ExplicitValuesSpec(name="window", values=(10, 20, 30)),
            },
            source=ExplicitValuesSpec(name="source", values=("open", "close")),
        ),
        GridSpec(
            indicator_id=IndicatorId("trend.adx"),
            params={
                "window": RangeValuesSpec(name="window", start=10, stop_inclusive=11, step=1),
                "smoothing": ExplicitValuesSpec(name="smoothing", values=(14,)),
            },
        ),
    )

    estimate = estimator.estimate_batch(
        indicator_grids=indicator_grids,
        sl_spec=ExplicitValuesSpec(name="sl", values=(0.01, 0.02)),
        tp_spec=ExplicitValuesSpec(name="tp", values=(0.03, 0.04, 0.05)),
        time_range=_time_range_1h(),
        timeframe=Timeframe("1m"),
    )

    assert estimate.total_variants == 72
    assert estimate.estimated_memory_bytes == 67_112_464
