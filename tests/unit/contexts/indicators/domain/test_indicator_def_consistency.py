import pytest

from trading.contexts.indicators.domain.entities import (
    IndicatorDef,
    IndicatorId,
    InputSeries,
    OutputSpec,
    ParamDef,
    ParamKind,
)


def _base_params() -> tuple[ParamDef, ...]:
    """
    Build reusable parameter definitions for test fixtures.

    Args:
        None.
    Returns:
        tuple[ParamDef, ...]: Two deterministic parameter definitions.
    Assumptions:
        Parameter names are unique and valid for IndicatorDef tests.
    Raises:
        None.
    Side Effects:
        None.
    """
    return (
        ParamDef(name="window", kind=ParamKind.INT, hard_min=2, hard_max=200, step=1),
        ParamDef(name="mult", kind=ParamKind.FLOAT, hard_min=0.1, hard_max=5.0, step=0.1),
    )


def test_indicator_def_accepts_axes_bound_to_params_and_source() -> None:
    """
    Verify acceptance of parameter axes plus the special source axis.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        "source" is allowed as a special axis by the walking-skeleton contract.
    Raises:
        AssertionError: If normalized axes do not match expected tuple.
    Side Effects:
        None.
    """
    indicator = IndicatorDef(
        indicator_id=IndicatorId("sma_v1"),
        title="Simple Moving Average",
        inputs=(InputSeries.CLOSE,),
        params=_base_params(),
        axes=("window", "source"),
        output=OutputSpec(names=("value",)),
    )
    assert indicator.axes == ("window", "source")


def test_indicator_def_rejects_unknown_axis_name() -> None:
    """
    Verify rejection for an axis name outside allowed parameter/special axes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Unknown axis names must fail IndicatorDef validation.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        IndicatorDef(
            indicator_id=IndicatorId("sma_v1"),
            title="Simple Moving Average",
            inputs=(InputSeries.CLOSE,),
            params=_base_params(),
            axes=("window", "unknown_axis"),
            output=OutputSpec(names=("value",)),
        )
