import pytest

from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, RangeValuesSpec


def test_explicit_values_spec_requires_non_empty_values() -> None:
    """
    Verify rejection when explicit values are empty.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        ExplicitValuesSpec requires at least one value.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        ExplicitValuesSpec(name="window", values=())


def test_range_values_spec_materializes_inclusive_stop_values() -> None:
    """
    Verify inclusive stop semantics for range materialization.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        RangeValuesSpec includes stop value when step lands on it.
    Raises:
        AssertionError: If materialized values differ from expected inclusive sequence.
    Side Effects:
        None.
    """
    spec = RangeValuesSpec(name="window", start=2, stop_inclusive=6, step=2)
    assert spec.materialize() == (2, 4, 6)


def test_range_values_spec_rejects_non_positive_step() -> None:
    """
    Verify rejection when range step is non-positive.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        RangeValuesSpec requires step > 0.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        RangeValuesSpec(name="window", start=2, stop_inclusive=10, step=0)
