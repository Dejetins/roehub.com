import pytest

from trading.contexts.indicators.domain.entities import AxisDef


def test_axis_def_accepts_exactly_one_value_family() -> None:
    """
    Verify axis acceptance with exactly one value family.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Axis one-of invariant allows one and only one populated value field.
    Raises:
        AssertionError: If axis length is unexpected.
    Side Effects:
        None.
    """
    axis = AxisDef(name="window", values_int=(5, 10, 20))
    assert axis.length() == 3


def test_axis_def_rejects_when_all_value_families_missing() -> None:
    """
    Verify rejection when all value families are missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Axis one-of invariant disallows zero populated value fields.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        AxisDef(name="window")


def test_axis_def_rejects_when_multiple_value_families_provided() -> None:
    """
    Verify rejection when multiple value families are populated.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Axis one-of invariant disallows multiple populated value fields.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        AxisDef(name="window", values_int=(5, 10), values_float=(5.0, 10.0))
