import pytest

from trading.contexts.indicators.domain.entities import ParamDef, ParamKind


def test_param_def_accepts_numeric_bounds_and_step() -> None:
    """
    Verify valid numeric parameter invariants.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        INT parameters accept hard bounds with positive step.
    Raises:
        AssertionError: If construction unexpectedly fails or fields mismatch.
    Side Effects:
        None.
    """
    param = ParamDef(
        name="window",
        kind=ParamKind.INT,
        hard_min=2,
        hard_max=200,
        step=1,
        default=14,
    )
    assert param.name == "window"
    assert param.step == 1


def test_param_def_rejects_invalid_numeric_range() -> None:
    """
    Verify rejection when hard_min is greater than hard_max.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Numeric range must satisfy hard_min <= hard_max.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        ParamDef(
            name="window",
            kind=ParamKind.INT,
            hard_min=20,
            hard_max=10,
            step=1,
        )


def test_param_def_rejects_non_positive_numeric_step() -> None:
    """
    Verify rejection when numeric step is non-positive.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Numeric parameters require positive step.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        ParamDef(
            name="alpha",
            kind=ParamKind.FLOAT,
            hard_min=0.0,
            hard_max=1.0,
            step=0.0,
        )


def test_param_def_requires_non_empty_enum_values() -> None:
    """
    Verify rejection when enum parameter has empty values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        ENUM parameters require non-empty enum_values.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        ParamDef(
            name="source",
            kind=ParamKind.ENUM,
            enum_values=(),
        )


def test_param_def_rejects_enum_default_outside_values() -> None:
    """
    Verify rejection when enum default is absent in enum_values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        ENUM defaults must belong to the declared enum list.
    Raises:
        AssertionError: If ValueError is not raised.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        ParamDef(
            name="source",
            kind=ParamKind.ENUM,
            enum_values=("close", "hlc3"),
            default="open",
        )
