from __future__ import annotations

import pytest

from trading.shared_kernel.primitives import PaidLevel


def test_paid_level_accepts_all_supported_levels() -> None:
    """
    Verify PaidLevel accepts all literal values from identity v1 contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Allowed literals are free/base/pro/ultra.
    Raises:
        AssertionError: If any valid level is rejected.
    Side Effects:
        None.
    """
    assert str(PaidLevel("free")) == "free"
    assert str(PaidLevel("base")) == "base"
    assert str(PaidLevel("pro")) == "pro"
    assert str(PaidLevel("ultra")) == "ultra"



def test_paid_level_rejects_unknown_level() -> None:
    """
    Verify PaidLevel rejects unsupported literal values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Unknown levels must fail fast with ValueError.
    Raises:
        AssertionError: If invalid level is accepted.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        PaidLevel("enterprise")
