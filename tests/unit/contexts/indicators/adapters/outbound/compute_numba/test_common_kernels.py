from __future__ import annotations

import pytest

from trading.contexts.indicators.adapters.outbound.compute_numba.kernels import (
    check_total_budget_or_raise,
    estimate_tensor_bytes,
    estimate_total_bytes,
)
from trading.contexts.indicators.domain.errors import ComputeBudgetExceeded


def test_estimate_total_bytes_uses_factor_and_fixed_reserve() -> None:
    """
    Verify total-bytes estimate uses proportional and fixed workspace reserve.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Formula is `bytes_out + ceil(bytes_out * factor) + fixed`.
    Raises:
        AssertionError: If estimate formula regresses.
    Side Effects:
        None.
    """
    bytes_out = estimate_tensor_bytes(t=100, variants=20)
    total = estimate_total_bytes(
        bytes_out=bytes_out,
        workspace_factor=0.20,
        workspace_fixed_bytes=64,
    )
    assert bytes_out == 8_000
    assert total == 9_664


def test_check_total_budget_or_raise_returns_deterministic_error_payload() -> None:
    """
    Verify budget guard raises `ComputeBudgetExceeded` with stable detail keys.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Details payload order is contractually stable for API diagnostics.
    Raises:
        AssertionError: If guard does not raise expected error or payload order.
    Side Effects:
        None.
    """
    with pytest.raises(ComputeBudgetExceeded) as exc_info:
        check_total_budget_or_raise(
            t=1_024,
            variants=2_048,
            bytes_out=8_388_608,
            bytes_total_est=10_066_944,
            max_compute_bytes_total=9_437_184,
        )

    details = exc_info.value.details
    assert list(details.keys()) == [
        "T",
        "V",
        "bytes_out",
        "bytes_total_est",
        "max_compute_bytes_total",
    ]
    assert details["T"] == 1_024
    assert details["V"] == 2_048
    assert details["bytes_out"] == 8_388_608
    assert details["bytes_total_est"] == 10_066_944
    assert details["max_compute_bytes_total"] == 9_437_184
