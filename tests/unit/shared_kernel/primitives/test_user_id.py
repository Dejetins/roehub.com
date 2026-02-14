from __future__ import annotations

from uuid import uuid4

import pytest

from trading.shared_kernel.primitives import UserId


def test_user_id_from_string_parses_uuid() -> None:
    """
    Verify UserId parses canonical UUID string values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        UUID string is valid.
    Raises:
        AssertionError: If parsing fails unexpectedly.
    Side Effects:
        None.
    """
    raw = str(uuid4())

    user_id = UserId.from_string(raw)

    assert str(user_id) == raw



def test_user_id_from_string_rejects_blank_value() -> None:
    """
    Verify UserId rejects blank UUID input.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Blank value is always invalid.
    Raises:
        AssertionError: If blank value does not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError):
        UserId.from_string(" ")
