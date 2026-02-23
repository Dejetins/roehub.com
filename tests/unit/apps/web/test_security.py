from __future__ import annotations

from apps.web.main.security import sanitize_next_path


def test_sanitize_next_path_allows_relative_route() -> None:
    """
    Verify open-redirect guard keeps valid relative targets unchanged.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Relative path starting with `/` is valid redirect target.
    Raises:
        AssertionError: If sanitization modifies valid relative path.
    Side Effects:
        None.
    """
    assert sanitize_next_path(raw_next="/strategies") == "/strategies"



def test_sanitize_next_path_rejects_absolute_external_target() -> None:
    """
    Verify open-redirect guard rejects external absolute URL and falls back to `/`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        External absolute URLs must never be accepted as redirect targets.
    Raises:
        AssertionError: If external URL is not replaced with fallback value.
    Side Effects:
        None.
    """
    assert sanitize_next_path(raw_next="https://evil.example/path") == "/"



def test_sanitize_next_path_rejects_protocol_relative_target() -> None:
    """
    Verify open-redirect guard rejects protocol-relative targets that bypass origin checks.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Values starting with `//` are not safe local routes.
    Raises:
        AssertionError: If protocol-relative URL is accepted.
    Side Effects:
        None.
    """
    assert sanitize_next_path(raw_next="//evil.example/path") == "/"
