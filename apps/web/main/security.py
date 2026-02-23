from __future__ import annotations

from urllib.parse import urlsplit

_DEFAULT_NEXT_PATH = "/"



def sanitize_next_path(*, raw_next: str | None, default_path: str = _DEFAULT_NEXT_PATH) -> str:
    """
    Sanitize login redirect target and block open-redirect attempts.

    Args:
        raw_next: User-supplied redirect target from query parameter.
        default_path: Fallback safe path used when `raw_next` is invalid.
    Returns:
        str: Safe relative redirect path.
    Assumptions:
        Only relative paths starting with `/` are allowed by web auth UX contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    candidate = "" if raw_next is None else raw_next.strip()
    if not candidate:
        return default_path

    parsed_target = urlsplit(candidate)
    if parsed_target.scheme or parsed_target.netloc:
        return default_path
    if not candidate.startswith("/"):
        return default_path
    if candidate.startswith("//"):
        return default_path
    return candidate
