"""
API application package.

This module exposes lazy exports for `app` and `create_app` to avoid
side effects during package import in tests and tooling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .main import app, create_app

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    """
    Lazily resolve package exports without eager FastAPI startup.

    Args:
        name: Requested attribute name.
    Returns:
        Any: Exported object for supported lazy names.
    Assumptions:
        `app` and `create_app` live in `apps.api.main`.
    Raises:
        AttributeError: If the requested attribute is not supported.
    Side Effects:
        Imports `apps.api.main` only when one of the lazy exports is requested.
    """
    if name == "app":
        from .main import app

        return app
    if name == "create_app":
        from .main import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
