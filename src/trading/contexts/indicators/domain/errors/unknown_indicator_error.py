from __future__ import annotations


class UnknownIndicatorError(LookupError):
    """
    Raised when an indicator id is not available in the registry.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: ..entities.indicator_id, ...application.ports.registry.indicator_registry
    """
