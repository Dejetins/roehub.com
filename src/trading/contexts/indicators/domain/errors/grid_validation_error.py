from __future__ import annotations


class GridValidationError(ValueError):
    """
    Raised when a grid definition violates domain invariants.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: ..specifications.grid_spec, ...application.ports.compute.indicator_compute
    """
