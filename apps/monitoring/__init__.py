"""Monitoring services and exporters package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .clickhouse_exporter import (
        ClickHouseExporterCollector,
        ClickHouseMetricsSnapshot,
        HttpClickHouseMetricsClient,
        main,
    )

__all__ = [
    "ClickHouseExporterCollector",
    "ClickHouseMetricsSnapshot",
    "HttpClickHouseMetricsClient",
    "main",
]


def __getattr__(name: str) -> Any:
    """
    Resolve monitoring exports lazily from `clickhouse_exporter`.

    Args:
        name: Export name requested from this package module.
    Returns:
        Any: Exported symbol from `apps.monitoring.clickhouse_exporter`.
    Assumptions:
        Public package exports stay aligned with `__all__`.
    Raises:
        AttributeError: If `name` is not a supported package export.
    Side Effects:
        Imports `apps.monitoring.clickhouse_exporter` on first access.
    """
    if name in __all__:
        module = import_module(".clickhouse_exporter", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
