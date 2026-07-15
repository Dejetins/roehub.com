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
    from .operational_health import (
        OperationalHealthService,
        OperationalManifest,
        OperationalProbe,
        OperationalSnapshot,
        OperationalStatus,
        create_operational_health_app,
    )

_EXPORT_MODULE = {
    "ClickHouseExporterCollector": ".clickhouse_exporter",
    "ClickHouseMetricsSnapshot": ".clickhouse_exporter",
    "HttpClickHouseMetricsClient": ".clickhouse_exporter",
    "OperationalHealthService": ".operational_health",
    "OperationalManifest": ".operational_health",
    "OperationalProbe": ".operational_health",
    "OperationalSnapshot": ".operational_health",
    "OperationalStatus": ".operational_health",
    "create_operational_health_app": ".operational_health",
    "main": ".clickhouse_exporter",
}

__all__ = [
    "ClickHouseExporterCollector",
    "ClickHouseMetricsSnapshot",
    "HttpClickHouseMetricsClient",
    "OperationalHealthService",
    "OperationalManifest",
    "OperationalProbe",
    "OperationalSnapshot",
    "OperationalStatus",
    "create_operational_health_app",
    "main",
]


def __getattr__(name: str) -> Any:
    """
    Resolve monitoring exports lazily from `clickhouse_exporter`.

    Args:
        name: Export name requested from this package module.
    Returns:
        Any: Exported monitoring symbol.
    Assumptions:
        Public package exports stay aligned with `__all__`.
    Raises:
        AttributeError: If `name` is not a supported package export.
    Side Effects:
        Imports the owning monitoring module on first access.
    """
    module_name = _EXPORT_MODULE.get(name)
    if module_name is not None:
        module = import_module(module_name, __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
