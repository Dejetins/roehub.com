"""Monitoring services and exporters package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ClickHouseExporterCollector",
    "ClickHouseMetricsSnapshot",
    "HttpClickHouseMetricsClient",
    "main",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".clickhouse_exporter", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
