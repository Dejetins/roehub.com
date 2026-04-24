from __future__ import annotations

import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping


def normalize_persisted_summary_metrics_v2(
    *,
    metrics: Mapping[str, Any],
) -> Mapping[str, float]:
    """Keep only finite numeric summary metrics in persisted JSON-facing payloads."""
    normalized: dict[str, float] = {}
    for raw_key, raw_value in metrics.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("persisted summary metric keys must be non-empty")
        if raw_value is None:
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise ValueError(f"persisted summary metric {key!r} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            continue
        normalized[key] = value
    return MappingProxyType(normalized)


__all__ = ["normalize_persisted_summary_metrics_v2"]
