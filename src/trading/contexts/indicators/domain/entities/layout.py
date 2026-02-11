from __future__ import annotations

from enum import Enum


class Layout(str, Enum):
    """
    Layout orientation of indicator tensor values.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .axis_def, ...application.dto.indicator_tensor
    """

    TIME_MAJOR = "time_major"
    VARIANT_MAJOR = "variant_major"
