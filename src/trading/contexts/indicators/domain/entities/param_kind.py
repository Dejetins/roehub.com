from __future__ import annotations

from enum import Enum


class ParamKind(str, Enum):
    """
    Supported indicator parameter kinds.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: .param_def, ..specifications.grid_param_spec
    """

    INT = "int"
    FLOAT = "float"
    ENUM = "enum"
