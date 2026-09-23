"""Strict typed raw-bit comparison, independent of public JSON normalization."""

from __future__ import annotations

import dataclasses
import datetime
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID


def encode(value: Any) -> Any:
    if isinstance(value, type):
        raise TypeError("Classes cannot be encoded")
    if dataclasses.is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, float):
        return {"f64": struct.pack(">d", value).hex()}
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Only string mapping keys are supported")
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, (Path, UUID, datetime.datetime, datetime.date)):
        return str(value)
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return {
            "ndarray": {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "raw_hex": value.tobytes(order="C").hex(),
            }
        }
    if hasattr(value, "tolist"):
        return encode(value.tolist())
    if hasattr(value, "item"):
        return encode(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(type(value).__name__)
