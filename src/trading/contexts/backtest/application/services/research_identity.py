from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from trading.shared_kernel.primitives import OrganizationId

RESEARCH_REQUEST_NAMESPACE = "research-request/v1"


def build_research_content_hash(*, payload: Mapping[str, Any]) -> str:
    """Hash organization-neutral research content using canonical JSON."""
    canonical_json = json.dumps(
        _normalize_json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def build_research_idempotency_key_hash(
    *,
    organization_id: OrganizationId,
    idempotency_key: str,
) -> str:
    normalized = idempotency_key.strip()
    if not normalized:
        raise ValueError("research idempotency key must be non-empty")
    namespaced = f"{RESEARCH_REQUEST_NAMESPACE}\0{organization_id}\0{normalized}"
    return hashlib.sha256(namespaced.encode("utf-8")).hexdigest()


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be hashed")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "RESEARCH_REQUEST_NAMESPACE",
    "build_research_content_hash",
    "build_research_idempotency_key_hash",
]
