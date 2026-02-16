from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from trading.contexts.strategy.domain.entities import StrategySpecV1


def estimate_strategy_warmup_bars(*, spec: StrategySpecV1) -> int:
    """
    Estimate deterministic warmup bars count from strategy indicators payload.

    Args:
        spec: Immutable strategy specification containing indicators configuration.
    Returns:
        int: Deterministic warmup bars estimate (`>= 1`).
    Assumptions:
        Positive numeric indicator params represent lookback-like warmup requirements in v1.
    Raises:
        ValueError: If indicator payload shape unexpectedly violates StrategySpec invariants.
    Side Effects:
        None.
    """
    candidates: list[int] = []
    for indicator in spec.indicators:
        params = indicator.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError("Strategy indicator params must be mapping")
        candidates.extend(_collect_warmup_candidates(value=params))

    if not candidates:
        return 1
    return max(candidates)



def _collect_warmup_candidates(*, value: Any) -> list[int]:
    """
    Collect positive numeric candidates from nested indicator params payload.

    Args:
        value: Nested JSON-compatible params payload.
    Returns:
        list[int]: Positive integer warmup candidate values.
    Assumptions:
        Dict keys are sorted to preserve deterministic traversal order.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, bool):
        return []

    if isinstance(value, int):
        return [value] if value > 0 else []

    if isinstance(value, float):
        if value <= 0 or math.isnan(value) or math.isinf(value):
            return []
        return [int(math.ceil(value))]

    if isinstance(value, Mapping):
        candidates: list[int] = []
        sorted_items = sorted(value.items(), key=lambda item: str(item[0]))
        for _, item_value in sorted_items:
            candidates.extend(_collect_warmup_candidates(value=item_value))
        return candidates

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        candidates = []
        for item in value:
            candidates.extend(_collect_warmup_candidates(value=item))
        return candidates

    return []
