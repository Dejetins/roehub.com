from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from trading.contexts.strategy.domain.entities.strategy_spec_v1 import StrategySpecV1
from trading.shared_kernel.primitives import UserId


def generate_strategy_name(*, user_id: UserId, spec: StrategySpecV1) -> str:
    """
    Build deterministic Strategy v1 name from `(user_id + spec_json)` and human summary.

    Example format:
    - `BTCUSDT 1m [MA(20,50)] #A1B2C3D4`

    Args:
        user_id: Strategy owner identifier.
        spec: Immutable v1 strategy spec.
    Returns:
        str: Deterministic strategy name with stable hash suffix.
    Assumptions:
        Name must be byte-stable for identical `user_id` and canonicalized `spec_json` payload.
    Raises:
        TypeError: If canonical payload cannot be serialized to JSON.
    Side Effects:
        None.

    Docs:
      - docs/architecture/strategy/strategy-domain-spec-immutable-storage-runs-events-v1.md
    Related:
      - src/trading/contexts/strategy/domain/entities/strategy.py
      - src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py
      - tests/unit/contexts/strategy/domain/test_strategy_domain.py
    """
    human_signature = _build_indicator_signature(indicators=spec.indicators)
    human_part = f"{spec.instrument_id.symbol} {spec.timeframe.code} [{human_signature}]"

    digest_input = {
        "user_id": str(user_id),
        "spec_json": spec.to_json(),
    }
    canonical = json.dumps(digest_input, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()[:8]
    return f"{human_part} #{digest}"



def _build_indicator_signature(*, indicators: Sequence[Mapping[str, Any]]) -> str:
    """
    Build deterministic compact indicator signature used by human-readable strategy name.

    Args:
        indicators: Normalized indicator payload list from immutable StrategySpecV1.
    Returns:
        str: Signature such as `MA(20,50)` or `MA(20,50), RSI(14)`.
    Assumptions:
        Every indicator has been validated by `StrategySpecV1`.
    Raises:
        ValueError: If list is unexpectedly empty.
    Side Effects:
        None.
    """
    if not indicators:
        raise ValueError(
            "Strategy name requires at least one indicator for deterministic human-part"
        )
    return ", ".join(_render_indicator(indicator=indicator) for indicator in indicators)



def _render_indicator(*, indicator: Mapping[str, Any]) -> str:
    """
    Render one indicator mapping into deterministic concise signature fragment.

    Args:
        indicator: Indicator payload with identifier and parameter mapping.
    Returns:
        str: Signature fragment, e.g. `MA(20,50)`.
    Assumptions:
        Identifier fields and params mapping were shape-validated upstream.
    Raises:
        ValueError: If identifier field is missing.
    Side Effects:
        None.
    """
    name_raw = indicator.get("name") or indicator.get("kind") or indicator.get("id")
    if not isinstance(name_raw, str) or not name_raw.strip():
        raise ValueError("Indicator payload requires non-empty name/kind/id")

    params_raw = indicator.get("params", {})
    if not isinstance(params_raw, Mapping):
        raise ValueError("Indicator params payload must be a mapping")

    normalized_name = name_raw.strip().upper()
    if not params_raw:
        return normalized_name

    rendered_values = ",".join(str(params_raw[key]) for key in sorted(params_raw))
    return f"{normalized_name}({rendered_values})"
