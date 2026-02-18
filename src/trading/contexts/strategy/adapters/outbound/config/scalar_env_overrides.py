from __future__ import annotations

from typing import Mapping

_TRUE_LITERALS = frozenset({"1", "true", "yes", "on"})
_FALSE_LITERALS = frozenset({"0", "false", "no", "off"})


def parse_bool_literal(*, raw_value: str, key: str) -> bool:
    """
    Parse strict boolean env literal used by runtime config overrides.

    Docs:
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - apps/api/wiring/modules/identity.py
      - src/trading/contexts/strategy/adapters/outbound/config/strategy_runtime_config.py

    Args:
        raw_value: Raw environment value.
        key: Environment variable key for deterministic diagnostics.
    Returns:
        bool: Parsed boolean value.
    Assumptions:
        Accepted true literals are `1,true,yes,on`.
        Accepted false literals are `0,false,no,off`.
    Raises:
        ValueError: If value is not one of strict boolean literals.
    Side Effects:
        None.
    """
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_LITERALS:
        return True
    if normalized in _FALSE_LITERALS:
        return False
    raise ValueError(
        f"{key} must be a boolean literal (1/0/true/false/yes/no/on/off), "
        f"got {raw_value!r}"
    )


def resolve_bool_override(
    *,
    environ: Mapping[str, str],
    key: str,
    default: bool,
) -> bool:
    """
    Resolve strict boolean override from environment mapping.

    Docs:
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/config/strategy_runtime_config.py
      - src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py

    Args:
        environ: Runtime environment mapping.
        key: Environment variable key.
        default: Fallback when override is missing or blank.
    Returns:
        bool: Resolved boolean value.
    Assumptions:
        Blank env values are treated as "not set".
    Raises:
        ValueError: If env value is present but not parseable as strict boolean.
    Side Effects:
        None.
    """
    raw_override = environ.get(key, "").strip()
    if not raw_override:
        return default
    return parse_bool_literal(raw_value=raw_override, key=key)


def resolve_positive_int_override(
    *,
    environ: Mapping[str, str],
    key: str,
    default: int,
) -> int:
    """
    Resolve positive integer override from environment mapping.

    Docs:
      - docs/architecture/strategy/strategy-runtime-config-v1.md
    Related:
      - src/trading/contexts/strategy/adapters/outbound/config/strategy_runtime_config.py
      - src/trading/contexts/strategy/adapters/outbound/config/live_runner_runtime_config.py

    Args:
        environ: Runtime environment mapping.
        key: Environment variable key.
        default: Fallback when override is missing or blank.
    Returns:
        int: Resolved positive integer value.
    Assumptions:
        Integer parsing uses base-10 representation.
    Raises:
        ValueError: If value is not an integer or is <= 0.
    Side Effects:
        None.
    """
    raw_override = environ.get(key, "").strip()
    if not raw_override:
        return default
    try:
        parsed = int(raw_override, 10)
    except ValueError as error:
        raise ValueError(f"{key} must be int, got {raw_override!r}") from error
    if parsed <= 0:
        raise ValueError(f"{key} must be > 0, got {parsed}")
    return parsed


__all__ = [
    "parse_bool_literal",
    "resolve_bool_override",
    "resolve_positive_int_override",
]
