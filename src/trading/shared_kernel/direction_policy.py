from __future__ import annotations

SHORT_DIRECTION_REQUIRES_FUTURES_MARKET = "short_direction_requires_futures_market"
SHORT_LIKE_DIRECTIONS = frozenset({"short", "long_short_reversal"})


def is_short_like_direction(direction: str) -> bool:
    return direction.strip().casefold() in SHORT_LIKE_DIRECTIONS


def short_direction_requires_futures_market(*, market_type: str, direction: str) -> bool:
    return market_type.strip().casefold() != "futures" and is_short_like_direction(direction)


__all__ = [
    "SHORT_DIRECTION_REQUIRES_FUTURES_MARKET",
    "SHORT_LIKE_DIRECTIONS",
    "is_short_like_direction",
    "short_direction_requires_futures_market",
]
