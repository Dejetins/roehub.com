from __future__ import annotations

from datetime import datetime, timezone


def ensure_tz_utc(value: datetime) -> datetime:
    """
    Normalize datetime to timezone-aware UTC.

    Parameters:
    - value: datetime from adapters, clocks, or storage.

    Returns:
    - UTC-aware datetime value.

    Assumptions/Invariants:
    - Naive datetime values are interpreted as UTC.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def floor_to_minute_utc(value: datetime) -> datetime:
    """
    Floor datetime to UTC minute bucket.

    Parameters:
    - value: datetime to normalize.

    Returns:
    - UTC datetime with seconds/microseconds zeroed.

    Assumptions/Invariants:
    - Input may be naive or timezone-aware.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    utc_value = ensure_tz_utc(value)
    return utc_value.replace(second=0, microsecond=0)


def minute_key(value: datetime) -> int:
    """
    Convert datetime into monotonic minute key.

    Parameters:
    - value: datetime value.

    Returns:
    - Integer minute key computed as `epoch_seconds // 60`.

    Assumptions/Invariants:
    - Naive datetimes are interpreted as UTC.

    Errors/Exceptions:
    - None.

    Side effects:
    - None.
    """
    return int(ensure_tz_utc(value).timestamp() // 60)

