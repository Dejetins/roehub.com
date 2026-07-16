from __future__ import annotations

from apps.cli.commands.rest_catchup_1m import (
    _parse_optional_time_range,
    _sleep_between_instruments,
)


def test_sleep_between_instruments_calls_sleep_when_next_exists(monkeypatch) -> None:
    """
    Ensure CLI applies configured delay when another instrument is pending.

    Parameters:
    - monkeypatch: pytest fixture for monkeypatching module attributes.

    Returns:
    - None.
    """
    calls: list[float] = []

    def _fake_sleep(delay: float) -> None:
        """
        Capture requested delay without blocking test runtime.

        Parameters:
        - delay: requested sleep duration in seconds.

        Returns:
        - None.
        """
        calls.append(delay)

    monkeypatch.setattr("apps.cli.commands.rest_catchup_1m.time.sleep", _fake_sleep)

    _sleep_between_instruments(delay_s=2.0, has_next=True)

    assert calls == [2.0]


def test_sleep_between_instruments_skips_sleep_without_next_or_delay(monkeypatch) -> None:
    """
    Ensure CLI does not sleep when delay is zero or when there is no next instrument.

    Parameters:
    - monkeypatch: pytest fixture for monkeypatching module attributes.

    Returns:
    - None.
    """
    calls: list[float] = []

    def _fake_sleep(delay: float) -> None:
        """
        Capture unexpected sleep requests.

        Parameters:
        - delay: requested sleep duration in seconds.

        Returns:
        - None.
        """
        calls.append(delay)

    monkeypatch.setattr("apps.cli.commands.rest_catchup_1m.time.sleep", _fake_sleep)

    _sleep_between_instruments(delay_s=0.0, has_next=True)
    _sleep_between_instruments(delay_s=2.0, has_next=False)

    assert calls == []


def test_parse_optional_time_range_requires_complete_utc_pair() -> None:
    """Ensure bounded operator fills cannot accidentally become unbounded catchups."""
    parsed = _parse_optional_time_range(
        start="2026-07-14T00:00:00Z",
        end="2026-07-15T00:00:00Z",
    )

    assert parsed is not None
    assert parsed.start.value.isoformat() == "2026-07-14T00:00:00+00:00"

    try:
        _parse_optional_time_range(start="2026-07-14T00:00:00Z", end=None)
    except SystemExit as exc:
        assert str(exc) == "--start and --end must be provided together"
    else:
        raise AssertionError("partial bounded fill range must fail")
