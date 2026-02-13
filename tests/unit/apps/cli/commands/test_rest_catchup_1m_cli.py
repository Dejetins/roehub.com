from __future__ import annotations

from apps.cli.commands.rest_catchup_1m import _sleep_between_instruments


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
